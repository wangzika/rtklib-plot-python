"""GIS, Web Map Tiles and 3D Google Earth KML / GeoJSON Exporter for RTKLIB-Plot-Python."""
from __future__ import annotations

import concurrent.futures
import io
import json
import math
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

CACHE_DIR = Path.home() / ".rtklib_tiles_cache"

TILE_PROVIDERS = {
    "Esri Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "CartoDB Dark": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    "CartoDB Light": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
}


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Convert WGS84 lat/lon in degrees to Web Mercator tile x, y indices."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Convert Web Mercator tile x, y to NW corner WGS84 lat/lon degrees."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def fetch_single_tile(provider_name: str, z: int, x: int, y: int, timeout: float = 3.0) -> Image.Image | None:
    """Download or retrieve a cached 256x256 map tile."""
    url_template = TILE_PROVIDERS.get(provider_name, TILE_PROVIDERS["Esri Satellite"])
    url = url_template.format(z=z, x=x, y=y)
    slug = provider_name.lower().replace(" ", "_")
    tile_file = CACHE_DIR / slug / str(z) / str(x) / f"{y}.png"

    if tile_file.exists():
        try:
            return Image.open(tile_file).convert("RGB")
        except Exception:
            pass

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RTKLIB-Plot-Python/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            tile_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(tile_file, "wb") as f:
                    f.write(data)
            except Exception:
                pass
            return img
    except Exception:
        return None


def _geodetic_to_enu_single(lat: float, lon: float, h: float,
                            lat0: float, lon0: float, h0: float) -> tuple[float, float, float]:
    """Convert a single (lat, lon, h) to local ENU relative to (lat0, lon0, h0)."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    p0, l0 = math.radians(lat0), math.radians(lon0)
    sin_p0, cos_p0 = math.sin(p0), math.cos(p0)
    sin_l0, cos_l0 = math.sin(l0), math.cos(l0)
    N0 = a / math.sqrt(1.0 - e2 * sin_p0 * sin_p0)
    x0 = (N0 + h0) * cos_p0 * cos_l0
    y0 = (N0 + h0) * cos_p0 * sin_l0
    z0 = (N0 * (1.0 - e2) + h0) * sin_p0

    p, l = math.radians(lat), math.radians(lon)
    sin_p, cos_p = math.sin(p), math.cos(p)
    sin_l, cos_l = math.sin(l), math.cos(l)
    N = a / math.sqrt(1.0 - e2 * sin_p * sin_p)
    x = (N + h) * cos_p * cos_l
    y = (N + h) * cos_p * sin_l
    z = (N * (1.0 - e2) + h) * sin_p

    dx = x - x0
    dy = y - y0
    dz = z - z0

    east = -sin_l0 * dx + cos_l0 * dy
    north = -sin_p0 * cos_l0 * dx - sin_p0 * sin_l0 * dy + cos_p0 * dz
    up = cos_p0 * cos_l0 * dx + cos_p0 * sin_l0 * dy + sin_p0 * dz
    return east, north, up


def get_stitched_tiles_enu(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    origin_lat: float, origin_lon: float, origin_h: float = 0.0,
    provider: str = "Esri Satellite", max_tiles: int = 72, margin: float = 0.06
) -> tuple[np.ndarray | None, list[float] | None]:
    """
    Fetch and stitch high-resolution map tiles spanning the bounding box, then calculate
    local ENU extent [e_min, e_max, n_min, n_max] relative to the origin.
    Uses multi-threaded concurrent downloading (max_workers=20) for lightning-fast loading.
    """
    if not (np.isfinite(lat_min) and np.isfinite(lat_max) and np.isfinite(lon_min) and np.isfinite(lon_max)):
        return None, None
    if not (np.isfinite(origin_lat) and np.isfinite(origin_lon)):
        return None, None

    dlat = max(lat_max - lat_min, 0.0003)
    dlon = max(lon_max - lon_min, 0.0003)
    lat_min_m = max(-85.0, lat_min - dlat * margin)
    lat_max_m = min(85.0, lat_max + dlat * margin)
    lon_min_m = max(-180.0, lon_min - dlon * margin)
    lon_max_m = min(180.0, lon_max + dlon * margin)

    # Search for maximum zoom level where total tiles <= max_tiles
    best_zoom = 1
    best_range = (0, 0, 0, 0)
    for z in range(19, 0, -1):
        x0, y1 = deg2num(lat_max_m, lon_min_m, z)
        x1, y0 = deg2num(lat_min_m, lon_max_m, z)
        nx = abs(x1 - x0) + 1
        ny = abs(y0 - y1) + 1
        if nx * ny <= max_tiles:
            best_zoom = z
            best_range = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            break

    xmin, xmax, ymin, ymax = best_range
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1

    tile_w, tile_h = 256, 256
    stitched = Image.new("RGB", (nx * tile_w, ny * tile_h), color=(240, 240, 240))
    any_success = False

    # Multi-threaded parallel tile downloading
    coords = [
        (ix, iy, x, y)
        for iy, y in enumerate(range(ymin, ymax + 1))
        for ix, x in enumerate(range(xmin, xmax + 1))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_map = {
            executor.submit(fetch_single_tile, provider, best_zoom, x, y): (ix, iy)
            for (ix, iy, x, y) in coords
        }
        for fut in concurrent.futures.as_completed(future_map):
            ix, iy = future_map[fut]
            try:
                tile_img = fut.result()
                if tile_img is not None:
                    stitched.paste(tile_img, (ix * tile_w, iy * tile_h))
                    any_success = True
            except Exception:
                pass

    if not any_success:
        return None, None

    # Corners in geodetic WGS84
    nw_lat, nw_lon = num2deg(xmin, ymin, best_zoom)
    se_lat, se_lon = num2deg(xmax + 1, ymax + 1, best_zoom)

    # Project corners into local ENU (m) relative to origin
    e_nw, n_nw, _ = _geodetic_to_enu_single(nw_lat, nw_lon, origin_h, origin_lat, origin_lon, origin_h)
    e_se, n_se, _ = _geodetic_to_enu_single(se_lat, se_lon, origin_h, origin_lat, origin_lon, origin_h)

    e_min, e_max = min(e_nw, e_se), max(e_nw, e_se)
    n_min, n_max = min(n_nw, n_se), max(n_nw, n_se)

    return np.asarray(stitched), [e_min, e_max, n_min, n_max]


def export_kml_3d(
    pos_df: pd.DataFrame,
    out_path: Path | str,
    segments: list[dict] | None = None,
    pos_name: str = "RTKLIB Solution",
    sample_step: int = 1
) -> Path:
    """
    Generate an academic-grade 3D Google Earth KML file with:
    - Status-colored paths (Q=1 FIX Green, Q=2 FLOAT Orange, Q=5 Single Grey)
    - 3D altitude curtain extrusion to ground (extrude=1, altitudeMode=absolute)
    - Time animation support (<TimeStamp><when>...)
    - Key pins for Start, End, and Problem Segments with rich HTML diagnostic cards.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = pos_df.copy()
    if sample_step > 1:
        p = p.iloc[::sample_step].reset_index(drop=True)

    # Determine coordinates
    has_llh = "lat" in p.columns and "lon" in p.columns and p["lat"].notna().any()
    if not has_llh:
        raise ValueError("Position data does not contain valid latitude/longitude coordinates for KML export.")

    # Time helper
    def get_iso_time(row) -> str:
        week = int(row.get("week", 0))
        tow = float(row.get("tow", 0.0))
        # GPS epoch 1980-01-06
        dt = datetime(1980, 1, 6) + timedelta(weeks=week, seconds=tow - 18.0) # approx GPS to UTC offset
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'  <name>{pos_name} — 3D Ground Track</name>',
        f'  <open>1</open>',
        f'  <description><![CDATA[',
        f'    <h3>RTKLIB GNSS 定位解算 3D 轨迹成果</h3>',
        f'    <p><b>解算文件</b>: {pos_name}</p>',
        f'    <p><b>历元总数</b>: {len(pos_df):,} 历元</p>',
        f'    <p><b>色彩图例</b>: <font color="#00C800">■ FIX (固定解)</font> | <font color="#FF8000">■ FLOAT (浮点解)</font> | <font color="#A0A0A0">■ SINGLE (单点)</font></p>',
        f'  ]]></description>',
        '',
        '  <!-- Styles for Solution Status -->',
        '  <Style id="fix_style">',
        '    <LineStyle><color>ff00c800</color><width>3.5</width></LineStyle>',
        '    <PolyStyle><color>4400c800</color></PolyStyle>',
        '  </Style>',
        '  <Style id="float_style">',
        '    <LineStyle><color>ff0080ff</color><width>3.5</width></LineStyle>',
        '    <PolyStyle><color>440080ff</color></PolyStyle>',
        '  </Style>',
        '  <Style id="single_style">',
        '    <LineStyle><color>ffa0a0a0</color><width>2.5</width></LineStyle>',
        '    <PolyStyle><color>44a0a0a0</color></PolyStyle>',
        '  </Style>',
        '  <Style id="other_style">',
        '    <LineStyle><color>ff9020a0</color><width>2.5</width></LineStyle>',
        '    <PolyStyle><color>449020a0</color></PolyStyle>',
        '  </Style>',
        '  <Style id="pin_start">',
        '    <IconStyle><color>ff00ff00</color><scale>1.2</scale>',
        '      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon>',
        '    </IconStyle>',
        '  </Style>',
        '  <Style id="pin_end">',
        '    <IconStyle><color>ff0000ff</color><scale>1.2</scale>',
        '      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>',
        '    </IconStyle>',
        '  </Style>',
        '  <Style id="pin_alert">',
        '    <IconStyle><color>ff0040ff</color><scale>1.1</scale>',
        '      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href></Icon>',
        '    </IconStyle>',
        '  </Style>',
        '',
        '  <!-- Trajectory Segments Grouped by Q-Status -->',
        '  <Folder>',
        '    <name>解算轨迹 (3D Trajectory Segments)</name>'
    ]

    # Partition continuous epochs with identical Q
    q_vals = p.q.to_numpy()
    transitions = np.where(q_vals[:-1] != q_vals[1:])[0] + 1
    split_indices = np.r_[0, transitions, len(p)]

    q_style_map = {1: "fix_style", 2: "float_style", 5: "single_style"}
    q_name_map = {1: "FIX", 2: "FLOAT", 5: "SINGLE"}

    for k in range(len(split_indices) - 1):
        s_idx = split_indices[k]
        e_idx = split_indices[k + 1]
        chunk = p.iloc[s_idx:e_idx]
        if chunk.empty:
            continue
        q = int(chunk.iloc[0].q)
        style_id = q_style_map.get(q, "other_style")
        q_label = q_name_map.get(q, f"Q={q}")

        t_start = get_iso_time(chunk.iloc[0])
        t_end = get_iso_time(chunk.iloc[-1])

        coords_str = " ".join(f"{row.lon:.8f},{row.lat:.8f},{float(row.get('h', 0.0)):.2f}" for _, row in chunk.iterrows())

        kml.append('    <Placemark>')
        kml.append(f'      <name>{q_label} #{k+1} ({len(chunk)} 历元)</name>')
        kml.append(f'      <styleUrl>#{style_id}</styleUrl>')
        kml.append(f'      <TimeSpan><begin>{t_start}</begin><end>{t_end}</end></TimeSpan>')
        kml.append('      <LineString>')
        kml.append('        <extrude>1</extrude>')
        kml.append('        <tessellate>1</tessellate>')
        kml.append('        <altitudeMode>absolute</altitudeMode>')
        kml.append(f'        <coordinates>{coords_str}</coordinates>')
        kml.append('      </LineString>')
        kml.append('    </Placemark>')

    kml.append('  </Folder>')

    # Start and End Placemarks
    p_start = p.iloc[0]
    p_end = p.iloc[-1]
    h_start = float(p_start.get("h", 0.0))
    h_end = float(p_end.get("h", 0.0))

    kml.append('  <Folder>')
    kml.append('    <name>关键地标点 (Key Waypoints)</name>')
    kml.append('    <Placemark>')
    kml.append('      <name>🚩 起始点 (Start)</name>')
    kml.append('      <styleUrl>#pin_start</styleUrl>')
    kml.append('      <description><![CDATA[')
    kml.append(f'        时间: {get_iso_time(p_start)} CST<br/>')
    kml.append(f'        坐标: {p_start.lat:.7f}°N, {p_start.lon:.7f}°E<br/>')
    kml.append(f'        大地高: {h_start:.2f} m')
    kml.append('      ]]></description>')
    kml.append('      <Point>')
    kml.append('        <altitudeMode>absolute</altitudeMode>')
    kml.append(f'        <coordinates>{p_start.lon:.8f},{p_start.lat:.8f},{h_start:.2f}</coordinates>')
    kml.append('      </Point>')
    kml.append('    </Placemark>')

    kml.append('    <Placemark>')
    kml.append('      <name>🏁 终点 (End)</name>')
    kml.append('      <styleUrl>#pin_end</styleUrl>')
    kml.append('      <description><![CDATA[')
    kml.append(f'        时间: {get_iso_time(p_end)} CST<br/>')
    kml.append(f'        坐标: {p_end.lat:.7f}°N, {p_end.lon:.7f}°E<br/>')
    kml.append(f'        大地高: {h_end:.2f} m')
    kml.append('      ]]></description>')
    kml.append('      <Point>')
    kml.append('        <altitudeMode>absolute</altitudeMode>')
    kml.append(f'        <coordinates>{p_end.lon:.8f},{p_end.lat:.8f},{h_end:.2f}</coordinates>')
    kml.append('      </Point>')
    kml.append('    </Placemark>')
    kml.append('  </Folder>')

    # Problem Segments Placemarks
    if segments:
        kml.append('  <Folder>')
        kml.append('    <name>自动检测异常区段 (Problematic Intervals)</name>')
        for idx, seg in enumerate(segments, 1):
            s_i = min(seg["start"], len(pos_df) - 1)
            e_i = min(seg["end"], len(pos_df) - 1)
            sub = pos_df.iloc[s_i:e_i + 1]
            if sub.empty:
                continue
            mid_row = sub.iloc[len(sub) // 2]
            mid_h = float(mid_row.get("h", 0.0))
            name = seg.get("name", f"Anomaly #{idx}")
            reason = seg.get("reason", "定位性能下降或掉固定")

            kml.append('    <Placemark>')
            kml.append(f'      <name>⚠️ {name}</name>')
            kml.append('      <styleUrl>#pin_alert</styleUrl>')
            kml.append('      <description><![CDATA[')
            kml.append(f'        <h4>异常诊断详情</h4>')
            kml.append(f'        <p><b>现象描述</b>: {reason}</p>')
            kml.append(f'        <p><b>持续历元</b>: {s_i+1} ~ {e_i+1} ({len(sub)} 历元)</p>')
            if "e3d" in sub and sub["e3d"].notna().any():
                kml.append(f'        <p><b>最大 3D 误差</b>: {sub.e3d.max():.3f} m</p>')
            kml.append(f'        <p><b>中心位置</b>: {mid_row.lat:.7f}°N, {mid_row.lon:.7f}°E</p>')
            kml.append('      ]]></description>')
            kml.append('      <Point>')
            kml.append('        <altitudeMode>absolute</altitudeMode>')
            kml.append(f'        <coordinates>{mid_row.lon:.8f},{mid_row.lat:.8f},{mid_h:.2f}</coordinates>')
            kml.append('      </Point>')
            kml.append('    </Placemark>')
        kml.append('  </Folder>')

    kml.append('</Document>')
    kml.append('</kml>')

    out_path.write_text("\n".join(kml), encoding="utf-8")
    return out_path


def export_truth_kml_3d(
    truth_df: pd.DataFrame,
    out_path: Path | str,
    truth_name: str = "Ground Truth Reference",
    sample_step: int = 1
) -> Path:
    """
    Generate a 3D Google Earth KML file for Ground Truth reference trajectory:
    - High-contrast pure white/gold line with 3D altitude extrusion curtain
    - Synchronized TimeSpan timestamps matching solution epochs
    - Start / End landmark placemarks with WGS84 coordinates and height
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = truth_df.copy()
    if sample_step > 1:
        p = p.iloc[::sample_step].reset_index(drop=True)

    has_llh = "lat" in p.columns and "lon" in p.columns and p["lat"].notna().any()
    if not has_llh:
        raise ValueError("Truth data does not contain valid latitude/longitude coordinates.")

    def get_iso_time(row) -> str:
        week = int(row.get("week", 0))
        tow = float(row.get("tow", 0.0))
        dt = datetime(1980, 1, 6) + timedelta(weeks=week, seconds=tow - 18.0)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'  <name>{truth_name} — 3D Reference Track</name>',
        f'  <open>1</open>',
        f'  <description><![CDATA[',
        f'    <h3>GNSS 真实基准参考轨迹 (Ground Truth 3D)</h3>',
        f'    <p><b>基准源</b>: {truth_name}</p>',
        f'    <p><b>历元总数</b>: {len(truth_df):,} 历元</p>',
        f'    <p><b>色彩说明</b>: <font color="#FFFFFF">■ 白色高亮 3D 幕墙参考轨迹</font></p>',
        f'  ]]></description>',
        '',
        '  <!-- Style for Ground Truth Track: High-contrast white line with extruded 3D curtain -->',
        '  <Style id="truth_style">',
        '    <LineStyle><color>ffffffff</color><width>4.0</width></LineStyle>',
        '    <PolyStyle><color>55ffffff</color></PolyStyle>',
        '  </Style>',
        '  <Style id="pin_truth_start">',
        '    <IconStyle><color>ff00ff00</color><scale>1.2</scale>',
        '      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href></Icon>',
        '    </IconStyle>',
        '  </Style>',
        '  <Style id="pin_truth_end">',
        '    <IconStyle><color>ff0000ff</color><scale>1.2</scale>',
        '      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-diamond.png</href></Icon>',
        '    </IconStyle>',
        '  </Style>',
        '',
        '  <Folder>',
        '    <name>真值基准轨迹 (3D Truth Track)</name>',
        '    <Placemark>',
        f'      <name>🚩 真值参考轨迹 ({len(p):,} 历元)</name>',
        '      <styleUrl>#truth_style</styleUrl>',
    ]
    t_start = get_iso_time(p.iloc[0])
    t_end = get_iso_time(p.iloc[-1])
    kml.append(f'      <TimeSpan><begin>{t_start}</begin><end>{t_end}</end></TimeSpan>')
    coords_str = " ".join(f"{row.lon:.8f},{row.lat:.8f},{float(row.get('h', 0.0)):.2f}" for _, row in p.iterrows())
    kml.extend([
        '      <LineString>',
        '        <extrude>1</extrude>',
        '        <tessellate>1</tessellate>',
        '        <altitudeMode>absolute</altitudeMode>',
        f'        <coordinates>{coords_str}</coordinates>',
        '      </LineString>',
        '    </Placemark>',
        '  </Folder>',
    ])

    p_start, p_end = p.iloc[0], p.iloc[-1]
    h_start = float(p_start.get("h", 0.0))
    h_end = float(p_end.get("h", 0.0))
    kml.extend([
        '  <Folder>',
        '    <name>真值起止参考点 (Truth Waypoints)</name>',
        '    <Placemark>',
        '      <name>🚩 真值起始点 (Truth Start)</name>',
        '      <styleUrl>#pin_truth_start</styleUrl>',
        '      <description><![CDATA[',
        f'        时间: {get_iso_time(p_start)} CST<br/>',
        f'        坐标: {p_start.lat:.7f}°N, {p_start.lon:.7f}°E<br/>',
        f'        大地高: {h_start:.2f} m',
        '      ]]></description>',
        '      <Point>',
        '        <altitudeMode>absolute</altitudeMode>',
        f'        <coordinates>{p_start.lon:.8f},{p_start.lat:.8f},{h_start:.2f}</coordinates>',
        '      </Point>',
        '    </Placemark>',
        '    <Placemark>',
        '      <name>🏁 真值终点 (Truth End)</name>',
        '      <styleUrl>#pin_truth_end</styleUrl>',
        '      <description><![CDATA[',
        f'        时间: {get_iso_time(p_end)} CST<br/>',
        f'        坐标: {p_end.lat:.7f}°N, {p_end.lon:.7f}°E<br/>',
        f'        大地高: {h_end:.2f} m',
        '      ]]></description>',
        '      <Point>',
        '        <altitudeMode>absolute</altitudeMode>',
        f'        <coordinates>{p_end.lon:.8f},{p_end.lat:.8f},{h_end:.2f}</coordinates>',
        '      </Point>',
        '    </Placemark>',
        '  </Folder>',
        '</Document>',
        '</kml>'
    ])
    out_path.write_text("\n".join(kml), encoding="utf-8")
    return out_path



def export_kml_2d_mymaps(
    pos_df: pd.DataFrame,
    out_path: Path | str,
    pos_name: str = "RTKLIB Solution",
    max_points: int = 8000
) -> Path:
    """
    Generate a 100% Google My Maps (我的地图) compliant 2D KML file:
    - Auto downsamples to stay well under My Maps' 10,000 points / 5MB limits
    - Strips unsupported 3D tags (<extrude>, <altitudeMode>, <TimeSpan>, <PolyStyle>)
    - Segments lines by Q status with pure 2D LineString
    - Adds Start/End waypoints
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = pos_df.copy()
    if len(p) > max_points:
        step = int(np.ceil(len(p) / max_points))
        p = p.iloc[::step].reset_index(drop=True)

    has_llh = "lat" in p.columns and "lon" in p.columns and p["lat"].notna().any()
    if not has_llh:
        raise ValueError("Position data does not contain valid latitude/longitude coordinates.")

    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'  <name>{pos_name} — 2D Trajectory</name>',
        f'  <description>RTKLIB GNSS Solution for Google My Maps</description>',
        '',
        '  <Style id="fix_style">',
        '    <LineStyle><color>ff00c800</color><width>4</width></LineStyle>',
        '  </Style>',
        '  <Style id="float_style">',
        '    <LineStyle><color>ff0080ff</color><width>4</width></LineStyle>',
        '  </Style>',
        '  <Style id="single_style">',
        '    <LineStyle><color>ffa0a0a0</color><width>3</width></LineStyle>',
        '  </Style>',
        '  <Style id="other_style">',
        '    <LineStyle><color>ff9020a0</color><width>3</width></LineStyle>',
        '  </Style>',
        '  <Style id="pin_start">',
        '    <IconStyle><Icon><href>https://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon></IconStyle>',
        '  </Style>',
        '  <Style id="pin_end">',
        '    <IconStyle><Icon><href>https://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle>',
        '  </Style>',
        ''
    ]

    q_vals = p.q.to_numpy()
    transitions = np.where(q_vals[:-1] != q_vals[1:])[0] + 1
    split_indices = np.r_[0, transitions, len(p)]

    q_style_map = {1: "fix_style", 2: "float_style", 5: "single_style"}
    q_name_map = {1: "FIX", 2: "FLOAT", 5: "SINGLE"}

    for k in range(len(split_indices) - 1):
        s_idx = split_indices[k]
        e_idx = split_indices[k + 1]
        chunk = p.iloc[s_idx:e_idx]
        if chunk.empty:
            continue
        q = int(chunk.iloc[0].q)
        style_id = q_style_map.get(q, "other_style")
        q_label = q_name_map.get(q, f"Q={q}")

        coords_str = " ".join(f"{row.lon:.7f},{row.lat:.7f},0" for _, row in chunk.iterrows())

        kml.append('  <Placemark>')
        kml.append(f'    <name>{q_label} #{k+1} ({len(chunk)} 历元)</name>')
        kml.append(f'    <styleUrl>#{style_id}</styleUrl>')
        kml.append('    <LineString>')
        kml.append('      <tessellate>1</tessellate>')
        kml.append(f'      <coordinates>{coords_str}</coordinates>')
        kml.append('    </LineString>')
        kml.append('  </Placemark>')

    p_start = p.iloc[0]
    p_end = p.iloc[-1]
    kml.append('  <Placemark>')
    kml.append('    <name>🚩 起始点 (Start)</name>')
    kml.append('    <styleUrl>#pin_start</styleUrl>')
    kml.append(f'    <description>起始坐标: {p_start.lat:.7f}°N, {p_start.lon:.7f}°E</description>')
    kml.append('    <Point>')
    kml.append(f'      <coordinates>{p_start.lon:.7f},{p_start.lat:.7f},0</coordinates>')
    kml.append('    </Point>')
    kml.append('  </Placemark>')

    kml.append('  <Placemark>')
    kml.append('    <name>🏁 终点 (End)</name>')
    kml.append('    <styleUrl>#pin_end</styleUrl>')
    kml.append(f'    <description>终点坐标: {p_end.lat:.7f}°N, {p_end.lon:.7f}°E</description>')
    kml.append('    <Point>')
    kml.append(f'      <coordinates>{p_end.lon:.7f},{p_end.lat:.7f},0</coordinates>')
    kml.append('    </Point>')
    kml.append('  </Placemark>')

    kml.append('</Document>')
    kml.append('</kml>')

    out_path.write_text("\n".join(kml), encoding="utf-8")
    return out_path


def export_truth_kml_2d_mymaps(
    truth_df: pd.DataFrame,
    out_path: Path | str,
    truth_name: str = "Ground Truth Reference",
    max_points: int = 8000
) -> Path:
    """
    Generate a 100% Google My Maps (我的地图) compliant 2D KML file for Ground Truth:
    - Downsampled to < 8,000 points
    - Clean dark/gold continuous 2D LineString
    - Start / End placemarks
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = truth_df.copy()
    if len(p) > max_points:
        step = int(np.ceil(len(p) / max_points))
        p = p.iloc[::step].reset_index(drop=True)

    has_llh = "lat" in p.columns and "lon" in p.columns and p["lat"].notna().any()
    if not has_llh:
        raise ValueError("Truth data does not contain valid latitude/longitude coordinates.")

    coords_str = " ".join(f"{row.lon:.7f},{row.lat:.7f},0" for _, row in p.iterrows())
    p_start, p_end = p.iloc[0], p.iloc[-1]

    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'  <name>{truth_name} — 2D Reference Track</name>',
        f'  <description>Ground Truth Reference Track for Google My Maps</description>',
        '',
        '  <Style id="truth_2d_style">',
        '    <LineStyle><color>ff000000</color><width>4</width></LineStyle>',
        '  </Style>',
        '  <Style id="pin_truth_start">',
        '    <IconStyle><Icon><href>https://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href></Icon></IconStyle>',
        '  </Style>',
        '  <Style id="pin_truth_end">',
        '    <IconStyle><Icon><href>https://maps.google.com/mapfiles/kml/paddle/wht-diamond.png</href></Icon></IconStyle>',
        '  </Style>',
        '',
        '  <Placemark>',
        f'    <name>🚩 真值参考轨迹 ({len(p):,} 点)</name>',
        '    <styleUrl>#truth_2d_style</styleUrl>',
        '    <LineString>',
        '      <tessellate>1</tessellate>',
        f'      <coordinates>{coords_str}</coordinates>',
        '    </LineString>',
        '  </Placemark>',
        '  <Placemark>',
        '    <name>🚩 真值起始点 (Truth Start)</name>',
        '    <styleUrl>#pin_truth_start</styleUrl>',
        f'    <description>起始坐标: {p_start.lat:.7f}°N, {p_start.lon:.7f}°E</description>',
        '    <Point>',
        f'      <coordinates>{p_start.lon:.7f},{p_start.lat:.7f},0</coordinates>',
        '    </Point>',
        '  </Placemark>',
        '  <Placemark>',
        '    <name>🏁 真值终点 (Truth End)</name>',
        '    <styleUrl>#pin_truth_end</styleUrl>',
        f'    <description>终点坐标: {p_end.lat:.7f}°N, {p_end.lon:.7f}°E</description>',
        '    <Point>',
        f'      <coordinates>{p_end.lon:.7f},{p_end.lat:.7f},0</coordinates>',
        '    </Point>',
        '  </Placemark>',
        '</Document>',
        '</kml>'
    ]
    out_path.write_text("\n".join(kml), encoding="utf-8")
    return out_path




def export_geojson(pos_df: pd.DataFrame, out_path: Path | str, sample_step: int = 1) -> Path:
    """Export standard GeoJSON FeatureCollection with line string and epoch properties."""
    import json
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = pos_df.copy()
    if sample_step > 1:
        p = p.iloc[::sample_step].reset_index(drop=True)

    coordinates = []
    features = []

    for idx, row in p.iterrows():
        lat = float(row.lat)
        lon = float(row.lon)
        alt = float(row.get("h", 0.0))
        coordinates.append([lon, lat, alt])

        # Sample point properties every 10 steps to keep GeoJSON file compact
        if idx % 10 == 0 or idx == len(p) - 1:
            props = {
                "epoch": int(idx + 1),
                "tow": float(row.tow),
                "q": int(row.q),
                "ns": int(row.ns) if "ns" in row and np.isfinite(row.ns) else None,
                "ratio": float(row.ratio) if "ratio" in row and np.isfinite(row.ratio) else None,
                "horizontal_err": float(row.eh) if "eh" in row and np.isfinite(row.eh) else None,
                "error_3d": float(row.e3d) if "e3d" in row and np.isfinite(row.e3d) else None,
            }
            pt_feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat, alt]
                },
                "properties": {k: v for k, v in props.items() if v is not None}
            }
            features.append(pt_feat)

    # LineString feature
    line_feat = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        },
        "properties": {
            "name": "RTKLIB Trajectory",
            "total_epochs": len(p),
            "start_time": float(p.iloc[0].tow),
            "end_time": float(p.iloc[-1].tow)
        }
    }
    features.insert(0, line_feat)

    geojson_doc = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2, ensure_ascii=False)

    return out_path


def export_web_3d_globe(
    pos_df: pd.DataFrame,
    out_path: Path | str,
    ref_df: pd.DataFrame | None = None,
    ground_truth: pd.DataFrame | None = None,
    title: str = "RTKLIB 3D 卫星实景数字地球",
    pos_name: str | None = None,
    sample_step: int = 1
) -> Path:
    """
    Generate an interactive, standalone 3D WebGL Digital Globe (HTML) powered by CesiumJS:
    - High-resolution global satellite photography (Esri World Imagery)
    - True 3D flight trajectory with extruded vertical height curtain walls to ground
    - Colormapped by 3D Positioning Error (or Solution Status Q)
    - Ground Truth 3D comparison path (gold) when reference data is available
    - 360° mouse orbit, zoom, pitch, fly-to view presets, and animated playback
    """
    if ref_df is None and ground_truth is not None:
        ref_df = ground_truth
    if pos_name is not None and title == "RTKLIB 3D 卫星实景数字地球":
        title = f"{pos_name} - 3D 卫星实景数字地球"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = pos_df.copy()
    if sample_step > 1:
        p = p.iloc[::sample_step].reset_index(drop=True)

    if "lat" not in p.columns or "lon" not in p.columns:
        raise ValueError("Position dataframe must contain 'lat' and 'lon' columns.")

    has_truth = "e3d" in p.columns and p["e3d"].notna().any()
    center_lat = float(p.lat.mean())
    center_lon = float(p.lon.mean())
    mean_h = float(p.h.mean()) if "h" in p else 0.0

    # Prepare trajectory points for Cesium
    pts_json = []
    for _, row in p.iterrows():
        lat = float(row.lat)
        lon = float(row.lon)
        alt = float(row.h) if "h" in row and pd.notna(row.h) else 0.0
        q = int(row.q) if "q" in row and pd.notna(row.q) else 1
        e3d = float(row.e3d) if has_truth and pd.notna(row.e3d) else 0.0
        eh = float(row.eh) if "eh" in row and pd.notna(row.eh) else 0.0
        tow = float(row.tow) if "tow" in row and pd.notna(row.tow) else 0.0
        pts_json.append({
            "lon": round(lon, 7),
            "lat": round(lat, 7),
            "alt": round(alt, 2),
            "q": q,
            "e3d": round(e3d, 3),
            "eh": round(eh, 3),
            "tow": round(tow, 2)
        })

    # Prepare Ground Truth line if available
    gt_pts_json = []
    if ref_df is not None and not ref_df.empty:
        r_sub = ref_df.iloc[::max(1, sample_step)]
        lat_c = "Latitude (deg)" if "Latitude (deg)" in r_sub.columns else ("lat" if "lat" in r_sub.columns else None)
        lon_c = "Longitude (deg)" if "Longitude (deg)" in r_sub.columns else ("lon" if "lon" in r_sub.columns else None)
        h_c = "Ellipsoid Height (m)" if "Ellipsoid Height (m)" in r_sub.columns else ("h" if "h" in r_sub.columns else None)
        if lat_c and lon_c:
            for _, rrow in r_sub.iterrows():
                gt_pts_json.append({
                    "lon": round(float(rrow[lon_c]), 7),
                    "lat": round(float(rrow[lat_c]), 7),
                    "alt": round(float(rrow[h_c]), 2) if h_c and pd.notna(rrow[h_c]) else round(mean_h, 2)
                })

    pts_str = json.dumps(pts_json)
    gt_pts_str = json.dumps(gt_pts_json)

    # Calculate summary metrics
    total_ep = len(p)
    fix_cnt = int((p.q == 1).sum()) if "q" in p else 0
    fix_rate = (fix_cnt / total_ep * 100.0) if total_ep > 0 else 0.0
    rmse_3d = float(np.sqrt(np.mean(p.e3d**2))) if has_truth else 0.0
    max_3d = float(p.e3d.max()) if has_truth else 0.0

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Cesium.js"></script>
  <style>
    html, body, #cesiumContainer {{
      width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
    }}
    .cesium-widget-credits {{ display: none !important; }}
    #hud-panel {{
      position: absolute; top: 16px; left: 16px; z-index: 99;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px);
      border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px;
      padding: 18px 22px; color: #f8fafc; width: 340px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
    }}
    #hud-panel h2 {{ margin: 0 0 10px 0; font-size: 17px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
    .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }}
    .stat-item {{ background: rgba(30, 41, 59, 0.7); padding: 8px 10px; border-radius: 6px; border-left: 3px solid #38bdf8; }}
    .stat-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; }}
    .stat-val {{ font-size: 15px; font-weight: 700; color: #f1f5f9; margin-top: 2px; }}
    .btn-group {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }}
    .hud-btn {{
      background: #0284c7; color: white; border: none; padding: 6px 12px;
      border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
      transition: all 0.2s;
    }}
    .hud-btn:hover {{ background: #0369a1; transform: translateY(-1px); }}
    .hud-btn.secondary {{ background: rgba(51, 65, 85, 0.85); }}
    .hud-btn.secondary:hover {{ background: rgba(71, 85, 105, 0.95); }}
    #legend {{
      position: absolute; bottom: 24px; right: 24px; z-index: 99;
      background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 8px;
      padding: 12px 16px; color: #e2e8f0; font-size: 12px;
    }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
    .legend-bar {{ width: 18px; height: 10px; border-radius: 2px; }}
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>

  <div id="hud-panel">
    <h2>🛰️ RTKLIB 三维实景数字地球</h2>
    <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 6px;">
      卫星遥感底图：Esri World Imagery (高清卫星遥感)
    </div>
    <div class="stat-grid">
      <div class="stat-item">
        <div class="stat-label">固定率 (Fix Rate)</div>
        <div class="stat-val" style="color: {'#4ade80' if fix_rate >= 85 else '#fbbf24'};">{fix_rate:.1f}%</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">历元总数 (Epochs)</div>
        <div class="stat-val">{total_ep:,}</div>
      </div>
      {f'''<div class="stat-item">
        <div class="stat-label">3D RMSE</div>
        <div class="stat-val" style="color: {'#4ade80' if rmse_3d < 1.0 else '#f87171'};">{rmse_3d:.3f} m</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">最大 3D 误差</div>
        <div class="stat-val" style="color: {'#fbbf24' if max_3d < 3.0 else '#f87171'};">{max_3d:.2f} m</div>
      </div>''' if has_truth else f'''<div class="stat-item">
        <div class="stat-label">平均大地高</div>
        <div class="stat-val">{mean_h:.2f} m</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">解状态模式</div>
        <div class="stat-val">FIX+FLOAT</div>
      </div>'''}
    </div>
    <div class="btn-group">
      <button class="hud-btn" onclick="flyToTrack()">🎯 视角对齐轨迹</button>
      <button class="hud-btn secondary" onclick="setViewTop()">📐 2D 顶视</button>
      <button class="hud-btn secondary" onclick="setViewIso()">🌐 3D 等轴</button>
      <button class="hud-btn secondary" onclick="toggleExtrude()">🏢 切换垂向挤出</button>
    </div>
  </div>

  <div id="legend">
    <div style="font-weight: 700; margin-bottom: 6px; color: #38bdf8;">{'定位误差分级' if has_truth else '解状态分类'}</div>
    {'''
    <div class="legend-row"><div class="legend-bar" style="background: #22c55e;"></div><span>高精度 (3D &lt; 0.5m)</span></div>
    <div class="legend-row"><div class="legend-bar" style="background: #eab308;"></div><span>中等误差 (0.5m ~ 2.0m)</span></div>
    <div class="legend-row"><div class="legend-bar" style="background: #ef4444;"></div><span>超差/异常 (&gt; 2.0m)</span></div>
    <div class="legend-row"><div class="legend-bar" style="background: #ffd700; height: 3px;"></div><span>参考真值 (Ground Truth)</span></div>
    ''' if has_truth else '''
    <div class="legend-row"><div class="legend-bar" style="background: #16a34a;"></div><span>Q=1 固定解 (FIX)</span></div>
    <div class="legend-row"><div class="legend-bar" style="background: #ea580c;"></div><span>Q=2 浮点解 (FLOAT)</span></div>
    <div class="legend-row"><div class="legend-bar" style="background: #64748b;"></div><span>Q=5 单点解 (SINGLE)</span></div>
    '''}
  </div>

  <script>
    const pts = {pts_str};
    const gtPts = {gt_pts_str};
    const hasTruth = {'true' if has_truth else 'false'};

    // Initialize Cesium viewer with Esri World Imagery
    const viewer = new Cesium.Viewer('cesiumContainer', {{
      imageryProvider: new Cesium.ArcGisMapServerImageryProvider({{
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
      }}),
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      infoBox: true,
      selectionIndicator: true,
      timeline: true,
      animation: true,
      sceneModePicker: true,
      navigationHelpButton: false
    }});

    // Helper for colors
    function getColor(pt) {{
      if (hasTruth) {{
        const e = pt.e3d;
        if (e < 0.5) return Cesium.Color.fromCssColorString('#22c55e');
        if (e < 2.0) return Cesium.Color.fromCssColorString('#eab308');
        return Cesium.Color.fromCssColorString('#ef4444');
      }} else {{
        if (pt.q === 1) return Cesium.Color.fromCssColorString('#16a34a');
        if (pt.q === 2) return Cesium.Color.fromCssColorString('#ea580c');
        return Cesium.Color.fromCssColorString('#64748b');
      }}
    }}

    // Construct 3D trajectory polyline segments with extruded curtain wall
    let wallEntities = [];
    let isExtruded = true;

    if (pts.length > 1) {{
      for (let i = 0; i < pts.length - 1; i++) {{
        const p1 = pts[i];
        const p2 = pts[i+1];
        const col = getColor(p1);

        // 3D Polyline segment
        viewer.entities.add({{
          polyline: {{
            positions: Cesium.Cartesian3.fromDegreesArrayHeights([
              p1.lon, p1.lat, p1.alt,
              p2.lon, p2.lat, p2.alt
            ]),
            width: 4.5,
            material: col
          }}
        }});

        // Vertical extrusion curtain wall to ground
        const wall = viewer.entities.add({{
          wall: {{
            positions: Cesium.Cartesian3.fromDegreesArrayHeights([
              p1.lon, p1.lat, p1.alt,
              p2.lon, p2.lat, p2.alt
            ]),
            minimumHeights: [0, 0],
            material: col.withAlpha(0.28)
          }}
        }});
        wallEntities.push(wall);
      }}

      // Start & End markers
      const pStart = pts[0];
      const pEnd = pts[pts.length - 1];

      viewer.entities.add({{
        position: Cesium.Cartesian3.fromDegrees(pStart.lon, pStart.lat, pStart.alt),
        point: {{ pixelSize: 14, color: Cesium.Color.LIME, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 }},
        label: {{ text: '🟢 起点 (Start)', font: '13px bold sans-serif', pixelOffset: new Cesium.Cartesian2(0, -22), fillColor: Cesium.Color.WHITE }}
      }});

      viewer.entities.add({{
        position: Cesium.Cartesian3.fromDegrees(pEnd.lon, pEnd.lat, pEnd.alt),
        point: {{ pixelSize: 14, color: Cesium.Color.RED, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 }},
        label: {{ text: '🔴 终点 (End)', font: '13px bold sans-serif', pixelOffset: new Cesium.Cartesian2(0, -22), fillColor: Cesium.Color.WHITE }}
      }});
    }}

    // Draw Ground Truth in 3D (Gold Line)
    if (gtPts.length > 1) {{
      const gtPos = [];
      gtPts.forEach(pt => {{
        gtPos.push(pt.lon, pt.lat, pt.alt);
      }});
      viewer.entities.add({{
        polyline: {{
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(gtPos),
          width: 3.0,
          material: new Cesium.PolylineDashMaterialProperty({{
            color: Cesium.Color.fromCssColorString('#ffd700'),
            dashLength: 16.0
          }})
        }}
      }});
    }}

    // Camera viewpoints
    function flyToTrack() {{
      if (!pts.length) return;
      const lons = pts.map(p => p.lon);
      const lats = pts.map(p => p.lat);
      const minLon = Math.min(...lons), maxLon = Math.max(...lons);
      const minLat = Math.min(...lats), maxLat = Math.max(...lats);

      viewer.camera.flyTo({{
        destination: Cesium.Rectangle.fromDegrees(minLon - 0.005, minLat - 0.005, maxLon + 0.005, maxLat + 0.005),
        duration: 1.8
      }});
    }}

    function setViewTop() {{
      const cLon = {center_lon}, cLat = {center_lat};
      viewer.camera.flyTo({{
        destination: Cesium.Cartesian3.fromDegrees(cLon, cLat, 4500.0),
        orientation: {{ heading: 0.0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0.0 }},
        duration: 1.2
      }});
    }}

    function setViewIso() {{
      const cLon = {center_lon}, cLat = {center_lat};
      viewer.camera.flyTo({{
        destination: Cesium.Cartesian3.fromDegrees(cLon - 0.02, cLat - 0.02, 2800.0),
        orientation: {{ heading: Cesium.Math.toRadians(45.0), pitch: Cesium.Math.toRadians(-35.0), roll: 0.0 }},
        duration: 1.2
      }});
    }}

    function toggleExtrude() {{
      isExtruded = !isExtruded;
      wallEntities.forEach(w => w.show = isExtruded);
    }}

    // Initial fly-in
    setTimeout(flyToTrack, 600);
  </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_path

