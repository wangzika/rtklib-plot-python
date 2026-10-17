"""GIS, Web Map Tiles and 3D Google Earth KML / GeoJSON Exporter for RTKLIB-Plot-Python."""
from __future__ import annotations

import io
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
    provider: str = "Esri Satellite", max_tiles: int = 16, margin: float = 0.20
) -> tuple[np.ndarray | None, list[float] | None]:
    """
    Fetch and stitch map tiles spanning the bounding box, then calculate
    local ENU extent [e_min, e_max, n_min, n_max] relative to the origin.
    """
    if not (np.isfinite(lat_min) and np.isfinite(lat_max) and np.isfinite(lon_min) and np.isfinite(lon_max)):
        return None, None
    if not (np.isfinite(origin_lat) and np.isfinite(origin_lon)):
        return None, None

    dlat = max(lat_max - lat_min, 0.0005)
    dlon = max(lon_max - lon_min, 0.0005)
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

    for iy, y in enumerate(range(ymin, ymax + 1)):
        for ix, x in enumerate(range(xmin, xmax + 1)):
            tile_img = fetch_single_tile(provider, best_zoom, x, y)
            if tile_img is not None:
                stitched.paste(tile_img, (ix * tile_w, iy * tile_h))
                any_success = True

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
