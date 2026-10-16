"""Modular RTKLIB diagnostic plots (01-15) and overview dashboard.

Metrics are computed from the documented RTKLIB .pos columns, matching GPST
to reference ECEF, and $SAT records in .stat. Unsupported diagnostics are
explicitly marked instead of being fabricated.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rtklib_map_tiles import export_kml_3d, export_geojson, get_stitched_tiles_enu, TILE_PROVIDERS
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POS = ROOT / "results_pos" / "6_RTK_Multi_L1L2.pos"
DEFAULT_REF = ROOT.parent / "Tokyo_Data" / "Tokyo_Data" / "Odaiba" / "reference.csv"
DEFAULT_OUT = ROOT / "plots_15_diagnostics"
EXPORT_FORMATS = ("png",)
EXPORT_DPI = 180

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 130,
                     "savefig.dpi": 180, "font.size": 9})
COLORS = {"E": "#2878B5", "N": "#E07A2D", "U": "#36965A",
          "GPS": "#2878B5", "GLONASS": "#E07A2D", "Galileo": "#36965A",
          "BeiDou": "#A34FA3", "QZSS": "#D3A11E", "SBAS": "#777777"}


def _geodetic_to_ecef(lat_deg: float | np.ndarray, lon_deg: float | np.ndarray, h: float | np.ndarray):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)
    return x, y, z


def _ecef_to_geodetic(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray):
    a, f = 6378137.0, 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(8):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + h)))
    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N
    return np.rad2deg(lat), np.rad2deg(lon), h


def _ecef_to_enu(xyz: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    p, l = np.deg2rad([lat_deg, lon_deg])
    rot = np.array([[-np.sin(l), np.cos(l), 0],
                    [-np.sin(p)*np.cos(l), -np.sin(p)*np.sin(l), np.cos(p)],
                    [np.cos(p)*np.cos(l), np.cos(p)*np.sin(l), np.sin(p)]])
    return (xyz @ rot.T)


def load_data(pos_path: Path, reference_path: Path | None = None,
              stat_path: Path | None = None) -> dict:
    """Load RTKLIB .pos and optional reference/.stat inputs into aligned frames."""
    from datetime import datetime
    pos_path = Path(pos_path)
    stat_path = Path(stat_path) if stat_path else Path(str(pos_path) + ".stat")
    rows = []
    for line in pos_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        f = line.split()
        if len(f) < 15:
            continue
        try:
            # 1. 兼容日期时间与 week/tow
            if "/" in f[0] or "-" in f[0]:
                t_str = f[0] + " " + f[1]
                try:
                    dt = datetime.strptime(t_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    dt = datetime.strptime(t_str, "%Y/%m/%d %H:%M:%S")
                diff = (dt - datetime(1980, 1, 6)).total_seconds()
                week = int(diff // 604800)
                tow = float(diff % 604800)
                v1, v2, v3 = float(f[2]), float(f[3]), float(f[4])
                q, ns = int(f[5]), int(f[6])
                s1, s2, s3 = float(f[7]), float(f[8]), float(f[9])
                s12, s23, s31 = float(f[10]), float(f[11]), float(f[12])
                age, ratio = float(f[13]), float(f[14])
            else:
                week = int(f[0])
                tow = float(f[1])
                v1, v2, v3 = float(f[2]), float(f[3]), float(f[4])
                q, ns = int(f[5]), int(f[6])
                s1, s2, s3 = float(f[7]), float(f[8]), float(f[9])
                s12, s23, s31 = float(f[10]), float(f[11]), float(f[12])
                age, ratio = float(f[13]), float(f[14])

            # 2. 兼容 LLH (纬度/经度/高度) 还是 ECEF (x/y/z)
            is_llh = abs(v1) <= 90.0 and abs(v2) <= 360.0
            if is_llh:
                lat, lon, h = v1, v2, v3
                x, y, z = _geodetic_to_ecef(lat, lon, h)
                sdn, sde, sdu = s1, s2, s3
                sdne, sdeu, sdun = s12, s23, s31
                sdx, sdy, sdz = s1, s2, s3
                sdxy, sdyz, sdzx = s12, s23, s31
            else:
                x, y, z = v1, v2, v3
                lat, lon, h = _ecef_to_geodetic(x, y, z)
                sdx, sdy, sdz = s1, s2, s3
                sdxy, sdyz, sdzx = s12, s23, s31
                sdn, sde, sdu = np.nan, np.nan, np.nan
                sdne, sdeu, sdun = np.nan, np.nan, np.nan

            rows.append([week, tow, x, y, z, lat, lon, h, q, ns,
                         sdx, sdy, sdz, sdxy, sdyz, sdzx,
                         sdn, sde, sdu, sdne, sdeu, sdun,
                         age, ratio, is_llh])
        except (ValueError, IndexError):
            continue

    cols = ["week", "tow", "x", "y", "z", "lat", "lon", "h", "q", "ns",
            "sdx", "sdy", "sdz", "sdxy", "sdyz", "sdzx",
            "sdn", "sde", "sdu", "sdne", "sdeu", "sdun",
            "age", "ratio", "is_llh"]
    pos = pd.DataFrame(rows, columns=cols).sort_values("tow").reset_index(drop=True)
    if pos.empty:
        raise ValueError(f"No position epochs parsed from {pos_path}")
    pos["time"] = pos.tow - pos.tow.iloc[0]

    ref = None
    if reference_path and Path(reference_path).exists():
        ref = pd.read_csv(reference_path, skipinitialspace=True)
        needed = ["GPS TOW (s)", "Latitude (deg)", "Longitude (deg)", "ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"]
        if all(c in ref for c in needed):
            ref = ref[needed].dropna().sort_values("GPS TOW (s)")
            xyz0 = ref[["ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"]].to_numpy()
            ref_lat0 = float(ref.iloc[0]["Latitude (deg)"])
            ref_lon0 = float(ref.iloc[0]["Longitude (deg)"])
            enu_ref = _ecef_to_enu(xyz0 - xyz0[0], ref_lat0, ref_lon0)
            xyz = pos[["x", "y", "z"]].to_numpy()
            enu_pos = _ecef_to_enu(xyz - xyz0[0], ref_lat0, ref_lon0)
            pos[["E", "N", "U"]] = enu_pos
            for i, c in enumerate(["E", "N", "U"]):
                pos[f"gt_{c}"] = np.interp(pos.tow, ref["GPS TOW (s)"], enu_ref[:, i], left=np.nan, right=np.nan)
                pos[f"d{c.lower()}"] = pos[c] - pos[f"gt_{c}"]
            pos["eh"] = np.hypot(pos.de, pos.dn)
            pos["e3d"] = np.sqrt(pos.de**2 + pos.dn**2 + pos.du**2)
        else:
            ref = None

    if "E" not in pos:
        ref_lat0 = float(pos.iloc[0]["lat"]) if np.isfinite(pos.iloc[0]["lat"]) else 0.0
        ref_lon0 = float(pos.iloc[0]["lon"]) if np.isfinite(pos.iloc[0]["lon"]) else 0.0
        xyz0 = pos[["x", "y", "z"]].iloc[0].to_numpy()
        enu_pos = _ecef_to_enu(pos[["x", "y", "z"]].to_numpy() - xyz0, ref_lat0, ref_lon0)
        pos[["E", "N", "U"]] = enu_pos

    if pos["is_llh"].iloc[0]:
        pos["sigE"] = pos["sde"]
        pos["sigN"] = pos["sdn"]
        pos["sigU"] = pos["sdu"]
        pos["covEN"] = np.sign(pos["sdne"]) * pos["sdne"]**2
        pos["rhoEN"] = pos["covEN"] / np.maximum(pos["sigE"] * pos["sigN"], 1e-15)
    else:
        cov_ecef = np.zeros((len(pos), 3, 3))
        cov_ecef[:, 0, 0], cov_ecef[:, 1, 1], cov_ecef[:, 2, 2] = pos.sdx**2, pos.sdy**2, pos.sdz**2
        cov_ecef[:, 0, 1] = cov_ecef[:, 1, 0] = np.sign(pos.sdxy)*pos.sdxy**2
        cov_ecef[:, 1, 2] = cov_ecef[:, 2, 1] = np.sign(pos.sdyz)*pos.sdyz**2
        cov_ecef[:, 2, 0] = cov_ecef[:, 0, 2] = np.sign(pos.sdzx)*pos.sdzx**2
        lat = float(ref.iloc[0]["Latitude (deg)"]) if ref is not None else (float(pos.iloc[0]["lat"]) if np.isfinite(pos.iloc[0]["lat"]) else 0.0)
        lon = float(ref.iloc[0]["Longitude (deg)"]) if ref is not None else (float(pos.iloc[0]["lon"]) if np.isfinite(pos.iloc[0]["lon"]) else 0.0)
        p_rad, l_rad = np.deg2rad([lat, lon])
        R = np.array([[-np.sin(l_rad), np.cos(l_rad), 0],
                      [-np.sin(p_rad)*np.cos(l_rad), -np.sin(p_rad)*np.sin(l_rad), np.cos(p_rad)],
                      [np.cos(p_rad)*np.cos(l_rad), np.cos(p_rad)*np.sin(l_rad), np.sin(p_rad)]])
        ce = np.einsum("ij,njk,lk->nil", R, cov_ecef, R)
        pos["sigE"], pos["sigN"], pos["sigU"] = np.sqrt(np.maximum(ce[:, 0, 0], 0)), np.sqrt(np.maximum(ce[:, 1, 1], 0)), np.sqrt(np.maximum(ce[:, 2, 2], 0))
        pos["covEN"] = ce[:, 0, 1]
        pos["rhoEN"] = ce[:, 0, 1] / np.maximum(pos.sigE * pos.sigN, 1e-15)
    sat = []
    if stat_path.exists():
        for line in stat_path.open(encoding="utf-8", errors="replace"):
            if not line.startswith("$SAT,"): continue
            f = line.rstrip().split(",")
            try:
                sat.append([int(f[1]), float(f[2]), f[3], int(f[4]), float(f[5]), float(f[6]), float(f[7]), float(f[8]), int(f[9]), float(f[10]), int(f[11]), int(f[12])])
            except (ValueError, IndexError): pass
    sats = pd.DataFrame(sat, columns=["week", "tow", "prn", "freq", "az", "el", "resp", "resc", "vsat", "snr", "fix", "slipc"])
    if not sats.empty:
        sats["time"] = sats.tow - pos.tow.iloc[0]
        sats["constellation"] = sats.prn.str[0].map({"G":"GPS", "R":"GLONASS", "E":"Galileo", "C":"BeiDou", "J":"QZSS", "S":"SBAS"}).fillna("Other")
    return {"pos": pos, "sat": sats, "reference": ref, "has_truth": "de" in pos,
            "pos_path": pos_path, "stat_path": stat_path}


def compute_segment_metrics(pos_df: pd.DataFrame, sat_df: pd.DataFrame | None = None) -> dict:
    """Compute key quantitative GNSS precision and availability metrics for a given epoch segment."""
    metrics = {
        "epochs": len(pos_df),
        "fix_rate": 0.0,
        "float_rate": 0.0,
        "has_truth": False,
        "rms_e": 0.0,
        "rms_n": 0.0,
        "rms_u": 0.0,
        "rms_3d": 0.0,
        "cep50": 0.0,
        "cep95": 0.0,
        "mean_ns": 0.0,
        "mean_ratio": 0.0,
    }
    if pos_df is None or pos_df.empty:
        return metrics

    if "q" in pos_df:
        metrics["fix_rate"] = float((pos_df["q"] == 1).mean() * 100)
        metrics["float_rate"] = float((pos_df["q"] == 2).mean() * 100)

    if "ns" in pos_df:
        metrics["mean_ns"] = float(pos_df["ns"].mean())

    if "ratio" in pos_df:
        metrics["mean_ratio"] = float(pos_df["ratio"].mean())

    has_truth = all(c in pos_df.columns for c in ("de", "dn", "du")) and pos_df["de"].notna().any()
    metrics["has_truth"] = has_truth
    if has_truth:
        de = pos_df["de"].dropna().to_numpy()
        dn = pos_df["dn"].dropna().to_numpy()
        du = pos_df["du"].dropna().to_numpy()
        if len(de) > 0:
            metrics["rms_e"] = float(np.sqrt(np.mean(de**2)))
            metrics["rms_n"] = float(np.sqrt(np.mean(dn**2)))
            metrics["rms_u"] = float(np.sqrt(np.mean(du**2)))
            metrics["rms_3d"] = float(np.sqrt(np.mean(de**2 + dn**2 + du**2)))
            eh = np.hypot(de, dn)
            metrics["cep50"] = float(np.quantile(eh, 0.50))
            metrics["cep95"] = float(np.quantile(eh, 0.95))
    else:
        if "sigE" in pos_df and pos_df["sigE"].notna().any():
            metrics["rms_e"] = float(np.sqrt(np.mean(pos_df["sigE"].dropna()**2)))
            metrics["rms_n"] = float(np.sqrt(np.mean(pos_df["sigN"].dropna()**2)))
            metrics["rms_u"] = float(np.sqrt(np.mean(pos_df["sigU"].dropna()**2)))
            metrics["rms_3d"] = float(np.sqrt(metrics["rms_e"]**2 + metrics["rms_n"]**2 + metrics["rms_u"]**2))
            metrics["cep50"] = float(metrics["rms_e"] * 0.5887 + metrics["rms_n"] * 0.5887)
            metrics["cep95"] = float(metrics["cep50"] * 2.08)

    return metrics


def render_hud_card(ax, metrics: dict, loc: str = "upper right", font_size: float = 8.5):
    """Render a clean, semi-transparent academic stats HUD card in the given subplot axes."""
    lines = [
        f"FIX Rate: {metrics['fix_rate']:.1f}%",
    ]
    if metrics.get("has_truth"):
        lines.append(f"RMS (E/N/U): {metrics['rms_e']:.3f} / {metrics['rms_n']:.3f} / {metrics['rms_u']:.3f} m")
        lines.append(f"RMS 3D: {metrics['rms_3d']:.3f} m")
        lines.append(f"CEP50 / 95: {metrics['cep50']:.3f} / {metrics['cep95']:.3f} m")
    elif metrics.get("rms_3d", 0) > 0:
        lines.append(f"Est 1σ (E/N/U): {metrics['rms_e']:.3f} / {metrics['rms_n']:.3f} / {metrics['rms_u']:.3f} m")
    lines.append(f"Mean Sats: {metrics['mean_ns']:.1f} | Mean AR: {metrics['mean_ratio']:.1f}")

    text = "\n".join(lines)
    x = 0.98 if "right" in loc else 0.02
    y = 0.97 if "upper" in loc else 0.05
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"

    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=font_size,
        fontfamily="monospace",
        fontweight="bold",
        color="#0B2B52",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#F8FAFC",
            edgecolor="#0070C0",
            alpha=0.88,
            linewidth=1.2,
        ),
        zorder=100
    )


def crop_and_export_pos(pos_path: Path, output_path: Path, start_tow: float, end_tow: float) -> int:
    """Crop the original .pos file within [start_tow, end_tow] while preserving original headers & formats."""
    from datetime import datetime
    pos_path = Path(pos_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    lo, hi = sorted((start_tow, end_tow))

    with pos_path.open("r", encoding="utf-8", errors="replace") as f_in, \
         output_path.open("w", encoding="utf-8", newline="\n") as f_out:
        for line in f_in:
            if line.startswith("%") or line.strip() == "":
                f_out.write(line)
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tow = None
            if "/" in parts[0] or "-" in parts[0]:
                t_str = parts[0] + " " + parts[1]
                try:
                    dt = datetime.strptime(t_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    try:
                        dt = datetime.strptime(t_str, "%Y/%m/%d %H:%M:%S")
                    except ValueError:
                        continue
                diff = (dt - datetime(1980, 1, 6)).total_seconds()
                tow = float(diff % 604800)
            else:
                try:
                    tow = float(parts[1])
                except ValueError:
                    continue

            if lo - 1e-4 <= tow <= hi + 1e-4:
                f_out.write(line)
                count += 1
    return count


def compute_quality_report(data: dict) -> dict:
    """Generate a comprehensive GNSS positioning health and quality report."""
    pos = data["pos"]
    sat = data.get("sat", None)
    has_truth = data.get("has_truth", False)
    n = len(pos)
    if n == 0:
        return {}

    duration_sec = float(pos.tow.iloc[-1] - pos.tow.iloc[0]) if n > 1 else 0.0
    fix_cnt = int((pos.q == 1).sum()) if "q" in pos else 0
    float_cnt = int((pos.q == 2).sum()) if "q" in pos else 0
    single_cnt = int((pos.q == 5).sum()) if "q" in pos else 0

    fix_rate = (fix_cnt / n) * 100
    float_rate = (float_cnt / n) * 100
    single_rate = (single_cnt / n) * 100

    metrics = compute_segment_metrics(pos, sat)

    age_valid = pos.age.dropna() if "age" in pos else pd.Series(dtype=float)
    age_mean = float(age_valid.mean()) if not age_valid.empty else 0.0
    age_max = float(age_valid.max()) if not age_valid.empty else 0.0
    age_delayed_pct = float((age_valid > 5.0).mean() * 100) if not age_valid.empty else 0.0

    const_info = {}
    if sat is not None and not sat.empty:
        s1 = sat[sat.freq == 1]
        for const, g in s1.groupby("constellation"):
            valid_snr = g.snr[g.snr > 0]
            const_info[const] = {
                "prns": int(g.prn.nunique()),
                "mean_snr": float(valid_snr.mean()) if not valid_snr.empty else 0.0,
                "slip_count": int((g.slipc > 0).sum()) if "slipc" in g else 0
            }

    score = fix_rate * 0.50
    if has_truth:
        rms_3d = metrics["rms_3d"]
        acc_score = max(0.0, min(30.0, 30.0 * (0.2 / max(rms_3d, 0.01))))
    else:
        sig_3d = metrics["rms_3d"]
        acc_score = max(0.0, min(30.0, 30.0 * (0.5 / max(sig_3d, 0.05))))
    score += acc_score
    score += min(10.0, (metrics["mean_ns"] / 20.0) * 10.0)
    score += min(10.0, (metrics["mean_ratio"] / 3.0) * 10.0)
    score = round(max(0.0, min(100.0, score)), 1)

    if score >= 90: grade = "卓越 (Grade A+)"
    elif score >= 80: grade = "良好 (Grade A)"
    elif score >= 65: grade = "中等 (Grade B)"
    else: grade = "警惕需排查 (Grade C - Attention Required)"

    return {
        "epochs": n,
        "duration_sec": duration_sec,
        "fix_rate": fix_rate,
        "float_rate": float_rate,
        "single_rate": single_rate,
        "score": score,
        "grade": grade,
        "metrics": metrics,
        "age_mean": age_mean,
        "age_max": age_max,
        "age_delayed_pct": age_delayed_pct,
        "constellations": const_info,
        "has_truth": has_truth
    }


def generate_markdown_report(rep: dict, pos_name: str = "solution") -> str:
    """Format the quality report dictionary into a publication-ready Markdown report."""
    if not rep:
        return "# RTKLIB 解算质量评估报告\n\n暂无有效数据。"

    m = rep["metrics"]
    lines = [
        f"# RTKLIB GNSS 解算综合诊断与质量评估报告",
        f"",
        f"- **数据集文件**: `{pos_name}`",
        f"- **总历元样本数**: `{rep['epochs']:,}` 历元 | **连续解算时长**: `{rep['duration_sec']:.1f}` 秒 ({rep['duration_sec']/60:.1f} 分钟)",
        f"- **综合健康度评分**: **{rep['score']} / 100** ({rep['grade']})",
        f"",
        f"---",
        f"## 1. 核心解算状态与可用性 (Solution Status & Availability)",
        f"",
        f"| 指标项 (Metric) | 统计数值 (Value) | 行业参考门限 (Benchmark) | 健康度评估 |",
        f"| :--- | :--- | :--- | :--- |",
        f"| **FIX 固定率** | **{rep['fix_rate']:.2f}%** | 建议 ≥ 90.0% | {'✅ 达标 (优质)' if rep['fix_rate']>=90 else ('⚠️ 偏低 (中等)' if rep['fix_rate']>=75 else '❌ 需优化')} |",
        f"| **FLOAT 浮点率** | **{rep['float_rate']:.2f}%** | 建议 ≤ 10.0% | {'✅ 良好' if rep['float_rate']<=10 else '⚠️ 偏高 (存在遮挡或周跳)'} |",
        f"| **SINGLE 单点率** | **{rep['single_rate']:.2f}%** | 建议 ≈ 0.0% | {'✅ 极佳' if rep['single_rate']<0.5 else '❌ 异常 (差分中断)'} |",
        f"| **平均可用卫星数** | **{m['mean_ns']:.1f} 颗** | 建议 ≥ 18 颗 | {'✅ 几何构型丰富' if m['mean_ns']>=18 else '⚠️ 卫星数不足'} |",
        f"| **平均 AR Ratio** | **{m['mean_ratio']:.2f}** | 检核门限 3.0 | {'✅ 模糊度解算稳固' if m['mean_ratio']>=3.0 else '⚠️ 接近临界门限'} |",
        f"",
        f"---",
        f"## 2. 空间定位精度与误差统计 (Positioning Accuracy & Statistics)",
        f"",
    ]
    if rep["has_truth"]:
        lines.extend([
            f"| 空间分量 (Component) | RMS 均方根误差 (m) | CEP 50% 误差 (m) | CEP 95% 置信误差 (m) |",
            f"| :--- | :--- | :--- | :--- |",
            f"| **水平平面 (Horizontal)** | **{np.hypot(m['rms_e'], m['rms_n']):.3f} m** | **{m['cep50']:.3f} m** | **{m['cep95']:.3f} m** |",
            f"| **东向分量 (East, ΔE)** | {m['rms_e']:.3f} m | — | — |",
            f"| **北向分量 (North, ΔN)** | {m['rms_n']:.3f} m | — | — |",
            f"| **高程天向 (Up, ΔU)** | {m['rms_u']:.3f} m | — | — |",
            f"| **空间三维 (3D RMS)** | **{m['rms_3d']:.3f} m** | — | — |",
        ])
    else:
        lines.extend([
            f"*(说明：当前未加载真值 CSV 轨迹，以下基于 RTKLIB 解算协方差阵估算)*  \n",
            f"| 空间分量 (Component) | 内部先验 1σ 误差 (m) | 名义 95% 置信包络 (1.96σ, m) |",
            f"| :--- | :--- | :--- |",
            f"| **水平先验 (Estimated H)** | {m['cep50']:.3f} m | {m['cep95']:.3f} m |",
            f"| **东向先验 (Est σE)** | {m['rms_e']:.3f} m | {m['rms_e']*1.96:.3f} m |",
            f"| **北向先验 (Est σN)** | {m['rms_n']:.3f} m | {m['rms_n']*1.96:.3f} m |",
            f"| **天向先验 (Est σU)** | {m['rms_u']:.3f} m | {m['rms_u']*1.96:.3f} m |",
        ])

    lines.extend([
        f"",
        f"---",
        f"## 3. 多系统卫星星历与信号质量 (Multi-Constellation Breakdown)",
        f"",
        f"| 星座系统 (Constellation) | 可见卫星总数 (PRNs) | 平均载噪比 C/N₀ (dB-Hz) | 累计标记周跳 (Cycle Slips) |",
        f"| :--- | :--- | :--- | :--- |",
    ])
    for const, info in rep["constellations"].items():
        lines.append(f"| **{const}** | {info['prns']} 颗 | {info['mean_snr']:.1f} dB-Hz | {info['slip_count']} 次 |")

    lines.extend([
        f"",
        f"---",
        f"## 4. 基准站差分链路健康度 (Differential Link & Latency)",
        f"",
        f"- **平均差分龄期 (Mean Age)**: `{rep['age_mean']:.2f} s`",
        f"- **最大差分延迟 (Max Latency)**: `{rep['age_max']:.2f} s`",
        f"- **差分延迟超标率 (Age > 5.0s)**: `{rep['age_delayed_pct']:.2f}%`",
        f"",
        f"---",
        f"*由 RTKLIB-Plot-Python 智能诊断与分析系统自动生成*"
    ])
    return "\n".join(lines)


def detect_problematic_segments(pos_df: pd.DataFrame, error_limit: float = 1.0, min_duration: float = 2.0) -> list[dict]:
    """Scan position epochs and automatically identify loss-of-fix or severe error intervals."""
    if pos_df is None or len(pos_df) < 5:
        return []

    times = pos_df.time.to_numpy()
    segments = []
    palette = ["#b23a48", "#d17b0f", "#ae5a9d", "#e07a2d", "#c00000", "#7b61a8"]

    if "q" in pos_df:
        not_fix = (pos_df.q != 1).to_numpy()
        diffs = np.diff(not_fix.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        if not_fix[0]: starts = np.r_[0, starts]
        ends = np.where(diffs == -1)[0]
        if not_fix[-1]: ends = np.r_[ends, len(not_fix)-1]

        for s_idx, e_idx in zip(starts, ends):
            dur = float(times[e_idx] - times[s_idx])
            if dur >= min_duration:
                q_sub = pos_df.q.iloc[s_idx:e_idx+1]
                q_type = "FLOAT" if (q_sub == 2).any() else ("SINGLE" if (q_sub == 5).any() else "NON-FIX")
                c = palette[len(segments) % len(palette)]
                segments.append({
                    "start": int(s_idx),
                    "end": int(e_idx),
                    "name": f"掉固定: {q_type}持续{dur:.1f}s",
                    "reason": f"{q_type} 持续 {dur:.1f} 秒 (历元 {s_idx+1}~{e_idx+1})",
                    "color": c,
                    "type": "loss_of_fix"
                })

    if "e3d" in pos_df and pos_df["e3d"].notna().any():
        high_err = (pos_df.e3d > error_limit).to_numpy()
        diffs = np.diff(high_err.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        if high_err[0]: starts = np.r_[0, starts]
        ends = np.where(diffs == -1)[0]
        if high_err[-1]: ends = np.r_[ends, len(high_err)-1]

        for s_idx, e_idx in zip(starts, ends):
            dur = float(times[e_idx] - times[s_idx])
            if dur >= min_duration:
                already = any(abs(seg["start"] - s_idx) < 10 and abs(seg["end"] - e_idx) < 10 for seg in segments)
                if not already:
                    peak = float(pos_df.e3d.iloc[s_idx:e_idx+1].max())
                    c = palette[len(segments) % len(palette)]
                    segments.append({
                        "start": int(s_idx),
                        "end": int(e_idx),
                        "name": f"超标: 3D误差峰值{peak:.2f}m",
                        "reason": f"3D误差超标 (>{error_limit}m) 持续 {dur:.1f} 秒，峰值 {peak:.2f}m",
                        "color": c,
                        "type": "high_error"
                    })

    segments.sort(key=lambda x: x["start"])
    return segments


def _save(fig, out, name):
    fig.tight_layout()
    for fmt in EXPORT_FORMATS:
        fig.savefig(out / f"{name}.{fmt}", format=fmt,
                    bbox_inches="tight", dpi=EXPORT_DPI if fmt == "png" else None)
    plt.close(fig)

def _status_shade(ax, p):
    if "q" not in p: return
    t = p.time.to_numpy(); q = p.q.to_numpy()
    for i in range(len(t)-1):
        ax.axvspan(t[i], t[i+1], color={1:"#61a96d",2:"#e9b44c"}.get(q[i],"#dddddd"), alpha=.10, lw=0)

def _title(fig, text): fig.suptitle(text, fontweight="bold", fontsize=15)

def plot_01(d, out):
    p=d["pos"]; fig,ax=plt.subplots(1,2,figsize=(12,4.2)); _title(fig,"01 · ENU 轨迹与高程")
    ax[0].plot(p.E,p.N,color="#444",lw=1,label="RTK")
    if d["has_truth"]: ax[0].plot(p.gt_E,p.gt_N,"--",color="#cf4b43",lw=1,label="真值")
    ax[0].set(xlabel="东向 E (m)",ylabel="北向 N (m)",title="水平轨迹"); ax[0].axis("equal"); ax[0].legend(); ax[0].grid(alpha=.25)
    _status_shade(ax[1],p); ax[1].plot(p.time,p.U,label="RTK U")
    if d["has_truth"]: ax[1].plot(p.time,p.gt_U,label="真值 U"); ax[1].legend()
    ax[1].set(xlabel="相对时间 (s)",ylabel="ENU 高程 (m)",title="高程与解算状态"); ax[1].grid(alpha=.25)
    _save(fig,out,"plot_01_enu_position")

def plot_02(d,out):
    p=d["pos"]
    if not d["has_truth"]: return _unavailable(out,"plot_02_enu_error","需要真值才能计算 ΔE/ΔN/ΔU")
    fig,axs=plt.subplots(3,1,figsize=(11,7),sharex=True); _title(fig,"02 · ENU 位置误差与 95% 置信区间")
    for ax,c,sig,col in zip(axs,["de","dn","du"],["sigE","sigN","sigU"],[COLORS["E"],COLORS["N"],COLORS["U"]]):
        x=p[c].dropna(); mu=x.mean(); sd=x.std(); ax.plot(p.time,p[c],lw=.65,color=col); ax.axhline(mu,color="black",ls="--",lw=.8,label=f"均值 {mu:.3f} m")
        ax.axhspan(mu-1.96*sd,mu+1.96*sd,color=col,alpha=.12,label=f"经验区间 μ±1.96σ [{mu-1.96*sd:.3f}, {mu+1.96*sd:.3f}] m")
        ax.plot(p.time,1.96*p[sig],color="#777",ls=":",lw=.7); ax.plot(p.time,-1.96*p[sig],color="#777",ls=":",lw=.7,label="滤波 ±1.96σ")
        ax.set_ylabel(c.upper()+" (m)"); ax.grid(alpha=.2); ax.legend(loc="upper right",fontsize=7,ncol=2)
    axs[-1].set_xlabel("相对时间 (s)"); _save(fig,out,"plot_02_enu_error")

def plot_03(d,out):
    p=d["pos"]
    if not d["has_truth"]: return _unavailable(out,"plot_03_horizontal_3d_error","需要真值才能计算水平/三维误差")
    fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"03 · 水平、三维误差与经验 CDF")
    ax[0].plot(p.time,p.eh,label="水平 √(ΔE²+ΔN²)"); ax[0].plot(p.time,p.e3d,label="三维 √(ΔE²+ΔN²+ΔU²)"); ax[0].set(xlabel="时间 (s)",ylabel="误差 (m)"); ax[0].legend(); ax[0].grid(alpha=.2)
    for c,label in [("eh","水平"),("e3d","三维")]:
        x=np.sort(p[c].dropna()); ax[1].plot(x,100*np.arange(1,len(x)+1)/len(x),label=label)
    ax[1].set(xlabel="误差 (m)",ylabel="经验累积概率 (%)",title=f"水平 RMSE={np.sqrt(np.mean(p.eh**2)):.3f} m · 3D RMSE={np.sqrt(np.mean(p.e3d**2)):.3f} m"); ax[1].legend(); ax[1].grid(alpha=.2)
    _save(fig,out,"plot_03_horizontal_3d_error")

def plot_04(d,out):
    p=d["pos"]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"04 · 解算状态与历元占比")
    ax[0].step(p.time,p.q,where="post",lw=.8); ax[0].set(xlabel="时间 (s)",ylabel="Q 状态",yticks=sorted(p.q.unique()),title="RTKLIB 状态时间序列"); ax[0].grid(alpha=.2)
    vc=p.q.value_counts().sort_index(); names={1:"FIX",2:"FLOAT",3:"SBAS",4:"DGPS",5:"单点",6:"PPP"}; ax[1].bar([names.get(i,str(i)) for i in vc.index],vc.values/len(p)*100,color=["#4b9b64" if i==1 else "#e5ae46" for i in vc.index]); ax[1].set(ylabel="历元比例 (%)",title="状态构成")
    for x,y in enumerate(vc.values/len(p)*100): ax[1].text(x,y,f"{y:.1f}%",ha="center",va="bottom")
    _save(fig,out,"plot_04_solution_status")

def plot_05(d,out):
    p=d["pos"]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"05 · 模糊度固定 Ratio")
    ax[0].plot(p.time,p.ratio,lw=.65); ax[0].axhline(3,color="red",ls="--",label="参考门限 3"); ax[0].set(xlabel="时间 (s)",ylabel="Ratio",ylim=(0,max(5,min(50,p.ratio.quantile(.995)))*1.1)); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].hist(p.ratio.clip(upper=p.ratio.quantile(.99)),bins=40,color="#607da8"); ax[1].axvline(3,color="red",ls="--"); ax[1].set(xlabel="Ratio (截于 99% 分位)",ylabel="历元数",title=f"中位数 {p.ratio.median():.2f} · >3 占比 {(p.ratio>3).mean()*100:.1f}%"); _save(fig,out,"plot_05_ar_ratio")

def _sat_epoch(d):
    s=d["sat"]
    if s.empty: return pd.DataFrame()
    return (s.groupby(["tow","time"])
            .apply(lambda g: pd.Series({"tracked": g.prn.nunique(),
                                        "used": g.loc[g.vsat == 1, "prn"].nunique()}),
                   include_groups=False).reset_index())

def plot_06(d,out):
    s=d["sat"]
    if s.empty: return _unavailable(out,"plot_06_satellite_counts",".stat 中无 $SAT 记录")
    e=_sat_epoch(d); fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True); _title(fig,"06 · 跟踪、使用卫星数与星座组成")
    ax[0].plot(e.time,e.tracked,label="跟踪 PRN 数"); ax[0].plot(e.time,e.used,label="使用频率记录数"); ax[0].set(ylabel="数量"); ax[0].legend(); ax[0].grid(alpha=.2)
    ss=s[(s.freq==1)&(s.vsat==1)].groupby(["time","constellation"]).prn.nunique().unstack(fill_value=0)
    ss.plot.area(ax=ax[1],stacked=True,alpha=.85,color=[COLORS.get(c) for c in ss.columns]); ax[1].set(xlabel="时间 (s)",ylabel="使用卫星数",title="按星座堆叠（L1 频率、vsat=1、按 PRN 去重）"); ax[1].legend(ncol=6,fontsize=7); _save(fig,out,"plot_06_satellite_counts")

def _compute_dop(s):
    rows=[]
    for (tow,t),g in s[(s.freq==1)&(s.vsat==1)].groupby(["tow","time"]):
        g=g.drop_duplicates("prn")
        if len(g)<4: continue
        az,el=np.deg2rad(g.az),np.deg2rad(g.el)
        A=np.c_[np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el),np.ones(len(g))]
        try:
            Q=np.linalg.inv(A.T@A); rows.append([tow,t,*np.sqrt(np.maximum([Q[0,0]+Q[1,1]+Q[2,2],Q[0,0]+Q[1,1],Q[2,2]],0))])
        except np.linalg.LinAlgError: pass
    return pd.DataFrame(rows,columns=["tow","time","PDOP","HDOP","VDOP"])

def plot_07(d,out):
    dop=_compute_dop(d["sat"])
    if dop.empty:return _unavailable(out,"plot_07_dop","有效 $SAT 几何不足以计算 DOP")
    fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"07 · 由卫星视线几何计算的 DOP")
    for c in ["PDOP","HDOP","VDOP"]: ax[0].plot(dop.time,dop[c],label=c)
    ax[0].axhspan(0,2,color="#55a868",alpha=.10); ax[0].axhspan(2,5,color="#ddaa33",alpha=.10); ax[0].axhspan(5,50,color="#c44e52",alpha=.08); ax[0].set(xlabel="时间 (s)",ylabel="DOP",ylim=(0,min(20,dop.PDOP.quantile(.99)*1.2))); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].hist(dop.PDOP,bins=35,color="#6688aa"); ax[1].set(xlabel="PDOP",ylabel="历元数",title=f"median={dop.PDOP.median():.2f} · 95%={dop.PDOP.quantile(.95):.2f}"); _save(fig,out,"plot_07_dop")

def plot_08(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_08_skyplot",".stat 中无卫星方位/高度角")
    fig=plt.figure(figsize=(7,7)); ax=fig.add_subplot(111,projection="polar"); _title(fig,"08 · 卫星天空图（方位角与高度角轨迹）")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(90,0); ax.set_yticks([0,30,60,90]); ax.set_yticklabels(["90°","60°","30°","0°"])
    s=s[s.freq==1]
    for (prn,con),g in s.groupby(["prn","constellation"]):
        g=g.sort_values("time"); th=np.deg2rad(g.az); r=90-g.el; color=COLORS.get(con,"#555")
        ax.plot(th,r,color=color,alpha=.45,lw=.6); ax.scatter(th.iloc[::max(1,len(g)//20)],r.iloc[::max(1,len(g)//20)],s=5,color=color)
        k=int(np.nanargmax(g.el.to_numpy())); ax.text(th.iloc[k],r.iloc[k],prn,fontsize=6,color=color)
    _save(fig,out,"plot_08_skyplot")

def plot_09(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_09_cn0",".stat 中无 C/N0")
    s=s[s.snr>0]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"09 · C/N₀ 信号质量")
    for f,g in s.groupby("freq"): ax[0].scatter(g.time,g.snr,s=2,alpha=.12,label=f"频率 {f}")
    ax[0].set(xlabel="时间 (s)",ylabel="C/N₀ (dB-Hz)"); ax[0].legend(); ax[0].grid(alpha=.2)
    groups=[(c,f,s[(s.constellation==c)&(s.freq==f)].snr.to_numpy()) for c in s.constellation.unique() for f in sorted(s.freq.unique())]; groups=[x for x in groups if len(x[2])]
    ax[1].boxplot([g[2] for g in groups],showfliers=False); ax[1].set_xticks(range(1,len(groups)+1),[f"{c}\nL{f}" for c,f,_ in groups],rotation=25); ax[1].set(ylabel="C/N₀ (dB-Hz)",title="按星座与频率分布"); ax[1].grid(axis="y",alpha=.2); _save(fig,out,"plot_09_cn0")

def plot_10(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_10_elevation",".stat 中无卫星高度角")
    s=s[s.freq==1]; fig,ax=plt.subplots(1,3,figsize=(14,4)); _title(fig,"10 · 卫星高度角与观测质量")
    for prn,g in s.groupby("prn"): ax[0].plot(g.time,g.el,lw=.45,alpha=.35)
    ax[0].set(xlabel="时间 (s)",ylabel="高度角 (°)",title="L1 卫星仰角"); ax[0].grid(alpha=.2)
    ax[1].scatter(s.el,s.snr,s=3,alpha=.18); ax[1].set(xlabel="高度角 (°)",ylabel="C/N₀ (dB-Hz)",title="高度角与信号强度"); ax[1].grid(alpha=.2)
    ax[2].scatter(s.el,s.resp.abs(),s=3,alpha=.18); ax[2].set(xlabel="高度角 (°)",ylabel="|伪距残差| (m)",title="高度角与伪距残差"); ax[2].grid(alpha=.2); _save(fig,out,"plot_10_elevation")

def plot_11(d,out):
    # $SAT residual is a post-fit residual, not a direct multipath observable.
    s=d["sat"]; s=s[(s.freq==1)&(s.resp!=0)]
    if s.empty:return _unavailable(out,"plot_11_multipath","无有效伪距残差；且 .stat 不含独立多路径组合")
    s=s.copy(); s["abs_resid"] = s.resp.abs(); s["elbin"]=pd.cut(s.el,np.arange(0,100,10),include_lowest=True)
    rms=s.groupby("elbin",observed=False).resp.apply(lambda x:np.sqrt(np.mean(x*x)))
    fig,ax=plt.subplots(1,2,figsize=(10,4)); _title(fig,"11 · 伪距残差诊断（多路径代理量，非 MP 组合）")
    ax[0].scatter(s.time,s.resp,s=2,alpha=.12); ax[0].set(xlabel="时间 (s)",ylabel="伪距残差 (m)"); ax[0].grid(alpha=.2)
    valid=[(x.mid,v) for x,v in rms.items() if not pd.isna(x) and np.isfinite(v)]
    if valid: ax[1].plot([x for x,_ in valid],[v for _,v in valid],"o-")
    ax[1].set(xlabel="高度角区间中点 (°)",ylabel="伪距残差 RMS (m)"); ax[1].grid(alpha=.2); _save(fig,out,"plot_11_multipath")

def plot_12(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_12_residuals",".stat 中无观测残差")
    fig,ax=plt.subplots(2,2,figsize=(11,7)); _title(fig,"12 · RTKLIB $SAT 观测残差")
    for a,col,label,scale in [(ax[0,0],"resp","伪距残差 (m)",1),(ax[1,0],"resp","伪距残差 (m)",1),(ax[0,1],"resc","载波残差 (m)",100),(ax[1,1],"resc","载波残差 (cm)",100)]:
        x=s.loc[s[col]!=0,col].to_numpy()*scale
        if a in (ax[0,0],ax[0,1]): a.scatter(s.loc[s[col]!=0,"time"],x,s=2,alpha=.1)
        elif len(x): a.hist(x,bins=60,density=True,alpha=.65); a.axvline(np.mean(x),color="red",ls="--",label=f"均值={np.mean(x):.3g}"); a.legend()
        a.set(xlabel="时间 (s)" if a in (ax[0,0],ax[0,1]) else label,ylabel=label if a in (ax[0,0],ax[0,1]) else "概率密度")
    _save(fig,out,"plot_12_residuals")

def plot_13(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_13_cycle_slip",".stat 中无周跳标记")
    x=s[s.slipc>0]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"13 · RTKLIB 周跳标记")
    if not x.empty: ax[0].scatter(x.time,x.prn,s=8,c=x.slipc,cmap="tab10");
    ax[0].set(xlabel="时间 (s)",ylabel="卫星 PRN",title=f"$SAT slipc>0: {len(x)} 条记录"); ax[0].grid(alpha=.2)
    count=s[s.slipc>0].groupby("time").prn.nunique(); all_t=np.sort(s.time.unique()); y=pd.Series(count,index=count.index).reindex(all_t,fill_value=0).to_numpy(); ax[1].step(all_t,y,where="post",label="每历元涉及卫星数"); ax[1].plot(all_t,np.cumsum(y),label="累计"); ax[1].set(xlabel="时间 (s)",ylabel="数量"); ax[1].legend(); ax[1].grid(alpha=.2); _save(fig,out,"plot_13_cycle_slip")

def plot_14(d,out):
    p=d["pos"]
    fig,ax=plt.subplots(2,2,figsize=(11,7)); _title(fig,"14 · .pos 协方差与误差覆盖")
    for a,c in zip(ax[0],["sigE","sigN","sigU"]): a.plot(p.time,p[c],label=c); a.set_yscale("log"); a.set(xlabel="时间 (s)",ylabel="标准差 (m)"); a.grid(alpha=.2); a.legend()
    ax[1,0].plot(p.time,p.rhoEN); ax[1,0].set(xlabel="时间 (s)",ylabel="相关系数",title="E-N 相关"); ax[1,0].grid(alpha=.2)
    if d["has_truth"]:
        ratio=np.abs(p.de)/np.maximum(1.96*p.sigE,1e-12); ax[1,1].plot(p.time,ratio,lw=.6); ax[1,1].axhline(1,color="red",ls="--"); ax[1,1].set(xlabel="时间 (s)",ylabel="|ΔE|/(1.96σE)",title=f"E 軸誤差包絡覆盖 {(ratio<=1).mean()*100:.1f}%")
    else: ax[1,1].text(.5,.5,"缺真值：无法计算误差覆盖率",ha="center",va="center",transform=ax[1,1].transAxes)
    _save(fig,out,"plot_14_covariance")

def _unavailable(out,name,message):
    fig,ax=plt.subplots(figsize=(8,3)); ax.axis("off"); ax.text(.5,.58,name,ha="center",fontsize=15,weight="bold"); ax.text(.5,.4,message,ha="center",wrap=True); _save(fig,out,name)

def plot_15(d,out):
    # RTKLIB .stat here exposes residuals but not innovations or their covariance.
    _unavailable(out,"plot_15_innovation_nis","输入文件没有滤波创新向量 v 或创新协方差 S；NIS=vᵀS⁻¹v 无法从 .pos/.stat 可靠恢复。")

def plot_16_cdf(d, out):
    p = d["pos"]
    if not d["has_truth"]:
        return _unavailable(out, "plot_16_error_cdf", "需要真值参考轨迹才能绘制误差累积概率分布 (CDF)")
    fig, ax = plt.subplots(figsize=(10, 5.5)); _title(fig, "16 · 定位误差累积分布函数 (CDF)")
    eh = np.sort(p.eh.dropna())
    e3d = np.sort(p.e3d.dropna())
    if len(eh) == 0:
        return _unavailable(out, "plot_16_error_cdf", "无有效误差数据")
    cdf_y = np.linspace(0, 100, len(eh))
    ax.plot(eh, cdf_y, lw=1.8, color="#2878B5", label=f"水平误差 (CEP50={np.quantile(eh, 0.5):.3f}m, CEP95={np.quantile(eh, 0.95):.3f}m)")
    ax.plot(e3d, cdf_y, lw=1.8, color="#E07A2D", label=f"3D误差 (P50={np.quantile(e3d, 0.5):.3f}m, P95={np.quantile(e3d, 0.95):.3f}m)")
    ax.axhline(50, color="gray", ls="--", lw=0.9, alpha=0.7)
    ax.axhline(68.3, color="green", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(95, color="red", ls="--", lw=0.9, alpha=0.7)
    ax.set(xlabel="误差 (m)", ylabel="累积概率 (%)", title="定位误差累积概率分布曲线 (CDF)")
    ax.set_xlim(0, max(0.1, float(np.quantile(e3d, 0.99) * 1.15)))
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out, "plot_16_error_cdf")

def plot_17_hist(d, out):
    p = d["pos"]
    if not d["has_truth"]:
        return _unavailable(out, "plot_17_error_histogram", "需要真值参考轨迹才能绘制误差分布直方图")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True); _title(fig, "17 · ENU 误差正态分布直方图")
    for ax, c, col, label in zip(axs, ["de", "dn", "du"], [COLORS["E"], COLORS["N"], COLORS["U"]], ["东向误差 ΔE", "北向误差 ΔN", "高程误差 ΔU"]):
        vals = p[c].dropna()
        if vals.empty: continue
        mu, std = float(vals.mean()), float(vals.std())
        n, bins, _ = ax.hist(vals, bins=40, density=True, alpha=0.6, color=col, edgecolor="white")
        x = np.linspace(bins[0], bins[-1], 200)
        if std > 1e-6:
            y = (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mu) / std)**2)
            ax.plot(x, y, color="#222", lw=1.4, label=f"拟合 N({mu:+.3f}, {std:.3f}²)")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set(xlabel=f"{label} (m)", ylabel="概率密度" if c=="de" else "")
        ax.set_title(f"{label} (均值={mu:+.3f}m, RMS={np.sqrt(np.mean(vals**2)):.3f}m)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    _save(fig, out, "plot_17_error_histogram")

def plot_18_constellation(d, out):
    s = d["sat"]
    if s.empty:
        return _unavailable(out, "plot_18_constellation", "缺少 .stat 卫星观测数据")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5)); _title(fig, "18 · 各星座系统卫星数与载噪比分布")
    s1 = s[s.freq == 1]
    counts = s1.groupby(["time", "constellation"]).prn.nunique().unstack(fill_value=0)
    for col in counts.columns:
        axs[0].plot(counts.index, counts[col], label=col, color=COLORS.get(col, "#666"), lw=1.0)
    axs[0].set(xlabel="时间 (s)", ylabel="可见卫星数 (颗)", title="各系统跟踪卫星数时序"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
    valid_snr = s1[s1.snr > 0]
    data_by_const = [valid_snr[valid_snr.constellation == c].snr for c in counts.columns if not valid_snr[valid_snr.constellation == c].empty]
    labels_by_const = [c for c in counts.columns if not valid_snr[valid_snr.constellation == c].empty]
    if data_by_const:
        axs[1].boxplot(data_by_const, tick_labels=labels_by_const, patch_artist=True)
    axs[1].set(xlabel="星座系统", ylabel="载噪比 C/N₀ (dB-Hz)", title="各系统 L1 载噪比分布盒须图"); axs[1].grid(alpha=0.25)
    _save(fig, out, "plot_18_constellation")

PLOTTERS=[plot_01,plot_02,plot_03,plot_04,plot_05,plot_06,plot_07,plot_08,plot_09,plot_10,plot_11,plot_12,plot_13,plot_14,plot_15,plot_16_cdf,plot_17_hist,plot_18_constellation]

def make_dashboard(d,out):
    p=d["pos"]; fig,ax=plt.subplots(2,3,figsize=(15,8)); _title(fig,"RTKLIB 解算质量总览 Dashboard")
    ax[0,0].plot(p.E,p.N,lw=.7); ax[0,0].set(title="ENU 轨迹",xlabel="E (m)",ylabel="N (m)"); ax[0,0].axis("equal")
    if d["has_truth"]:
        ax[0,1].plot(p.time,p.eh,lw=.6,label="水平误差"); ax[0,1].plot(p.time,p.e3d,lw=.6,label="3D误差"); ax[0,1].legend(); ax[0,1].set(title="定位误差（纵轴限至 99.5% 分位）",xlabel="时间 (s)",ylabel="m"); ax[0,1].set_ylim(0,p.e3d.quantile(.995)*1.1)
    else: ax[0,1].text(.5,.5,"无真值",ha="center",transform=ax[0,1].transAxes)
    ax[0,2].step(p.time,p.q,where="post"); ax[0,2].set(title=f"FIX 比例 {(p.q==1).mean()*100:.1f}%",xlabel="时间 (s)",ylabel="Q")
    ax[1,0].plot(p.time,p.ratio,lw=.5); ax[1,0].axhline(3,ls="--",color="red"); ax[1,0].set(title="AR Ratio",xlabel="时间 (s)",ylabel="Ratio")
    if not d["sat"].empty:
        e=_sat_epoch(d); ax[1,1].plot(e.time,e.tracked,label="tracked"); ax[1,1].plot(e.time,e.used,label="used"); ax[1,1].legend(); ax[1,1].set(title="卫星数",xlabel="时间 (s)",ylabel="颗")
        ss=d["sat"].query("freq == 1 and snr > 0"); ax[1,2].scatter(ss.el,ss.snr,s=2,alpha=.12); ax[1,2].set(title="高度角与 C/N₀（L1，有效信号）",xlabel="高度角 (°)",ylabel="dB-Hz")
    for a in ax.flat: a.grid(alpha=.18)
    _save(fig,out,"dashboard")

def main():
    global EXPORT_FORMATS, EXPORT_DPI
    ap=argparse.ArgumentParser(description="RTKLIB 结果文件诊断绘图工具：读取 .pos/.stat/真值并导出分析图")
    ap.add_argument("--pos",type=Path,default=DEFAULT_POS,help="RTKLIB .pos 解算结果")
    ap.add_argument("--reference",type=Path,default=DEFAULT_REF,help="可选真值 CSV；设为 none 可禁用")
    ap.add_argument("--stat",type=Path,default=None,help="可选 .pos.stat 文件，默认自动读取 <pos>.stat")
    ap.add_argument("--output",type=Path,default=DEFAULT_OUT,help="图片输出目录")
    ap.add_argument("--plots",default="all",help="all 或逗号分隔的图号，例如 1,2,3")
    ap.add_argument("--format",default="png",help="导出格式，可多选：png,pdf,svg，例如 png,pdf")
    ap.add_argument("--dpi",type=int,default=180,help="PNG 分辨率")
    args=ap.parse_args()
    EXPORT_FORMATS=tuple(dict.fromkeys(x.strip().lower() for x in args.format.split(",") if x.strip()))
    bad=set(EXPORT_FORMATS)-{"png","pdf","svg"}
    if not EXPORT_FORMATS or bad: ap.error(f"不支持的格式：{', '.join(sorted(bad))}")
    if args.dpi < 72: ap.error("--dpi 至少为 72")
    EXPORT_DPI=args.dpi
    if not args.pos.exists(): ap.error(f"找不到 .pos 文件：{args.pos}")
    ref=None if str(args.reference).lower()=="none" else args.reference
    d=load_data(args.pos,ref,args.stat); args.output.mkdir(parents=True,exist_ok=True)
    chosen=range(1,16) if args.plots.lower()=="all" else sorted(set(int(x) for x in args.plots.split(",")))
    if any(i < 1 or i > 15 for i in chosen): ap.error("图号范围为 1–15")
    for i in chosen:
        PLOTTERS[i-1](d,args.output); print(f"Plot {i:02d}: done")
    make_dashboard(d,args.output); print(f"Dashboard: {args.output/f'dashboard.{EXPORT_FORMATS[0]}'}")
    print(f"Epochs: {len(d['pos'])}; satellite records: {len(d['sat'])}; truth matched: {d['has_truth']}")

if __name__=="__main__": main()
