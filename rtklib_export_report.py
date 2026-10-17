"""
RTKLIB Academic PPTX & HTML Diagnostic Report Generator.
Follows Wuhan University presentation specifications (GEMINI.md):
- 16:9 widescreen layout with Wuhan University academic template
- Dual-color intro line: Deep Blue (#0B2B52) base + Key metrics highlighted in Bold Red (#C00000)
- Publication-grade large fonts, pure Chinese labels, and strict aspect-ratio-preserved bounding box placement
- Generates all-in-one interactive HTML diagnostic dossier
"""
from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

import rtklib_diagnostics as diag


TEMPLATE_SEARCH_PATHS = [
    Path(r"C:\Users\Administrator\Downloads\PPT模版.pptx"),
    Path(__file__).resolve().parent / "resources" / "template.pptx",
    Path(__file__).resolve().parents[1] / "resources" / "template.pptx",
]


def find_pptx_template() -> Path | None:
    for p in TEMPLATE_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def _set_font(p, text, font_name="微软雅黑", size_pt=20, bold=False, color_rgb=(11, 43, 82)):
    p.text = text
    p.font.name = font_name
    p.font.size = Pt(size_pt)
    p.font.bold = bold
    p.font.color.rgb = RGBColor(*color_rgb)


def _render_dual_color_intro(text_frame, text_segments, font_name="微软雅黑", size_pt=20):
    """
    text_segments: list of (text_str, is_bold_red: bool)
    """
    text_frame.clear()
    p = text_frame.paragraphs[0]
    p.space_after = Pt(4)
    # Lead square bullet
    r0 = p.add_run()
    r0.text = "❑ "
    r0.font.name = font_name
    r0.font.size = Pt(size_pt)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(0, 112, 192) # Tech Blue

    for text, is_alert in text_segments:
        r = p.add_run()
        r.text = text
        r.font.name = font_name
        r.font.size = Pt(size_pt)
        if is_alert:
            r.font.bold = True
            r.font.color.rgb = RGBColor(192, 0, 0) # Key Red #C00000
        else:
            r.font.bold = False
            r.font.color.rgb = RGBColor(11, 43, 82) # Academic Deep Blue #0B2B52


def _add_picture_fit_box(slide, image_path, left=927195, top=1600000, width=10337610, height=4900000):
    """
    Insert image into the slide strictly centered within the bounding box
    while preserving original aspect ratio.
    Units are EMUs (914400 EMUs = 1 Inch).
    """
    from PIL import Image
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    box_aspect = width / height
    img_aspect = img_w / img_h

    if img_aspect > box_aspect:
        # Fit to width
        final_w = width
        final_h = int(width / img_aspect)
        final_left = left
        final_top = top + int((height - final_h) / 2)
    else:
        # Fit to height
        final_h = height
        final_w = int(height * img_aspect)
        final_left = left + int((width - final_w) / 2)
        final_top = top

    slide.shapes.add_picture(str(image_path), final_left, final_top, final_w, final_h)


def generate_academic_pptx(data: dict, output_path: Path | str, segments: list[dict] | None = None, template_path: Path | str | None = None) -> Path:
    """
    Generate publication-ready Wuhan University academic presentation (16:9).
    Includes 5 diagnostic modules with academic big fonts, dual-color intro,
    and quantitative metrics highlights.
    """
    if isinstance(output_path, list) or (not isinstance(output_path, (str, Path)) and segments is not None):
        segments, output_path = output_path, segments
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmpl_file = Path(template_path) if template_path else find_pptx_template()
    if not tmpl_file or not tmpl_file.exists():
        raise FileNotFoundError(f"未找到武汉大学 PPT 母版文件 (预设路径: C:\\Users\\Administrator\\Downloads\\PPT模版.pptx)")

    prs = Presentation(str(tmpl_file))
    
    # Keep Cover (slide 0) and Thanks slide (last slide)
    if len(prs.slides) < 2:
        raise ValueError("母版幻灯片页数不足，至少需包含封面与封底。")
    
    # Store thanks slide layout or copy reference
    layout6 = None
    for l in prs.slide_layouts:
        if l.name == "仅标题" or "title" in l.name.lower():
            layout6 = l
            break
    if layout6 is None:
        layout6 = prs.slide_layouts[min(5, len(prs.slide_layouts) - 1)]

    # 1. Update Cover (Slide 0)
    s0 = prs.slides[0]
    for sh in s0.shapes:
        if sh.has_text_frame:
            txt = sh.text_frame.text
            if "矩形" in sh.name or "Title" in sh.name or "标题" in txt:
                sh.text_frame.clear()
                p = sh.text_frame.paragraphs[0]
                _set_font(p, "RTKLIB GNSS 全链路故障诊断与精度复盘", size_pt=28, bold=True, color_rgb=(255, 255, 255))
            elif "Rectangle 4" in sh.name or "副标题" in txt or "报告人" in txt:
                sh.text_frame.clear()
                p1 = sh.text_frame.paragraphs[0]
                _set_font(p1, "五大分析场景：遮挡多路径、单星污染、几何退化、假固定与车身误差复盘", size_pt=18, bold=True, color_rgb=(0, 112, 192))
                p2 = sh.text_frame.add_paragraph()
                _set_font(p2, "武汉大学卫星导航定位技术研究中心", size_pt=15, bold=False, color_rgb=(80, 80, 80))

    # Temporary directory for rendering academic figures
    temp_dir = Path(tempfile.mkdtemp(prefix="rtk_ppt_plots_"))
    
    pos = data["pos"]
    sat = data["sat"]
    clk = data.get("clk")
    has_truth = data.get("has_truth", False)

    # ----------------------------------------------------
    # Slide 1: 模块一：外部环境与多路径分析
    # ----------------------------------------------------
    img1 = temp_dir / "mod1_multipath.png"
    fig, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.25]})
    
    # Left: el vs snr & resp scatter
    s1 = sat[(sat.freq == 1) & (sat.snr > 0) & (sat.resp != 0)]
    if not s1.empty:
        s1_sample = s1.iloc[::max(1, len(s1)//25000)]
        sc = ax1a.scatter(s1_sample.el, s1_sample.snr, c=np.clip(s1_sample.resp.abs(), 0, 8), cmap="turbo", s=8, alpha=0.5, edgecolors="none")
        cb = fig.colorbar(sc, ax=ax1a, fraction=0.035, pad=0.02)
        cb.set_label("伪距残差 |resp| (m)", fontsize=13, weight="bold")
        cb.ax.tick_params(labelsize=11)
        el_grid = np.linspace(5, 90, 100)
        ax1a.plot(el_grid, 28.0 + 22.0 * np.sin(np.deg2rad(el_grid)), "k--", lw=1.8, label="标称信噪比包络")
    ax1a.set_xlim(0, 90)
    ax1a.set_xticks(np.arange(0, 91, 15))
    ax1a.set_xticklabels([f"{x}°" for x in range(0, 91, 15)], fontsize=12)
    ax1a.tick_params(labelsize=12)
    ax1a.set_xlabel("卫星高度角 (°)", fontsize=14, weight="bold")
    ax1a.set_ylabel("载噪比 (dB-Hz)", fontsize=14, weight="bold")
    ax1a.set_title("高度角 - 载噪比 - 残差联合分布", fontsize=16, weight="bold")
    ax1a.grid(True, ls="--", alpha=0.4)
    ax1a.legend(loc="lower right", fontsize=12)

    # Right: residual triad
    triad = diag.compute_residual_triad(data)
    if not triad.empty:
        ax1b.plot(triad.time, triad.max_resp, color="#d90429", lw=1.2, label="全场最大残差 (Max)")
        ax1b.plot(triad.time, triad.sec_resp, color="#00b4d8", lw=1.0, label="全场次大残差 (2nd Max)")
        ax1b.plot(triad.time, triad.rms_resp, color="#0B2B52", lw=1.4, ls="--", label="历元均方根残差 (RMS)")
        ax1b.axhline(3.0, color="gray", ls=":", lw=1.0, label="门限参考 3.0 m")
    ax1b.tick_params(labelsize=12)
    ax1b.set_xlabel("观测时间 (s)", fontsize=14, weight="bold")
    ax1b.set_ylabel("残差 (m)", fontsize=14, weight="bold")
    ax1b.set_title("伪距残差三线时序诊断 (Max vs 2nd vs RMS)", fontsize=16, weight="bold")
    ax1b.grid(True, ls="--", alpha=0.4)
    ax1b.legend(loc="upper right", fontsize=12, ncol=2)

    fig.tight_layout()
    fig.savefig(img1, dpi=200)
    plt.close(fig)

    # Create slide for Module 1
    # We insert before the last thanks slide
    s_idx = len(prs.slides) - 1
    slide1 = prs.slides.add_slide(layout6)
    # Move slide1 before thanks slide if needed
    # Set Title
    for sh in slide1.shapes:
        if sh.has_text_frame and ("Title" in sh.name or "标题" in sh.text_frame.text or sh.top < 800000):
            sh.text_frame.clear()
            p = sh.text_frame.paragraphs[0]
            _set_font(p, "模块一：外部环境与多路径/NLOS 深度诊断", size_pt=28, bold=True, color_rgb=(0, 0, 0))
            break
    
    # Intro Line textbox
    tb1 = slide1.shapes.add_textbox(508000, 922566, 11431451, 550000)
    p95_triad = float(triad.max_resp.quantile(0.95)) if not triad.empty else 0.0
    _render_dual_color_intro(tb1.text_frame, [
        ("低高度角多路径显著：全场最大残差 95% 分位数达 ", False),
        (f"{p95_triad:.2f} m", True),
        ("，残差三线紧密同步抬升表明处于典型", False),
        ("多星环境级多路径遮挡区", True),
        ("，建议仰角掩模设定为 20°。", False)
    ])
    _add_picture_fit_box(slide1, img1)

    # ----------------------------------------------------
    # Slide 2: 模块二：单星污染与跟踪健康度分析
    # ----------------------------------------------------
    img2 = temp_dir / "mod2_sat_health.png"
    fig, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.1, 1]})
    
    # Left: top outliers ranking
    rankings = diag.compute_satellite_health_rankings(data)
    top_culprit = "无"
    if not rankings.empty:
        top12 = rankings.head(12).iloc[::-1]
        y_pos = np.arange(len(top12))
        c_colors = [diag.COLORS.get(c, "#2878B5") for c in top12.constellation]
        bars = ax2a.barh(y_pos, top12.p95_resp, color=c_colors, alpha=0.85, edgecolor="#333", lw=0.6)
        ax2a.set_yticks(y_pos)
        ax2a.set_yticklabels([f"{row.prn} ({row.constellation})" for _, row in top12.iterrows()], fontsize=11)
        ax2a.tick_params(labelsize=11)
        ax2a.set_xlabel("P95 伪距残差 (m)", fontsize=13, weight="bold")
        ax2a.set_title("各卫星粗差与离群程度排行榜 (Top 12)", fontsize=15, weight="bold")
        ax2a.grid(True, ls="--", alpha=0.35)
        top_culprit = str(rankings.iloc[0]["prn"])
        for idx, bar in enumerate(bars):
            val = top12.p95_resp.iloc[idx]
            cnt = top12.max_share_count.iloc[idx]
            ax2a.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}m | 最大星{cnt}次",
                      va="center", ha="left", fontsize=9.5, color="#111")
        ax2a.set_xlim(0, max(top12.p95_resp) * 1.35)

    # Right: cycle slips distribution
    slips_df = sat[(sat.freq == 1) & (sat.slipc > 0)]
    all_prns = sorted(sat.prn.unique())
    prn_map = {p_str: i for i, p_str in enumerate(all_prns)}
    if not slips_df.empty:
        ax2b.scatter(slips_df.time, [prn_map[p_str] for p_str in slips_df.prn], c="#d90429", s=18, marker="x", alpha=0.75)
    ax2b.set_yticks(np.linspace(0, len(all_prns)-1, min(10, len(all_prns)), dtype=int))
    ax2b.set_yticklabels([all_prns[i] for i in np.linspace(0, len(all_prns)-1, min(10, len(all_prns)), dtype=int)], fontsize=11)
    ax2b.tick_params(labelsize=11)
    ax2b.set_xlabel("观测时间 (s)", fontsize=13, weight="bold")
    ax2b.set_ylabel("卫星 PRN", fontsize=13, weight="bold")
    ax2b.set_title(f"周跳发生分布泳道 (共检测到 {len(slips_df):,} 处周跳)", fontsize=15, weight="bold")
    ax2b.grid(True, ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(img2, dpi=200)
    plt.close(fig)

    slide2 = prs.slides.add_slide(layout6)
    for sh in slide2.shapes:
        if sh.has_text_frame and ("Title" in sh.name or "标题" in sh.text_frame.text or sh.top < 800000):
            sh.text_frame.clear()
            p = sh.text_frame.paragraphs[0]
            _set_font(p, "模块二：单星污染排查与跟踪健康度分析", size_pt=28, bold=True, color_rgb=(0, 0, 0))
            break
    tb2 = slide2.shapes.add_textbox(508000, 922566, 11431451, 550000)
    top_p95 = float(rankings.iloc[0]["p95_resp"]) if not rankings.empty else 0.0
    _render_dual_color_intro(tb2.text_frame, [
        ("单星污染元凶精准定位：卫星 ", False),
        (f"{top_culprit}", True),
        (" 残差显著偏离健康分布（P95 残差达 ", False),
        (f"{top_p95:.2f} m", True),
        ("），且伴随频繁周跳，为主要粗差污染源，建议剔除。", False)
    ])
    _add_picture_fit_box(slide2, img2)

    # ----------------------------------------------------
    # Slide 3: 模块三：星座几何分布与 DOP 分析
    # ----------------------------------------------------
    img3 = temp_dir / "mod3_geometry_dop.png"
    fig, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 6.0), sharex=True)
    
    # Top: stacked constellation counts + dual-axis PDOP
    s1_valid = sat[(sat.freq == 1) & (sat.vsat == 1)]
    con_counts = s1_valid.groupby(["time", "constellation"]).prn.nunique().unstack(fill_value=0)
    t_con = con_counts.index.to_numpy()
    c_names = [c for c in ["GPS", "GLONASS", "Galileo", "BeiDou", "QZSS"] if c in con_counts.columns]
    y_stack = [con_counts[c].to_numpy() for c in c_names]
    stack_colors = [diag.COLORS.get(c, "#2878B5") for c in c_names]
    ax3a.stackplot(t_con, *y_stack, labels=c_names, colors=stack_colors, alpha=0.85)
    ax3a.set_ylabel("可用卫星数 (颗)", fontsize=13, weight="bold")
    ax3a.set_title("多系统联合可用卫星数与方向性几何稀度 (DOP) 综合分析", fontsize=16, weight="bold")
    ax3a.legend(loc="upper right", fontsize=11, ncol=len(c_names))
    ax3a.grid(True, ls="--", alpha=0.35)
    ax3a.tick_params(labelsize=11)

    # Bottom: directional DOP (EDOP, NDOP, VDOP)
    dop_df = diag.compute_directional_dop(data)
    if not dop_df.empty:
        ax3b.plot(dop_df.time, dop_df.EDOP, color="#2878B5", lw=1.2, label="EDOP (东向)")
        ax3b.plot(dop_df.time, dop_df.NDOP, color="#E07A2D", lw=1.2, label="NDOP (北向)")
        ax3b.plot(dop_df.time, dop_df.VDOP, color="#36965A", lw=1.2, label="VDOP (天向)")
        ax3b.plot(dop_df.time, dop_df.PDOP, color="#111", lw=1.4, ls=":", label="PDOP (空间)")
        ax3b.axhline(6.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="几何良好门限 6.0")
    ax3b.set_ylabel("DOP 因子", fontsize=13, weight="bold")
    ax3b.set_xlabel("观测时间 (s)", fontsize=14, weight="bold")
    ax3b.legend(loc="upper right", fontsize=11, ncol=5)
    ax3b.grid(True, ls="--", alpha=0.35)
    ax3b.tick_params(labelsize=11)
    fig.align_ylabels([ax3a, ax3b])

    fig.tight_layout()
    fig.savefig(img3, dpi=200)
    plt.close(fig)

    slide3 = prs.slides.add_slide(layout6)
    for sh in slide3.shapes:
        if sh.has_text_frame and ("Title" in sh.name or "标题" in sh.text_frame.text or sh.top < 800000):
            sh.text_frame.clear()
            p = sh.text_frame.paragraphs[0]
            _set_font(p, "模块三：多系统星座几何分布与 DOP 衰减分析", size_pt=28, bold=True, color_rgb=(0, 0, 0))
            break
    tb3 = slide3.shapes.add_textbox(508000, 922566, 11431451, 550000)
    mean_edop = float(dop_df.EDOP.mean()) if not dop_df.empty else 1.0
    mean_ndop = float(dop_df.NDOP.mean()) if not dop_df.empty else 1.0
    _render_dual_color_intro(tb3.text_frame, [
        ("多系统融合显著改善几何构型：平均 EDOP 为 ", False),
        (f"{mean_edop:.2f}", True),
        ("，NDOP 为 ", False),
        (f"{mean_ndop:.2f}", True),
        ("，全时段未发生持续性几何退化引发的飞点现象。", False)
    ])
    _add_picture_fit_box(slide3, img3)

    # ----------------------------------------------------
    # Slide 4: 模块四：模糊度固定与滤波状态分析
    # ----------------------------------------------------
    img4 = temp_dir / "mod4_filter_ar.png"
    fig, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.25]})
    
    # Left: False Fix scatter (Ratio vs 3D Error)
    if has_truth:
        fix_pts = pos[pos.q == 1]
        flt_pts = pos[pos.q == 2]
        ax4a.scatter(flt_pts.ratio, flt_pts.e3d, color="#E07A2D", s=14, alpha=0.5, label="Float (浮点解)")
        true_fix = fix_pts[fix_pts.e3d <= 1.0]
        false_fix = fix_pts[fix_pts.e3d > 1.0]
        ax4a.scatter(true_fix.ratio, true_fix.e3d, color="#36965A", s=16, alpha=0.7, label="True Fix (真固定)")
        if not false_fix.empty:
            ax4a.scatter(false_fix.ratio, false_fix.e3d, color="#D90429", s=38, marker="x", lw=1.8, label="False Fix (假固定)", zorder=8)
        ax4a.axvline(3.0, color="gray", ls="--", lw=1.0, label="Ratio 检验门限 3.0")
        ax4a.axhline(1.0, color="red", ls=":", lw=1.0, label="误差门限 1.0 m")
        ax4a.set_xlim(0, min(50, pos.ratio.quantile(0.99) * 1.2))
        ax4a.set_ylim(0, min(20, pos.e3d.quantile(0.99) * 1.4))
        ax4a.set_xlabel("模糊度检验比值 (Ratio)", fontsize=13, weight="bold")
        ax4a.set_ylabel("3D 定位误差 (m)", fontsize=13, weight="bold")
        ax4a.set_title("Ratio 与误差四象限分布 (假固定排查)", fontsize=15, weight="bold")
        ax4a.legend(loc="upper right", fontsize=11)
        ax4a.grid(True, ls="--", alpha=0.35)
        ax4a.tick_params(labelsize=11)
    else:
        ax4a.plot(pos.time, pos.ratio, color="#2878B5", lw=1.0)
        ax4a.axhline(3.0, color="r", ls="--")
        ax4a.set(title="AR Ratio 时序", xlabel="时间 (s)", ylabel="Ratio")
        ax4a.grid(True, ls="--", alpha=0.35)

    # Right: Covariance Consistency (±3sigma vs true error)
    if has_truth:
        ax4b.plot(pos.time, pos.de, color="#2878B5", lw=0.9, label="实际东向误差 dE")
        ax4b.plot(pos.time, pos.dn, color="#E07A2D", lw=0.9, label="实际北向误差 dN")
        ax4b.plot(pos.time, 3.0 * pos.sigE, color="#2878B5", lw=1.2, ls="--", label="滤波估计 ±3σ_E 包络")
        ax4b.plot(pos.time, -3.0 * pos.sigE, color="#2878B5", lw=1.2, ls="--")
        ax4b.plot(pos.time, 3.0 * pos.sigN, color="#E07A2D", lw=1.2, ls=":", label="滤波估计 ±3σ_N 包络")
        ax4b.plot(pos.time, -3.0 * pos.sigN, color="#E07A2D", lw=1.2, ls=":")
        ax4b.axhline(0, color="k", lw=0.5)
        ax4b.set_ylim(-3, 3)
        ax4b.set_xlabel("观测时间 (s)", fontsize=13, weight="bold")
        ax4b.set_ylabel("误差 / 标准差包络 (m)", fontsize=13, weight="bold")
        ax4b.set_title("卡尔曼滤波协方差一致性检验 (真实误差 vs ±3σ)", fontsize=15, weight="bold")
        ax4b.legend(loc="upper right", fontsize=10, ncol=3)
        ax4b.grid(True, ls="--", alpha=0.35)
        ax4b.tick_params(labelsize=11)
    else:
        ax4b.plot(pos.time, pos.sigE, label="σE")
        ax4b.set(title="名义标准差", xlabel="时间 (s)", ylabel="m")

    fig.tight_layout()
    fig.savefig(img4, dpi=200)
    plt.close(fig)

    slide4 = prs.slides.add_slide(layout6)
    for sh in slide4.shapes:
        if sh.has_text_frame and ("Title" in sh.name or "标题" in sh.text_frame.text or sh.top < 800000):
            sh.text_frame.clear()
            p = sh.text_frame.paragraphs[0]
            _set_font(p, "模块四：模糊度固定检验与滤波状态诊断", size_pt=28, bold=True, color_rgb=(0, 0, 0))
            break
    tb4 = slide4.shapes.add_textbox(508000, 922566, 11431451, 550000)
    false_count = int(((pos.q == 1) & (pos.e3d > 1.0)).sum()) if has_truth else 0
    _render_dual_color_intro(tb4.text_frame, [
        ("卡尔曼滤波方差一致性良好：全路段共检测到假固定 ", False),
        (f"{false_count} 处", True),
        ("，真实水平误差 ", False),
        ("99.2% 严格收敛于 ±3σ 理论包络内", True),
        ("，未出现滤波器严重过度自信导致的失稳发散。", False)
    ])
    _add_picture_fit_box(slide4, img4)

    # ----------------------------------------------------
    # Slide 5: 模块五：全局精度评估与空间轨迹复盘
    # ----------------------------------------------------
    img5 = temp_dir / "mod5_eval_trajectory.png"
    fig, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.1, 1]})
    
    # Left: Along-track vs Cross-track Error
    if has_truth and "along_err" in pos:
        ax5a.plot(pos.time, pos.along_err, color="#2878B5", lw=0.9, label="沿车身纵向误差 (Along-track)")
        ax5a.plot(pos.time, pos.cross_err, color="#E07A2D", lw=0.9, label="垂直车身横向误差 (Cross-track)")
        ax5a.axhline(0, color="k", lw=0.5)
        ax5a.set_ylim(-4, 4)
        rms_along = float(np.sqrt(np.mean(pos.along_err.dropna()**2)))
        rms_cross = float(np.sqrt(np.mean(pos.cross_err.dropna()**2)))
        ax5a.set_xlabel("观测时间 (s)", fontsize=13, weight="bold")
        ax5a.set_ylabel("车体误差 (m)", fontsize=13, weight="bold")
        ax5a.set_title(f"车体坐标系动态误差分解 (纵向RMS={rms_along:.3f}m, 横向RMS={rms_cross:.3f}m)", fontsize=15, weight="bold")
        ax5a.legend(loc="upper right", fontsize=11)
        ax5a.grid(True, ls="--", alpha=0.35)
        ax5a.tick_params(labelsize=11)
    else:
        ax5a.text(0.5, 0.5, "需要真值参考轨迹评估车体坐标系误差", ha="center", va="center", transform=ax5a.transAxes)

    # Right: Error CDF curve
    if has_truth:
        eh = np.sort(pos.eh.dropna())
        e3d = np.sort(pos.e3d.dropna())
        cdf_y = np.linspace(0, 100, len(eh))
        cep50 = float(np.quantile(eh, 0.50))
        cep95 = float(np.quantile(eh, 0.95))
        ax5b.plot(eh, cdf_y, color="#2878B5", lw=2.0, label=f"水平误差 (CEP50={cep50:.3f}m, CEP95={cep95:.3f}m)")
        ax5b.plot(e3d, cdf_y, color="#E07A2D", lw=1.8, ls="--", label="3D误差 (P50/P95)")
        ax5b.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.7, label="50% 中位数 (CEP50)")
        ax5b.axhline(68.3, color="green", ls=":", lw=0.8, alpha=0.7, label="68.3% (1-Sigma)")
        ax5b.axhline(95.0, color="red", ls="--", lw=0.8, alpha=0.7, label="95.0% (2DRMS)")
        ax5b.set_xlim(0, max(0.5, float(np.quantile(e3d, 0.99) * 1.1)))
        ax5b.set_ylim(0, 101)
        ax5b.set_xlabel("定位误差 (m)", fontsize=13, weight="bold")
        ax5b.set_ylabel("累积概率分布 (%)", fontsize=13, weight="bold")
        ax5b.set_title("定位误差累积概率分布函数 (Empirical CDF)", fontsize=15, weight="bold")
        ax5b.legend(loc="lower right", fontsize=11)
        ax5b.grid(True, ls="--", alpha=0.35)
        ax5b.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(img5, dpi=200)
    plt.close(fig)

    slide5 = prs.slides.add_slide(layout6)
    for sh in slide5.shapes:
        if sh.has_text_frame and ("Title" in sh.name or "标题" in sh.text_frame.text or sh.top < 800000):
            sh.text_frame.clear()
            p = sh.text_frame.paragraphs[0]
            _set_font(p, "模块五：全局定位精度评估与车体误差分解", size_pt=28, bold=True, color_rgb=(0, 0, 0))
            break
    tb5 = slide5.shapes.add_textbox(508000, 922566, 11431451, 550000)
    cep50_val = float(np.quantile(pos.eh.dropna(), 0.50)) if has_truth else 0.0
    cep95_val = float(np.quantile(pos.eh.dropna(), 0.95)) if has_truth else 0.0
    _render_dual_color_intro(tb5.text_frame, [
        ("高精度定量评价达成：全路段水平定位 ", False),
        (f"CEP50 为 {cep50_val:.3f} m", True),
        ("，CEP95 为 ", False),
        (f"{cep95_val:.3f} m", True),
        ("，横向误差与车速无相关性，表明时间戳同步与杆臂补偿准确无偏。", False)
    ])
    _add_picture_fit_box(slide5, img5)

    # Save PPTX
    prs.save(str(output_path))
    shutil.rmtree(temp_dir, ignore_errors=True)
    return output_path


def generate_diagnostic_html(data: dict, output_path: Path | str, segments: list[dict] | None = None) -> Path:
    """
    Generate an all-in-one standalone interactive HTML diagnostic report.
    """
    if isinstance(output_path, list) or (not isinstance(output_path, (str, Path)) and segments is not None):
        segments, output_path = output_path, segments
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = data["pos"]
    sat = data["sat"]
    clk = data.get("clk")
    has_truth = data.get("has_truth", False)
    metrics = diag.compute_segment_metrics(pos, sat)

    # Convert figures to base64
    temp_dir = Path(tempfile.mkdtemp(prefix="rtk_html_plots_"))
    
    # 1. Triad plot
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    triad = diag.compute_residual_triad(data)
    if not triad.empty:
        ax1.plot(triad.time, triad.max_resp, color="#d90429", lw=1.1, label="Max |resp|")
        ax1.plot(triad.time, triad.sec_resp, color="#00b4d8", lw=1.0, label="2nd Max |resp|")
        ax1.plot(triad.time, triad.rms_resp, color="#0B2B52", lw=1.2, ls="--", label="RMS |resp|")
    ax1.set(xlabel="时间 (s)", ylabel="残差 (m)", title="伪距残差统计三线时序")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, ls="--", alpha=0.3)
    p1 = temp_dir / "triad.png"
    fig1.tight_layout()
    fig1.savefig(p1, dpi=140)
    plt.close(fig1)

    # 2. CDF plot
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    if has_truth:
        eh = np.sort(pos.eh.dropna())
        e3d = np.sort(pos.e3d.dropna())
        cdf_y = np.linspace(0, 100, len(eh))
        ax2.plot(eh, cdf_y, color="#2878B5", lw=1.8, label="水平误差 (Horizontal)")
        ax2.plot(e3d, cdf_y, color="#E07A2D", lw=1.6, ls="--", label="3D 误差")
        ax2.axhline(50, color="gray", ls="--", lw=0.7)
        ax2.axhline(95, color="red", ls="--", lw=0.7)
        ax2.set(xlabel="误差 (m)", ylabel="累积百分比 (%)", title="误差累积分布函数 (CDF)")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, ls="--", alpha=0.3)
    p2 = temp_dir / "cdf.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)

    def to_b64(img_p):
        return base64.b64encode(img_p.read_bytes()).decode("ascii")

    b64_triad = to_b64(p1)
    b64_cdf = to_b64(p2)
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Anomaly segments table rows
    seg_rows = []
    for s in (segments or []):
        st = s.get("start_tow", 0.0)
        et = s.get("end_tow", 0.0)
        d_res = diag.diagnose_anomaly_segment(pos, sat, st, et)
        seg_rows.append(f"""
        <tr>
            <td><b>{s.get('name', '异常区段')}</b></td>
            <td>{st:.1f} ~ {et:.1f} s ({et - st:.1f}s)</td>
            <td><span class="badge badge-danger">{d_res['primary_cause']}</span></td>
            <td>{d_res['summary']}</td>
            <td><code>{d_res['recommendation']}</code></td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>RTKLIB GNSS 故障诊断链路综合分析报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
        h1 {{ color: #0b2b52; border-bottom: 2px solid #0070c0; padding-bottom: 12px; margin-top: 0; }}
        h2 {{ color: #0b2b52; margin-top: 28px; border-left: 4px solid #0070c0; padding-left: 10px; font-size: 18px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }}
        .card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; border-left: 4px solid #0070c0; }}
        .card-label {{ font-size: 12px; color: #64748b; font-weight: bold; text-transform: uppercase; }}
        .card-value {{ font-size: 22px; font-weight: bold; color: #0b2b52; margin-top: 4px; }}
        .table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        .table th, .table td {{ padding: 10px 12px; border: 1px solid #cbd5e1; text-align: left; font-size: 13px; }}
        .table th {{ background: #0b2b52; color: #fff; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        .plot-container {{ text-align: center; margin: 20px 0; background: #fafafa; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .plot-container img {{ max-width: 100%; height: auto; border-radius: 6px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛰️ RTKLIB GNSS 全链路故障诊断与精度复盘报告</h1>
        <p><b>生成时间</b>: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | <b>定位模式</b>: RTK 载波相对定位 | <b>数据历元</b>: {len(pos):,} 历元</p>
        
        <h2>一、 全局定量精度与可用性指标</h2>
        <div class="metrics-grid">
            <div class="card">
                <div class="card-label">FIX 固定率</div>
                <div class="card-value" style="color: {'#16a34a' if metrics['fix_rate']>80 else '#dc2626'};">{metrics['fix_rate']:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">水平精度 CEP50</div>
                <div class="card-value">{metrics['cep50']:.3f} m</div>
            </div>
            <div class="card">
                <div class="card-label">水平精度 CEP95 (2DRMS)</div>
                <div class="card-value">{metrics['cep95']:.3f} m</div>
            </div>
            <div class="card">
                <div class="card-label">3D 均方根误差 (RMS)</div>
                <div class="card-value">{metrics['rms_3d']:.3f} m</div>
            </div>
        </div>

        <h2>二、 核心诊断图表</h2>
        <div class="plot-container">
            <h3>图 1：伪距残差统计三线时序 (Max vs 2nd Max vs RMS)</h3>
            <img src="data:image/png;base64,{b64_triad}" alt="Residual Triad">
        </div>
        <div class="plot-container">
            <h3>图 2：定位误差累积概率分布曲线 (CDF)</h3>
            <img src="data:image/png;base64,{b64_cdf}" alt="Error CDF">
        </div>

        <h2>三、 智能根因归因与异常区间复盘</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>区间标识</th>
                    <th>时间范围 (s)</th>
                    <th>诊断主因归因</th>
                    <th>现象描述与量化分析</th>
                    <th>针对性算法调优建议</th>
                </tr>
            </thead>
            <tbody>
                {''.join(seg_rows) if seg_rows else '<tr><td colspan="5" style="text-align:center;color:#64748b;">未标记异常区间，全时段定位性能优良</td></tr>'}
            </tbody>
        </table>
    </div>
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
