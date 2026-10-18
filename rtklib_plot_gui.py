"""Interactive RTKLIB-style plotter for .pos and optional .stat/reference files."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import rtklib_diagnostics as diag
import rtklib_map_tiles as mt

def insert_time_gaps(times, *arrays, max_gap_factor=2.5, min_gap_s=0.25):
    """Insert np.nan wherever there is a time outage so matplotlib disconnects lines."""
    if times is None or len(times) <= 1:
        return (times,) + tuple(arrays)
    t = np.asarray(times, dtype=float)
    dt = np.diff(t)
    pos_dt = dt[dt > 0]
    dt0 = float(np.median(pos_dt)) if len(pos_dt) > 0 else 1.0
    gap_thresh = max(max_gap_factor * dt0, min_gap_s)
    gap_idx = np.where(dt > gap_thresh)[0]
    if len(gap_idx) == 0:
        return (t,) + tuple(np.asarray(a, dtype=float) for a in arrays)
    insert_pos = gap_idx + 1
    new_t = np.insert(t, insert_pos, np.nan)
    new_arrs = [np.insert(np.asarray(a, dtype=float), insert_pos, np.nan) for a in arrays]
    return (new_t,) + tuple(new_arrs)

Q_LABELS = {1: "FIX", 2: "FLOAT", 3: "SBAS", 4: "DGPS", 5: "SINGLE", 6: "PPP"}
Q_COLORS = {1: "#07851d", 2: "#dc8a12", 3: "#5b83c4", 4: "#9b59b6", 5: "#777777", 6: "#17a2a4"}
MODES = [
    "Gnd Trk", "Trajectory Error Heatmap", "ENU Position", "Height", "ENU Error", "Solution Status",
    "AR Ratio", "Differential Age", "Satellites", "DOP", "Directional DOP",
    "Skyplot & C/N0 (Env)", "Skyplot", "C/N0", "Satellite Elevation",
    "Residuals", "Residual Triad", "El vs SNR & Resp", "Top Outlier Satellites",
    "Multipath Proxy", "Multipath Skyplot", "Cycle Slips", "Satellite Visibility Gantt",
    "Per-PRN Residuals", "Covariance", "False Fix Analysis", "Innovation / NIS",
    "Error CDF", "Error Histogram", "Body-Frame Error", "Clock Bias & ISB", "Env & Geometry Combined"
]

NON_TIME_SERIES_MODES = (
    "Gnd Trk", "Trajectory Error Heatmap", "Skyplot", "Multipath Skyplot", "Multipath Proxy",
    "Per-PRN Residuals", "False Fix Analysis", "Innovation / NIS", "Error CDF",
    "Error Histogram",
    "El vs SNR & Resp", "Top Outlier Satellites", "Env & Geometry Combined", "Skyplot & C/N0 (Env)"
)

PLOT_CATEGORIES = {
    "核心诊断链路 (Survey Diagnostics)": [
        "Env & Geometry Combined", "Skyplot & C/N0 (Env)", "Satellite Visibility Gantt", "Top Outlier Satellites",
        "False Fix Analysis", "Covariance", "Trajectory Error Heatmap",
        "Residual Triad", "El vs SNR & Resp", "Directional DOP", "Body-Frame Error"
    ],
    "轨迹与空间位置 (Trajectory & Position)": [
        "Gnd Trk", "Trajectory Error Heatmap", "ENU Position", "Height", "Body-Frame Error"
    ],
    "定位误差与精度评估 (Accuracy & Errors)": [
        "ENU Error", "Trajectory Error Heatmap", "False Fix Analysis", "Error CDF", "Error Histogram", "Covariance", "Residual Triad", "Body-Frame Error"
    ],
    "解算状态与可靠性 (Status & Reliability)": [
        "Solution Status", "AR Ratio", "Differential Age", "False Fix Analysis", "Innovation / NIS"
    ],
    "卫星观测与信号质量 (Satellites & Signal Quality)": [
        "Satellites", "DOP", "Directional DOP", "Skyplot & C/N0 (Env)", "Skyplot", "C/N0", "Satellite Elevation",
        "Residuals", "Residual Triad", "Multipath Proxy", "Multipath Skyplot", "Cycle Slips",
        "Satellite Visibility Gantt", "Per-PRN Residuals", "Top Outlier Satellites", "El vs SNR & Resp", "Clock Bias & ISB"
    ]
}

MODE_BILINGUAL = {
    "Custom Analysis": "自定义网格分析 (Custom Analysis)",
    "Dashboard": "综合六宫格仪表盘 (Dashboard)",
    "Multi-Compare": "多方案横向对比 (Multi-Compare)",
    "Env & Geometry Combined": "环境与几何综合 (Env & Geometry)",
    "Skyplot & C/N0 (Env)": "星空图与载噪比综合 (Skyplot & C/N0)",
    "Skyplot": "卫星星空图 (Skyplot)",
    "Multipath Skyplot": "多路径星空图 (Multipath Skyplot)",
    "C/N0": "载噪比时序 (C/N0)",
    "Satellite Elevation": "卫星高度角时序 (Elevation)",
    "Gnd Trk": "地面轨迹 (Gnd Trk)",
    "Trajectory Error Heatmap": "轨迹误差热力图 (Trajectory Error)",
    "ENU Position": "ENU位置时序 (ENU Pos)",
    "Height": "大地高程 (Height)",
    "ENU Error": "ENU定位误差 (ENU Error)",
    "Solution Status": "解状态 (Solution Status)",
    "AR Ratio": "Ratio比值检验 (AR Ratio)",
    "Differential Age": "差分时延与解状态 (Age & Q)",
    "Satellites": "卫星数量 (Satellites)",
    "DOP": "DOP精度因子 (DOP)",
    "Directional DOP": "方向DOP几何因子 (Directional DOP)",
    "Residuals": "伪距与载波残差 (Residuals)",
    "Residual Triad": "残差统计三线时序 (Residual Triad)",
    "El vs SNR & Resp": "高度角-载噪比-残差 (El vs SNR & Resp)",
    "Top Outlier Satellites": "粗差卫星贡献排行 (Top Outliers)",
    "Multipath Proxy": "多路径效应 (Multipath)",
    "Cycle Slips": "周跳探测 (Cycle Slips)",
    "Satellite Visibility Gantt": "残差与周跳泳道图 (Visibility & Slips)",
    "Per-PRN Residuals": "单星残差画像 (Per-PRN Res)",
    "Covariance": "滤波协方差一致性 (Covariance)",
    "False Fix Analysis": "假固定检验 (False Fix Analysis)",
    "Innovation / NIS": "新息与NIS (Innovation / NIS)",
    "Error CDF": "误差累计分布 (Error CDF)",
    "Error Histogram": "误差直方图 (Error Histogram)",
    "Body-Frame Error": "车体坐标系误差 (Body-Frame Error)",
    "Clock Bias & ISB": "接收机钟差与ISB (Clock Bias & ISB)",
}

BILINGUAL_TO_MODE = {v: k for k, v in MODE_BILINGUAL.items()}
for k in list(MODE_BILINGUAL.keys()):
    BILINGUAL_TO_MODE[k] = k

DIAGNOSTIC_WORKSPACES = [
    {
        "id": 1,
        "name": "① 外部环境与多路径",
        "desc": "星空图与载噪比综合(地固+车体+联合散点)、高度角残差分布与残差三线",
        "preset": "Focus Left (左主右副)",
        "modes": ["Skyplot & C/N0 (Env)", "El vs SNR & Resp", "Residual Triad"]
    },
    {
        "id": 2,
        "name": "② 单星污染与健康度",
        "desc": "全星座可见性/周跳泳道图、粗差卫星贡献度排行榜与伪距残差",
        "preset": "Focus Top (上主下副)",
        "modes": ["Satellite Visibility Gantt", "Top Outlier Satellites", "Residuals"]
    },
    {
        "id": 3,
        "name": "③ 星座几何与DOP",
        "desc": "可见卫星数量（按系统堆叠）与方向 DOP / 8扇区覆盖",
        "preset": "Vertical Stack (纵向流)",
        "modes": ["Satellites", "Directional DOP"]
    },
    {
        "id": 4,
        "name": "④ 模糊度固定与滤波",
        "desc": "假固定点空间识别 (1m门限)、状态协方差 ±3σ 检验与残差三线",
        "preset": "Focus Left (左主右副)",
        "modes": ["False Fix Analysis", "Covariance", "Residual Triad"]
    },
    {
        "id": 5,
        "name": "⑤ 全局精度与轨迹复盘",
        "desc": "2D航迹着色图、车体坐标系(沿航向/跨航向/垂直)误差分解与误差CDF",
        "preset": "Focus Left (左主右副)",
        "modes": ["Trajectory Error Heatmap", "Body-Frame Error", "Error CDF"]
    }
]

PLOT_UNIT_GROUPS = {
    # 距离定位误差 (米) - 以 0 为中心对称
    "ENU Error": "error_m",
    "Trajectory Error Heatmap": "error_m",
    "Body-Frame Error": "error_m",
    "Compare Error": "error_m",
    "False Fix Analysis": "error_m",
    # 测量残差 (米) - 以 0 为中心对称
    "Residuals": "residual_m",
    "Residual Triad": "triad_m",
    "Per-PRN Residuals": "prn_residual_m",
    # 卫星颗数
    "Satellites": "sat_count",
    # 几何精度衰减因子 (无量纲)
    "DOP": "dop_ratio",
    "Directional DOP": "dop_ratio",
    # AR 模糊度固定比值 (无量纲)
    "AR Ratio": "ar_ratio",
    # 载噪比信号强度 (dB-Hz)
    "C/N0": "cn0_dbhz",
    # 解状态
    "Solution Status": "sol_q",
}


@lru_cache(maxsize=16384)
def gpst_to_local_datetime(week, tow):
    """Convert GPST week/TOW to CST, applying the applicable GPS-UTC offset."""
    gps_epoch=datetime(1980,1,6)
    gpst=gps_epoch+timedelta(weeks=int(week),seconds=float(tow))
    # Effective UTC date and GPS-UTC offset after each leap second.
    transitions=[("1981-07-01",1),("1982-07-01",2),("1983-07-01",3),
        ("1985-07-01",4),("1988-01-01",5),("1990-01-01",6),("1991-01-01",7),
        ("1992-07-01",8),("1993-07-01",9),("1994-07-01",10),("1996-01-01",11),
        ("1997-07-01",12),("1999-01-01",13),("2006-01-01",14),("2009-01-01",15),
        ("2012-07-01",16),("2015-07-01",17),("2017-01-01",18)]
    offset=0
    for date,seconds in transitions:
        effective_gpst=datetime.strptime(date,"%Y-%m-%d")+timedelta(seconds=seconds)
        if gpst<effective_gpst:break
        offset=seconds
    return gpst-timedelta(seconds=offset)+timedelta(hours=8)


def ecef_to_geodetic(x, y, z):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f); lon = np.arctan2(y, x); p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(8):
        n = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
    n = a / np.sqrt(1 - e2 * np.sin(lat)**2); h = p / np.cos(lat) - n
    return np.rad2deg(lat), np.rad2deg(lon), h


class RTKPlotApp:
    def __init__(self, root, pos=None, reference=None, initial_plot="Gnd Trk"):
        self.root = root; self.root.title("RTKLIB Plot - Python"); self.root.geometry("1500x920"); self.root.minsize(1100, 700)
        self.data = None; self.pos_path = None; self.ref_path = reference; self.stat_path = None; self.playing = False; self.play_job = None
        self.mode = tk.StringVar(value=initial_plot); self.qfilter = tk.StringVar(value="All Q"); self.grid_on = tk.BooleanVar(value=True)
        self.track_color_by_error = tk.BooleanVar(value=True)
        self.zoom_axis = tk.StringVar(value="Both axes"); self.equal_scale = tk.BooleanVar(value=False); self.speed = tk.StringVar(value="10x")
        self.follow_mouse = tk.BooleanVar(value=True); self.click_to_maximize = tk.BooleanVar(value=False)
        self.vehicle_marker = None; self.vehicle_trail = None; self.vehicle_label = None
        self.truth_marker = None; self.error_link = None
        self.dashboard_modes_default = ["Gnd Trk", "Position Error", "Solution Status", "AR Ratio", "Satellites", "Elevation vs C/N0"]
        self.dashboard_modes = list(self.dashboard_modes_default)
        self.selected_dashboard_index = 0; self.focused_dashboard_subplot = None
        self.multi_solutions = []; self.multi_compare_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#17becf"]
        self.selected_compare_subplot = 0; self.focused_compare_subplot = None
        self.time_cursors=[];self.sky_current=None;self.sky_labels=[];self.sky_plots=[];self.custom_modes=[];self.custom_panels=[];self.selected_custom_panel=0;self.focused_custom_panel=None;self.visible_panel_indices=[];self._drag_plot_name=None
        self.custom_page=0;self.custom_page_size=6
        self.custom_updaters=[];self.custom_dynamic_artists=[];self.dop_frames={};self.cov_twin_axes={}
        self.dynamic_artists=[];self._blit_background=None;self._cursor_job=None;self._last_cursor_index=None;self._press_state=None
        self.false_fix_limit=tk.DoubleVar(value=1.0)
        self.segments=[];self.segment_start_idx=None;self.segment_palette=["#7b61a8","#168aad","#d17b0f","#b23a48","#4f772d","#ae5a9d"]
        self._drag_epoch=False;self.track_axes=[];self.datetime_axes=[];self.segment_track_lines=[];self.segment_sky_artists=[];self.segment_artists=[]
        self._shift_drag_start=None;self._shift_drag_patch=None;self._shift_drag_ax=None
        self.line_scope=tk.StringVar(value="All Subplots")
        self.line_style=tk.StringVar(value="Solid (―)")
        self.line_width=tk.DoubleVar(value=1.2)
        self.marker_style=tk.StringVar(value="None")
        self.marker_size=tk.DoubleVar(value=3.0)
        self.curve_styles={}
        self.basemap_provider=tk.StringVar(value="None")
        self.basemap_alpha=tk.DoubleVar(value=0.85)
        self.cached_basemap=None
        self.custom_preset=tk.StringVar(value="Auto Grid (自适应等分)")
        self.sync_time_axes=tk.BooleanVar(value=True)
        self.sync_y_axes=tk.BooleanVar(value=True)
        self.robust_view=tk.BooleanVar(value=True)
        self.layout_profiles={}
        self.selected_profile_name=tk.StringVar(value="")
        self._load_layout_profiles()
        self._current_category = "全部"
        self._current_listed_modes = list(MODES)
        self._expand_segments_card = None
        # Reserve the bottom status/epoch area before packing the expanding
        # canvas.  Packing the expanding plot first can push these controls
        # below the visible window on high-DPI displays.
        self._build_menu(); self._build_controls(); self._build_status(); self._build_plot()
        self.root.bind("<Control-o>", lambda _e: self.open_file()); self.root.bind("<Control-s>", lambda _e: self.export_plot())
        self.root.bind("<Control-e>", lambda _e: self.open_batch_export_dialog())
        self.root.bind("<Control-r>", lambda _e: self.open_report_dialog())
        self.root.bind("<Delete>", lambda _e: self.delete_selected_subplot())
        self.root.bind("<space>", lambda _e: self.toggle_play())
        self.root.bind("<Left>", lambda _e: self._step_epoch(-1))
        self.root.bind("<Right>", lambda _e: self._step_epoch(1))
        self.root.bind("<Shift-Left>", lambda _e: self._step_epoch(-10))
        self.root.bind("<Shift-Right>", lambda _e: self._step_epoch(10))
        self.root.bind("<Home>", lambda _e: self.fit_view())
        self.root.bind("<r>", lambda _e: self.redraw())
        self.root.bind("<R>", lambda _e: self.redraw())
        self.root.bind("<y>", lambda _e: self._toggle_global_robust_view())
        self.root.bind("<Y>", lambda _e: self._toggle_global_robust_view())
        if pos and Path(pos).exists(): self.load_solution(Path(pos), reference)
        elif diag.DEFAULT_POS.exists(): self.load_solution(diag.DEFAULT_POS, diag.DEFAULT_REF if diag.DEFAULT_REF.exists() else None)
        else: self.draw_empty("打开 RTKLIB .pos 文件开始绘图")

    def _step_epoch(self, step):
        if self.data is None: return
        n = len(self.data["pos"])
        cur = int(self.slider.get())
        nxt = max(0, min(cur + step, n - 1))
        self.slider.set(nxt)

    def _build_menu(self):
        bar=tk.Menu(self.root)
        filem=tk.Menu(bar,tearoff=False)
        filem.add_command(label="Open Solution (.pos)…",command=self.open_file,accelerator="Ctrl+O")
        filem.add_command(label="Open Satellite Status (.stat)…",command=self.open_stat_file)
        filem.add_command(label="Open Ground Truth (.csv)…",command=self.open_reference_file)
        filem.add_separator()
        filem.add_command(label="Generate Diagnostic Report…",command=self.open_report_dialog,accelerator="Ctrl+R")
        filem.add_command(label="Export Academic Defense PPTX (武汉大学母版)…",command=self.export_academic_pptx)
        filem.add_command(label="Export Diagnostic HTML Dossier (综合分析HTML)…",command=self.export_diagnostic_html)
        filem.add_command(label="Export Current Plot…",command=self.export_plot,accelerator="Ctrl+S")
        filem.add_command(label="Batch Export Subplots…",command=self.open_batch_export_dialog,accelerator="Ctrl+E")
        filem.add_command(label="Export Google Earth KML (3D)…",command=self.export_kml)
        filem.add_command(label="Export Google My Maps KML (2D)…",command=self.export_kml_2d)
        filem.add_command(label="Export Standard GeoJSON…",command=self.export_geojson)
        filem.add_separator(); filem.add_command(label="Exit",command=self.root.destroy); bar.add_cascade(label="File",menu=filem)
        edit=tk.Menu(bar,tearoff=False); edit.add_command(label="Save Plot As…",command=self.export_plot); edit.add_command(label="Reset View",command=self.redraw); bar.add_cascade(label="Edit",menu=edit)
        
        # Diagnostics Menu (五大故障诊断场景工作区)
        diag_m = tk.Menu(bar, tearoff=False)
        for idx, ws in enumerate(DIAGNOSTIC_WORKSPACES):
            diag_m.add_command(label=f"{ws['name']} ({ws['desc'][:16]}…)", command=lambda i=idx: self.switch_workspace(i))
        diag_m.add_separator()
        diag_m.add_command(label="Auto Detect Issues (自动探测异常区段)", command=self.auto_detect_issues)
        diag_m.add_command(label="Quality Assessment Report (数据质量诊断评估)…", command=self.open_report_dialog)
        diag_m.add_separator()
        diag_m.add_command(label="Export Academic Defense PPTX (武大母版)…", command=self.export_academic_pptx)
        diag_m.add_command(label="Export Diagnostic HTML Dossier (综合分析)…", command=self.export_diagnostic_html)
        bar.add_cascade(label="Diagnostics (故障诊断)", menu=diag_m)

        # Plots Menu with Categorized Submenus
        plots_m=tk.Menu(bar,tearoff=False)
        for cat_name, m_list in PLOT_CATEGORIES.items():
            cat_menu=tk.Menu(plots_m,tearoff=False)
            for m in m_list:
                cat_menu.add_radiobutton(label=MODE_BILINGUAL.get(m, m),variable=self.mode,value=m,command=self.redraw)
            plots_m.add_cascade(label=cat_name,menu=cat_menu)
        plots_m.add_separator()
        plots_m.add_radiobutton(label="Dashboard (综合六宫格仪表盘)",variable=self.mode,value="Dashboard",command=self.redraw)
        plots_m.add_radiobutton(label="Custom Analysis (自定义分区网格)",variable=self.mode,value="Custom Analysis",command=self.redraw)
        plots_m.add_radiobutton(label="Multi-Compare (多方案横向对比)",variable=self.mode,value="Multi-Compare",command=self.redraw)
        bar.add_cascade(label="Plots (图表库)",menu=plots_m)

        view=tk.Menu(bar,tearoff=False)
        view.add_checkbutton(label="Grid (网格)",variable=self.grid_on,command=self.redraw)
        view.add_checkbutton(label="Hover scrubs timeline (悬停同步时间轴)",variable=self.follow_mouse)
        view.add_checkbutton(label="Click to maximize / restore (点击放大/还原)",variable=self.click_to_maximize)
        
        basemap_m=tk.Menu(view,tearoff=False)
        for p_name in ["None","Esri Satellite","OpenStreetMap","CartoDB Dark"]:
            basemap_m.add_radiobutton(label=p_name,variable=self.basemap_provider,value=p_name,command=self._on_basemap_changed)
        view.add_cascade(label="Basemap Layer (底图图层)",menu=basemap_m)

        zoom_m=tk.Menu(view,tearoff=False)
        for z_opt in ["Both axes","X axis","Y axis"]:
            zoom_m.add_radiobutton(label=z_opt,variable=self.zoom_axis,value=z_opt)
        view.add_cascade(label="Zoom Axis Constraint (缩放轴约束)",menu=zoom_m)

        view.add_checkbutton(label="Equal scale (1:1 物理等比)",variable=self.equal_scale,command=self._apply_aspect)
        view.add_checkbutton(label="Track Error Colorbar / Heatmap (航迹误差指示bar与热力)",variable=self.track_color_by_error,command=self.redraw)
        view.add_checkbutton(label="Robust 95% Y-Scale (95% 稳健主体量程) [Y]",variable=self.robust_view,command=self.redraw)
        view.add_command(label="Toggle Epoch slider (底端滑块显隐)",command=self._toggle_epoch_slider)
        view.add_separator()
        layout_m=tk.Menu(view,tearoff=False)
        layout_m.add_command(label="Restore Default Dashboard (恢复默认六宫格)",command=self.reset_subplots_layout)
        layout_m.add_separator()

        # Presets
        preset_m=tk.Menu(layout_m,tearoff=False)
        for p_name in ["Auto Grid (自适应等分)", "Focus Left (左主右副)", "Vertical Stack (纵向流)", "Focus Top (上主下副)"]:
            preset_m.add_radiobutton(label=p_name,variable=self.custom_preset,value=p_name,command=self._on_custom_preset_changed)
        layout_m.add_cascade(label="Layout Presets (排版预设)",menu=preset_m)
        layout_m.add_checkbutton(label="Sync Time Axes (多图时间轴联动)",variable=self.sync_time_axes,command=self.redraw)
        layout_m.add_checkbutton(label="Sync Same-Unit Y Axes (同单位纵轴联动)",variable=self.sync_y_axes,command=self.redraw)
        layout_m.add_separator()

        # Profiles
        self.profile_menu=tk.Menu(layout_m,tearoff=False)
        self._rebuild_profile_menu()
        layout_m.add_cascade(label="Layout Profiles (排版方案库)",menu=self.profile_menu)

        layout_m.add_separator()
        layout_m.add_command(label="Promote to Primary (设为主图)",command=self._promote_custom_panel_to_primary)
        layout_m.add_command(label="Move Selected Up (上移)",command=lambda:self._move_custom_panel(-1))
        layout_m.add_command(label="Move Selected Down (下移)",command=lambda:self._move_custom_panel(1))
        layout_m.add_command(label="➕ Add Subplot (添加新子图…)",command=self.add_selected_plot)
        layout_m.add_command(label="Repack Custom Grid (重排自定义网格)",command=self._reset_custom_grid)
        layout_m.add_command(label="Delete Selected Subplot (删除选中子图)",command=self.delete_selected_subplot,accelerator="Del")
        view.add_cascade(label="Subplots Layout (多图排版管理)",menu=layout_m)
        view.add_command(label="Fit to Data (全图视野自适应)",command=self.fit_view); bar.add_cascade(label="View",menu=view)

        windows=tk.Menu(bar,tearoff=False)
        windows.add_command(label="Dashboard",command=lambda:self.set_mode("Dashboard"))
        windows.add_command(label="Custom Analysis",command=lambda:self.set_mode("Custom Analysis"))
        windows.add_command(label="Multi-Compare",command=lambda:self.set_mode("Multi-Compare"))
        bar.add_cascade(label="Windows",menu=windows)
        helpm=tk.Menu(bar,tearoff=False); helpm.add_command(label="About",command=lambda:messagebox.showinfo("About RTKLIB Plot - Python","Interactive RTKLIB result viewer\nReads .pos, optional .pos.stat and reference CSV.")); bar.add_cascade(label="Help",menu=helpm)
        self.root.config(menu=bar)

    def _build_controls(self):
        # 1. Main Action Bar (Main Flow)
        row1=ttk.Frame(self.root,padding=(8,3)); row1.pack(side=tk.TOP,fill=tk.X)
        ttk.Button(row1,text="Open…",command=self.open_file).pack(side=tk.LEFT,padx=(0,6))
        ttk.Label(row1,text="Plot:").pack(side=tk.LEFT)
        combo_modes = ['Custom Analysis', 'Dashboard', 'Multi-Compare'] + [m for m in MODES if m not in ('Custom Analysis', 'Dashboard', 'Multi-Compare')]
        self._mode_combo_values = [MODE_BILINGUAL.get(m, m) for m in combo_modes]
        self.mode_box = ttk.Combobox(row1, values=self._mode_combo_values, state="readonly", width=25)
        init_disp = MODE_BILINGUAL.get(self.mode.get(), self.mode.get())
        self.mode_box.set(init_disp)
        self.mode_box.pack(side=tk.LEFT, padx=4)
        def _on_mode_combo_select(_e=None):
            sel_text = self.mode_box.get()
            internal = BILINGUAL_TO_MODE.get(sel_text, sel_text)
            self.set_mode(internal)
        self.mode_box.bind("<<ComboboxSelected>>", _on_mode_combo_select)
        ttk.Label(row1,text="Q:").pack(side=tk.LEFT,padx=(6,0))
        self.qbox=ttk.Combobox(row1,textvariable=self.qfilter,values=["All Q"]+[f"Q={q}: {name}" for q,name in Q_LABELS.items()],state="readonly",width=10)
        self.qbox.pack(side=tk.LEFT,padx=3); self.qbox.bind("<<ComboboxSelected>>",lambda _e:self.redraw())
        ttk.Checkbutton(row1,text="Grid",variable=self.grid_on,command=self.redraw).pack(side=tk.LEFT,padx=6)
        ttk.Label(row1,text="Map:").pack(side=tk.LEFT,padx=(6,2))
        self.basemap_combo=ttk.Combobox(row1,textvariable=self.basemap_provider,values=["None","Esri Satellite","OpenStreetMap","CartoDB Dark"],state="readonly",width=12)
        self.basemap_combo.pack(side=tk.LEFT,padx=3)
        self.basemap_combo.bind("<<ComboboxSelected>>",self._on_basemap_changed)
        ttk.Separator(row1,orient=tk.VERTICAL).pack(side=tk.LEFT,fill=tk.Y,padx=6,pady=2)
        ttk.Button(row1,text="Fit",command=self.fit_view).pack(side=tk.LEFT,padx=2)
        ttk.Button(row1,text="➕ Add",command=self.add_selected_plot).pack(side=tk.LEFT,padx=2)
        ttk.Button(row1,text="✕ Del",command=self.delete_selected_subplot).pack(side=tk.LEFT,padx=2)
        ttk.Button(row1,text="↺ Reset",command=self.reset_subplots_layout).pack(side=tk.LEFT,padx=2)
        ttk.Label(row1,text="Speed:").pack(side=tk.LEFT,padx=(6,0))
        self.speed_box=ttk.Combobox(row1,textvariable=self.speed,values=["1x","10x","50x"],state="readonly",width=5)
        self.speed_box.pack(side=tk.LEFT,padx=2)
        self.play_btn=ttk.Button(row1,text="▶ Play",command=self.toggle_play); self.play_btn.pack(side=tk.LEFT,padx=4)
        ttk.Button(row1,text="Export Image…",command=self.export_plot).pack(side=tk.RIGHT,padx=4)
        ttk.Button(row1,text="Batch Export…",command=self.open_batch_export_dialog).pack(side=tk.RIGHT,padx=4)

        # 2. Mark & Diagnostics Bar
        row2=ttk.Frame(self.root,padding=(8,2)); row2.pack(side=tk.TOP,fill=tk.X)
        ttk.Button(row2,text="Set start",command=self.mark_segment_start).pack(side=tk.LEFT,padx=(0,3))
        ttk.Button(row2,text="Set end & add",command=self.add_segment).pack(side=tk.LEFT,padx=3)
        ttk.Button(row2,text="Clear",command=self.clear_segments).pack(side=tk.LEFT,padx=3)
        ttk.Button(row2,text="Auto Detect Issues",command=self.auto_detect_issues).pack(side=tk.LEFT,padx=4)
        ttk.Button(row2,text="Diagnostic Report",command=self.open_report_dialog).pack(side=tk.LEFT,padx=4)
        ttk.Button(row2,text="Google Earth 3D…",command=self.export_kml).pack(side=tk.LEFT,padx=3)
        ttk.Button(row2,text="My Maps 2D…",command=self.export_kml_2d).pack(side=tk.LEFT,padx=3)
        ttk.Button(row2,text="GeoJSON…",command=self.export_geojson).pack(side=tk.LEFT,padx=3)
        self.segment_status=ttk.Label(row2,text="右键子图快捷菜单 · 双击满屏放大/还原 · 列表双击聚焦区段 · Shift+拖拽标记区段",anchor="w")
        self.segment_status.pack(side=tk.LEFT,padx=8,fill=tk.X,expand=True)

        # 3. Diagnostic Workspaces Bar (五大故障诊断场景工作区)
        row_ws = ttk.Frame(self.root, padding=(8, 2))
        row_ws.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row_ws, text="诊断工作区:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row_ws, text="①+③ 环境与几何", command=lambda: self.set_mode("Env & Geometry Combined")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_ws, text="① 外部多路径", command=lambda: self.switch_workspace(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_ws, text="② 单星污染", command=lambda: self.switch_workspace(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_ws, text="③ 星座几何与DOP", command=lambda: self.switch_workspace(2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_ws, text="④ 模糊度与滤波", command=lambda: self.switch_workspace(3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_ws, text="⑤ 精度与轨迹复盘", command=lambda: self.switch_workspace(4)).pack(side=tk.LEFT, padx=2)
        ttk.Separator(row_ws, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(row_ws, text="📊 导出答辩PPTX", command=self.export_academic_pptx).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_ws, text="🌐 导出HTML报告", command=self.export_diagnostic_html).pack(side=tk.LEFT, padx=2)

    def _create_accordion_card(self, parent, title, is_expanded=True, on_toggle=None):
        card = ttk.Frame(parent)
        card.pack(fill=tk.X, pady=(0, 6))

        # Modern header
        header = tk.Frame(card, bg="#f1f5f9", cursor="hand2", padx=6, pady=4, relief=tk.FLAT)
        header.pack(fill=tk.X)

        arrow_var = tk.StringVar(value="▼ " if is_expanded else "▶ ")
        arrow_lbl = tk.Label(header, textvariable=arrow_var, bg="#f1f5f9", fg="#475569", font=("Segoe UI", 9, "bold"))
        arrow_lbl.pack(side=tk.LEFT)
        title_lbl = tk.Label(header, text=title, bg="#f1f5f9", fg="#0f172a", font=("Segoe UI", 9, "bold"))
        title_lbl.pack(side=tk.LEFT)

        content = ttk.Frame(card, padding=(2, 4))
        state = {"expanded": is_expanded}

        def do_toggle(event=None):
            if state["expanded"]:
                content.pack_forget()
                arrow_var.set("▶ ")
                state["expanded"] = False
            else:
                content.pack(fill=tk.X, expand=True)
                arrow_var.set("▼ ")
                state["expanded"] = True
            self._update_sidebar_scroll()
            if on_toggle:
                on_toggle(state["expanded"])

        def do_expand():
            if not state["expanded"]:
                do_toggle()

        def do_collapse():
            if state["expanded"]:
                do_toggle()

        header.bind("<Button-1>", do_toggle)
        arrow_lbl.bind("<Button-1>", do_toggle)
        title_lbl.bind("<Button-1>", do_toggle)

        def on_enter(e):
            header.configure(bg="#e2e8f0")
            arrow_lbl.configure(bg="#e2e8f0")
            title_lbl.configure(bg="#e2e8f0")

        def on_leave(e):
            header.configure(bg="#f1f5f9")
            arrow_lbl.configure(bg="#f1f5f9")
            title_lbl.configure(bg="#f1f5f9")

        header.bind("<Enter>", on_enter)
        header.bind("<Leave>", on_leave)
        arrow_lbl.bind("<Enter>", on_enter)
        arrow_lbl.bind("<Leave>", on_leave)
        title_lbl.bind("<Enter>", on_enter)
        title_lbl.bind("<Leave>", on_leave)

        if is_expanded:
            content.pack(fill=tk.X, expand=True)

        return content, do_expand, do_collapse

    def _update_sidebar_scroll(self):
        if hasattr(self, "sidebar_canvas") and hasattr(self, "scrollable_content"):
            self.root.update_idletasks()
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_category_selected(self):
        cat = self._cat_var.get()
        self._current_category = cat
        self._refresh_plot_list()

    def _refresh_plot_list(self):
        if not hasattr(self, "plot_list"):
            return
        cat = getattr(self, "_current_category", "全部")
        if "全部" in cat:
            listed = list(MODES)
        elif "诊断" in cat:
            listed = [m for m in MODES if m in PLOT_CATEGORIES["核心诊断链路 (Survey Diagnostics)"]]
        elif "轨迹" in cat:
            listed = [m for m in MODES if m in PLOT_CATEGORIES["轨迹与空间位置 (Trajectory & Position)"]]
        elif "精度" in cat:
            listed = [m for m in MODES if m in PLOT_CATEGORIES["定位误差与精度评估 (Accuracy & Errors)"]]
        elif "状态" in cat:
            listed = [m for m in MODES if m in PLOT_CATEGORIES["解算状态与可靠性 (Status & Reliability)"]]
        elif "卫星" in cat:
            listed = [m for m in MODES if m in PLOT_CATEGORIES["卫星观测与信号质量 (Satellites & Signal Quality)"]]
        else:
            listed = list(MODES)

        self._current_listed_modes = listed

        # Find currently active modes on canvas
        active_modes = set()
        curr_mode = self.mode.get() if hasattr(self, "mode") else ""
        if curr_mode == "Custom Analysis":
            active_modes = {p.get("mode") for p in self.custom_panels if p.get("mode")}
        elif curr_mode == "Dashboard":
            active_modes = set(self.dashboard_modes)
        elif curr_mode in MODES:
            active_modes = {curr_mode}

        prev_selected = set()
        for idx in self.plot_list.curselection():
            if idx < len(self._current_listed_modes):
                prev_selected.add(self._current_listed_modes[idx])

        self.plot_list.delete(0, tk.END)
        for idx, m in enumerate(self._current_listed_modes):
            bilingual_name = MODE_BILINGUAL.get(m, m)
            is_active = m in active_modes
            prefix = "● " if is_active else "  "
            self.plot_list.insert(tk.END, f"{prefix}{bilingual_name}")
            if is_active:
                self.plot_list.itemconfig(idx, fg="#0b6bcb")
            else:
                self.plot_list.itemconfig(idx, fg="#1e293b")
            if m in prev_selected:
                self.plot_list.select_set(idx)

        self._update_sidebar_scroll()

    def _build_plot(self):
        self.fig=plt.Figure(figsize=(10,6),dpi=100,facecolor="white"); self.ax=self.fig.add_subplot(111)
        self.main_pane=ttk.Panedwindow(self.root,orient=tk.HORIZONTAL);self.main_pane.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        self.sidebar=ttk.Frame(self.main_pane,padding=2,width=285);self.plot_frame=ttk.Frame(self.main_pane)
        self.main_pane.add(self.sidebar,weight=0);self.main_pane.add(self.plot_frame,weight=1)

        # Scrollable sidebar container
        self.sidebar_canvas = tk.Canvas(self.sidebar, borderwidth=0, highlightthickness=0, width=285)
        self.sidebar_scroll = ttk.Scrollbar(self.sidebar, orient=tk.VERTICAL, command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scroll.set)
        self.sidebar_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollable_content = ttk.Frame(self.sidebar_canvas, padding=(2, 2))
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")

        self.scrollable_content.bind("<Configure>", lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda e: self.sidebar_canvas.itemconfig(self.sidebar_window, width=e.width))

        def _on_wheel(event):
            w = self.root.winfo_containing(event.x_root, event.y_root)
            if not w: return
            if w in (self.plot_list, getattr(self, "segment_box", None)):
                w.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return "break"
            curr = w
            while curr:
                if curr == self.sidebar:
                    self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    return "break"
                curr = getattr(curr, "master", None)

        self.root.bind_all("<MouseWheel>", _on_wheel, add="+")

        # Standard sidebar (Available plots, Custom layout, Line styles, Segments)
        self.standard_sidebar = ttk.Frame(self.scrollable_content)
        self.standard_sidebar.pack(fill=tk.BOTH, expand=True)

        # Card 1: 可用图表库 (Available Plots)
        card1_content, self._expand_card1, self._collapse_card1 = self._create_accordion_card(
            self.standard_sidebar, "可用图表库 (Available Plots)", is_expanded=True
        )

        # 1. Category Filter Pills
        cat_frame = ttk.Frame(card1_content)
        cat_frame.pack(fill=tk.X, pady=(0, 4))
        self._cat_var = tk.StringVar(value="全部")
        categories = [
            ("全部", "全部"),
            ("诊断", "诊断"),
            ("轨迹", "轨迹"),
            ("精度", "精度"),
            ("状态", "状态"),
            ("卫星", "卫星"),
        ]
        for key, disp in categories:
            rb = ttk.Radiobutton(
                cat_frame, text=key, value=key,
                variable=self._cat_var, style="Toolbutton",
                command=self._on_category_selected
            )
            rb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 2. Listbox & Scrollbar
        list_frame = ttk.Frame(card1_content)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.plot_list = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, exportselection=False, width=28, height=12)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.plot_list.yview)
        self.plot_list.configure(yscrollcommand=list_scroll.set)
        self.plot_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.plot_list.bind("<ButtonPress-1>", self._begin_plot_drag, add="+")
        self.plot_list.bind("<ButtonRelease-1>", self._drop_plot_on_panel, add="+")
        self.root.bind_all("<ButtonRelease-1>", self._drop_plot_on_panel, add="+")
        self.plot_list.bind("<Double-Button-1>", lambda _e: self.add_selected_plot())

        # 3. Action Buttons
        btn_box = ttk.Frame(card1_content)
        btn_box.pack(fill=tk.X, pady=(4, 1))
        ttk.Button(btn_box, text="➕ 追加 (Add)", command=self.add_selected_plot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_box, text="⟳ 替换 (Show)", command=self.show_selected).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        btn_sel_box = ttk.Frame(card1_content)
        btn_sel_box.pack(fill=tk.X, pady=1)
        ttk.Button(btn_sel_box, text="全选 (All)", command=lambda: self.plot_list.select_set(0, tk.END)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_sel_box, text="清空 (Clear)", command=lambda: self.plot_list.selection_clear(0, tk.END)).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Card 2: 多图排版与方案 (Custom Layout & Profiles)
        card2_content, self._expand_card2, self._collapse_card2 = self._create_accordion_card(
            self.standard_sidebar, "多图排版与方案 (Layout & Profiles)", is_expanded=True
        )

        p_row = ttk.Frame(card2_content); p_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(p_row, text="Preset:").pack(side=tk.LEFT)
        self.preset_combo = ttk.Combobox(
            p_row, textvariable=self.custom_preset,
            values=["Auto Grid (自适应等分)", "Focus Left (左主右副)", "Vertical Stack (纵向流)", "Focus Top (上主下副)"],
            state="readonly", width=15
        )
        self.preset_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_custom_preset_changed)

        sync_f = ttk.Frame(card2_content); sync_f.pack(fill=tk.X, pady=(1, 2))
        ttk.Checkbutton(sync_f, text="⏱ 时间联动 (Sync X)", variable=self.sync_time_axes, command=self.redraw).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(sync_f, text="📏 同单位纵轴联动 (Sync Y)", variable=self.sync_y_axes, command=self.redraw).pack(side=tk.LEFT)

        order_f = ttk.Frame(card2_content); order_f.pack(fill=tk.X, pady=(1, 2))
        ttk.Button(order_f, text="➕ 添加", command=self.add_selected_plot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(order_f, text="▲ 上移", command=lambda: self._move_custom_panel(-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(order_f, text="▼ 下移", command=lambda: self._move_custom_panel(1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(order_f, text="★ 主图", command=self._promote_custom_panel_to_primary).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=1)

        prof_f = ttk.Frame(card2_content); prof_f.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(prof_f, text="Profile:").pack(side=tk.LEFT)
        self.profile_combo = ttk.Combobox(prof_f, textvariable=self.selected_profile_name, values=list(self.layout_profiles.keys()), state="readonly", width=11)
        self.profile_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_layout_profile(self.selected_profile_name.get()))
        ttk.Button(prof_f, text="💾", width=3, command=self._save_current_layout_profile).pack(side=tk.RIGHT)

        page_f = ttk.Frame(card2_content); page_f.pack(fill=tk.X, pady=(2, 1))
        ttk.Button(page_f, text="◀ Page", command=lambda: self._change_custom_page(-1)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.custom_page_label = ttk.Label(page_f, text="Page 1 / 1", anchor="center", width=8); self.custom_page_label.pack(side=tk.LEFT, padx=1)
        ttk.Button(page_f, text="Page ▶", command=lambda: self._change_custom_page(1)).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=1)

        # Card 3: 曲线与线型样式 (Line & Curve Styles)
        card3_content, self._expand_card3, self._collapse_card3 = self._create_accordion_card(
            self.standard_sidebar, "曲线与线型样式 (Line Styles)", is_expanded=False
        )

        r_s1 = ttk.Frame(card3_content); r_s1.pack(fill=tk.X, pady=1)
        ttk.Label(r_s1, text="Target:").pack(side=tk.LEFT)
        ttk.Combobox(r_s1, textvariable=self.line_scope, values=["All Subplots", "Current Subplot"], state="readonly", width=13).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        r_s2 = ttk.Frame(card3_content); r_s2.pack(fill=tk.X, pady=1)
        ttk.Label(r_s2, text="Line:").pack(side=tk.LEFT)
        c_ls = ttk.Combobox(r_s2, textvariable=self.line_style, values=["Solid (―)", "Dashed (---)", "Dotted (···)", "Dash-Dot (-·-)", "Scatter Only", "Line + Marker"], state="readonly", width=13)
        c_ls.pack(side=tk.RIGHT, fill=tk.X, expand=True); c_ls.bind("<<ComboboxSelected>>", lambda _e: self._on_style_changed())

        r_s3 = ttk.Frame(card3_content); r_s3.pack(fill=tk.X, pady=1)
        ttk.Label(r_s3, text="Width:").pack(side=tk.LEFT)
        sp_w = ttk.Spinbox(r_s3, from_=0.5, to=5.0, increment=0.25, textvariable=self.line_width, width=4, command=self._on_style_changed)
        sp_w.pack(side=tk.LEFT, padx=2); sp_w.bind("<Return>", lambda _e: self._on_style_changed())
        ttk.Label(r_s3, text="Marker:").pack(side=tk.LEFT, padx=(2, 0))
        c_m = ttk.Combobox(r_s3, textvariable=self.marker_style, values=["None", "Circle (o)", "Square (s)", "Point (.)", "Cross (+)"], state="readonly", width=8)
        c_m.pack(side=tk.RIGHT, fill=tk.X, expand=True); c_m.bind("<<ComboboxSelected>>", lambda _e: self._on_style_changed())

        r_s4 = ttk.Frame(card3_content); r_s4.pack(fill=tk.X, pady=(2, 1))
        ttk.Button(r_s4, text="Reset", command=self._reset_curve_styles).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(r_s4, text="Apply", command=self._on_style_changed).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=1)

        # Card 4: 时间区段标记与导出 (Marked Time Segments)
        card4_content, self._expand_segments_card, self._collapse_segments_card = self._create_accordion_card(
            self.standard_sidebar, "时间区段标记与导出 (Segments)", is_expanded=False
        )

        seg_frame = ttk.Frame(card4_content); seg_frame.pack(fill=tk.X, pady=(1, 0))
        self.segment_box = tk.Listbox(seg_frame, height=5, width=28, exportselection=False)
        self.segment_box.bind("<<ListboxSelect>>", self.on_segment_select)
        self.segment_box.bind("<ButtonRelease-1>", self.on_segment_select)
        self.segment_box.bind("<Double-Button-1>", self.on_segment_double_click)
        seg_scroll = ttk.Scrollbar(seg_frame, orient=tk.VERTICAL, command=self.segment_box.yview); self.segment_box.configure(yscrollcommand=seg_scroll.set)
        self.segment_box.pack(side=tk.LEFT, fill=tk.X, expand=True); seg_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        btn_row1 = ttk.Frame(card4_content); btn_row1.pack(fill=tk.X, pady=(2, 1))
        ttk.Button(btn_row1, text="⚡ 智能探测", command=self.auto_detect_issues).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(btn_row1, text="🌐 复位全图", command=self.reset_segment_focus).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(btn_row1, text="✕ 删除", command=self.delete_segment).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_row1, text="清空", command=self.clear_segments).pack(side=tk.RIGHT, padx=1)

        btn_row2 = ttk.Frame(card4_content); btn_row2.pack(fill=tk.X, pady=(1, 1))
        ttk.Button(btn_row2, text="💾 保存区段", command=self.save_segments_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(btn_row2, text="📂 导入区段", command=self.load_segments_dialog).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=1)

        ttk.Button(card4_content, text="📷 导出区段多图 (Export)…", command=self.open_batch_export_dialog).pack(fill=tk.X, pady=(2, 1))

        # Initial populate plot list
        self._refresh_plot_list()

        # Multi-compare sidebar (Solution overlay manager)
        self.compare_sidebar = ttk.Frame(self.scrollable_content)
        ttk.Label(self.compare_sidebar, text="Multi-Compare", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(self.compare_sidebar, text="Overlay 2–4 solutions side-by-side with ground truth.", wraplength=180, justify=tk.LEFT).pack(anchor="w", pady=(0, 6))
        ttk.Button(self.compare_sidebar, text="+ Add Solution (.pos)…", command=self.add_compare_solution).pack(fill=tk.X, pady=2)
        ttk.Button(self.compare_sidebar, text="Open Reference (.csv)…", command=self.open_reference_file).pack(fill=tk.X, pady=2)
        ttk.Separator(self.compare_sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(self.compare_sidebar, text="Loaded Solutions:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.compare_list_container = ttk.Frame(self.compare_sidebar); self.compare_list_container.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        ttk.Button(self.compare_sidebar, text="Clear Extra Solutions", command=self.clear_compare_solutions).pack(fill=tk.X, pady=2)

        self.canvas=FigureCanvasTkAgg(self.fig,master=self.plot_frame); self.canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        self.canvas.mpl_connect("draw_event",self._on_canvas_draw)
        self.canvas.mpl_connect("scroll_event",self.on_scroll)
        self.canvas.mpl_connect("button_press_event",self.on_mouse_down)
        self.canvas.mpl_connect("button_release_event",self.on_mouse_up)
        self.canvas.mpl_connect("motion_notify_event",self.on_mouse_move)
        self.toolbar=NavigationToolbar2Tk(self.canvas,self.plot_frame,pack_toolbar=False); self.toolbar.update(); self.toolbar.pack(side=tk.BOTTOM,fill=tk.X)

    def _build_status(self):
        self.status_bar_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=(4, 2))
        self.status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status = ttk.Label(self.status_bar_frame, text="No solution loaded", anchor="w")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        zoom_frame = ttk.Frame(self.status_bar_frame)
        zoom_frame.pack(side=tk.RIGHT, padx=(6, 2))
        ttk.Label(zoom_frame, text="🔍 缩放约束:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 2))

        ZOOM_MAP = {"Both axes": "双轴 (XY)", "X axis": "X轴 (时间)", "Y axis": "Y轴 (数值)"}
        ZOOM_REV = {"双轴 (XY)": "Both axes", "X轴 (时间)": "X axis", "Y轴 (数值)": "Y axis"}

        self.zoom_status_combo = ttk.Combobox(
            zoom_frame,
            values=["双轴 (XY)", "X轴 (时间)", "Y轴 (数值)"],
            state="readonly",
            width=10,
            font=("Segoe UI", 8)
        )
        self.zoom_status_combo.set(ZOOM_MAP.get(self.zoom_axis.get(), "双轴 (XY)"))
        self.zoom_status_combo.pack(side=tk.LEFT)

        def _on_zoom_combo_changed(_e=None):
            val = self.zoom_status_combo.get()
            internal = ZOOM_REV.get(val, "Both axes")
            self.zoom_axis.set(internal)

        self.zoom_status_combo.bind("<<ComboboxSelected>>", _on_zoom_combo_changed)

        def _sync_zoom_var(*_args):
            cur = self.zoom_axis.get()
            disp = ZOOM_MAP.get(cur, "双轴 (XY)")
            if self.zoom_status_combo.get() != disp:
                self.zoom_status_combo.set(disp)

        self.zoom_axis.trace_add("write", _sync_zoom_var)
        row=ttk.Frame(self.root,padding=(8,3)); row.pack(side=tk.BOTTOM,fill=tk.X);self.epoch_row=row;self.epoch_row_visible=True
        ttk.Label(row,text="Epoch:").pack(side=tk.LEFT)
        self.epoch=tk.IntVar(value=0); self.slider=ttk.Scale(row,from_=0,to=1,orient=tk.HORIZONTAL,command=self.on_scrub); self.slider.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=8)
        self.epoch_label=ttk.Label(row,text="0 / 0",width=12,anchor="w");self.epoch_label.pack(side=tk.LEFT)
        self.time_label=ttk.Label(row,text="—",width=28,anchor="e"); self.time_label.pack(side=tk.RIGHT)
        truth_frame=ttk.LabelFrame(self.root,text="RTK / Ground Truth Comparison",padding=(8,4));truth_frame.pack(side=tk.BOTTOM,fill=tk.X,padx=7,pady=(3,2))
        self.truth_info=ttk.Label(truth_frame,text="Ground truth not loaded",justify=tk.LEFT,anchor="w");self.truth_info.pack(side=tk.LEFT,fill=tk.X,expand=True)
        analysis_frame=ttk.LabelFrame(self.root,text="Chart analysis",padding=(8,4));analysis_frame.pack(side=tk.BOTTOM,fill=tk.X,padx=7,pady=(2,3))
        self.analysis_info=ttk.Label(analysis_frame,text="Load a solution to see summary statistics and interpretation.",justify=tk.LEFT,anchor="w",wraplength=1250)
        self.analysis_info.pack(side=tk.LEFT,fill=tk.X,expand=True)
        analysis_frame.bind("<Configure>",lambda e:self.analysis_info.configure(wraplength=max(500,e.width-24)))

    def _toggle_epoch_slider(self):
        """Show or hide the epoch slider independently from hover sync."""
        if self.epoch_row_visible:
            self.epoch_row.pack_forget();self.epoch_toggle_btn.configure(text="Show Epoch slider")
        else:
            # Keep the slider in the bottom-control reservation established
            # before the expanding plot pane.  Repacking at the end otherwise
            # lets the plot pane consume the space and hides the slider.
            self.epoch_row.pack(side=tk.BOTTOM,fill=tk.X,before=self.main_pane);self.epoch_toggle_btn.configure(text="Hide Epoch slider")
        self.epoch_row_visible=not self.epoch_row_visible
        self.root.update_idletasks();self.canvas.draw_idle()

    def open_file(self):
        name=filedialog.askopenfilename(title="Open RTKLIB solution",filetypes=[("RTKLIB position files","*.pos *.txt *.csv"),("All files","*.*")])
        if not name:return
        pos=Path(name); stat=Path(str(pos)+".stat")
        if not stat.exists():
            if messagebox.askyesno("Satellite status file (.stat)", f"No matching status file found at:\n{stat.name}\n\nDo you want to select a .stat file manually?"):
                stat_chosen=filedialog.askopenfilename(title="Select satellite status (.stat) file",filetypes=[("RTKLIB status files","*.stat *.txt"),("All files","*.*")])
                stat=Path(stat_chosen) if stat_chosen else None
            else:
                stat=None
        ref=self._find_reference(pos)
        if ref is None and messagebox.askyesno("Reference data","Select a ground-truth CSV for position-error plots?"):
            chosen=filedialog.askopenfilename(title="Select reference CSV",filetypes=[("CSV files","*.csv"),("All files","*.*")]); ref=Path(chosen) if chosen else None
        self.load_solution(pos,ref,stat if (stat and Path(stat).exists()) else None)

    def open_stat_file(self):
        if self.data is None or self.pos_path is None:
            messagebox.showwarning("Open .stat", "Please open a .pos solution file first.")
            return
        name=filedialog.askopenfilename(title="Open Satellite Status (.stat) file",filetypes=[("RTKLIB status files","*.stat *.txt"),("All files","*.*")])
        if not name:return
        self.load_solution(self.pos_path,self.ref_path,Path(name))

    def open_reference_file(self):
        if self.data is None or self.pos_path is None:
            messagebox.showwarning("Open reference", "Please open a .pos solution file first.")
            return
        name=filedialog.askopenfilename(title="Open Ground Truth reference (.csv) file",filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not name:return
        self.load_solution(self.pos_path,Path(name),self.stat_path)

    def _find_reference(self,pos):
        for p in [pos.parent/"reference.csv",pos.parent.parent/"reference.csv",diag.DEFAULT_REF]:
            if p.exists():return p
        return None

    def load_solution(self,pos,reference=None,stat=None):
        try:
            self.data=diag.load_data(pos,reference,stat); self.pos_path=Path(pos); self.ref_path=reference; self.stat_path=Path(stat) if stat else None
        except Exception as exc:
            messagebox.showerror("Could not read solution", str(exc)); return
        self.segment_start_idx=None
        seg_file = self._get_default_segments_path()
        if seg_file and seg_file.exists():
            self._load_segments_from_file(seg_file, silent=True)
        else:
            self.segments.clear()
            limit = float(self.false_fix_limit.get()) if hasattr(self, "false_fix_limit") else 1.0
            detected = diag.detect_problematic_segments(
                self.data["pos"],
                sat_df=self.data.get("sat"),
                error_limit=limit,
                e3d_limit=3.0,
                resp_limit=3.0,
                min_duration=2.0,
                max_gap=5.0
            )
            if detected:
                self.segments = detected
                if seg_file:
                    self._save_segments_to_file(seg_file, silent=True)
            self._refresh_segment_list()
        p=self.data["pos"]; self.root.title(f"{self.pos_path} - RTKLIB Plot - Python")
        sat=self.data["sat"]
        self._pos_times=p.time.to_numpy(copy=True)
        self.sky_l1=sat[sat.freq==1].sort_values("tow").reset_index(drop=True) if not sat.empty else sat
        self._sky_tows=self.sky_l1.tow.to_numpy() if not self.sky_l1.empty else np.array([])
        self._sat_sorted=sat.sort_values("tow").reset_index(drop=True) if not sat.empty else sat
        self._sat_tows=self._sat_sorted.tow.to_numpy() if not self._sat_sorted.empty else np.array([])
        self._sat_by_time=sat.sort_values("time").reset_index(drop=True) if not sat.empty else sat
        self._sat_times=self._sat_by_time.time.to_numpy() if not self._sat_by_time.empty else np.array([])
        self.slider.configure(from_=0,to=max(0,len(p)-1)); self.slider.set(0); self.epoch.set(0)
        self.multi_solutions=[{
            "name": self.pos_path.stem,
            "path": self.pos_path,
            "data": self.data,
            "color": self.multi_compare_colors[0],
            "visible": tk.BooleanVar(value=True)
        }]
        self._refresh_multi_solution_ui()
        self.status.configure(text=f"{self.pos_path}    |    {len(p):,} epochs    |    {len(self.data['sat']):,} satellite records    |    Reference: {'loaded' if self.data['has_truth'] else 'not loaded'}")
        self.redraw()

    def _toggle_sidebar_mode(self, mode_name):
        if mode_name == "multi_compare":
            if hasattr(self, "standard_sidebar") and self.standard_sidebar.winfo_manager():
                self.standard_sidebar.pack_forget()
            if hasattr(self, "compare_sidebar") and not self.compare_sidebar.winfo_manager():
                self.compare_sidebar.pack(fill=tk.BOTH, expand=True)
                self._refresh_multi_solution_ui()
        else:
            if hasattr(self, "compare_sidebar") and self.compare_sidebar.winfo_manager():
                self.compare_sidebar.pack_forget()
            if hasattr(self, "standard_sidebar") and not self.standard_sidebar.winfo_manager():
                self.standard_sidebar.pack(fill=tk.BOTH, expand=True)
        self._update_sidebar_scroll()

    def _refresh_multi_solution_ui(self):
        if not hasattr(self, "compare_list_container"): return
        for child in self.compare_list_container.winfo_children():
            child.destroy()
        for idx, sol in enumerate(self.multi_solutions):
            row = ttk.Frame(self.compare_list_container, padding=(2, 2))
            row.pack(fill=tk.X, pady=2)
            c_canvas = tk.Canvas(row, width=13, height=13, bg=sol["color"], highlightthickness=1, relief=tk.SOLID)
            c_canvas.pack(side=tk.LEFT, padx=(0, 5))
            disp_name = sol["name"] if len(sol["name"]) <= 14 else sol["name"][:13] + "…"
            cb = ttk.Checkbutton(row, text=disp_name, variable=sol["visible"], command=self.redraw)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if idx > 0:
                del_btn = ttk.Button(row, text="✕", width=2, command=lambda i=idx: self.remove_compare_solution(i))
                del_btn.pack(side=tk.RIGHT)
        self._update_sidebar_scroll()

    def add_compare_solution(self):
        if self.data is None:
            messagebox.showwarning("Add solution", "Please open a primary .pos solution file first.")
            return
        if len(self.multi_solutions) >= 6:
            messagebox.showinfo("Limit reached", "You have reached the limit of 6 overlay solutions.")
            return
        name = filedialog.askopenfilename(title="Add RTKLIB solution for comparison", filetypes=[("RTKLIB position files", "*.pos *.txt *.csv"), ("All files", "*.*")])
        if not name: return
        pos_path = Path(name)
        ref_path = self.ref_path or self._find_reference(pos_path)
        try:
            sol_data = diag.load_data(pos_path, ref_path)
        except Exception as exc:
            messagebox.showerror("Failed to load solution", str(exc)); return
        idx = len(self.multi_solutions)
        color = self.multi_compare_colors[idx % len(self.multi_compare_colors)]
        self.multi_solutions.append({
            "name": pos_path.stem,
            "path": pos_path,
            "data": sol_data,
            "color": color,
            "visible": tk.BooleanVar(value=True)
        })
        self._refresh_multi_solution_ui()
        self.redraw()

    def remove_compare_solution(self, index):
        if 0 <= index < len(self.multi_solutions):
            removed = self.multi_solutions.pop(index)
            self._refresh_multi_solution_ui()
            self.segment_status.configure(text=f"Removed comparison solution: {removed['name']}.")
            self.redraw()

    def clear_compare_solutions(self):
        if len(self.multi_solutions) > 1:
            self.multi_solutions = self.multi_solutions[:1]
            self._refresh_multi_solution_ui()
            self.segment_status.configure(text="Cleared extra comparison solutions.")
            self.redraw()

    def set_mode(self, name):
        internal = BILINGUAL_TO_MODE.get(name, name)
        self.mode.set(internal)
        disp = MODE_BILINGUAL.get(internal, internal)
        if hasattr(self, "mode_box") and self.mode_box.get() != disp:
            self.mode_box.set(disp)
        self.redraw()

    def _epoch_datetime_label(self,index):
        r=self.data["pos"].iloc[int(index)];dt=gpst_to_local_datetime(r.week,r.tow)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def mark_segment_start(self):
        if self.data is None:return
        self.segment_start_idx=int(self.slider.get())
        self.segment_status.configure(text=f"Segment start A: epoch {self.segment_start_idx+1:,} · {self._epoch_datetime_label(self.segment_start_idx)} CST")

    def add_segment(self):
        if self.data is None:return
        if self.segment_start_idx is None:
            self.mark_segment_start();return
        end=int(self.slider.get());start=self.segment_start_idx
        if end==start:
            self.segment_status.configure(text="Move to a different epoch, then set the segment end.");return
        lo,hi=sorted((start,end));number=len(self.segments)+1
        color=self.segment_palette[(number-1)%len(self.segment_palette)]
        self.segments.append({"start":lo,"end":hi,"name":f"Segment {number}","color":color})
        self.segment_start_idx=None;self._refresh_segment_list()
        self._auto_save_segments()
        if hasattr(self, "_expand_segments_card") and callable(self._expand_segments_card):
            self._expand_segments_card()
        self.segment_status.configure(text=f"Added Segment {number}: {self._epoch_datetime_label(lo)} — {self._epoch_datetime_label(hi)} CST")
        self._refresh_segment_overlays()

    def _refresh_segment_list(self):
        if not hasattr(self,"segment_box"):return
        self.segment_box.delete(0,tk.END)
        for i,seg in enumerate(self.segments):
            if self.data is not None and "pos" in self.data and len(self.data["pos"]) > 0:
                lbl = f"{seg['name']}: {self._epoch_datetime_label(seg['start'])} – {self._epoch_datetime_label(seg['end'])}"
            else:
                lbl = f"{seg['name']}: Epoch {seg['start']} – {seg['end']}"
            self.segment_box.insert(tk.END, lbl)
            self.segment_box.itemconfig(i,fg=seg.get("color", "#333"))
        if len(self.segments) > 0 and hasattr(self, "_expand_segments_card") and callable(self._expand_segments_card):
            self._expand_segments_card()
        self._update_sidebar_scroll()

    def delete_segment(self):
        selected=list(self.segment_box.curselection())
        if not selected:return
        for i in reversed(selected):del self.segments[i]
        for i,seg in enumerate(self.segments):seg["name"]=f"Segment {i+1}"
        self._refresh_segment_list();self._refresh_segment_overlays()
        self._auto_save_segments()

    def clear_segments(self):
        self.segments.clear();self.segment_start_idx=None;self._refresh_segment_list()
        self.segment_status.configure(text="All marked segments cleared.");self._refresh_segment_overlays()
        self._auto_save_segments()

    def on_segment_select(self, event=None):
        if not hasattr(self, "segment_box"): return
        sel = self.segment_box.curselection()
        if not sel or not self.segments or self.data is None: return
        idx = sel[0]
        if idx >= len(self.segments): return
        self._focus_on_segment(idx)

    def on_segment_double_click(self, event=None):
        if not hasattr(self, "segment_box"): return
        sel = self.segment_box.curselection()
        if not sel or not self.segments or self.data is None: return
        idx = sel[0]
        if idx >= len(self.segments): return
        # Double clicking the currently focused segment toggles back to full view
        if getattr(self, "_selected_segment_idx", None) == idx:
            self.reset_segment_focus()
        else:
            self._focus_on_segment(idx)

    def reset_segment_focus(self):
        """Restore full trajectory and timeline view across all subplots."""
        self._selected_segment_idx = None
        if hasattr(self, "segment_box"):
            self.segment_box.selection_clear(0, tk.END)
        self.fit_view()
        self._refresh_segment_overlays()
        self.segment_status.configure(text="已复位全图视图。在左侧区段列表点击任意时段可立即联动聚焦。")

    def _focus_on_segment(self, idx):
        if self.data is None or not self.segments or idx >= len(self.segments): return
        seg = self.segments[idx]
        self._selected_segment_idx = idx
        p = self.data["pos"]
        t0 = float(p.iloc[seg["start"]].time)
        t1 = float(p.iloc[seg["end"]].time)
        dur = max(0.1, t1 - t0)
        pad = max(2.0, dur * 0.12)

        # 1. Zoom timeline datetime axes
        for ax in self.datetime_axes:
            ax.set_xlim(t0 - pad, t1 + pad)

        # 2. Zoom ground track axes
        g = p.iloc[seg["start"]:seg["end"]+1]
        for ax in self.track_axes:
            e_min, e_max = float(g.E.min()), float(g.E.max())
            n_min, n_max = float(g.N.min()), float(g.N.max())
            span_e = max(1.0, e_max - e_min)
            span_n = max(1.0, n_max - n_min)
            center_e = (e_min + e_max) / 2.0
            center_n = (n_min + n_max) / 2.0
            half_span = max(span_e * 0.65, span_n * 0.65, 8.0)
            ax.set_xlim(center_e - half_span, center_e + half_span)
            ax.set_ylim(center_n - half_span, center_n + half_span)
            if getattr(self, "equal_scale", None) and self.equal_scale.get():
                ax.set_aspect("equal", adjustable="box")

        # 3. Position epoch slider to start of segment and scrub
        self.slider.set(seg["start"])
        self.epoch.set(seg["start"])
        self.on_scrub(seg["start"])

        # 4. Refresh segment overlays with distinct highlight
        self._refresh_segment_overlays()

        # 5. Fault diagnosis explanation
        st = float(p.iloc[seg["start"]].tow)
        et = float(p.iloc[seg["end"]].tow)
        d_res = diag.diagnose_anomaly_segment(p, self.data.get("sat"), st, et)
        self.segment_status.configure(text=f"🔍 聚焦 [{seg['name']}]: 【{d_res['primary_cause']}】{d_res['summary']} | 建议: {d_res['recommendation']}")
        if hasattr(self, "analysis_info"):
            self.analysis_info.configure(
                text=f"当前聚焦时段 [{seg['name']}]: 历元 {seg['start']+1}~{seg['end']+1} · 持续 {dur:.1f}s | "
                     f"故障诊断: 【{d_res['primary_cause']}】{d_res['summary']} | 优化建议: {d_res['recommendation']}"
            )
        self.canvas.draw_idle()

    def auto_detect_issues(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先加载 .pos 解算数据后再进行智能异常探测。")
            return
        limit = float(self.false_fix_limit.get()) if hasattr(self, "false_fix_limit") else 1.0
        detected, meta = diag.detect_problematic_segments(
            self.data["pos"],
            sat_df=self.data.get("sat"),
            error_limit=limit,
            e3d_limit=3.0,
            resp_limit=3.0,
            min_duration=2.0,
            max_gap=5.0,
            return_details=True
        )
        if not detected:
            if meta.get("has_truth"):
                msg = f"全文未探测到持续水平误差 > {limit}m 或 3D 误差 > 3.0m 的异常时段，解算质量表现优异！"
            else:
                msg = "全文未探测到持续伪距残差粗差（>3.0m）时段。\n\n提示：加载参考真值（reference.csv）可获得真实毫米/分米级定位误差评估并精准识别假固定。"
            messagebox.showinfo("智能诊断结果", msg)
            return

        # 智能替换：清空旧探测并生成全新的精准超标区段列表（避免遗留之前无逻辑的旧区段）
        self.segments = detected
        self._refresh_segment_list()
        self._refresh_segment_overlays()
        self._auto_save_segments()
        if hasattr(self, "_expand_segments_card") and callable(self._expand_segments_card):
            self._expand_segments_card()

        if meta.get("has_truth"):
            ff_count = sum(1 for s in detected if s.get("type") == "false_fix")
            info_text = (
                f"【智能异常探测完成】\n\n"
                f"• 判定准则：水平误差 > {limit}m 或 3D误差 > 3.0m（平滑容差 5.0s，持续 ≥ 2.0s）\n"
                f"• 检出区段：共精确定位 {len(detected)} 处异常时段\n"
                f"• 假固定警报：发现 {ff_count} 处危险假固定（Fix但严重超标）\n\n"
                f"提示：双击左侧时段列表中的任意条目，右侧图表与轨迹将自动缩放聚焦！"
            )
            status_text = f"已智能定位 {len(detected)} 处超标时段（含 {ff_count} 处假固定），双击列表条目可自动缩放聚焦。"
        else:
            info_text = (
                f"【智能异常探测完成（回退模式）】\n\n"
                f"• 判定准则：未加载真值，基于「伪距残差 > 3.0m」粗差探测（持续 ≥ 2s，平滑容差 5s）\n"
                f"• 检出区段：共发现 {len(detected)} 处卫星残差粗差时段\n\n"
                f"💡 建议：加载参考真值（reference.csv）可获取真实水平与空间误差评估并精准定位假固定！"
            )
            status_text = f"（无真值回退模式）已根据伪距残差粗差标记 {len(detected)} 处异常时段。加载真值可获得真实误差。"

        messagebox.showinfo("智能探测完成", info_text)
        self.segment_status.configure(text=status_text)

    def _get_default_segments_path(self):
        if not hasattr(self, "pos_path") or not self.pos_path:
            return None
        return self.pos_path.with_suffix(".segments.json")

    def _auto_save_segments(self):
        seg_file = self._get_default_segments_path()
        if seg_file:
            self._save_segments_to_file(seg_file, silent=True)

    def _save_segments_to_file(self, path, silent=False):
        if not self.data or "pos" not in self.data:
            return
        p = self.data["pos"]
        data_to_save = []
        for seg in self.segments:
            s_idx = int(seg.get("start", 0))
            e_idx = int(seg.get("end", 0))
            s_idx = max(0, min(s_idx, len(p) - 1))
            e_idx = max(0, min(e_idx, len(p) - 1))
            item = {
                "start": s_idx,
                "end": e_idx,
                "name": str(seg.get("name", "Segment")),
                "color": str(seg.get("color", "#b23a48")),
                "cause": str(seg.get("cause", "")),
            }
            if "tow" in p.columns:
                item["tow_start"] = float(p.iloc[s_idx].tow)
                item["tow_end"] = float(p.iloc[e_idx].tow)
            data_to_save.append(item)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            if not silent:
                messagebox.showinfo("保存成功", f"时间区段已成功保存至:\n{path}")
        except Exception as exc:
            if not silent:
                messagebox.showerror("保存失败", f"保存区段失败: {exc}")

    def _load_segments_from_file(self, path, silent=False):
        if not Path(path).exists() or not self.data or "pos" not in self.data:
            return False
        p = self.data["pos"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            loaded = []
            for item in raw_list:
                s_idx = item.get("start", 0)
                e_idx = item.get("end", 0)
                if "tow" in p.columns and "tow_start" in item and "tow_end" in item:
                    tows = p.tow.to_numpy()
                    if s_idx >= len(p) or abs(p.iloc[min(s_idx, len(p)-1)].tow - item["tow_start"]) > 0.5:
                        s_idx = int(np.clip(np.searchsorted(tows, item["tow_start"]), 0, len(p) - 1))
                        e_idx = int(np.clip(np.searchsorted(tows, item["tow_end"]), 0, len(p) - 1))
                s_idx = max(0, min(int(s_idx), len(p) - 1))
                e_idx = max(0, min(int(e_idx), len(p) - 1))
                if e_idx < s_idx:
                    s_idx, e_idx = e_idx, s_idx
                loaded.append({
                    "start": s_idx,
                    "end": e_idx,
                    "name": item.get("name", f"Segment {len(loaded)+1}"),
                    "color": item.get("color", self.segment_palette[len(loaded) % len(self.segment_palette)]),
                    "cause": item.get("cause", "")
                })
            self.segments = loaded
            self._refresh_segment_list()
            self._refresh_segment_overlays()
            if not silent:
                messagebox.showinfo("导入成功", f"已成功加载 {len(loaded)} 个时间区段！")
            return True
        except Exception as exc:
            if not silent:
                messagebox.showerror("导入失败", f"加载区段失败: {exc}")
            return False

    def save_segments_dialog(self):
        if not self.segments:
            messagebox.showinfo("提示", "当前没有已标记的时间区段可供保存。")
            return
        default_name = f"{self.pos_path.stem}.segments.json" if hasattr(self, "pos_path") and self.pos_path else "segments.json"
        out_file = filedialog.asksaveasfilename(
            title="保存时间区段 (JSON)",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile=default_name
        )
        if out_file:
            self._save_segments_to_file(Path(out_file), silent=False)

    def load_segments_dialog(self):
        in_file = filedialog.askopenfilename(
            title="导入时间区段 (JSON)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if in_file:
            self._load_segments_from_file(Path(in_file), silent=False)

    def open_report_dialog(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先加载 .pos 解算数据后再生成诊断报告。")
            return
        DiagnosticReportDialog(self)

    def show_selected(self):
        if self.data is None:return
        selected=[self._current_listed_modes[i] for i in self.plot_list.curselection() if i < len(self._current_listed_modes)]
        if not selected:return
        self.custom_modes=selected;self.custom_panels=self._initial_custom_panels(selected, self.custom_preset.get());self.custom_page=0;self.selected_custom_panel=0;self.focused_custom_panel=None
        self.mode.set("Custom Analysis");self.redraw()

    @staticmethod
    def _initial_custom_panels(modes, preset="Auto Grid (自适应等分)"):
        n = len(modes)
        if n == 0:
            return []

        panels = []
        p_lower = str(preset).lower()
        if "focus left" in p_lower or "左主右副" in p_lower:
            layout_type = "focus_left"
        elif "vertical" in p_lower or "纵向" in p_lower:
            layout_type = "vertical_stack"
        elif "focus top" in p_lower or "上主下副" in p_lower:
            layout_type = "focus_top"
        else:
            layout_type = "auto_grid"

        if layout_type == "vertical_stack":
            page_size = 4
            for i, name in enumerate(modes):
                page_idx = i % page_size
                page_total = min(page_size, n - (i // page_size) * page_size)
                k = max(1, page_total)
                gap_y = 0.05
                top, bottom, left, width = 0.93, 0.09, 0.075, 0.89
                total_h = top - bottom
                h = (total_h - (k - 1) * gap_y) / k
                y = top - (page_idx + 1) * h - page_idx * gap_y
                panels.append({"mode": name, "rect": (left, y, width, h)})

        elif layout_type == "focus_left":
            page_size = 7
            for i, name in enumerate(modes):
                page_idx = i % page_size
                page_total = min(page_size, n - (i // page_size) * page_size)
                if page_idx == 0 or page_total == 1:
                    if page_total == 1:
                        panels.append({"mode": name, "rect": (0.075, 0.09, 0.89, 0.83)})
                    else:
                        panels.append({"mode": name, "rect": (0.065, 0.09, 0.54, 0.83)})
                else:
                    sub_idx = page_idx - 1
                    sub_total = page_total - 1
                    if sub_total <= 3:
                        gap_y = 0.065
                        top, bottom, right_x, right_w = 0.92, 0.08, 0.67, 0.295
                        h = (top - bottom - (sub_total - 1) * gap_y) / sub_total
                        y = top - (sub_idx + 1) * h - sub_idx * gap_y
                        panels.append({"mode": name, "rect": (right_x, y, right_w, h)})
                    else:
                        r_cols = 2
                        r_rows = (sub_total + 1) // 2
                        r_row, r_col = divmod(sub_idx, r_cols)
                        gap_x, gap_y = 0.035, 0.055
                        top, bottom, right_x = 0.92, 0.08, 0.64
                        total_w = 0.33
                        sub_w = (total_w - gap_x) / 2
                        h = (top - bottom - (r_rows - 1) * gap_y) / r_rows
                        x = right_x + r_col * (sub_w + gap_x)
                        y = top - (r_row + 1) * h - r_row * gap_y
                        panels.append({"mode": name, "rect": (x, y, sub_w, h)})

        elif layout_type == "focus_top":
            page_size = 5
            for i, name in enumerate(modes):
                page_idx = i % page_size
                page_total = min(page_size, n - (i // page_size) * page_size)
                if page_idx == 0 or page_total == 1:
                    if page_total == 1:
                        panels.append({"mode": name, "rect": (0.075, 0.08, 0.89, 0.84)})
                    else:
                        panels.append({"mode": name, "rect": (0.075, 0.53, 0.89, 0.38)})
                else:
                    sub_idx = page_idx - 1
                    sub_total = page_total - 1
                    gap_x = 0.045
                    left, right, bottom, h = 0.075, 0.965, 0.08, 0.36
                    sub_w = (right - left - (sub_total - 1) * gap_x) / sub_total
                    x = left + sub_idx * (sub_w + gap_x)
                    panels.append({"mode": name, "rect": (x, bottom, sub_w, h)})

        else: # auto_grid
            if n == 1:
                panels.append({"mode": modes[0], "rect": (0.075, 0.08, 0.89, 0.84)})
            elif n == 2:
                w = 0.425; gap_x = 0.06; left = 0.075
                panels.append({"mode": modes[0], "rect": (left, 0.08, w, 0.83)})
                panels.append({"mode": modes[1], "rect": (left + w + gap_x, 0.08, w, 0.83)})
            elif n == 3:
                w_top = 0.425; gap_x = 0.06; left = 0.075
                panels.append({"mode": modes[0], "rect": (left, 0.525, w_top, 0.375)})
                panels.append({"mode": modes[1], "rect": (left + w_top + gap_x, 0.525, w_top, 0.375)})
                panels.append({"mode": modes[2], "rect": (left, 0.08, 0.89, 0.375)})
            elif n == 4:
                w = 0.425; gap_x = 0.06; left = 0.075
                h = 0.375; gap_y = 0.075
                panels.append({"mode": modes[0], "rect": (left, 0.53, w, h)})
                panels.append({"mode": modes[1], "rect": (left + w + gap_x, 0.53, w, h)})
                panels.append({"mode": modes[2], "rect": (left, 0.08, w, h)})
                panels.append({"mode": modes[3], "rect": (left + w + gap_x, 0.08, w, h)})
            else:
                page_size = 6
                cols = 3
                rows = 2
                left, right, bottom, top = 0.065, 0.985, 0.08, 0.91
                gap_x, gap_y = 0.045, 0.075
                width = (right - left - gap_x * (cols - 1)) / cols
                height = (top - bottom - gap_y * (rows - 1)) / rows
                for i, name in enumerate(modes):
                    page_idx = i % page_size
                    row, col = divmod(page_idx, cols)
                    x = left + col * (width + gap_x)
                    y = top - (row + 1) * height - row * gap_y
                    panels.append({"mode": name, "rect": (x, y, width, height)})

        return panels

    def _change_custom_page(self,step):
        if self.mode.get()!="Custom Analysis" or not self.custom_panels:return
        pages=max(1,(len(self.custom_panels)+self.custom_page_size-1)//self.custom_page_size)
        self.custom_page=(self.custom_page+step)%pages
        self.selected_custom_panel=min(self.custom_page*self.custom_page_size,len(self.custom_panels)-1)
        self.focused_custom_panel=None;self.redraw()

    def _begin_plot_drag(self,event):
        index=self.plot_list.nearest(event.y)
        if 0<=index<len(self._current_listed_modes) and self.plot_list.bbox(index):
            self._drag_plot_name=self._current_listed_modes[index]

    def _drop_plot_on_panel(self,event):
        name=self._drag_plot_name;self._drag_plot_name=None
        if name is None or self.mode.get()!="Custom Analysis" or not self.custom_panels:return
        target=self.root.winfo_containing(event.x_root,event.y_root);widget=self.canvas.get_tk_widget()
        if target is not widget:return
        width=max(1,widget.winfo_width());height=max(1,widget.winfo_height())
        x=(event.x_root-widget.winfo_rootx())*self.fig.bbox.width/width
        y=(height-(event.y_root-widget.winfo_rooty()))*self.fig.bbox.height/height
        for i,ax in enumerate(self.plot_axes):
            if ax.bbox.contains(x,y):
                panel_index=self.visible_panel_indices[i] if i<len(self.visible_panel_indices) else i
                self.selected_custom_panel=panel_index;self.custom_panels[panel_index]["mode"]=name
                self.custom_modes=[panel["mode"] for panel in self.custom_panels]
                self.segment_status.configure(text=f"Panel {panel_index+1} now displays {name}.")
                self.redraw();return

    def _assign_selected_plot(self):
        """Place the selected list item into the active panel without dragging."""
        if self.mode.get()!="Custom Analysis" or not self.custom_panels:return
        selected=self.plot_list.curselection()
        if not selected:return
        panel_index=min(max(0,self.selected_custom_panel),len(self.custom_panels)-1)
        sel_idx = selected[-1]
        if sel_idx < len(self._current_listed_modes):
            name=self._current_listed_modes[sel_idx]
            self.custom_panels[panel_index]["mode"]=name
            self.custom_modes=[panel["mode"] for panel in self.custom_panels]
            self.segment_status.configure(text=f"Panel {panel_index+1} now displays {name}.")
            self.redraw()

    def _focus_custom_panel(self):
        if self.mode.get()!="Custom Analysis" or not self.custom_panels:return
        self.focused_custom_panel=min(max(0,self.selected_custom_panel),len(self.custom_panels)-1)
        self.redraw()

    def _split_custom_panel(self,orientation):
        if self.mode.get()!="Custom Analysis" or not self.custom_panels:return
        i=min(max(0,self.selected_custom_panel),len(self.custom_panels)-1);panel=self.custom_panels[i]
        x,y,w,h=panel["rect"];gap=.025
        if orientation=="vertical":
            half=(w-gap)/2;panel["rect"]=(x,y,half,h);new_rect=(x+half+gap,y,half,h)
        else:
            half=(h-gap)/2;panel["rect"]=(x,y+half+gap,w,half);new_rect=(x,y,w,half)
        self.custom_panels.insert(i+1,{"mode":None,"rect":new_rect});self.custom_modes=[p["mode"] for p in self.custom_panels]
        self.selected_custom_panel=i+1;self.focused_custom_panel=None;self.redraw()

    def _clear_custom_panel(self):
        if self.mode.get()!="Custom Analysis" or not self.custom_panels:return
        i=min(max(0,self.selected_custom_panel),len(self.custom_panels)-1)
        self.custom_panels[i]["mode"]=None;self.custom_modes=[p["mode"] for p in self.custom_panels];self.focused_custom_panel=None;self.redraw()

    def _reset_custom_grid(self):
        if self.mode.get() != "Custom Analysis":
            return
        modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
        if not modes:
            modes = [self._current_listed_modes[i] for i in self.plot_list.curselection() if i < len(self._current_listed_modes)]
        if not modes:
            return
        self.custom_modes = modes
        self.custom_panels = self._initial_custom_panels(modes, self.custom_preset.get())
        self.custom_page = 0
        self.selected_custom_panel = 0
        self.focused_custom_panel = None
        self.redraw()

    def _on_custom_preset_changed(self, event=None):
        if self.mode.get() != "Custom Analysis":
            self.mode.set("Custom Analysis")
        modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
        if not modes:
            modes = [self._current_listed_modes[i] for i in self.plot_list.curselection() if i < len(self._current_listed_modes)]
        if not modes:
            modes = ["Gnd Trk", "ENU Error", "Solution Status", "Satellites"]
        self.custom_modes = modes
        self.custom_panels = self._initial_custom_panels(modes, self.custom_preset.get())
        self.custom_page = 0
        self.selected_custom_panel = 0
        self.focused_custom_panel = None
        self.segment_status.configure(text=f"布局预设已切换为: {self.custom_preset.get()} ({len(modes)} 个子图)")
        self.redraw()

    def _move_custom_panel(self, direction):
        if self.mode.get() != "Custom Analysis" or len(self.custom_panels) < 2:
            return
        idx = min(max(0, self.selected_custom_panel), len(self.custom_panels) - 1)
        new_idx = idx + direction
        if 0 <= new_idx < len(self.custom_panels):
            self.custom_panels[idx], self.custom_panels[new_idx] = self.custom_panels[new_idx], self.custom_panels[idx]
            modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
            self.custom_modes = modes
            self.custom_panels = self._initial_custom_panels(modes, self.custom_preset.get())
            self.selected_custom_panel = new_idx
            self.focused_custom_panel = None
            self.segment_status.configure(text=f"已移动子图: {self.custom_modes[new_idx]} 到槽位 {new_idx+1}")
            self.redraw()

    def _promote_custom_panel_to_primary(self, ax=None):
        if self.mode.get() != "Custom Analysis" or len(self.custom_panels) < 2:
            return
        if ax in self.plot_axes:
            idx = self.plot_axes.index(ax)
            panel_idx = self.visible_panel_indices[idx] if idx < len(self.visible_panel_indices) else idx
        else:
            panel_idx = self.selected_custom_panel
        if panel_idx <= 0 or panel_idx >= len(self.custom_panels):
            return
        promoted = self.custom_panels.pop(panel_idx)
        self.custom_panels.insert(0, promoted)
        modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
        self.custom_modes = modes
        self.custom_panels = self._initial_custom_panels(modes, self.custom_preset.get())
        self.selected_custom_panel = 0
        self.focused_custom_panel = None
        self.segment_status.configure(text=f"已将 [{promoted.get('mode')}] 设为主图 (首位槽位)")
        self.redraw()

    def _swap_custom_panels(self, idx1, idx2):
        if self.mode.get() != "Custom Analysis":
            return
        if 0 <= idx1 < len(self.custom_panels) and 0 <= idx2 < len(self.custom_panels) and idx1 != idx2:
            self.custom_panels[idx1], self.custom_panels[idx2] = self.custom_panels[idx2], self.custom_panels[idx1]
            modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
            self.custom_modes = modes
            self.custom_panels = self._initial_custom_panels(modes, self.custom_preset.get())
            self.selected_custom_panel = idx2
            self.focused_custom_panel = None
            self.segment_status.configure(text=f"已互换槽位: {self.custom_modes[idx2]} ⇄ {self.custom_modes[idx1]}")
            self.redraw()

    def _load_layout_profiles(self):
        self.layout_profiles = {}
        prof_file = HERE / "configs" / "custom_layout_profiles.json"
        if prof_file.exists():
            try:
                import json
                with open(prof_file, "r", encoding="utf-8") as f:
                    self.layout_profiles = json.load(f)
            except Exception as e:
                print(f"Error loading layout profiles: {e}")

    def _rebuild_profile_menu(self):
        if not hasattr(self, "profile_menu"):
            return
        self.profile_menu.delete(0, tk.END)
        self.profile_menu.add_command(label="💾 保存当前排版为新方案…", command=self._save_current_layout_profile)
        self.profile_menu.add_separator()
        if self.layout_profiles:
            for p_name in self.layout_profiles:
                self.profile_menu.add_command(
                    label=p_name,
                    command=lambda name=p_name: self._apply_layout_profile(name)
                )
        else:
            self.profile_menu.add_command(label="(暂无保存方案)", state="disabled")

    def _update_profile_combobox(self):
        if hasattr(self, "profile_combo"):
            keys = list(self.layout_profiles.keys())
            self.profile_combo.configure(values=keys)
        self._rebuild_profile_menu()

    def _save_current_layout_profile(self):
        modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
        if not modes:
            messagebox.showwarning("提示", "当前没有正在显示的自定义子图，请先选择子图后再保存方案。")
            return
        name_win = tk.Toplevel(self.root)
        name_win.title("保存自定义多图方案")
        name_win.geometry("400x170")
        name_win.resizable(False, False)
        name_win.transient(self.root)
        name_win.grab_set()

        ttk.Label(name_win, text="请输入方案名称 (例如: 车载RTK动态评估、高程精度对比):", wraplength=360).pack(padx=16, pady=(16, 6), anchor="w")
        name_var = tk.StringVar(value=f"方案 {len(self.layout_profiles)+1}")
        entry = ttk.Entry(name_win, textvariable=name_var, width=34)
        entry.pack(padx=16, pady=4, fill=tk.X)
        entry.focus_set()

        def do_save():
            p_name = name_var.get().strip()
            if not p_name:
                messagebox.showwarning("提示", "方案名称不能为空。")
                return
            self.layout_profiles[p_name] = {
                "preset": self.custom_preset.get(),
                "modes": modes
            }
            prof_file = HERE / "configs" / "custom_layout_profiles.json"
            try:
                prof_file.parent.mkdir(parents=True, exist_ok=True)
                import json
                with open(prof_file, "w", encoding="utf-8") as f:
                    json.dump(self.layout_profiles, f, ensure_ascii=False, indent=2)
                self.segment_status.configure(text=f"多图方案 [{p_name}] 已成功保存")
                self._update_profile_combobox()
                self.selected_profile_name.set(p_name)
                name_win.destroy()
                messagebox.showinfo("成功", f"多图方案 [{p_name}] 已保存至配置库！")
            except Exception as exc:
                messagebox.showerror("保存失败", str(exc))

        btn_f = ttk.Frame(name_win)
        btn_f.pack(padx=16, pady=14, fill=tk.X)
        ttk.Button(btn_f, text="确定保存", command=do_save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_f, text="取消", command=name_win.destroy).pack(side=tk.RIGHT)

    def _apply_layout_profile(self, profile_name):
        if profile_name not in self.layout_profiles:
            return
        prof = self.layout_profiles[profile_name]
        preset = prof.get("preset", "Auto Grid (自适应等分)")
        modes = prof.get("modes", ["Gnd Trk", "ENU Error"])
        self.custom_preset.set(preset)
        self.custom_modes = list(modes)
        self.custom_panels = self._initial_custom_panels(modes, preset)
        self.custom_page = 0
        self.selected_custom_panel = 0
        self.focused_custom_panel = None
        self.selected_profile_name.set(profile_name)
        self.mode.set("Custom Analysis")
        self.segment_status.configure(text=f"已成功载入多图方案: [{profile_name}]")
        self.redraw()

    def _get_active_layout_modes(self) -> list[str]:
        """Return list of active mode names corresponding to the currently rendered view."""
        cur_mode = self.mode.get()
        if cur_mode == "Dashboard":
            return list(self.dashboard_modes)
        elif cur_mode == "Env & Geometry Combined":
            return ["Multipath Skyplot", "El vs SNR & Resp", "Satellites", "Directional DOP", "Residual Triad"]
        elif cur_mode == "Skyplot & C/N0 (Env)":
            return ["Multipath Skyplot", "C/N0", "El vs SNR & Resp"]
        elif cur_mode == "Custom Analysis":
            if self.custom_panels:
                return [p["mode"] for p in self.custom_panels if p.get("mode")]
            return list(self.custom_modes) if self.custom_modes else ["Gnd Trk"]
        else:
            return [cur_mode]

    def add_selected_plot(self, mode_name=None):
        """Append currently selected item(s) from 'Available plots' list into active layout."""
        if self.data is None:
            messagebox.showwarning("提示", "请先打开解算数据文件。")
            return
        if mode_name is not None:
            modes_to_add = [mode_name]
        else:
            selected_indices = self.plot_list.curselection()
            if selected_indices:
                modes_to_add = [self._current_listed_modes[i] for i in selected_indices if i < len(self._current_listed_modes)]
            else:
                self._popup_add_subplot_picker()
                return

        mode = self.mode.get()
        if mode == "Dashboard":
            to_add = [m for m in modes_to_add if m not in self.dashboard_modes]
            if not to_add:
                for idx, m in enumerate(self.dashboard_modes):
                    if m in modes_to_add:
                        self.selected_dashboard_index = idx
                        break
                self.segment_status.configure(text=f"提示: 所选图表已存在于 Dashboard 中，已直接聚焦。")
                self.redraw()
                return
            for m in to_add:
                self.dashboard_modes.append(m)
            self.selected_dashboard_index = len(self.dashboard_modes) - 1
            self.focused_dashboard_subplot = None
            self.segment_status.configure(text=f"已向 Dashboard 追加 {len(to_add)} 个新子图，当前共 {len(self.dashboard_modes)} 个。")
            self.redraw()
        else:
            current_modes = self._get_active_layout_modes()
            to_add = [m for m in modes_to_add if m not in current_modes]
            if not to_add:
                if mode == "Custom Analysis":
                    for idx, p in enumerate(self.custom_panels):
                        if p.get("mode") in modes_to_add:
                            self.selected_custom_panel = idx
                            break
                disp_name = MODE_BILINGUAL.get(modes_to_add[0], modes_to_add[0])
                self.segment_status.configure(text=f"提示: 图表 [{disp_name}] 已存在于当前排版中，已直接聚焦定位，未重复追加。")
                self.redraw()
                return
            new_modes = list(current_modes) + to_add
            self.mode.set("Custom Analysis")
            self.custom_modes = new_modes
            self.custom_panels = self._initial_custom_panels(new_modes, self.custom_preset.get())
            self.selected_custom_panel = len(self.custom_panels) - 1
            self.focused_custom_panel = None
            self.segment_status.configure(text=f"已成功追加 {len(to_add)} 个新子图，当前共 {len(self.custom_panels)} 个。")
            self.redraw()

    def _popup_add_subplot_picker(self, insert_idx=None):
        """Show categorized popup menu to pick and add a new subplot."""
        picker_menu = tk.Menu(self.root, tearoff=False)
        picker_menu.add_command(label="➕ 选择要添加的子图类型:", state="disabled")
        picker_menu.add_separator()
        for cat_name, modes in PLOT_CATEGORIES.items():
            cat_menu = tk.Menu(picker_menu, tearoff=False)
            for m in modes:
                cat_menu.add_command(
                    label=MODE_BILINGUAL.get(m, m),
                    command=lambda name=m: self._insert_or_append_subplot(name, insert_idx)
                )
            picker_menu.add_cascade(label=cat_name, menu=cat_menu)
        try:
            picker_menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            picker_menu.grab_release()

    def _insert_or_append_subplot(self, mode_name, insert_idx=None):
        if self.data is None:
            return
        mode = self.mode.get()
        if mode == "Dashboard":
            if mode_name in self.dashboard_modes:
                self.selected_dashboard_index = self.dashboard_modes.index(mode_name)
                self.segment_status.configure(text=f"提示: 子图 [{mode_name}] 已存在于 Dashboard 中，已直接聚焦。")
                self.redraw()
                return
            if insert_idx is not None and 0 <= insert_idx < len(self.dashboard_modes):
                self.dashboard_modes.insert(insert_idx + 1, mode_name)
                self.selected_dashboard_index = insert_idx + 1
            else:
                self.dashboard_modes.append(mode_name)
                self.selected_dashboard_index = len(self.dashboard_modes) - 1
            self.focused_dashboard_subplot = None
            self.segment_status.configure(text=f"已添加子图 [{mode_name}] 到 Dashboard，当前共 {len(self.dashboard_modes)} 个。")
            self.redraw()
        else:
            current_modes = self._get_active_layout_modes()
            if mode_name in current_modes:
                if mode == "Custom Analysis":
                    self.selected_custom_panel = current_modes.index(mode_name)
                disp = MODE_BILINGUAL.get(mode_name, mode_name)
                self.segment_status.configure(text=f"提示: 子图 [{disp}] 已存在于当前排版中，已直接聚焦。")
                self.redraw()
                return
            if insert_idx is not None and 0 <= insert_idx < len(current_modes):
                current_modes.insert(insert_idx + 1, mode_name)
                new_sel = insert_idx + 1
            else:
                current_modes.append(mode_name)
                new_sel = len(current_modes) - 1
            self.mode.set("Custom Analysis")
            self.custom_modes = current_modes
            self.custom_panels = self._initial_custom_panels(current_modes, self.custom_preset.get())
            self.selected_custom_panel = new_sel
            self.focused_custom_panel = None
            disp = MODE_BILINGUAL.get(mode_name, mode_name)
            self.segment_status.configure(text=f"已添加子图 [{disp}]，当前共 {len(self.custom_panels)} 个。")
            self.redraw()

    def delete_selected_subplot(self):
        if self.data is None:return
        mode=self.mode.get()
        if mode=="Dashboard":
            if not self.dashboard_modes:return
            idx=min(max(0,self.selected_dashboard_index),len(self.dashboard_modes)-1)
            removed=self.dashboard_modes.pop(idx)
            self.selected_dashboard_index=max(0,min(self.selected_dashboard_index,len(self.dashboard_modes)-1))
            self.focused_dashboard_subplot=None
            self.segment_status.configure(text=f"Removed dashboard subplot: {removed}. {len(self.dashboard_modes)} remaining.")
            self.redraw()
        elif mode=="Custom Analysis":
            if not self.custom_panels:return
            idx=min(max(0,self.selected_custom_panel),len(self.custom_panels)-1)
            self.custom_panels.pop(idx)
            self.custom_modes=[p["mode"] for p in self.custom_panels]
            self.focused_custom_panel=None
            if self.custom_panels:
                modes=[p["mode"] for p in self.custom_panels if p["mode"] is not None]
                if not modes:modes=[None]*len(self.custom_panels)
                self.custom_panels=self._initial_custom_panels(modes, self.custom_preset.get())
                self.selected_custom_panel=max(0,min(idx,len(self.custom_panels)-1))
            else:
                self.selected_custom_panel=0
            self.segment_status.configure(text=f"Removed custom panel {idx+1}. {len(self.custom_panels)} remaining.")
            self.redraw()
        else:
            messagebox.showinfo("Delete subplot","Delete subplot is available in Dashboard and Custom Analysis modes.")

    def reset_subplots_layout(self):
        mode=self.mode.get()
        if mode=="Dashboard":
            self.dashboard_modes=list(self.dashboard_modes_default)
            self.focused_dashboard_subplot=None
            self.selected_dashboard_index=0
            self.segment_status.configure(text="Restored default Dashboard subplots.")
            self.redraw()
        elif mode=="Custom Analysis":
            self._reset_custom_grid()
        else:
            self.fit_view()

    def _filtered(self):
        p=self.data["pos"]; val=self.qfilter.get()
        if val=="All Q":return p
        q=int(val.split("=")[1].split(":")[0]); return p[p.q==q]

    def draw_empty(self,text):
        self.ax.clear(); self.ax.text(.5,.5,text,ha="center",va="center",transform=self.ax.transAxes,fontsize=15,color="#666"); self.ax.set_axis_off(); self.canvas.draw_idle()

    def redraw(self):
        if self.data is None:self.draw_empty("打开 RTKLIB .pos 文件开始绘图"); return
        self._blit_background=None;self._last_cursor_index=None;self.dynamic_artists=[]
        self.vehicle_marker=None;self.vehicle_trail=None;self.vehicle_label=None;self.truth_marker=None;self.error_link=None
        self.sky_current=None;self.sky_labels=[];self.sky_plots=[];self.time_cursors=[];self.custom_updaters=[];self.custom_dynamic_artists=[];self.dop_frames={};self.cov_twin_axes={}
        self.track_axes=[];self.datetime_axes=[];self.segment_track_lines=[];self.segment_sky_artists=[];self.segment_artists=[]
        self.ax.clear(); self.fig.clear(); mode=self.mode.get(); p=self._filtered(); full=self.data["pos"]; s=self.data["sat"]
        if hasattr(self, "mode_box"):
            disp = MODE_BILINGUAL.get(mode, mode)
            if self.mode_box.get() != disp:
                self.mode_box.set(disp)
        self._refresh_plot_list()
        if mode=="Multi-Compare":
            self._toggle_sidebar_mode("multi_compare")
            self._draw_multi_compare();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        else:
            self._toggle_sidebar_mode("standard")
        if mode=="Dashboard":
            self._draw_dashboard();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        if mode=="Custom Analysis":
            self._draw_custom_grid();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        if mode=="Env & Geometry Combined":
            self._draw_env_geometry_combined();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        if mode in ("Skyplot & C/N0 (Env)",):
            self._draw_skyplot_cn0_3in1();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        if mode == "Skyplot":
            self.ax = self.fig.add_subplot(111, projection="polar"); self.plot_axes = [self.ax]
            self._draw_sky(self.ax); self._apply_segment_overlays(); self._register_dynamic_artists(); self.canvas.draw_idle(); self._update_cursor(); self.update_analysis(); return
        if mode == "Multipath Skyplot":
            self.ax = self.fig.add_subplot(111, projection="polar"); self.plot_axes = [self.ax]
            self._draw_multipath_skyplot(self.ax); self._apply_segment_overlays(); self._register_dynamic_artists(); self.canvas.draw_idle(); self._update_cursor(); self.update_analysis(); return
        if mode in ("ENU Position","ENU Error","Error Histogram","Body-Frame Error"):
            sharex = (mode != "Error Histogram")
            axes=self.fig.subplots(3,1,sharex=sharex);self.plot_axes=list(axes);self.ax=axes[0]
        elif mode in ("Directional DOP",):
            axes=self.fig.subplots(2,1,sharex=True);self.plot_axes=list(axes);self.ax=axes[0]
        else:
            self.ax=self.fig.add_subplot(111);self.plot_axes=[self.ax]
        ax=self.ax
        if mode=="Gnd Trk":self._draw_track(ax,p,full)
        elif mode=="Trajectory Error Heatmap":self._draw_trajectory_error_heatmap(ax,p,full)
        elif mode=="ENU Position":self._draw_enu_positions(self.plot_axes,p)
        elif mode=="Height":
            me=max(1,len(p)//120)
            t_gap, u_gap = insert_time_gaps(p.time, p.U)
            ax.plot(t_gap,u_gap,label="RTK height U",color="#16853b",marker=".",ms=3,markevery=me)
            if self.data["has_truth"]:
                t_gt_gap, gt_u_gap = insert_time_gaps(p.time, p.gt_U)
                ax.plot(t_gt_gap,gt_u_gap,label="Ground truth U",color="#252525",lw=.8,ls="--",marker=".",ms=2,markevery=me)
        elif mode=="ENU Error":self._draw_errors(self.plot_axes,p)
        elif mode=="Solution Status":ax.step(p.time,p.q,where="post",lw=.8,marker=".",ms=2,markevery=max(1,len(p)//120));ax.set(xlabel="Time from start (s)",ylabel="Q",yticks=sorted(full.q.unique()),title="Solution Status")
        elif mode=="AR Ratio":self._draw_ar_ratio(ax,p);ax.set_xlabel("时间 (s)")
        elif mode=="Differential Age":self._draw_differential_age(ax,p)
        elif mode=="Satellites":self._draw_sats(ax)
        elif mode=="DOP":self._draw_dop(ax)
        elif mode=="Residuals":self._draw_residuals(ax)
        elif mode=="Multipath Proxy":self._draw_multipath(ax)
        elif mode=="Cycle Slips":self._draw_slips(ax)
        elif mode=="Satellite Visibility Gantt":self._draw_satellite_visibility_gantt(ax,s)
        elif mode=="Per-PRN Residuals":self._draw_per_prn_residuals(ax)
        elif mode=="Covariance":self._draw_covariance(ax,p)
        elif mode=="False Fix Analysis":self._draw_false_fix(ax,p)
        elif mode=="Innovation / NIS":self._draw_nis(ax)
        elif mode=="Error CDF":self._draw_error_cdf(ax,p)
        elif mode=="Error Histogram":self._draw_error_histogram(self.plot_axes,p)
        elif mode=="Residual Triad":self._draw_residual_triad(ax)
        elif mode=="El vs SNR & Resp":self._draw_el_snr_resp(ax)
        elif mode=="Top Outlier Satellites":self._draw_top_outliers(ax)
        elif mode=="Directional DOP":
            dop_df = diag.compute_directional_dop(self.data)
            if not dop_df.empty and len(self.plot_axes) >= 2:
                sec_cols = [c for c in dop_df.columns if c.startswith("sector_")]
                if sec_cols:
                    sec_labels = ["北(N)", "东北(NE)", "东(E)", "东南(SE)", "南(S)", "西南(SW)", "西(W)", "西北(NW)"]
                    sec_colors = ["#2878B5", "#00b4d8", "#36965A", "#8ac926", "#ffca3a", "#ff924c", "#ff595e", "#6a4c93"]
                    self.plot_axes[0].stackplot(dop_df.time, *[dop_df[c] for c in sec_cols], labels=sec_labels, colors=sec_colors, alpha=0.7)
                    self.plot_axes[0].set(ylabel="各扇区星数 (颗)", title="8 扇区空间几何覆盖与方向 DOP 精度因子")
                    self.plot_axes[0].legend(fontsize=7, ncol=4, loc="upper right")
                else:
                    self.plot_axes[0].plot(dop_df.time, dop_df.HDOP, color="#2878B5", label="HDOP")
                    self.plot_axes[0].plot(dop_df.time, dop_df.VDOP, color="#36965A", label="VDOP")
                    self.plot_axes[0].set(ylabel="DOP", title="方向 DOP 几何精度因子")
                    self.plot_axes[0].legend(fontsize=8)
                self.plot_axes[1].plot(dop_df.time, dop_df.EDOP, color="#2878B5", lw=1.1, label="EDOP (东向跨街分量)")
                self.plot_axes[1].plot(dop_df.time, dop_df.NDOP, color="#E07A2D", lw=1.1, label="NDOP (北向沿街分量)")
                self.plot_axes[1].plot(dop_df.time, dop_df.VDOP, color="#36965A", lw=1.1, label="VDOP (天向高程分量)")
                self.plot_axes[1].set(xlabel="相对时间 (s)", ylabel="方向 DOP 分量")
                lim_y = min(float(dop_df[["EDOP", "NDOP", "VDOP"]].quantile(0.99).max()) * 1.5, 30.0)
                self.plot_axes[1].set_ylim(0, max(5.0, lim_y))
            else:
                self._draw_directional_dop(ax)
        elif mode=="Body-Frame Error":
            if self.data.get("has_truth") and "along_err" in p.columns and len(self.plot_axes) >= 3:
                t_gap, along_gap, cross_gap, vert_gap = insert_time_gaps(p.time, p.along_err, p.cross_err, p.vert_err)
                self.plot_axes[0].plot(t_gap, along_gap, color="#2878B5", lw=0.8, label="Along-track (沿航向/纵向)")
                self.plot_axes[0].axhline(0, color="k", lw=0.5, ls=":")
                self.plot_axes[0].set(ylabel="沿向误差 (m)", title="车体坐标系误差分解 (Along-track / Cross-track / Vertical)")
                self.plot_axes[0].legend(fontsize=8)
                self.plot_axes[1].plot(t_gap, cross_gap, color="#d90429", lw=0.8, label="Cross-track (跨航向/横向)")
                self.plot_axes[1].axhline(0, color="k", lw=0.5, ls=":")
                self.plot_axes[1].set(ylabel="横向误差 (m)")
                self.plot_axes[1].legend(fontsize=8)
                self.plot_axes[2].plot(t_gap, vert_gap, color="#36965A", lw=0.8, label="Vertical (垂直高程/天向)")
                self.plot_axes[2].axhline(0, color="k", lw=0.5, ls=":")
                self.plot_axes[2].set(xlabel="相对时间 (s)", ylabel="天向误差 (m)")
                self.plot_axes[2].legend(fontsize=8)
            else:
                self._draw_body_frame_error(ax, p)
        elif mode in ("Satellite Elevation", "Elevation"):
            self._draw_elevation(ax)
        elif mode in ("Clock Bias & ISB", "Clock Bias"):
            self._draw_clock_bias_isb(ax)
        elif mode in ("C/N0", "C/N₀"):
            self._draw_cn0(ax)
        for plot_ax in self.plot_axes:plot_ax.grid(self.grid_on.get(),linestyle="--",alpha=.45)
        if mode not in NON_TIME_SERIES_MODES:
            for plot_ax in self.plot_axes:self.time_cursors.append(plot_ax.axvline(0,color="#333",ls="--",lw=.8,alpha=.7,zorder=10))
        if mode in ("ENU Position","ENU Error","Height","Solution Status","AR Ratio","Differential Age","Satellites","DOP","Residuals","Cycle Slips","Satellite Visibility Gantt","Covariance","Residual Triad","Directional DOP","Body-Frame Error","Satellite Elevation","Clock Bias & ISB","C/N0"):
            for plot_ax in self.plot_axes:self._set_datetime_axis(plot_ax)
        if mode in ("El vs SNR & Resp", "False Fix Analysis", "Covariance", "Satellite Visibility Gantt", "Trajectory Error Heatmap"):
            self._register_custom_dynamic(mode, ax, p, full)
        self._apply_segment_overlays();self._register_dynamic_artists()
        self.fig.tight_layout();self.canvas.draw_idle();self._update_cursor();self.update_analysis()

    def _register_dynamic_artists(self):
        """Keep moving cursors separate from the static plot for fast blitting."""
        self._home_limits={id(ax):(ax.get_xlim(),ax.get_ylim()) for ax in self.fig.axes}
        candidates=list(self.time_cursors)+list(self.custom_dynamic_artists)
        candidates.extend(a for a in (self.vehicle_marker,self.vehicle_trail,self.vehicle_label,self.truth_marker,self.error_link) if a is not None)
        for ax,points,_labels in self.sky_plots:candidates.extend((points,ax.title))
        seen=set()
        for artist in candidates:
            if id(artist) in seen:continue
            seen.add(id(artist));artist.set_animated(True);self.dynamic_artists.append(artist)

    def _schedule_cursor_update(self):
        """Coalesce rapid slider/mouse events so only the newest epoch is drawn."""
        if self._cursor_job is None:
            self._cursor_job=self.root.after(30,self._run_cursor_update)

    def _run_cursor_update(self):
        self._cursor_job=None
        self._update_cursor()

    def _on_canvas_draw(self,event):
        """Cache the static figure after a full draw, then paint moving overlays."""
        try:
            self._blit_background=self.canvas.copy_from_bbox(self.fig.bbox)
            self._blit_dynamic()
        except Exception:
            self._blit_background=None

    def _blit_dynamic(self):
        if self._blit_background is None:
            self.canvas.draw_idle();return
        self.canvas.restore_region(self._blit_background)
        for artist in self.dynamic_artists:
            if artist.get_visible() and artist.axes is not None:artist.axes.draw_artist(artist)
        self.canvas.blit(self.fig.bbox)

    def _set_datetime_axis(self,ax):
        p=self.data["pos"];first=p.iloc[0]
        def fmt(seconds,_pos):
            dt=gpst_to_local_datetime(first.week,first.tow+float(seconds))
            return dt.strftime("%Y-%m-%d\n%H:%M:%S")
        ax.xaxis.set_major_formatter(FuncFormatter(fmt));ax.set_xlabel("日期时间（中国标准时间 UTC+08:00）")
        if ax not in self.datetime_axes:self.datetime_axes.append(ax)

    def _sat_records_at_tow(self,tow):
        if not self._sat_tows.size:return self._sat_sorted.iloc[0:0]
        j=int(np.clip(np.searchsorted(self._sat_tows,tow),0,len(self._sat_tows)-1))
        if j and abs(self._sat_tows[j-1]-tow)<abs(self._sat_tows[j]-tow):j-=1
        t=self._sat_tows[j];lo=np.searchsorted(self._sat_tows,t-1e-7,"left");hi=np.searchsorted(self._sat_tows,t+1e-7,"right")
        return self._sat_sorted.iloc[lo:hi]

    def _apply_segment_overlays(self):
        if not self.segments or self.data is None: return
        p = self.data["pos"]
        focused_idx = getattr(self, "_selected_segment_idx", None)
        has_focused = (focused_idx is not None and 0 <= focused_idx < len(self.segments))

        for ax in self.datetime_axes:
            for i, seg in enumerate(self.segments):
                t0 = float(p.iloc[seg["start"]].time)
                t1 = float(p.iloc[seg["end"]].time)
                is_foc = (has_focused and i == focused_idx)
                alpha = 0.35 if is_foc else (0.07 if has_focused else 0.15)
                lw = 1.4 if is_foc else 0.5
                patch = ax.axvspan(t0, t1, facecolor=seg["color"], alpha=alpha, edgecolor=seg["color"], linewidth=lw, zorder=2.5 if is_foc else 1.5)
                self.segment_artists.append(patch)

        for ax in self.track_axes:
            # 1. First draw background non-focused segments softly
            for i, seg in enumerate(self.segments):
                if has_focused and i == focused_idx:
                    continue
                g = p.iloc[seg["start"]:seg["end"]+1]
                alpha = 0.25 if has_focused else 0.70
                lw = 1.4 if has_focused else 2.5
                line, = ax.plot(g.E, g.N, color=seg["color"], lw=lw, alpha=alpha, zorder=6)
                self.segment_track_lines.append(line); self.segment_artists.append(line)
                first = ax.scatter([g.E.iloc[0]], [g.N.iloc[0]], s=16, marker="o", facecolors="white", edgecolors=seg["color"], linewidths=0.9, alpha=alpha, zorder=7)
                last = ax.scatter([g.E.iloc[-1]], [g.N.iloc[-1]], s=16, marker="s", facecolors=seg["color"], edgecolors="white", linewidths=0.6, alpha=alpha, zorder=7)
                self.segment_artists.extend((first, last))

            # 2. Draw focused segment prominently with high contrast and labels
            if has_focused:
                seg = self.segments[focused_idx]
                g = p.iloc[seg["start"]:seg["end"]+1]
                halo, = ax.plot(g.E, g.N, color="#ffffff", lw=7.0, alpha=0.95, zorder=8)
                line, = ax.plot(g.E, g.N, color=seg["color"], lw=4.2, alpha=1.0, zorder=9, label=f"★聚焦: {seg['name']}")
                self.segment_track_lines.append(line); self.segment_artists.extend((halo, line))
                s_pt = ax.scatter([g.E.iloc[0]], [g.N.iloc[0]], s=65, marker="o", facecolors="white", edgecolors=seg["color"], linewidths=2.5, zorder=11)
                e_pt = ax.scatter([g.E.iloc[-1]], [g.N.iloc[-1]], s=65, marker="s", facecolors=seg["color"], edgecolors="white", linewidths=1.8, zorder=11)
                self.segment_artists.extend((s_pt, e_pt))
                
                ann_s = ax.annotate("起点", (g.E.iloc[0], g.N.iloc[0]), xytext=(7, 7), textcoords="offset points", fontsize=8.5, weight="bold", color="#111", bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=seg["color"], lw=1.2, alpha=0.92), zorder=12)
                ann_e = ax.annotate("终点", (g.E.iloc[-1], g.N.iloc[-1]), xytext=(7, -12), textcoords="offset points", fontsize=8.5, weight="bold", color="#111", bbox=dict(boxstyle="round,pad=0.2", fc=seg["color"], ec="#ffffff", lw=1.2, alpha=0.92), zorder=12)
                self.segment_artists.extend((ann_s, ann_e))
                ax.legend(fontsize=7.5, loc="best")
            else:
                dummy, = ax.plot([], [], color="#C00000", lw=2.5, label=f"异常时段标注 (共{len(self.segments)}处)")
                self.segment_artists.append(dummy)
                ax.legend(fontsize=7.5, loc="upper left", ncol=2)

        # Multipath proxy uses elevation (not time) on its x-axis, so overlay
        # the observations from each selected interval as colored points.
        axes_by_mode={self.custom_modes[panel_index]:ax for panel_index,ax in zip(self.visible_panel_indices,self.plot_axes)
                      if panel_index<len(self.custom_modes)} if self.mode.get()=="Custom Analysis" else {}
        proxy_axes=[axes_by_mode["Multipath Proxy"]] if "Multipath Proxy" in axes_by_mode else ([self.ax] if self.mode.get()=="Multipath Proxy" else [])
        for ax in proxy_axes:
            for seg in self.segments:
                t0=float(p.iloc[seg["start"]].time);t1=float(p.iloc[seg["end"]].time)
                g=self._segment_sat_records(t0,t1);g=g[(g.freq==1)&(g.resp!=0)]
                if len(g)>1800:g=g.iloc[::int(np.ceil(len(g)/1800))]
                if not g.empty:self.segment_artists.append(ax.scatter(g.el,g.resp.abs(),s=8,color=seg["color"],alpha=.65,label=seg["name"],zorder=5))
        if self.mode.get()=="Dashboard" and len(self.plot_axes)>5:
            ax=self.plot_axes[5]
            for seg in self.segments:
                t0=float(p.iloc[seg["start"]].time);t1=float(p.iloc[seg["end"]].time);g=self._segment_sat_records(t0,t1)
                g=g[(g.freq==1)&(g.snr>0)]
                if len(g)>1800:g=g.iloc[::int(np.ceil(len(g)/1800))]
                if not g.empty:self.segment_artists.append(ax.scatter(g.el,g.snr,s=8,color=seg["color"],alpha=.55,label=seg["name"],zorder=5))

    def _segment_sat_records(self,t0,t1):
        """Return a time-window slice from the cached, sorted satellite table."""
        if not self._sat_times.size:return self._sat_by_time.iloc[0:0]
        lo=int(np.searchsorted(self._sat_times,t0,side="left"));hi=int(np.searchsorted(self._sat_times,t1,side="right"))
        return self._sat_by_time.iloc[lo:hi]

    def _refresh_segment_overlays(self):
        """Update marked intervals in-place without rebuilding the underlying plots."""
        for artist in self.segment_artists:
            try:artist.remove()
            except (ValueError, AttributeError):pass
        self.segment_artists=[];self.segment_track_lines=[];self.segment_sky_artists=[]
        for ax in self.track_axes:
            legend=ax.get_legend()
            if legend is not None:legend.remove()
        self._apply_segment_overlays()
        # Recreate base track legends after the overlays have been replaced.
        for ax in self.track_axes:
            if ax.get_legend() is None:ax.legend(fontsize=6,ncol=3)
        self.canvas.draw_idle()

    def _apply_robust_ylim(self, ax, series_list, is_symmetric=False, non_negative=False, min_span=1.0, default_max=None):
        """Apply a robust 95% quantile vertical range to prevent isolated outliers from blowing up the scale."""
        robust_var = getattr(self, "robust_view", None)
        default_robust = robust_var.get() if robust_var is not None else True
        is_robust = getattr(ax, "_robust_mode", default_robust)

        valid_arrays = []
        for s in series_list:
            if s is None: continue
            arr = np.asarray(s)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0: valid_arrays.append(arr)
        if not valid_arrays: return
        all_vals = np.concatenate(valid_arrays)
        if len(all_vals) == 0: return

        full_min = float(np.min(all_vals))
        full_max = float(np.max(all_vals))
        ax._full_ylim = (full_min, full_max)

        if not is_robust:
            if is_symmetric:
                m = max(abs(full_min), abs(full_max), min_span)
                margin = m * 0.05
                ax.set_ylim(-m - margin, m + margin)
            elif non_negative:
                ax.set_ylim(0, max(full_max * 1.05, min_span))
            else:
                span = max(full_max - full_min, min_span)
                ax.set_ylim(full_min - 0.05 * span, full_max + 0.05 * span)
            return

        has_clipped = False
        if is_symmetric:
            abs_vals = np.abs(all_vals)
            q95 = float(np.percentile(abs_vals, 95))
            span = max(q95 * 1.25, min_span)
            ax.set_ylim(-span, span)
            has_clipped = (full_min < -span) or (full_max > span)
            ax._robust_ylim = (-span, span)
        elif non_negative:
            q95 = float(np.percentile(all_vals, 95))
            high = max(q95 * 1.25, min_span)
            if default_max is not None:
                high = max(high, default_max)
            ax.set_ylim(0, high)
            has_clipped = full_max > high
            ax._robust_ylim = (0, high)
        else:
            q05 = float(np.percentile(all_vals, 5))
            q95 = float(np.percentile(all_vals, 95))
            span = max(q95 - q05, min_span)
            low = q05 - 0.15 * span
            high = q95 + 0.15 * span
            ax.set_ylim(low, high)
            has_clipped = (full_min < low) or (full_max > high)
            ax._robust_ylim = (low, high)

        ax._has_outliers = has_clipped

    def _toggle_subplot_robust_view(self, ax):
        cur = getattr(ax, "_robust_mode", getattr(self, "robust_view", type("V", (), {"get": lambda s: True})()).get())
        ax._robust_mode = not cur
        if not ax._robust_mode and hasattr(ax, "_full_ylim"):
            fmin, fmax = ax._full_ylim
            span = max(1e-3, fmax - fmin)
            ax.set_ylim(fmin - 0.05 * span, fmax + 0.05 * span)
        elif ax._robust_mode and hasattr(ax, "_robust_ylim"):
            ax.set_ylim(ax._robust_ylim)
        else:
            self.redraw()
            return
        self.canvas.draw_idle()

    def _toggle_global_robust_view(self):
        new_val = not self.robust_view.get()
        self.robust_view.set(new_val)
        self.redraw()
        st = "已开启 95% 稳健主体量程聚焦 (已滤除极端飞点)" if new_val else "已恢复全量极值视图"
        if hasattr(self, "segment_status"):
            self.segment_status.configure(text=st)

    def _draw_false_fix(self, ax, p):
        if not self.data["has_truth"]:
            ax.text(0.5, 0.5, "需要参考真值轨迹以评估 3D 定位误差与假固定\n(Ground truth is required for False Fix Analysis)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666")
            ax.set_title("假固定与 AR Ratio 检验 (False Fix Analysis)")
            return
        limit = float(self.false_fix_limit.get()) if hasattr(self, "false_fix_limit") else 1.0
        limit = max(0.001, limit)
        ratio_thres = 3.0

        ratio_clipped = p.ratio.fillna(1.0).clip(1.0, 15.0)
        e3d_clipped = p.e3d.fillna(0.01).clip(0.01, 3000.0)

        # 1. Float points (Q != 1)
        m_float = (p.q != 1)
        # 2. True Fix (Q == 1 and e3d <= limit)
        m_true_fix = (p.q == 1) & (p.e3d <= limit)
        # 3. False Fix (Q == 1 and e3d > limit)
        m_false_fix = (p.q == 1) & (p.e3d > limit)

        n_float = int(m_float.sum())
        n_true = int(m_true_fix.sum())
        n_false = int(m_false_fix.sum())

        if n_float > 0:
            ax.scatter(ratio_clipped[m_float], e3d_clipped[m_float],
                       s=14, marker="o", color="#ff9f1c", alpha=0.55, edgecolors="none",
                       label=f"Float (Q!=1, n={n_float})")
        if n_true > 0:
            ax.scatter(ratio_clipped[m_true_fix], e3d_clipped[m_true_fix],
                       s=16, marker="o", color="#2a9d8f", alpha=0.75, edgecolors="none",
                       label=f"True Fix (Err<={limit:g}m, n={n_true})")
        if n_false > 0:
            ax.scatter(ratio_clipped[m_false_fix], e3d_clipped[m_false_fix],
                       s=36, marker="^", color="#d90429", alpha=0.92, edgecolors="#660000", linewidths=0.5,
                       zorder=5, label=f"False Fix! (Err>{limit:g}m, n={n_false})")

        # Ratio threshold vertical dashed line
        ax.axvline(ratio_thres, color="#222222", ls="--", lw=1.2, label=f"Ratio Thres = {ratio_thres:g}")
        # Error limit horizontal dotted line
        ax.axhline(limit, color="#d90429", ls=":", lw=1.3, label=f"{limit:g}m Error Line")

        ax.set_yscale("log")
        ax.set_xlim(1.0, 15.5)
        max_err = float(e3d_clipped.max()) if not e3d_clipped.empty else 100.0
        ax.set_ylim(0.01, max(3000.0, max_err * 1.5))
        ax.set_xlabel("AR Ratio Factor (clipped at 15)")
        ax.set_ylabel("3D Position Error (m, log)")
        ax.set_title("假固定与 AR Ratio 检验 (False Fix Analysis)")
        ax.grid(self.grid_on.get(), which="both", ls=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    def _draw_error_cdf(self, ax, p):
        if not self.data["has_truth"]:
            ax.text(0.5, 0.5, "需要真值参考轨迹才能绘制误差累积概率分布 (CDF)", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            return
        eh = np.sort(p.eh.dropna())
        e3d = np.sort(p.e3d.dropna())
        if len(eh) == 0:
            ax.text(0.5, 0.5, "无有效误差数据", ha="center", va="center", transform=ax.transAxes)
            return
        cdf_y = np.linspace(0, 100, len(eh))
        cep50 = float(np.quantile(eh, 0.5))
        cep95 = float(np.quantile(eh, 0.95))
        p50 = float(np.quantile(e3d, 0.5))
        p95 = float(np.quantile(e3d, 0.95))
        kw_eh = self._get_line_kwargs("Error CDF", default_ls="-", default_lw=1.8)
        ax.plot(eh, cdf_y, color="#2878B5", label=f"水平误差 (CEP50={cep50:.3f}m, CEP95={cep95:.3f}m)", **kw_eh)
        ax.plot(e3d, cdf_y, color="#E07A2D", lw=kw_eh.get("lw", 1.8), ls="--", label=f"3D误差 (P50={p50:.3f}m, P95={p95:.3f}m)")
        ax.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.7, label="50% (中位数 / CEP50)")
        ax.axhline(68.3, color="green", ls=":", lw=0.8, alpha=0.7, label="68.3% (1-Sigma)")
        ax.axhline(95, color="red", ls="--", lw=0.8, alpha=0.7, label="95.0% (2DRMS / 2-Sigma)")
        ax.set_xlabel("误差 (m)")
        ax.set_ylabel("累积概率分布 (%)")
        ax.set_title("定位误差累积分布函数 (Empirical CDF)", fontsize=11, weight="bold")
        ax.set_xlim(0, max(0.08, float(np.quantile(e3d, 0.99) * 1.15)))
        ax.set_ylim(0, 101)
        ax.grid(self.grid_on.get(), alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    def _draw_error_histogram(self, axes, p):
        if not self.data["has_truth"]:
            target_ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes
            target_ax.text(0.5, 0.5, "需要真值参考轨迹才能绘制误差直方图", ha="center", va="center", transform=target_ax.transAxes, fontsize=10)
            return
        if not isinstance(axes, (list, tuple, np.ndarray)) or len(axes) < 3:
            ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes
            for c, col, label in [("de", "#2878B5", "ΔE"), ("dn", "#E07A2D", "ΔN"), ("du", "#36965A", "ΔU")]:
                vals = p[c].dropna()
                if vals.empty: continue
                ax.hist(vals, bins=35, density=True, alpha=0.45, color=col, edgecolor="white", label=f"{label} (RMS={np.sqrt(np.mean(vals**2)):.3f}m)")
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.set(xlabel="误差 (m)", ylabel="概率密度", title="ENU 误差分布直方图")
            ax.legend(fontsize=7); ax.grid(self.grid_on.get(), alpha=0.3)
            return

        specs = [("de", "东向误差 ΔE", "#2878B5"), ("dn", "北向误差 ΔN", "#E07A2D"), ("du", "高程误差 ΔU", "#36965A")]
        for ax, (c, label, col) in zip(axes, specs):
            vals = p[c].dropna()
            if vals.empty: continue
            mu, std = float(vals.mean()), float(vals.std())
            n, bins, _ = ax.hist(vals, bins=35, density=True, alpha=0.6, color=col, edgecolor="white")
            x = np.linspace(bins[0], bins[-1], 200)
            if std > 1e-6:
                y = (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mu) / std)**2)
                ax.plot(x, y, color="#222", lw=1.3, label=f"高斯拟合 N({mu:+.3f}, {std:.3f}²)")
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.set_xlabel(f"{label} (m)")
            if c == "de": ax.set_ylabel("概率密度")
            ax.set_title(f"{label} (偏置={mu:+.3f}m, RMS={np.sqrt(np.mean(vals**2)):.3f}m)", fontsize=9)
            ax.legend(fontsize=7); ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_constellation_summary(self, axes, s):
        if s is None or s.empty:
            target_ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes
            target_ax.text(0.5, 0.5, "缺少 .stat 卫星观测数据", ha="center", va="center", transform=target_ax.transAxes, fontsize=10)
            return
        s1 = s[s.freq == 1]
        valid_snr = s1[s1.snr > 0]
        con_cols = [c for c in diag.COLORS if c in valid_snr.constellation.values] or sorted(valid_snr.constellation.unique())

        if not isinstance(axes, (list, tuple, np.ndarray)) or len(axes) < 2:
            ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes
            data_by_const = [valid_snr[valid_snr.constellation == c].snr.to_numpy() for c in con_cols if not valid_snr[valid_snr.constellation == c].empty]
            labels_by_const = [c for c in con_cols if not valid_snr[valid_snr.constellation == c].empty]
            if data_by_const:
                bp = ax.boxplot(data_by_const, tick_labels=labels_by_const, patch_artist=True)
                for patch, c_name in zip(bp['boxes'], labels_by_const):
                    patch.set_facecolor(diag.COLORS.get(c_name, "#488f31"))
                    patch.set_alpha(0.7)
            ax.set(xlabel="导航星座", ylabel="载噪比 C/N₀ (dB-Hz)", title="各星座 L1 载噪比分布 (Boxplot)")
            ax.grid(self.grid_on.get(), alpha=0.3)
            return

        if "_cached_con_counts" not in self.data:
            self.data["_cached_con_counts"] = s1.groupby(["time", "constellation"]).prn.nunique().unstack(fill_value=0)
        counts = self.data["_cached_con_counts"]

        for col in counts.columns:
            axes[0].plot(counts.index, counts[col], label=col, color=diag.COLORS.get(col, "#666"), lw=1.0)
        axes[0].set(xlabel="相对时间 (s)", ylabel="可见卫星数 (颗)", title="各星座跟踪卫星数时序变化")
        axes[0].legend(fontsize=7, ncol=2); axes[0].grid(self.grid_on.get(), alpha=0.3)
        self._set_datetime_axis(axes[0])

        data_by_const = [valid_snr[valid_snr.constellation == c].snr.to_numpy() for c in con_cols if not valid_snr[valid_snr.constellation == c].empty]
        labels_by_const = [c for c in con_cols if not valid_snr[valid_snr.constellation == c].empty]
        if data_by_const:
            bp = axes[1].boxplot(data_by_const, tick_labels=labels_by_const, patch_artist=True)
            for patch, c_name in zip(bp['boxes'], labels_by_const):
                patch.set_facecolor(diag.COLORS.get(c_name, "#488f31"))
                patch.set_alpha(0.7)
        axes[1].set(xlabel="导航星座", ylabel="载噪比 C/N₀ (dB-Hz)", title="各星座 L1 载噪比分布 (Boxplot)")
        axes[1].grid(self.grid_on.get(), alpha=0.3)

    def switch_workspace(self, ws_idx: int):
        """Switch to one of the 5 Diagnostic Workspaces."""
        if not (0 <= ws_idx < len(DIAGNOSTIC_WORKSPACES)):
            return
        ws = DIAGNOSTIC_WORKSPACES[ws_idx]
        modes = list(ws["modes"])
        preset = ws["preset"]
        self.custom_preset.set(preset)
        self.custom_modes = list(modes)
        self.custom_panels = self._initial_custom_panels(modes, preset)
        self.custom_page = 0
        self.selected_custom_panel = 0
        self.focused_custom_panel = None
        self.mode.set("Custom Analysis")
        self.segment_status.configure(text=f"已切换至诊断工作区 [{ws['name']}]: {ws['desc']}")
        self.redraw()

    def export_academic_pptx(self):
        """Export publication-ready Wuhan University academic presentation."""
        if self.data is None:
            messagebox.showwarning("提示", "请先打开解算数据文件。")
            return
        import rtklib_export_report as rep
        init_name = f"{self.pos_path.stem}_故障诊断学术汇报.pptx" if self.pos_path else "RTKLIB_故障诊断学术汇报.pptx"
        out_file = filedialog.asksaveasfilename(
            title="导出武汉大学学术答辩 PPTX (16:9)",
            defaultextension=".pptx",
            filetypes=[("PowerPoint Presentation", "*.pptx"), ("All Files", "*.*")],
            initialfile=init_name
        )
        if not out_file:
            return
        try:
            self.segment_status.configure(text="正在生成武汉大学学术答辩 PPTX，请稍候...")
            self.root.update_idletasks()
            rep.generate_academic_pptx(self.data, out_file, segments=self.segments)
            self.segment_status.configure(text=f"学术答辩 PPTX 已成功导出至: {out_file}")
            if messagebox.askyesno("导出成功", f"武汉大学学术汇报 PPTX 已成功生成！\n\n路径: {out_file}\n\n是否立即打开查看？"):
                os.startfile(out_file)
        except Exception as exc:
            messagebox.showerror("导出失败", f"生成 PPTX 发生异常:\n{exc}")

    def export_diagnostic_html(self):
        """Export standalone interactive HTML diagnostic report."""
        if self.data is None:
            messagebox.showwarning("提示", "请先打开解算数据文件。")
            return
        import rtklib_export_report as rep
        init_name = f"{self.pos_path.stem}_全链路故障诊断报告.html" if self.pos_path else "RTKLIB_全链路故障诊断报告.html"
        out_file = filedialog.asksaveasfilename(
            title="导出全链路故障诊断 HTML 综合分析报告",
            defaultextension=".html",
            filetypes=[("HTML File", "*.html"), ("All Files", "*.*")],
            initialfile=init_name
        )
        if not out_file:
            return
        try:
            self.segment_status.configure(text="正在生成全链路故障诊断 HTML 报告，请稍候...")
            self.root.update_idletasks()
            rep.generate_diagnostic_html(self.data, out_file, segments=self.segments)
            self.segment_status.configure(text=f"全链路诊断 HTML 报告已成功导出至: {out_file}")
            if messagebox.askyesno("导出成功", f"全链路故障诊断 HTML 综合报告已成功生成！\n\n路径: {out_file}\n\n是否立即在浏览器中打开？"):
                import webbrowser
                webbrowser.open(Path(out_file).as_uri())
        except Exception as exc:
            messagebox.showerror("导出失败", f"生成 HTML 发生异常:\n{exc}")

    def _draw_residual_triad(self, ax):
        if "_cached_triad" not in self.data:
            self.data["_cached_triad"] = diag.compute_residual_triad(self.data)
        triad = self.data["_cached_triad"]
        if triad.empty:
            ax.text(0.5, 0.5, "无有效伪距残差数据 (需 .stat 文件)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("伪距残差统计三线时序")
            return
        t_gap, max_g, sec_g, rms_g = insert_time_gaps(triad.time, triad.max_resp, triad.sec_resp, triad.rms_resp)
        ax.plot(t_gap, max_g, color="#d90429", lw=1.1, label="第一大残差 (Max |resp|)")
        ax.plot(t_gap, sec_g, color="#00b4d8", lw=1.0, label="第二大残差 (2nd Max)")
        ax.plot(t_gap, rms_g, color="#0B2B52", lw=1.2, ls="--", label="全场RMS (RMS |resp|)")
        if hasattr(self, "_apply_robust_ylim"):
            self._apply_robust_ylim(ax, [triad.max_resp], non_negative=True, min_span=5.0, default_max=15.0)
        ax.set(xlabel="时间 (s)", ylabel="伪距残差 (m)", title="伪距残差统计三线时序 (Max / 2nd Max / RMS)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_el_snr_resp(self, ax):
        sat = self.data.get("sat")
        if sat is None or sat.empty:
            ax.text(0.5, 0.5, "无卫星观测数据 (需 .stat 文件)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("高度角-载噪比-残差联合分布")
            return
        s1 = sat[(sat.freq == 1) & (sat.snr > 0) & (sat.el > 0)].copy()
        if not s1.empty:
            s1_sub = s1.iloc[::max(1, len(s1)//25000)]
            resp_clip = np.clip(s1_sub.resp.fillna(0).abs(), 0, 15.0) if "resp" in s1_sub.columns else np.zeros(len(s1_sub))
            sc = ax.scatter(s1_sub.el, s1_sub.snr, c=resp_clip, cmap="turbo", s=14, vmin=0, vmax=15.0, alpha=0.88, edgecolors="none")
            fig_target = ax.figure if getattr(ax, "figure", None) is not None else getattr(self, "fig", None)
            if fig_target is not None:
                cb = fig_target.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
                cb.set_label("|Pseudorange Residual| (m)")
            el_grid = np.linspace(5, 90, 100)
            ax.plot(el_grid, 28.0 + 22.0 * np.sin(np.deg2rad(el_grid)), "k--", lw=1.5, label="标称信噪比包络")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, 90)
        ax.set(xlabel="高度角 (°)", ylabel="载噪比 (dB-Hz)", title="高度角 - 载噪比 - 残差联合分布")
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_top_outliers(self, ax):
        sat = self.data.get("sat")
        if sat is None or sat.empty:
            ax.text(0.5, 0.5, "无卫星残差统计数据 (需 .stat 文件)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("粗差与多路径污染卫星排行 (Top Outliers)")
            return

        s1 = sat[(sat.freq == 1) & (sat.resp != 0)]
        if s1.empty:
            s1 = sat[sat.resp != 0]
        if s1.empty:
            ax.text(0.5, 0.5, "无有效伪距残差记录", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("粗差与多路径污染卫星排行 (Top Outliers)")
            return

        prn_metrics = []
        for prn, grp in s1.groupby("prn"):
            arr = grp.resp.abs().dropna().to_numpy(dtype=float)
            if len(arr) < 5:
                continue
            prn_metrics.append({
                "prn": prn,
                "data": arr,
                "mean": float(np.mean(arr)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(np.max(arr))
            })
        if not prn_metrics:
            ax.text(0.5, 0.5, "样本量不足", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("粗差与多路径污染卫星排行 (Top Outliers)")
            return

        prn_metrics.sort(key=lambda m: (m["p75"], m["mean"]))
        top_metrics = prn_metrics[-15:]

        top_prns = [m["prn"] for m in top_metrics]
        box_data = [m["data"] for m in top_metrics]
        mean_vals = [m["mean"] for m in top_metrics]

        bp = ax.boxplot(
            box_data, positions=range(len(top_prns)), vert=False, patch_artist=True,
            showfliers=False, widths=0.55,
            boxprops=dict(facecolor="#ffccd5", edgecolor="#e63946", linewidth=1.2),
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="#333333", linewidth=1.0),
            capprops=dict(color="#333333", linewidth=1.0)
        )

        ax.scatter(mean_vals, range(len(top_prns)), marker="D", color="#0077b6", s=28, zorder=6, label="Mean |Resp|")
        ax.axvline(5.0, color="#e63946", ls="--", lw=1.2, label="5m Error Line")

        ax.set_yticks(range(len(top_prns)))
        ax.set_yticklabels(top_prns, fontsize=8)
        ax.set_xlabel("|Pseudorange Residual| Distribution (m, outliers hidden for scale)")
        ax.set_ylabel("Satellite PRN")
        ax.set_title("粗差与多路径污染卫星排行 (Top Outliers)")

        max_disp = max(11.0, max(m["p95"] for m in top_metrics) * 1.25)
        ax.set_xlim(0, max_disp)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        ax.grid(self.grid_on.get(), alpha=0.3, axis="x", ls=":")

    def _draw_directional_dop(self, ax):
        if "_cached_directional_dop" not in self.data:
            self.data["_cached_directional_dop"] = diag.compute_directional_dop(self.data)
        dop_df = self.data["_cached_directional_dop"]
        if dop_df.empty:
            ax.text(0.5, 0.5, "无法计算方向 DOP (需 .stat 卫星仰角方位角)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("方向 DOP 几何精度因子")
            return
        hdop_col = dop_df.HDOP if "HDOP" in dop_df.columns else np.full(len(dop_df), np.nan)
        t_gap, edop_g, ndop_g, vdop_g, hdop_g = insert_time_gaps(
            dop_df.time, dop_df.EDOP, dop_df.NDOP, dop_df.VDOP, hdop_col
        )
        ax.plot(t_gap, edop_g, color="#2878B5", lw=1.1, label="EDOP (东向)")
        ax.plot(t_gap, ndop_g, color="#E07A2D", lw=1.1, label="NDOP (北向)")
        ax.plot(t_gap, vdop_g, color="#36965A", lw=1.1, label="VDOP (天向)")
        if "HDOP" in dop_df.columns:
            ax.plot(t_gap, hdop_g, color="#666666", lw=0.9, ls="--", label="HDOP (水平)")
        ax.set(xlabel="时间 (s)", ylabel="DOP 值", title="方向 DOP 几何精度因子 (EDOP / NDOP / VDOP)")
        lim_y = min(float(dop_df[["EDOP", "NDOP", "VDOP"]].quantile(0.99).max()) * 1.5, 30.0)
        ax.set_ylim(0, max(5.0, lim_y))
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_clock_bias_isb(self, ax):
        clk = self.data.get("clk")
        if clk is None or clk.empty:
            ax.text(0.5, 0.5, "无接收机钟差数据 (未在 .stat 中找到 $CLK)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("接收机钟差漂移与系统间偏差 (Clock Bias & ISB)")
            return
        t0 = self.data["pos"].tow.iloc[0]
        clk_t = clk.tow - t0
        has_isb = False
        if "isb_glo" in clk.columns and clk.isb_glo.abs().max() > 0:
            ax.plot(clk_t, clk.isb_glo, label="ISB GLONASS-GPS", color="#9b59b6", lw=1.0)
            has_isb = True
        if "isb_gal" in clk.columns and clk.isb_gal.abs().max() > 0:
            ax.plot(clk_t, clk.isb_gal, label="ISB Galileo-GPS", color="#36965A", lw=1.0)
            has_isb = True
        if "isb_bds" in clk.columns and clk.isb_bds.abs().max() > 0:
            ax.plot(clk_t, clk.isb_bds, label="ISB BDS-GPS", color="#d90429", lw=1.0)
            has_isb = True
        if "isb_qzs" in clk.columns and clk.isb_qzs.abs().max() > 0:
            ax.plot(clk_t, clk.isb_qzs, label="ISB QZSS-GPS", color="#E07A2D", lw=1.0)
            has_isb = True
        if "drift" in clk.columns and clk.drift.abs().max() > 0:
            ax.plot(clk_t, clk.drift, label="钟漂 (ns/s)", color="#2878B5", lw=1.0, ls="--")
        elif not has_isb and "clk_gps" in clk.columns:
            ax.plot(clk_t, clk.clk_gps, label="接收机钟差 (ns)", color="#2878B5", lw=1.0)
        ax.set(xlabel="时间 (s)", ylabel="偏差 / 漂移 (m 或 ns)", title="接收机钟差漂移与系统间偏差 ($CLK)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_body_frame_error(self, ax, p):
        if not self.data.get("has_truth"):
            ax.text(0.5, 0.5, "需要真值参考轨迹才能分解车体坐标系误差", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("车体坐标系误差分解 (Along-track / Cross-track / Vertical)")
            return
        has_body = ("along_err" in p.columns and "cross_err" in p.columns)
        if not has_body:
            ax.text(0.5, 0.5, "缺少航向角数据，无法投影至车体坐标系", ha="center", va="center", transform=ax.transAxes)
            return
        t_gap, along_gap, cross_gap, vert_gap = insert_time_gaps(p.time, p.along_err, p.cross_err, p.vert_err)
        ax.plot(t_gap, along_gap, label="沿航向误差 (Along-track)", color="#2878B5", lw=0.9)
        ax.plot(t_gap, cross_gap, label="跨航向误差 (Cross-track)", color="#d90429", lw=0.9)
        ax.plot(t_gap, vert_gap, label="垂直高程误差 (Vertical)", color="#36965A", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        if hasattr(self, "_apply_robust_ylim"):
            self._apply_robust_ylim(ax, [p.along_err, p.cross_err, p.vert_err], is_symmetric=True, min_span=2.0)
        ax.set(xlabel="时间 (s)", ylabel="误差 (m)", title="车体坐标系误差分解 (Along-track / Cross-track / Vertical)")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_ar_ratio(self, ax, p):
        t_gap, ratio_gap = insert_time_gaps(p.time, p.ratio)
        ax.plot(t_gap, ratio_gap, color="#1f77b4", lw=0.75, label="AR Ratio (模糊度检验值)")
        ax.axhline(3.0, color="#d90429", ls="--", lw=1.2, alpha=0.85, label="判定门限 (Threshold = 3.0)")
        if self.data.get("has_truth"):
            eh = p.eh if "eh" in p.columns else np.hypot(p.de, p.dn)
            e3d = p.e3d if "e3d" in p.columns else np.sqrt(p.de**2 + p.dn**2 + p.du**2)
            m_ff = (p.ratio >= 3.0) & ((eh > 1.0) | (e3d > 3.0))
            if m_ff.any():
                ff_p = p[m_ff]
                ax.scatter(ff_p.time, ff_p.ratio, marker="^", s=45, color="#d90429",
                           edgecolors="white", linewidths=0.9, zorder=6,
                           label=f"假固定 (Ratio≥3 且 水平>1m 或 3D>3m, n={len(ff_p)})")
        if hasattr(self, "_apply_robust_ylim"):
            self._apply_robust_ylim(ax, [p.ratio], non_negative=True, min_span=5.0, default_max=10.0)
        ax.set(ylabel="Ratio 值", title="模糊度解算 Ratio 时序与假固定检验 (AR Ratio)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_skyplot_cn0_3in1(self):
        """Draw 3-panel combined Earth-Frame Skyplot, Body-Frame Skyplot, and Elevation vs SNR scatter."""
        sat = self.data.get("sat")
        pos = self.data.get("pos")
        if sat is None or sat.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "无卫星观测与残差数据 (需 .stat 文件)", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            self.plot_axes = [ax]; self.ax = ax
            return

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 3, figure=self.fig, width_ratios=[1, 1, 1.25], wspace=0.28, left=0.05, right=0.96, top=0.90, bottom=0.12)

        s_valid = sat[(sat.freq == 1) & (sat.snr > 0) & (sat.el > 0)].copy()
        s_sub = s_valid.iloc[::max(1, len(s_valid) // 18000)].copy()
        c_vals = np.clip(s_sub.resp.fillna(0).abs(), 0, 15) if "resp" in s_sub.columns else np.zeros(len(s_sub))

        # 1A. Earth-Frame Skyplot
        ax1 = self.fig.add_subplot(gs[0], projection="polar")
        theta1 = np.deg2rad(s_sub.az.to_numpy())
        r1 = 90.0 - s_sub.el.to_numpy()
        ax1.scatter(theta1, r1, c=c_vals, cmap="turbo", s=14, alpha=0.88, edgecolors="none", vmin=0, vmax=15)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_ylim(0, 90)
        ax1.set_yticks([15, 30, 45, 60, 75])
        ax1.set_yticklabels(["75°", "60°", "45°", "30°", "15°"], fontsize=8, color="#555")
        ax1.set_rlabel_position(45)
        ax1.tick_params(labelsize=8.5)
        ax1.set_title("地固系星空图 (Earth Skyplot)", fontsize=11, weight="bold", pad=10)

        # 1B. Body-Frame Skyplot (0° = Vehicle Heading)
        ax2 = self.fig.add_subplot(gs[1], projection="polar")
        if pos is not None and not pos.empty and "heading" in pos.columns:
            hd_interp = np.interp(s_sub.tow, pos.tow, pos.heading)
            az_body = (s_sub.az - hd_interp) % 360.0
        else:
            az_body = s_sub.az
        theta2 = np.deg2rad(az_body)
        ax2.scatter(theta2, r1, c=c_vals, cmap="turbo", s=14, alpha=0.88, edgecolors="none", vmin=0, vmax=15)
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_ylim(0, 90)
        ax2.set_yticks([15, 30, 45, 60, 75])
        ax2.set_yticklabels(["75°", "60°", "45°", "30°", "15°"], fontsize=8, color="#555")
        ax2.set_rlabel_position(45)
        ax2.set_thetagrids(np.arange(0, 360, 45), labels=["Front\n(0°)", "45°", "Right\n(90°)", "135°", "Rear\n(180°)", "225°", "Left\n(270°)", "315°"], fontsize=8.5)
        ax2.set_title("车体系星空图 (Body-Frame Skyplot)", fontsize=11, weight="bold", pad=10)

        # 1C. Elevation vs SNR Joint Scatter
        ax3 = self.fig.add_subplot(gs[2])
        sc = ax3.scatter(s_sub.el, s_sub.snr, c=c_vals, cmap="turbo", s=14, alpha=0.88, edgecolors="none", vmin=0, vmax=15)
        ax3.axvline(20, color="k", ls="--", lw=1.2, label="Elev = 20°")
        ax3.axhline(35, color="r", ls="--", lw=1.2, label="SNR Mask = 35 dB-Hz")
        ax3.set_xlim(10, 90)
        ax3.set_ylim(20, 55)
        ax3.set_xlabel("Elevation Angle (deg)", fontsize=10, weight="bold")
        ax3.set_ylabel("SNR / C/N0 (dB-Hz)", fontsize=10, weight="bold")
        ax3.set_title("高度角与载噪比分布 (Elevation vs SNR)", fontsize=11, weight="bold")
        ax3.legend(loc="lower right", fontsize=8.5)
        ax3.grid(self.grid_on.get(), ls="--", alpha=0.35)

        cb = self.fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
        cb.set_label("|Pseudorange Residual| (m, clipped at 15m)", fontsize=9, weight="bold")

        # Setup dynamic satellite artists for 1A, 1B, 1C
        sc_sky1 = ax1.scatter([], [], s=85, facecolors="none", edgecolors=[], linewidths=2.0, zorder=15)
        labels_sky1 = [ax1.text(0, 0, "", fontsize=7.5, weight="bold", color="#111",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#dc2626", alpha=0.9, lw=1.1), zorder=16)
                       for _ in range(35)]

        sc_sky2 = ax2.scatter([], [], s=85, facecolors="none", edgecolors=[], linewidths=2.0, zorder=15)
        labels_sky2 = [ax2.text(0, 0, "", fontsize=7.5, weight="bold", color="#111",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#2563eb", alpha=0.9, lw=1.1), zorder=16)
                       for _ in range(35)]

        sc_snr = ax3.scatter([], [], s=95, facecolors="none", edgecolors=[], linewidths=2.0, zorder=15)
        labels_snr = [ax3.text(0, 0, "", fontsize=7.5, weight="bold", color="#111",
                               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#dc2626", alpha=0.9, lw=1.1), zorder=16)
                      for _ in range(35)]

        dyn_artists_3in1 = [sc_sky1] + labels_sky1 + [sc_sky2] + labels_sky2 + [sc_snr] + labels_snr
        self.custom_dynamic_artists.extend(dyn_artists_3in1)

        def update_3in1(i, r, current_sat):
            cur = current_sat[(current_sat.freq == 1) & (current_sat.snr > 0) & (current_sat.el > 0)].drop_duplicates("prn")
            if not cur.empty:
                import matplotlib.pyplot as _plt
                turbo = _plt.cm.turbo
                az = cur.az.to_numpy()
                el = cur.el.to_numpy()
                snr = cur.snr.to_numpy()
                prn = cur.prn.to_numpy()
                th1 = np.deg2rad(az)
                r1_curr = 90.0 - el
                sc_sky1.set_offsets(np.c_[th1, r1_curr])

                hd = float(getattr(r, "heading", 0.0)) if ("heading" in r and np.isfinite(r.heading)) else 0.0
                az_body = (az - hd) % 360.0
                th2 = np.deg2rad(az_body)
                sc_sky2.set_offsets(np.c_[th2, r1_curr])

                sc_snr.set_offsets(np.c_[el, snr])

                sat_edgecolors = []
                sat_facecolors = []
                for k, (_, sat_row) in enumerate(cur.iterrows()):
                    resp_val = float(sat_row.resp) if ("resp" in sat_row and pd.notna(sat_row.resp)) else 0.0
                    abs_r = abs(resp_val)
                    col = turbo(np.clip(abs_r / 15.0, 0.0, 1.0))
                    sat_edgecolors.append(col)
                    sat_facecolors.append([col[0], col[1], col[2], 0.22])

                    if k < 35:
                        labels_sky1[k].set_position((th1[k], r1_curr[k]))
                        labels_sky1[k].set_text(prn[k])
                        labels_sky1[k].get_bbox_patch().set_edgecolor(col)
                        labels_sky1[k].set_visible(True)

                        labels_sky2[k].set_position((th2[k], r1_curr[k]))
                        labels_sky2[k].set_text(prn[k])
                        labels_sky2[k].get_bbox_patch().set_edgecolor(col)
                        labels_sky2[k].set_visible(True)

                        labels_snr[k].set_position((el[k] + 0.8, snr[k] + 0.5))
                        labels_snr[k].set_text(prn[k])
                        labels_snr[k].get_bbox_patch().set_edgecolor(col)
                        labels_snr[k].set_visible(True)

                for sc_item in [sc_sky1, sc_sky2, sc_snr]:
                    sc_item.set_edgecolors(sat_edgecolors)
                    sc_item.set_facecolors(sat_facecolors)

                for k in range(len(cur), 35):
                    labels_sky1[k].set_visible(False)
                    labels_sky2[k].set_visible(False)
                    labels_snr[k].set_visible(False)
            else:
                sc_sky1.set_offsets(np.empty((0, 2)))
                sc_sky2.set_offsets(np.empty((0, 2)))
                sc_snr.set_offsets(np.empty((0, 2)))
                for k in range(35):
                    labels_sky1[k].set_visible(False)
                    labels_sky2[k].set_visible(False)
                    labels_snr[k].set_visible(False)

        self.custom_updaters.append(update_3in1)

        self.plot_axes = [ax1, ax2, ax3]
        self.ax = ax1

    def _draw_env_geometry_combined(self):
        """Draw 5-in-1 integrated Module 1 (Environment) & Module 3 (Geometry) figure."""
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        sat = self.data.get("sat")
        if sat is None or sat.empty:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "无卫星观测与残差数据 (需 .stat 文件)", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            self.plot_axes = [ax]; self.ax = ax
            return

        gs_main = GridSpec(1, 2, figure=self.fig, width_ratios=[1.1, 1.4], wspace=0.22)
        gs_left = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], hspace=0.45)
        gs_right = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1], hspace=0.32)

        # 1. Left-Top: Polar Multipath Skyplot
        ax_sky = self.fig.add_subplot(gs_left[0], projection="polar")
        s_valid = sat[(sat.freq == 1) & (sat.resp != 0)]
        if not s_valid.empty:
            s_sub = s_valid.iloc[::max(1, len(s_valid) // 12000)]
            theta = np.deg2rad(s_sub.az)
            r = 90.0 - s_sub.el
            sc1 = ax_sky.scatter(theta, r, c=np.clip(s_sub.resp.abs(), 0, 15.0), cmap="turbo", vmin=0.0, vmax=15.0, s=9, alpha=0.65, edgecolors="none")
            cb1 = self.fig.colorbar(sc1, ax=ax_sky, orientation="vertical", fraction=0.045, pad=0.08)
            cb1.set_label("伪距残差 |resp| (m)", fontsize=9, weight="bold")
            cb1.ax.tick_params(labelsize=8)
        ax_sky.set_theta_zero_location("N")
        ax_sky.set_theta_direction(-1)
        ax_sky.set_ylim(0, 90)
        ax_sky.set_yticks([30, 60, 90])
        ax_sky.set_yticklabels(['60°', '30°', '0°'], fontsize=8.5, color='#555')
        ax_sky.set_rlabel_position(135)
        ax_sky.tick_params(labelsize=9)
        ax_sky.set_title('(a) 空间多路径与建筑物遮挡星空图', fontsize=11, weight='bold', pad=10)

        # Dynamic satellite artists for active epoch
        sky_curr = ax_sky.scatter([], [], s=100, marker="o", facecolors="none", edgecolors=[], linewidths=2.2, zorder=12)
        sky_labels = [ax_sky.text(0, 0, "", fontsize=7.5, weight="bold", color="#111",
                                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#d90429", lw=1.2, alpha=0.92), zorder=14) for _ in range(35)]
        self.sky_plots.append((ax_sky, sky_curr, sky_labels))

        # 2. Left-Bottom: El vs SNR & Resp
        ax_el = self.fig.add_subplot(gs_left[1])
        s_snr = sat[(sat.freq == 1) & (sat.snr > 0) & (sat.el > 0)]
        if not s_snr.empty:
            s_sub2 = s_snr.iloc[::max(1, len(s_snr) // 15000)]
            resp_clip = np.clip(s_sub2.resp.fillna(0).abs(), 0, 15.0) if "resp" in s_sub2.columns else np.zeros(len(s_sub2))
            sc2 = ax_el.scatter(s_sub2.el, s_sub2.snr, c=resp_clip, cmap="turbo", s=14, vmin=0.0, vmax=15.0, alpha=0.88, edgecolors="none")
            el_grid = np.linspace(5, 90, 100)
            ax_el.plot(el_grid, 28.0 + 22.0 * np.sin(np.deg2rad(el_grid)), 'k--', lw=1.5, label='标称信噪比包络')
            ax_el.legend(loc='lower right', fontsize=8.5)
        ax_el.set_xlim(0, 90)
        ax_el.set_xticks(np.arange(0, 91, 15))
        ax_el.set_xticklabels([f'{x}°' for x in range(0, 91, 15)], fontsize=9)
        ax_el.tick_params(labelsize=9)
        ax_el.set_xlabel('卫星高度角 (°)', fontsize=10, weight='bold')
        ax_el.set_ylabel('载噪比 (dB-Hz)', fontsize=10, weight='bold')
        ax_el.set_title('(b) 卫星高度角 - 载噪比 - 残差联合分布', fontsize=11, weight='bold', pad=6)
        ax_el.grid(self.grid_on.get(), ls='--', alpha=0.35)

        # 3. Right-Top: Satellites Stack
        ax_sat = self.fig.add_subplot(gs_right[0])
        s1_valid = sat[(sat.freq == 1) & (sat.vsat == 1)]
        if not s1_valid.empty:
            con_counts = s1_valid.groupby(['time', 'constellation']).prn.nunique().unstack(fill_value=0)
            t_con = con_counts.index.to_numpy()
            c_names = [c for c in ['GPS', 'GLONASS', 'Galileo', 'BeiDou', 'QZSS'] if c in con_counts.columns]
            y_stack = [con_counts[c].to_numpy() for c in c_names]
            stack_colors = [diag.COLORS.get(c, '#2878B5') for c in c_names]
            ax_sat.stackplot(t_con, *y_stack, labels=c_names, colors=stack_colors, alpha=0.85)
            ax_sat.legend(loc='upper right', fontsize=8, ncol=len(c_names))
        ax_sat.set_ylabel('可用星数 (颗)', fontsize=10, weight='bold')
        ax_sat.set_title('(c) 多系统跟踪与参与解算有效卫星数时序', fontsize=11, weight='bold')
        ax_sat.grid(self.grid_on.get(), ls='--', alpha=0.35)
        ax_sat.tick_params(labelsize=9, labelbottom=False)

        # 4. Right-Middle: Directional DOP
        ax_dop = self.fig.add_subplot(gs_right[1], sharex=ax_sat)
        dop_df = diag.compute_directional_dop(self.data)
        if not dop_df.empty:
            t_gap, edop_g, ndop_g, vdop_g, pdop_g = insert_time_gaps(
                dop_df.time, dop_df.EDOP, dop_df.NDOP, dop_df.VDOP, dop_df.PDOP
            )
            ax_dop.plot(t_gap, edop_g, color='#2878B5', lw=1.1, label='EDOP (东向跨街)')
            ax_dop.plot(t_gap, ndop_g, color='#E07A2D', lw=1.1, label='NDOP (北向沿街)')
            ax_dop.plot(t_gap, vdop_g, color='#36965A', lw=1.1, label='VDOP (天向高程)')
            ax_dop.plot(t_gap, pdop_g, color='#111111', lw=1.2, ls=':', label='PDOP (空间几何)')
            ax_dop.axhline(6.0, color='gray', ls='--', lw=0.9, alpha=0.7, label='良好门限 6.0')
            ax_dop.set_ylim(0, min(float(dop_df[['EDOP', 'NDOP', 'VDOP']].quantile(0.99).max()) * 1.5, 25.0))
            ax_dop.legend(loc='upper right', fontsize=8, ncol=5)
        ax_dop.set_ylabel('DOP 精度因子', fontsize=10, weight='bold')
        ax_dop.set_title('(d) 方向 DOP 空间几何对称性衰减时序', fontsize=11, weight='bold')
        ax_dop.grid(self.grid_on.get(), ls='--', alpha=0.35)
        ax_dop.tick_params(labelsize=9, labelbottom=False)

        # 5. Right-Bottom: Residual Triad
        ax_tri = self.fig.add_subplot(gs_right[2], sharex=ax_sat)
        triad = diag.compute_residual_triad(self.data)
        if not triad.empty:
            t_gap, max_g, sec_g, rms_g = insert_time_gaps(
                triad.time, triad.max_resp, triad.sec_resp, triad.rms_resp
            )
            ax_tri.plot(t_gap, max_g, color='#d90429', lw=1.1, label='第一大残差 (Max |resp|)')
            ax_tri.plot(t_gap, sec_g, color='#00b4d8', lw=0.95, label='第二大残差 (2nd Max)')
            ax_tri.plot(t_gap, rms_g, color='#0B2B52', lw=1.2, ls='--', label='历元均方根 (RMS)')
            ax_tri.axhline(3.0, color='gray', ls=':', lw=0.9, label='粗差门限 3.0 m')
            ax_tri.legend(loc='upper right', fontsize=8, ncol=4)
        ax_tri.set_ylabel('伪距残差 (m)', fontsize=10, weight='bold')
        ax_tri.set_xlabel('观测时间 (s)', fontsize=10, weight='bold')
        ax_tri.set_title('(e) 伪距残差三线联合响应诊断 (Max vs 2nd Max vs RMS)', fontsize=11, weight='bold')
        ax_tri.grid(self.grid_on.get(), ls='--', alpha=0.35)
        ax_tri.tick_params(labelsize=9)

        self.fig.align_ylabels([ax_sat, ax_dop, ax_tri])
        self.plot_axes = [ax_sky, ax_el, ax_sat, ax_dop, ax_tri]
        self.ax = ax_sky

        for p_ax in (ax_sat, ax_dop, ax_tri):
            self.time_cursors.append(p_ax.axvline(0, color="#333", ls="--", lw=0.8, alpha=0.7, zorder=10))
            self._set_datetime_axis(p_ax)
        for p_ax in (ax_sat, ax_dop):
            p_ax.set_xlabel("")
            p_ax.tick_params(labelbottom=False)

    def _draw_custom_grid(self):
        modes=self.custom_modes
        if not modes or not self.custom_panels:
            ax=self.fig.add_subplot(111);ax.text(.5,.5,"Select plot types from the left list.",ha="center",va="center",transform=ax.transAxes);return
        self.fig.clear();self.plot_axes=[];self.custom_sky=[];p=self._filtered();full=self.data["pos"];s=self.data["sat"]
        self.custom_modes=[panel["mode"] for panel in self.custom_panels]
        pages=max(1,(len(self.custom_panels)+self.custom_page_size-1)//self.custom_page_size)
        self.custom_page=max(0,min(self.custom_page,pages-1))
        if hasattr(self,"custom_page_label"):self.custom_page_label.configure(text=f"Page {self.custom_page+1} / {pages}")
        page_start=self.custom_page*self.custom_page_size
        self.visible_panel_indices=([self.focused_custom_panel] if self.focused_custom_panel is not None
                                    and 0<=self.focused_custom_panel<len(self.custom_panels)
                                    else list(range(page_start,min(page_start+self.custom_page_size,len(self.custom_panels)))))
        base_time_ax = None
        group_base_ax = {}
        for i in self.visible_panel_indices:
            panel=self.custom_panels[i]
            name=panel["mode"]
            rect=(.065,.09,.92,.82) if self.focused_custom_panel is not None else panel["rect"]
            is_time_series = name not in NON_TIME_SERIES_MODES
            share = base_time_ax if (self.sync_time_axes.get() and is_time_series and self.focused_custom_panel is None) else None
            unit_grp = PLOT_UNIT_GROUPS.get(name)
            sharey = group_base_ax.get(unit_grp) if (self.sync_y_axes.get() and unit_grp and self.focused_custom_panel is None) else None
            ax=self.fig.add_axes(rect,projection="polar" if name in ("Skyplot", "Multipath Skyplot", "Skyplot & C/N0 (Env)") else None, sharex=share, sharey=sharey)
            ax._panel_mode = name
            if self.sync_time_axes.get() and is_time_series and base_time_ax is None and self.focused_custom_panel is None:
                base_time_ax = ax
            if self.sync_y_axes.get() and unit_grp and unit_grp not in group_base_ax and self.focused_custom_panel is None:
                group_base_ax[unit_grp] = ax
            self.plot_axes.append(ax)
            for spine in ax.spines.values():spine.set_edgecolor("#168aad" if i==self.selected_custom_panel else "#b9c0c8");spine.set_linewidth(1.6 if i==self.selected_custom_panel else .75)
            if not name:
                # Keep the drop hint inside its own axes. A long title overlaps
                # neighboring panels after a narrow split.
                ax.set_xticks([]);ax.set_yticks([])
                ax.text(.5,.5,"Drop plot here",ha="center",va="center",transform=ax.transAxes,fontsize=8,color="#777")
                ax.patch.set_facecolor("#fafafa");ax.patch.set_edgecolor("#cccccc")
                continue
            if name=="Gnd Trk":
                self.track_axes.append(ax)
                if self.data["has_truth"]:
                    gt_t, gt_e, gt_n = insert_time_gaps(full.time, full.gt_E, full.gt_N)
                    ax.plot(gt_e, gt_n, "k--", lw=.7, label="Truth")
                for q,g in p.groupby("q"):ax.scatter(g.E,g.N,s=2,color=Q_COLORS.get(int(q),"#777"),label=Q_LABELS.get(int(q),str(q)))
                ax.set(xlabel="E (m)",ylabel="N (m)",title=name);ax.legend(fontsize=6,ncol=2);ax.set_aspect("equal" if self.equal_scale.get() else "auto")
            elif name=="ENU Position":
                t_gap, e_g, n_g, u_g = insert_time_gaps(p.time, p.E, p.N, p.U)
                for y_arr, col, lbl in zip([e_g, n_g, u_g], ["#2878B5","#E07A2D","#36965A"], ["E", "N", "U"]):
                    ax.plot(t_gap, y_arr, lw=.65, marker=".", ms=2, markevery=max(1, len(p)//50), label=lbl, color=col)
                ax.set(title=name, ylabel="Position (m)"); ax.legend(fontsize=6, ncol=3)
            elif name=="Height":
                t_gap, u_g = insert_time_gaps(p.time, p.U)
                ax.plot(t_gap, u_g, lw=.7, marker=".", ms=2, markevery=max(1, len(p)//50), label="RTK U")
                if self.data["has_truth"]:
                    t_gap_gt, gt_u_g = insert_time_gaps(p.time, p.gt_U)
                    ax.plot(t_gap_gt, gt_u_g, lw=.7, ls="--", label="Truth U")
                ax.set(title=name, ylabel="Up (m)"); ax.legend(fontsize=6)
            elif name=="ENU Error":
                if self.data["has_truth"]:
                    t_gap, de_g, dn_g, du_g = insert_time_gaps(p.time, p.de, p.dn, p.du)
                    for y_arr, col, lbl in zip([de_g, dn_g, du_g], ["#2878B5","#E07A2D","#36965A"], ["DE", "DN", "DU"]):
                        ax.plot(t_gap, y_arr, lw=.65, label=lbl, color=col)
                    ax.axhline(0, color="k", lw=.5); ax.legend(fontsize=6, ncol=3)
                    self._apply_robust_ylim(ax, [p.de, p.dn, p.du], is_symmetric=True, min_span=2.0)
                else:ax.text(.5,.5,"No truth",ha="center",transform=ax.transAxes)
                ax.set(title=name,ylabel="Error (m)")
            elif name=="Solution Status":
                t_gap, q_g = insert_time_gaps(p.time, p.q)
                ax.step(t_gap, q_g, where="post", lw=.65, marker=".", ms=2, markevery=max(1, len(p)//50))
                ax.set(title=name, ylabel="Q")
            elif name=="AR Ratio":self._draw_ar_ratio(ax,p)
            elif name=="Differential Age":self._draw_differential_age(ax,p)
            elif name=="Satellites":self._draw_sats(ax)
            elif name=="DOP":self._draw_dop(ax)
            elif name in ("Skyplot & C/N0 (Env)", "Skyplot"):
                if getattr(ax, "name", "") == "polar":
                    self._draw_sky(ax)
                else:
                    ax.text(0.5, 0.5, "星空图需要极坐标投影\n(双击放大查看)", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title("Skyplot & C/N0")
            elif name == "Multipath Skyplot":
                if getattr(ax, "name", "") == "polar":
                    self._draw_multipath_skyplot(ax)
                else:
                    ax.text(0.5, 0.5, "多路径星空图需要极坐标投影\n(双击放大查看)", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title("Multipath Skyplot")
            elif name in ("C/N0", "C/N₀"):
                self._draw_cn0(ax)
                ax.set_title("载噪比与信号强度 (C/N0)")
            elif name in ("Satellite Elevation", "Elevation"):
                self._draw_elevation(ax)
            elif name in ("Clock Bias & ISB", "Clock Bias"):
                self._draw_clock_bias_isb(ax)
            elif name=="Residuals":self._draw_residuals(ax)
            elif name=="Multipath Proxy":self._draw_multipath(ax)
            elif name=="Cycle Slips":self._draw_slips(ax)
            elif name=="Satellite Visibility Gantt":self._draw_satellite_visibility_gantt(ax,s)
            elif name=="Per-PRN Residuals":self._draw_per_prn_residuals(ax)
            elif name=="Covariance":self._draw_covariance(ax,p)
            elif name=="False Fix Analysis":self._draw_false_fix(ax,p)
            elif name=="Innovation / NIS":self._draw_nis(ax)
            elif name=="Error CDF":self._draw_error_cdf(ax,p)
            elif name=="Error Histogram":self._draw_error_histogram([ax],p)
            elif name=="Residual Triad":self._draw_residual_triad(ax)
            elif name=="Env & Geometry Combined":self._draw_residual_triad(ax)
            elif name=="El vs SNR & Resp":self._draw_el_snr_resp(ax)
            elif name=="Top Outlier Satellites":self._draw_top_outliers(ax)
            elif name=="Directional DOP":self._draw_directional_dop(ax)
            elif name=="Body-Frame Error":self._draw_body_frame_error(ax,p)
            elif name=="Trajectory Error Heatmap":self._draw_trajectory_error_heatmap(ax,p,full)
            ax.grid(self.grid_on.get(),alpha=.28)
            if is_time_series:
                self._set_datetime_axis(ax);ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis="x",labelsize=7,labelrotation=0)
                self.time_cursors.append(ax.axvline(0,color="#333",ls="--",lw=.7,alpha=.65))
            self._register_custom_dynamic(name,ax,p,full)
        for ax in self.plot_axes:
            box = ax.get_position()
            p_mode = getattr(ax, "_panel_mode", "")
            is_ts = p_mode not in NON_TIME_SERIES_MODES
            has_panel_below = any(
                other is not ax
                and other.get_position().y1 < box.y0 - .005
                and min(box.x1, other.get_position().x1) - max(box.x0, other.get_position().x0) > .2 * min(box.width, other.get_position().width)
                for other in self.plot_axes
            )
            has_ts_panel_below = any(
                other is not ax and getattr(other, "_panel_mode", "") not in NON_TIME_SERIES_MODES
                and other.get_position().y1 < box.y0 - .005
                and min(box.x1, other.get_position().x1) - max(box.x0, other.get_position().x0) > .2 * min(box.width, other.get_position().width)
                for other in self.plot_axes
            )
            if has_panel_below:
                # Unconditionally clear the bulky date-time xlabel whenever another subplot is underneath
                ax.set_xlabel("")
                if is_ts:
                    if has_ts_panel_below and self.sync_time_axes.get():
                        ax.tick_params(axis="x", labelbottom=False)
                    elif self.sync_time_axes.get():
                        has_bottom_ts = any(
                            getattr(o, "_panel_mode", "") not in NON_TIME_SERIES_MODES
                            and not any(b is not o and b.get_position().y1 < o.get_position().y0 - .005 for b in self.plot_axes)
                            for o in self.plot_axes
                        )
                        if has_bottom_ts:
                            ax.tick_params(axis="x", labelbottom=False)
                        else:
                            ax.tick_params(axis="x", labelsize=6.5, pad=1)
                    else:
                        ax.tick_params(axis="x", labelsize=6.5, pad=1)
        self.fig.suptitle("Custom RTKLIB Analysis",fontsize=13,weight="bold",y=0.985)

    def _register_custom_dynamic(self,name,ax,p,full):
        """Add a current-epoch highlight to every plot in a custom grid or single plot view."""
        def point(color="#111",marker="o",size=7,target=None):
            target=target if target is not None else ax
            return target.plot([],[],linestyle="None",marker=marker,ms=size,mfc=color,mec="white",mew=.7,zorder=12)[0]
        artists=[]
        if name in ("Gnd Trk", "Trajectory Error Heatmap"):
            trail,=ax.plot([],[],color="#ed3030",lw=1.4,zorder=8);artists=[trail,point("#ed3030","o",9)]
            if self.data["has_truth"]:artists.extend([point("#ffd92f","*",11),ax.plot([],[],color="#d22",ls=":",lw=1,zorder=9)[0]])
        elif name=="ENU Position":artists=[point(c) for c in ["#2878B5","#E07A2D","#36965A"]]
        elif name=="Height":
            artists=[point("#16853b")]
            if self.data["has_truth"]:artists.append(point("#252525"))
        elif name=="ENU Error" and self.data["has_truth"]:artists=[point(c) for c in ["#2878B5","#E07A2D","#36965A"]]
        elif name=="Solution Status":artists=[point("#333","o",8)]
        elif name=="AR Ratio":
            p_ratio = point("#1f77b4", "o", 8)
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", alpha=0.9, lw=0.8), zorder=15)
            artists = [p_ratio, badge]
        elif name=="Differential Age":artists=[point("#dc2626","o",8)]
        elif name=="Satellites":
            p_trk = point("#2878B5", "o", 8)
            p_usd = point("#E07A2D", "o", 8)
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2878B5", alpha=0.9, lw=0.8), zorder=15)
            artists = [p_trk, p_usd, badge]
        elif name=="DOP":
            pts = [point(c, "o", 7) for c in ["#2878B5","#E07A2D","#36965A"]]
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", alpha=0.9, lw=0.8), zorder=15)
            artists = pts + [badge]
        elif name=="Directional DOP":
            pts = [point(c, "o", 7) for c in ["#2878B5", "#E07A2D", "#36965A"]]
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2878B5", alpha=0.9, lw=0.8), zorder=15)
            artists = pts + [badge]
        elif name=="Residual Triad":
            p_max = point("#d90429", "o", 8)
            p_2nd = point("#0077b6", "o", 7)
            p_rms = point("#0B2B52", "o", 7)
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d90429", alpha=0.9, lw=0.8), zorder=15)
            artists = [p_max, p_2nd, p_rms, badge]
        elif name=="Body-Frame Error":
            pts = [point(c, "o", 7) for c in ["#2878B5", "#E07A2D", "#36965A"]]
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d90429", alpha=0.9, lw=0.8), zorder=15)
            artists = pts + [badge]
        elif name=="C/N0":artists=[ax.scatter([],[],s=24,edgecolors="#222",linewidths=.5,zorder=11) for _ in range(2)]
        elif name=="Elevation":artists=[ax.scatter([],[],s=18,marker="o",facecolors="none",edgecolors="#222",zorder=11)]
        elif name=="Residuals":artists=[ax.scatter([],[],s=24,marker="o",facecolors="none",edgecolors=c,zorder=11) for c in ["#2878B5","#E07A2D"]]
        elif name=="Multipath Proxy":artists=[ax.scatter([],[],s=24,marker="o",facecolors="none",edgecolors="#222",zorder=11)]
        elif name=="Cycle Slips":artists=[ax.scatter([],[],s=32,marker="x",color="#d00000",zorder=11)]
        elif name=="Satellite Visibility Gantt":
            v_line = ax.axvline(np.nan, color="#e63946", ls="--", lw=1.6, alpha=0.95, zorder=12)
            sc_active = ax.scatter([], [], s=65, marker="o", facecolors=[], edgecolors="white", linewidths=1.3, zorder=14)
            badge = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d90429", alpha=0.9, lw=0.8), zorder=15)
            artists = [v_line, sc_active, badge]
        elif name=="Multipath Skyplot":
            artists=[ax.scatter([],[],s=60,marker="*",edgecolors="#fff",linewidths=0.8,color="#dc2626",zorder=12)]
        elif name=="Per-PRN Residuals":
            artists=[point("#dc2626","o",7)]
        elif name=="Covariance":
            p_cov = point("#1f77b4" if self.data["has_truth"] else "#2878B5","o",8)
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", alpha=0.9, lw=0.8), zorder=15)
            artists = [p_cov, badge]
        elif name=="False Fix Analysis":
            p_ff = point("#ffd92f","*",14)
            badge = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=8, weight="bold", color="#111",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#333", alpha=0.9, lw=0.8), zorder=15)
            artists = [p_ff, badge]
        elif name=="El vs SNR & Resp":
            sc_curr = ax.scatter([], [], s=110, facecolors="none", edgecolors=[], linewidths=2.2, zorder=15)
            texts = [ax.text(0, 0, "", fontsize=8, weight="bold", color="#111",
                             bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#dc2626", alpha=0.88, lw=1.2),
                             zorder=16) for _ in range(35)]
            artists = [sc_curr] + texts
        elif name=="Innovation / NIS":
            artists=[ax.text(.5,.12,"",transform=ax.transAxes,ha="center",va="center",fontsize=9,color="#555")]
        if not artists:return
        self.custom_dynamic_artists.extend(artists)

        def set_point(artist,x,y,show=True):
            if hasattr(artist,"set_offsets"):
                artist.set_offsets(np.array([[x,y]]) if show else np.empty((0,2)))
            else:artist.set_data([x],[y]) if show else artist.set_data([],[])
        sat_frame=self._sat_sorted; sat_tows=self._sat_tows
        track_e=full.E.to_numpy();track_n=full.N.to_numpy()
        dop=self.dop_frames.get(ax)
        sat_epoch=diag._sat_epoch(self.data) if name=="Satellites" else None

        def update(i,r,current_sat):
            visible=self.qfilter.get()=="All Q" or int(r.q)==int(self.qfilter.get().split("=")[1].split(":")[0])
            x=float(r.time);y=None
            dt=gpst_to_local_datetime(r.week,r.tow)
            if name in ("Gnd Trk", "Trajectory Error Heatmap"):
                start=max(0,i-120);stride=max(1,(i-start+1)//300)
                sub_t = full.time.iloc[start:i+1:stride]
                sub_e = track_e[start:i+1:stride]
                sub_n = track_n[start:i+1:stride]
                _t_gap, e_gap, n_gap = insert_time_gaps(sub_t, sub_e, sub_n)
                artists[0].set_data(e_gap, n_gap)
                set_point(artists[1],r.E,r.N,visible)
                if self.data["has_truth"]:
                    set_point(artists[2],r.gt_E,r.gt_N,visible and np.isfinite(r.gt_E) and np.isfinite(r.gt_N))
                    artists[3].set_data([r.E,r.gt_E],[r.N,r.gt_N]) if visible and np.isfinite(r.gt_E) else artists[3].set_data([],[])
            elif name=="ENU Position":
                for art,col in zip(artists,["E","N","U"]):set_point(art,x,r[col],visible)
            elif name=="Height":
                set_point(artists[0],x,r.U,visible)
                if len(artists)>1:set_point(artists[1],x,r.gt_U,visible and np.isfinite(r.gt_U))
            elif name=="ENU Error" and self.data["has_truth"]:
                for art,col in zip(artists,["de","dn","du"]):set_point(art,x,r[col],visible and np.isfinite(r[col]))
            elif name=="Solution Status":set_point(artists[0],x,r.q,visible)
            elif name=="AR Ratio":
                rat_val = float(getattr(r, "ratio", np.nan))
                if visible and np.isfinite(rat_val):
                    set_point(artists[0], x, rat_val, True)
                    eh_val = float(getattr(r, "eh", np.nan))
                    e3d_val = float(getattr(r, "e3d", np.nan))
                    is_false_fix = (rat_val >= 3.0) and (
                        (np.isfinite(e3d_val) and e3d_val > 3.0) or (np.isfinite(eh_val) and eh_val > 1.0)
                    )
                    if is_false_fix:
                        artists[0].set_color("#d90429")
                        artists[0].set_marker("^")
                        artists[0].set_markersize(11)
                        eh_str = f"水平={eh_val:.2f}m" if np.isfinite(eh_val) else ""
                        e3_str = f"3D={e3d_val:.2f}m" if np.isfinite(e3d_val) else ""
                        err_str = f" ({eh_str}, {e3_str})" if (eh_str or e3_str) else ""
                        artists[1].set_text(f"Ratio: {rat_val:.1f} · ⚠️ 假固定!{err_str}")
                        artists[1].get_bbox_patch().set_edgecolor("#d90429")
                    elif rat_val >= 3.0:
                        artists[0].set_color("#16a34a")
                        artists[0].set_marker("o")
                        artists[0].set_markersize(8)
                        artists[1].set_text(f"Ratio: {rat_val:.1f} · ✅ 正常固定")
                        artists[1].get_bbox_patch().set_edgecolor("#16a34a")
                    else:
                        artists[0].set_color("#1f77b4")
                        artists[0].set_marker("o")
                        artists[0].set_markersize(8)
                        artists[1].set_text(f"Ratio: {rat_val:.1f} · 浮点解 (Float)")
                        artists[1].get_bbox_patch().set_edgecolor("#1f77b4")
                    artists[1].set_visible(True)
                else:
                    set_point(artists[0], 0, 0, False)
                    artists[1].set_visible(False)
                age_val=float(getattr(r,"age",np.nan))
                set_point(artists[0],x,age_val,visible and np.isfinite(age_val))
            elif name=="Satellites" and sat_epoch is not None and not sat_epoch.empty:
                j=int(np.clip(np.searchsorted(sat_epoch.time.to_numpy(),x),0,len(sat_epoch)-1));row=sat_epoch.iloc[j]
                set_point(artists[0],x,row.tracked,visible);set_point(artists[1],x,row.used,visible)
                artists[2].set_text(f"跟踪: {int(row.tracked)}颗 · 参与解算: {int(row.used)}颗")
                artists[2].set_visible(visible)
            elif name=="DOP" and dop is not None and not dop.empty:
                j=int(np.clip(np.searchsorted(dop.time.to_numpy(),x),0,len(dop)-1));row=dop.iloc[j]
                for art,col in zip(artists[:3],["PDOP","HDOP","VDOP"]):set_point(art,x,row[col],visible and np.isfinite(row[col]))
                artists[3].set_text(f"PDOP: {row.PDOP:.2f} · HDOP: {row.HDOP:.2f} · VDOP: {row.VDOP:.2f}")
                artists[3].set_visible(visible)
            elif name=="Directional DOP":
                dop_df = self.data.get("_cached_directional_dop")
                if visible and dop_df is not None and not dop_df.empty:
                    j = int(np.clip(np.searchsorted(dop_df.time.to_numpy(), x), 0, len(dop_df) - 1))
                    row = dop_df.iloc[j]
                    set_point(artists[0], x, row.EDOP, True)
                    set_point(artists[1], x, row.NDOP, True)
                    set_point(artists[2], x, row.VDOP, True)
                    artists[3].set_text(f"EDOP: {row.EDOP:.2f} · NDOP: {row.NDOP:.2f} · VDOP: {row.VDOP:.2f}")
                    artists[3].set_visible(True)
                else:
                    for art in artists[:3]: set_point(art, 0, 0, False)
                    artists[3].set_visible(False)
            elif name=="Residual Triad":
                triad = self.data.get("_cached_triad")
                if visible and triad is not None and not triad.empty:
                    j = int(np.clip(np.searchsorted(triad.time.to_numpy(), x), 0, len(triad) - 1))
                    row = triad.iloc[j]
                    set_point(artists[0], x, row.max_resp, True)
                    set_point(artists[1], x, row.sec_resp, True)
                    set_point(artists[2], x, row.rms_resp, True)
                    artists[3].set_text(f"残差: Max={row.max_resp:.2f}m · 2nd={row.sec_resp:.2f}m · RMS={row.rms_resp:.2f}m")
                    artists[3].set_visible(True)
                else:
                    for art in artists[:3]: set_point(art, 0, 0, False)
                    artists[3].set_visible(False)
            elif name=="Body-Frame Error":
                along = float(getattr(r, "along_err", np.nan))
                cross = float(getattr(r, "cross_err", np.nan))
                vert = float(getattr(r, "vert_err", np.nan))
                if visible and np.isfinite(along):
                    set_point(artists[0], x, along, True)
                    set_point(artists[1], x, cross, True)
                    set_point(artists[2], x, vert, True)
                    artists[3].set_text(f"沿迹: {along:+.2f}m · 横向: {cross:+.2f}m · 垂向: {vert:+.2f}m")
                    artists[3].set_visible(True)
                else:
                    for art in artists[:3]: set_point(art, 0, 0, False)
                    artists[3].set_visible(False)
            elif name in ("C/N0","Elevation","Residuals","Multipath Proxy","Cycle Slips"):
                cs=current_sat
                if name=="C/N0":
                    for freq,art in zip([1,2],artists):
                        g=cs[(cs.freq==freq)&(cs.snr>0)];art.set_offsets(np.c_[g.time,g.snr] if visible and not g.empty else np.empty((0,2)))
                elif name=="Elevation":
                    g=cs[(cs.freq==1)&(cs.el>=0)];artists[0].set_offsets(np.c_[g.time,g.el] if visible and not g.empty else np.empty((0,2)))
                elif name=="Residuals":
                    g=cs[(cs.freq==1)&(cs.resp!=0)];h=cs[(cs.freq==1)&(cs.resc!=0)]
                    artists[0].set_offsets(np.c_[g.time,g.resp] if visible and not g.empty else np.empty((0,2)))
                    artists[1].set_offsets(np.c_[h.time,h.resc*100] if visible and not h.empty else np.empty((0,2)))
                elif name=="Multipath Proxy":
                    g=cs[(cs.freq==1)&(cs.resp!=0)];artists[0].set_offsets(np.c_[g.el,g.resp.abs()] if visible and not g.empty else np.empty((0,2)))
                elif name=="Cycle Slips":
                    g=cs[cs.slipc>0];prns=sorted(sat_frame.prn.unique());yi={prn:k for k,prn in enumerate(prns)}
                    offsets=np.c_[g.time, [yi[z] for z in g.prn]] if visible and not g.empty else np.empty((0,2))
                    artists[0].set_offsets(offsets)
            elif name=="Satellite Visibility Gantt":
                artists[0].set_xdata([x,x]) if visible else artists[0].set_xdata([np.nan,np.nan])
                prn_map = getattr(ax, "_gantt_prn_to_y", getattr(self, "_gantt_prn_to_y", {}))
                active_prns = [p for p in current_sat.prn.unique() if p in prn_map]
                if visible and len(active_prns) > 0:
                    active_y = [prn_map[p] for p in active_prns]
                    artists[1].set_offsets(np.c_[[x] * len(active_y), active_y])
                    import matplotlib.pyplot as _plt
                    turbo = _plt.cm.turbo
                    sat_colors = []
                    for p in active_prns:
                        sub_sat = current_sat[current_sat.prn == p]
                        sub_l1 = sub_sat[sub_sat.freq == 1]
                        row = sub_l1.iloc[0] if not sub_l1.empty else sub_sat.iloc[0]
                        resp_val = float(row.resp) if ("resp" in row and pd.notna(row.resp)) else 0.0
                        abs_r = abs(resp_val)
                        sat_colors.append(turbo(np.clip(abs_r / 15.0, 0.0, 1.0)))
                    artists[1].set_facecolors(sat_colors)
                    artists[1].set_edgecolors("white")
                    artists[1].set_linewidths(1.3)
                    artists[2].set_text(f"历元 {i+1} · 跟踪: {len(active_y)} 颗星")
                    artists[2].set_visible(True)
                else:
                    artists[1].set_offsets(np.empty((0, 2)))
                    artists[2].set_visible(False)
            elif name=="Multipath Skyplot":
                cs=current_sat
                g=cs[(cs.freq==1)&(cs.resp!=0)]
                if visible and not g.empty:
                    th=np.deg2rad(g.az.to_numpy())
                    r_val=90.0-g.el.to_numpy()
                    artists[0].set_offsets(np.c_[th, r_val])
                else:
                    artists[0].set_offsets(np.empty((0,2)))
            elif name=="Covariance":
                val = float(getattr(r, "de", np.nan)) if self.data["has_truth"] else float(getattr(r, "sigE", np.nan))
                if visible and np.isfinite(val):
                    set_point(artists[0], x, val, True)
                    sde_val = float(getattr(r, "sde", np.nan))
                    b_str = f" · 3σ: ±{3*sde_val:.2f}m" if np.isfinite(sde_val) else ""
                    artists[1].set_text(f"东向误差: {val:+.2f}m{b_str}")
                    artists[1].set_visible(True)
                else:
                    set_point(artists[0], 0, 0, False)
                    artists[1].set_visible(False)
            elif name=="False Fix Analysis":
                rat = min(float(getattr(r, "ratio", 1.0)), 15.0)
                e3 = max(float(getattr(r, "e3d", 0.01)), 0.01) if self.data["has_truth"] else np.nan
                if visible and np.isfinite(rat) and np.isfinite(e3):
                    set_point(artists[0], rat, e3, True)
                    is_false = (int(r.q) == 1 and e3 > 1.0)
                    status_desc = "⚠️ 假固定 (False Fix!)" if is_false else ("✅ 正常固定" if int(r.q)==1 else "浮点解 (Float)")
                    artists[1].set_text(f"Ratio: {rat:.1f} · 3D误差: {e3:.2f}m · {status_desc}")
                    artists[1].set_visible(True)
                else:
                    set_point(artists[0], 0, 0, False)
                    artists[1].set_visible(False)
            elif name=="El vs SNR & Resp":
                cs=current_sat
                g=cs[(cs.freq==1)&(cs.snr>0)&(cs.el>0)].drop_duplicates("prn") if (cs is not None and not cs.empty) else pd.DataFrame()
                if visible and not g.empty:
                    import matplotlib.pyplot as _plt
                    turbo = _plt.cm.turbo
                    artists[0].set_offsets(np.c_[g.el, g.snr])
                    sat_edgecolors = []
                    sat_facecolors = []
                    for k, (_, sat_row) in enumerate(g.iterrows()):
                        resp_val = float(sat_row.resp) if ("resp" in sat_row and pd.notna(sat_row.resp)) else 0.0
                        abs_r = abs(resp_val)
                        col = turbo(np.clip(abs_r / 15.0, 0.0, 1.0))
                        sat_edgecolors.append(col)
                        sat_facecolors.append([col[0], col[1], col[2], 0.22])
                        if k < 35:
                            artists[k + 1].set_position((sat_row.el + 0.8, sat_row.snr + 0.5))
                            is_used = ("resp" in sat_row and sat_row.resp != 0)
                            tag = sat_row.prn if is_used else f"{sat_row.prn} (跟踪)"
                            artists[k + 1].set_text(tag)
                            artists[k + 1].get_bbox_patch().set_edgecolor(col)
                            artists[k + 1].get_bbox_patch().set_linewidth(1.3)
                            artists[k + 1].set_visible(True)
                    artists[0].set_edgecolors(sat_edgecolors)
                    artists[0].set_facecolors(sat_facecolors)
                    for k in range(len(g), 35):
                        artists[k + 1].set_visible(False)
                else:
                    artists[0].set_offsets(np.empty((0, 2)))
                    for k in range(35):
                        artists[k + 1].set_visible(False)
            elif name=="Innovation / NIS":
                artists[0].set_text(f"当前历元：{dt:%Y-%m-%d %H:%M:%S} CST · Q={int(r.q)}")

        self.custom_updaters.append(update)

    @staticmethod
    def _rms(values):
        x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
        return float(np.sqrt(np.mean(x*x))) if len(x) else float("nan")

    @staticmethod
    def _max_run(mask):
        best=run=0
        for v in np.asarray(mask,dtype=bool):
            run=run+1 if v else 0;best=max(best,run)
        return best

    def update_analysis(self):
        if self.data is None:return
        d=self.data;p=d["pos"];s=d["sat"];mode=self.mode.get();lines=[]
        if mode in ("Gnd Trk","Dashboard"):
            rates=p.q.value_counts(normalize=True);fix=float(rates.get(1,0))*100;fl=float(rates.get(2,0))*100
            lines.append(f"解算状态：FIX {fix:.1f}%、FLOAT {fl:.1f}%；共有 {len(p):,} 个历元，轨迹按解状态着色。")
            if d["has_truth"]:
                lines.append(f"与真值比较：水平 RMSE {self._rms(p.eh):.3f} m，3D RMSE {self._rms(p.e3d):.3f} m；水平误差 P95={p.eh.quantile(.95):.3f} m。")
                worst=p.e3d.idxmax();lines.append(f"最大 3D 误差 {p.loc[worst,'e3d']:.2f} m，发生在 TOW {p.loc[worst,'tow']:.2f} s（Q={int(p.loc[worst,'q'])}）。")
        elif mode=="ENU Position":
            ranges=[]
            for c,label in [("E","东向"),("N","北向"),("U","天向")]:
                v=p[c];ranges.append(f"{label}范围 {v.min():.2f} 至 {v.max():.2f} m")
            lines.append("；".join(ranges)+"。这是相对 ENU 坐标变化，表示运动轨迹，不等同于定位误差。")
        elif mode=="Height":
            if d["has_truth"]:
                du=p.du.dropna();lines.append(f"垂向误差：偏差 {du.mean():+.3f} m，RMSE {self._rms(du):.3f} m，绝对误差 P95={du.abs().quantile(.95):.3f} m。")
            else:lines.append(f"没有真值：RTK Up 分量范围 {p.U.min():.2f} 至 {p.U.max():.2f} m；只能观察变化，不能解释为高程误差。")
        elif mode=="ENU Error":
            if d["has_truth"]:
                axes=[]
                for c,label in [("de","E"),("dn","N"),("du","U")]:axes.append(f"{label}: bias {p[c].mean():+.3f} m, RMSE {self._rms(p[c]):.3f} m")
                lines.append("真值差（RTK−真值）："+"；".join(axes)+f"。水平误差 RMSE {self._rms(p.eh):.3f} m，P95 {p.eh.quantile(.95):.3f} m；偏差为有符号平均误差。")
            else:lines.append("缺少可匹配的真值历元，无法计算 ENU 位置误差。")
        elif mode=="Solution Status":
            counts=p.q.value_counts(normalize=True);parts=[f"{Q_LABELS.get(int(q),str(q))} {v*100:.1f}%" for q,v in counts.sort_index().items()]
            transitions=int((p.q.to_numpy()[1:]!=p.q.to_numpy()[:-1]).sum());lines.append("状态比例："+"、".join(parts)+f"；状态切换 {transitions} 次；最长连续 FIX {self._max_run(p.q==1)} 历元。")
        elif mode=="AR Ratio":
            med=float(p.ratio.median());pct=float((p.ratio>=3).mean()*100);lines.append(f"Ratio 中位数 {med:.2f}，P95={p.ratio.quantile(.95):.2f}，达到图示参考线 3 的历元占 {pct:.1f}%。Ratio 是模糊度检验统计量，单独超过门限不代表真值误差必然较小。")
        elif mode=="Differential Age":
            if "age" in p.columns and not p.empty:
                valid_age = p.age.dropna()
                med_age = float(valid_age.median())
                p95_age = float(valid_age.quantile(0.95))
                max_age = float(valid_age.max())
                over2 = float((valid_age > 2.0).mean() * 100)
                over5 = float((valid_age > 5.0).mean() * 100)
                lines.append(f"差分改正数时延 (Differential Age)：中位数={med_age:.2f} s，P95={p95_age:.2f} s，最大={max_age:.2f} s。")
                lines.append(f"时延预警统计：Age > 2.0s 历元占比 {over2:.1f}%；Age > 5.0s (易导致丢定/降级) 占比 {over5:.1f}%。底色带联动对应解状态 Q (绿=FIX, 黄=FLOAT, 红=SINGLE)。")
            else:
                lines.append("当前 .pos 解算数据未包含有效差分时延 (age) 记录。")
        elif mode=="Satellites":
            if not s.empty:
                e=diag._sat_epoch(d);lines.append(f"跟踪卫星数：中位数 {e.tracked.median():.1f}，范围 {e.tracked.min()}–{e.tracked.max()}；有效参与数：中位数 {e.used.median():.1f}，最低 {e.used.min()}。")
                l1=s[(s.freq==1)&(s.vsat==1)].drop_duplicates(["tow","prn"]);parts=l1.groupby("constellation").prn.nunique().sort_values(ascending=False)
                lines.append("L1 有效卫星记录按星座统计："+"、".join(f"{k} {v}" for k,v in parts.items())+"（为记录数，不是同时可见星数）。")
            else:lines.append("未加载 .stat 卫星记录。")
        elif mode=="DOP":
            dop=diag._compute_dop(s)
            if not dop.empty:
                high=float((dop.PDOP>5).mean()*100);lines.append(f"几何 DOP：PDOP 中位数 {dop.PDOP.median():.2f}，P95={dop.PDOP.quantile(.95):.2f}；PDOP>5 的历元占 {high:.1f}%。>5 仅作常用几何质量提示，DOP 不包含观测噪声与多路径影响。")
            else:lines.append("有效卫星几何记录不足，无法计算 DOP。")
        elif mode=="Skyplot":
            if not s.empty:
                g=s[s.freq==1].drop_duplicates(["prn"]);parts=g.groupby("constellation").prn.nunique()
                lines.append(f"天空图包含 {g.prn.nunique()} 颗不同 L1 PRN："+"、".join(f"{k} {v}" for k,v in parts.items())+"。仰角与方位角来自 .stat；轨迹点密度也受采样率影响。")
        elif mode=="C/N0":
            valid=s[s.snr>0]
            if not valid.empty:
                bands=[f"频率 {f} 中位数 {g.snr.median():.1f} dB-Hz" for f,g in valid.groupby("freq")]
                weak=float((valid.snr<30).mean()*100);lines.append("；".join(bands)+f"。C/N₀<30 dB-Hz 的有效记录占 {weak:.1f}%（仅作弱信号提示，阈值依设备与环境而变）。")
        elif mode=="Elevation":
            v=s[(s.freq==1)&(s.el>=0)]
            if not v.empty:
                low=float((v.el<15).mean()*100);rho=v[["el","snr"]].corr(method="spearman").iloc[0,1]
                lines.append(f"L1 仰角中位数 {v.el.median():.1f}°；低于 15° 的记录占 {low:.1f}%；仰角与 C/N₀ Spearman 相关系数 {rho:.2f}。相关性描述样本关联，不证明因果。")
        elif mode=="Residuals":
            if not s.empty:
                rp=s.loc[(s.freq==1)&(s.resp!=0),"resp"];rc=s.loc[(s.freq==1)&(s.resc!=0),"resc"]*100
                if len(rp):lines.append(f"L1 伪距后验残差：N={len(rp):,}，bias={rp.mean():+.3f} m，RMS={self._rms(rp):.3f} m，|残差| P95={rp.abs().quantile(.95):.3f} m。")
                if len(rc):lines.append(f"载波后验残差：N={len(rc):,}，RMS={self._rms(rc):.2f} cm。残差不是创新量，也不能直接等同多路径误差。")
        elif mode=="Multipath Proxy":
            g=s[(s.freq==1)&(s.resp!=0)]
            if not g.empty:lines.append(f"L1 伪距后验残差绝对值中位数 {g.resp.abs().median():.3f} m，P95={g.resp.abs().quantile(.95):.3f} m。该图只能作为多路径/遮挡的残差代理，不能当成独立 MP 组合测量。")
        elif mode=="Cycle Slips":
            if not s.empty:
                slip=s[s.slipc>0];lines.append(f".stat 中 slipc>0 的卫星频率记录 {len(slip):,} 条，涉及 {slip.prn.nunique()} 颗 PRN、{slip.tow.nunique()} 个历元。多个频率标记可能对应同一颗卫星同一历元。")
        elif mode=="Satellite Visibility Gantt":
            if not s.empty:
                s1 = s[s.freq == 1]
                prns = s1.prn.unique()
                slips = s[s.slipc > 0]
                lines.append(f"卫星可见度与周跳甘特图：共跟踪 {len(prns)} 颗卫星，涵盖 GPS/BDS/GLONASS/Galileo/QZSS 星座。")
                lines.append(f"粗条带表示有效参与解算历元 (vsat=1)，细浅灰带为未参与解算历元 (vsat=0)；红色竖线标记周跳历元 (共 {len(slips):,} 处周跳记录)。")
            else:
                lines.append("未加载 .stat 卫星记录，无法生成可见度与周跳甘特图。")
        elif mode=="Multipath Skyplot":
            if not s.empty:
                g = s[(s.freq == 1) & (s.resp != 0)]
                if not g.empty:
                    sectors = {
                        "北 (N)": g[(g.az >= 337.5) | (g.az < 22.5)],
                        "东北 (NE)": g[(g.az >= 22.5) & (g.az < 67.5)],
                        "东 (E)": g[(g.az >= 67.5) & (g.az < 112.5)],
                        "东南 (SE)": g[(g.az >= 112.5) & (g.az < 157.5)],
                        "南 (S)": g[(g.az >= 157.5) & (g.az < 202.5)],
                        "西南 (SW)": g[(g.az >= 202.5) & (g.az < 247.5)],
                        "西 (W)": g[(g.az >= 247.5) & (g.az < 292.5)],
                        "西北 (NW)": g[(g.az >= 292.5) & (g.az < 337.5)]
                    }
                    sector_means = {k: float(v.resp.abs().mean()) for k, v in sectors.items() if len(v) > 20}
                    worst_sec = max(sector_means, key=sector_means.get) if sector_means else "未知"
                    worst_val = sector_means[worst_sec] if sector_means else 0.0

                    low_el = float(g[g.el < 20].resp.abs().mean()) if not g[g.el < 20].empty else 0.0
                    high_el = float(g[g.el > 50].resp.abs().mean()) if not g[g.el > 50].empty else 0.0

                    lines.append(f"多路径星空热力图：共统计 {len(g):,} 条有效 L1 伪距残差记录。")
                    lines.append(f"反射方位角诊断：最大多路径残差聚集在 【{worst_sec}】 方位扇区 (平均残差 {worst_val:.2f} m)，提示该方向存在强反射结构物（建筑幕墙/立面）。")
                    lines.append(f"仰角衰减特征：高度角 <20° 平均残差 {low_el:.2f} m；>50° 平均残差 {high_el:.2f} m。")
            else:
                lines.append("未加载 .stat 卫星记录。")
        elif mode=="Per-PRN Residuals":
            if "_cached_prn_stats" in self.data:
                df_stat = self.data["_cached_prn_stats"]
                worst_rms = df_stat.loc[df_stat['rms'].idxmax()]
                worst_p95 = df_stat.loc[df_stat['p95'].idxmax()]
                severe_sats = df_stat[df_stat['rms'] > 3.0]
                lines.append(f"单星残差画像与污染诊断：共统计 {len(df_stat)} 颗卫星观测质量。")
                if not severe_sats.empty:
                    bad_list = ", ".join(f"{row.prn} (RMS {row.rms:.2f}m)" for row in severe_sats.itertuples())
                    lines.append(f"[警告] 严重污染卫星预警 (RMS > 3.0m)：{bad_list}。建议在配置中临时排除该星测试是否恢复固定解。")
                else:
                    lines.append("全场卫星无孤立严重污染卫星 (所有星 RMS 均在 3.0m 门限内)。")
                lines.append(f"残差最高卫星：{worst_rms['prn']} (RMS {worst_rms['rms']:.2f}m, 中位数 {worst_rms['median']:.2f}m)；P95 最高卫星：{worst_p95['prn']} (P95 {worst_p95['p95']:.2f}m)。")
            else:
                lines.append("尚未生成单星残差统计数据。")
        elif mode=="Covariance":
            lines.append(f"ENU 估计标准差中位数：E {p.sigE.median():.4f} m、N {p.sigN.median():.4f} m、U {p.sigU.median():.4f} m。")
            if d["has_truth"]:
                cover=float((np.abs(p.de)<=1.96*p.sigE).mean()*100);lines.append(f"东向误差落在 ±1.96σ 内的比例 {cover:.1f}%（名义 95% 仅在误差近似正态且协方差校准时适用）。")
            lines.append("图中曲线是 RTKLIB .pos 解算协方差传播得到的标准差，不是实测误差；右轴为 |ΔE|/(1.96σE)，小于等于 1 表示东向误差落在该历元的名义 95% 包络内。")
        elif mode=="False Fix Analysis":
            if d["has_truth"]:
                fixed=p[p.q==1];limit=max(.001,float(self.false_fix_limit.get()));false=fixed[fixed.e3d>limit]
                rate=len(false)/len(fixed)*100 if len(fixed) else float("nan")
                lines.append(f"按用户设定的 3D 误差门限 {limit:.3f} m：FIX {len(fixed):,} 个历元中，{len(false):,} 个超过门限；经验错误 FIX 率={rate:.2f}%，达标 FIX 率={100-rate:.2f}%。")
                lines.append("该比率是相对于当前真值数据和所选门限的经验统计，不是 RTKLIB 内部置信度；请按应用精度要求调整门限。")
            else:lines.append("错误 FIX 率需要真值轨迹；请加载与解算时间同步的 reference.csv。")
        elif mode=="Custom Analysis":
            modes=[name for name in self.custom_modes if name]
            lines.append(f"自定义分析网格包含 {len(modes)} 种图："+"、".join(modes)+"。空白区域可通过左侧分区按钮拆分，并拖入图表。")
        elif mode=="Innovation / NIS":
            lines.append("当前 .pos/.stat 没有滤波创新向量 v 和创新协方差 S，不能可靠计算 NIS=vᵀS⁻¹v；后验残差不能替代 NIS。")
        elif mode=="Error CDF":
            if d["has_truth"]:
                eh=p.eh.dropna(); e3d=p.e3d.dropna()
                lines.append(f"误差累积分布 (CDF)：水平 CEP50={eh.quantile(0.5):.3f} m，CEP95={eh.quantile(0.95):.3f} m；3D 误差 P50={e3d.quantile(0.5):.3f} m，P95={e3d.quantile(0.95):.3f} m。")
            else:lines.append("未加载真值参考轨迹，无法计算定位误差 CDF。")
        elif mode=="Error Histogram":
            if d["has_truth"]:
                lines.append(f"ENU 误差正态分布：东向偏置 {p.de.mean():+.3f} m (RMS {self._rms(p.de):.3f} m)；北向偏置 {p.dn.mean():+.3f} m (RMS {self._rms(p.dn):.3f} m)；高程偏置 {p.du.mean():+.3f} m (RMS {self._rms(p.du):.3f} m)。")
            else:lines.append("未加载真值参考轨迹，无法绘制误差直方图与高斯拟合。")
        elif mode=="Constellation Summary":
            if not s.empty:
                s1=s[s.freq==1]; con_list=list(s1.constellation.unique())
                lines.append(f"按星座统览：涵盖 {len(con_list)} 个系统 ({', '.join(con_list)})；左侧为各星座跟踪卫星数时序，右侧为 L1 载噪比 (C/N₀) 箱线分布。")
            else:lines.append("未加载 .stat 卫星记录。")
        elif mode=="Residual Triad":
            triad = diag.compute_residual_triad(self.data)
            if not triad.empty:
                lines.append(f"伪距残差三线统计：全场第一大残差中位数 {triad.max_resp.median():.2f}m (峰值 {triad.max_resp.max():.2f}m)；第二大残差中位数 {triad.sec_resp.median():.2f}m；RMS中位数 {triad.rms_resp.median():.2f}m。")
                lines.append("诊断启示：当最大与第二大残差同步冲高时，判定为大范围多路径/多星污染；若仅单星突兀冲高而第二大残差平稳，判定为偶发粗差/单星失锁。")
            else:lines.append("未加载 .stat 卫星残差数据。")
        elif mode=="El vs SNR & Resp":
            lines.append("高度角-载噪比-残差联合分布：正常情况下仰角越高信噪比越大；若高仰角(>45°)出现异常低信噪比(<35dB-Hz)且伴随大残差，指示头顶遮挡或强反射多路径。")
        elif mode=="Top Outlier Satellites":
            df_rank = diag.compute_satellite_health_rankings(self.data)
            if not df_rank.empty:
                worst = df_rank.iloc[0]
                lines.append(f"粗差贡献度榜首：卫星 【{worst.prn}】 (P95残差={worst.p95_resp:.2f}m, 占全场最大残差 {worst.max_resp_share*100:.1f}%, 周跳={int(worst.slip_count)}次)。")
                lines.append("调优建议：若某颗卫星持续独占最大残差，可在配置文件中将其添加至 exclsats 予以剔除排除实验。")
            else:lines.append("未加载 .stat 卫星数据。")
        elif mode=="Directional DOP":
            dop_df = diag.compute_directional_dop(self.data)
            if not dop_df.empty:
                lines.append(f"方向 DOP 评估：EDOP中位数 {dop_df.EDOP.median():.2f}，NDOP中位数 {dop_df.NDOP.median():.2f}，VDOP中位数 {dop_df.VDOP.median():.2f}。")
                if dop_df.EDOP.median() > dop_df.NDOP.median() * 1.5:
                    lines.append("构型退化提示：东向 EDOP 显著恶化于北向 NDOP，呈现典型南北向街道“一线天”遮挡引起的几何对称性缺失。")
            else:lines.append("无法计算方向 DOP。")
        elif mode=="Clock Bias & ISB":
            clk = self.data.get("clk")
            if clk is not None and not clk.empty:
                lines.append(f"接收机钟差状态：有效 $CLK 历元数 {len(clk):,}；GPS钟差范围 {clk.clk_gps.min():.1f} ~ {clk.clk_gps.max():.1f} ns。")
                if "drift" in clk.columns:
                    lines.append(f"钟漂评估：平均钟漂 {clk.drift.mean():+.2f} ns/s，最大钟漂 {clk.drift.abs().max():.2f} ns/s。钟漂平稳表明接收机晶振健康且无严重伪距粗差污染时钟状态。")
            else:lines.append("当前数据未包含 $CLK 接收机时钟参数。")
        elif mode=="Body-Frame Error":
            if d["has_truth"] and "along_err" in p.columns:
                lines.append(f"车体坐标系误差分解：沿航向(纵向) RMSE {self._rms(p.along_err):.3f} m (偏置 {p.along_err.mean():+.3f}m)；跨航向(横向) RMSE {self._rms(p.cross_err):.3f} m (偏置 {p.cross_err.mean():+.3f}m)；高程 RMSE {self._rms(p.vert_err):.3f} m。")
                lines.append("工程诊断：若纵向误差随车速线性放大，说明存在时间戳硬同步时延；若横向误差呈转弯正弦振荡，说明存在天线杆臂(Lever-arm)未精确补偿。")
            else:lines.append("需要真值参考轨迹才能投影分解车体坐标系误差。")
        if not lines:lines=["该视图当前没有足够数据生成统计摘要。"]
        self.analysis_info.configure(text="\n".join(lines))

    def _draw_enu_positions(self,axes,p):
        specs=[("E","East (m)","#2878B5"),("N","North (m)","#E07A2D"),("U","Up (m)","#36965A")]
        kw=self._get_line_kwargs("ENU Position",default_ls="-",default_lw=.8)
        t_gap, e_gap, n_gap, u_gap = insert_time_gaps(p.time, p.E, p.N, p.U)
        for ax,(col,label,color),y_arr in zip(axes,specs,[e_gap, n_gap, u_gap]):
            ax.plot(t_gap,y_arr,color=color,**kw)
            ax.set_ylabel(label);ax.set_title(f"{label.split()[0]} component",loc="left",fontsize=9);ax.grid(self.grid_on.get(),ls="--",alpha=.4)
        axes[0].set_title("ENU Position Components",loc="center",fontsize=12,weight="bold")
        axes[-1].set_xlabel("Time from start (s)")

    def _draw_track(self,ax,p,full):
        self.track_axes.append(ax)
        self._track_e=full.E.to_numpy();self._track_n=full.N.to_numpy()
        self._apply_basemap(ax, full)
        if self.data["has_truth"]:
            gt_t, gt_e, gt_n = insert_time_gaps(full.time, full.gt_E, full.gt_N)
            ax.plot(gt_e,gt_n,color="#252525",lw=1.05,ls="--",alpha=.72,label="Ground truth")

        color_by_err = (getattr(self, "track_color_by_error", None) and self.track_color_by_error.get() and self.data["has_truth"])
        if color_by_err:
            ax.set_title("Ground Track / ENU Position — colored by 3D Error")
            err = np.clip(p.e3d.fillna(0).to_numpy(dtype=float), 0, 15.0)
            sc = ax.scatter(p.E, p.N, c=err, cmap="turbo", s=8, vmin=0, vmax=15.0, alpha=0.9, zorder=4)
            fig_target = ax.figure if getattr(ax, "figure", None) is not None else getattr(self, "fig", None)
            if fig_target is not None:
                cb = fig_target.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
                cb.set_label("3D Error (m, clipped at 15m)")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_title("Ground Track / ENU Position — colored by solution status")
            for q,g in p.groupby("q"):
                label=f"Q={q}: {Q_LABELS.get(q,'Other')}"; ax.plot(g.E,g.N,".",ms=3,color=Q_COLORS.get(q,"#666"),label=label,alpha=.9)
            ax.legend(loc="upper left",fontsize=8)

        ax.set_xlabel("East (m)");ax.set_ylabel("North (m)");ax.set_aspect("equal" if self.equal_scale.get() else "auto",adjustable="box")
        idx=min(int(self.slider.get()),len(full)-1); cur=full.iloc[idx]
        stride=max(1,(idx+1)//2500)
        sub_t = full.time.iloc[:idx+1:stride]
        sub_e = self._track_e[:idx+1:stride]
        sub_n = self._track_n[:idx+1:stride]
        _t_gap, e_gap, n_gap = insert_time_gaps(sub_t, sub_e, sub_n)
        self.vehicle_trail,=ax.plot(e_gap, n_gap, color="#ed3030", lw=1.8, alpha=.9, zorder=5)
        self.vehicle_marker,=ax.plot([cur.E],[cur.N],marker=(3,0,0),ms=17,mfc="#ff2727",mec="black",mew=1.0,linestyle="None",zorder=8)
        self.vehicle_label=ax.annotate(f"  {idx+1}/{len(full)}  Q={int(cur.q)}:{Q_LABELS.get(int(cur.q),'?')}",(cur.E,cur.N),xytext=(9,8),textcoords="offset points",fontsize=8,weight="bold",bbox=dict(boxstyle="round,pad=.25",fc="white",ec="#888",alpha=.9),zorder=9)
        if self.data["has_truth"] and np.isfinite(cur.gt_E) and np.isfinite(cur.gt_N):
            self.truth_marker,=ax.plot([cur.gt_E],[cur.gt_N],marker="*",ms=14,mfc="#ffd92f",mec="#222",mew=.8,linestyle="None",zorder=9,label="Current truth")
            self.error_link,=ax.plot([cur.E,cur.gt_E],[cur.N,cur.gt_N],color="#df2525",lw=1.5,ls=":",zorder=7)
        self._draw_stats(ax,full)

    def _draw_trajectory_error_heatmap(self, ax, p, full):
        self.track_axes.append(ax)
        self._track_e = full.E.to_numpy()
        self._track_n = full.N.to_numpy()
        self._apply_basemap(ax, full)
        ax.set_title("二维轨迹误差热力图 (Trajectory Error Heatmap)")

        if self.data["has_truth"]:
            gt_t, gt_e, gt_n = insert_time_gaps(full.time, full.gt_E, full.gt_N)
            ax.plot(gt_e, gt_n, color="#444444", lw=1.1, ls="--", alpha=0.75, label="Reference GT")
            err = np.clip(p.e3d.fillna(0).to_numpy(dtype=float), 0, 15.0)
            sc = ax.scatter(p.E, p.N, c=err, cmap="turbo", s=8, vmin=0, vmax=15.0, alpha=0.9, zorder=4)
            fig_target = ax.figure if getattr(ax, "figure", None) is not None else getattr(self, "fig", None)
            if fig_target is not None:
                cb = fig_target.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
                cb.set_label("3D Error (m, clipped at 15m)")
            ax.legend(loc="upper right", fontsize=8.5)
        else:
            for q, g in p.groupby("q"):
                label = f"Q={q}: {Q_LABELS.get(q, 'Other')}"
                ax.plot(g.E, g.N, ".", ms=3, color=Q_COLORS.get(q, "#666"), label=label, alpha=0.9)
            ax.text(0.5, 0.5, "未加载参考真值 (Ground Truth is required for Error Heatmap)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.legend(loc="upper left", fontsize=8)

        ax.set_xlabel("East Offset from Start (m)")
        ax.set_ylabel("North Offset from Start (m)")
        equal_val = self.equal_scale.get() if hasattr(self, "equal_scale") else True
        ax.set_aspect("equal" if equal_val else "auto", adjustable="box")
        grid_val = self.grid_on.get() if hasattr(self, "grid_on") else True
        ax.grid(grid_val, alpha=0.3, ls=":")

        if hasattr(self, "slider"):
            idx = min(int(self.slider.get()), len(full) - 1)
            cur = full.iloc[idx]
            stride = max(1, (idx + 1) // 2500)
            sub_t = full.time.iloc[:idx+1:stride]
            sub_e = self._track_e[:idx+1:stride]
            sub_n = self._track_n[:idx+1:stride]
            _t_gap, e_gap, n_gap = insert_time_gaps(sub_t, sub_e, sub_n)
            self.vehicle_trail, = ax.plot(e_gap, n_gap, color="#ed3030", lw=1.8, alpha=0.9, zorder=5)
            self.vehicle_marker, = ax.plot([cur.E], [cur.N], marker=(3, 0, 0), ms=17, mfc="#ff2727", mec="black", mew=1.0, linestyle="None", zorder=8)
            self.vehicle_label = ax.annotate(f"  {idx+1}/{len(full)}  Q={int(cur.q)}:{Q_LABELS.get(int(cur.q),'?')}", (cur.E, cur.N), xytext=(9, 8), textcoords="offset points", fontsize=8, weight="bold", bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#888", alpha=.9), zorder=9)
            if self.data["has_truth"] and np.isfinite(cur.gt_E) and np.isfinite(cur.gt_N):
                self.truth_marker, = ax.plot([cur.gt_E], [cur.gt_N], marker="*", ms=14, mfc="#ffd92f", mec="#222", mew=0.8, linestyle="None", zorder=9, label="Current truth")
                self.error_link, = ax.plot([cur.E, cur.gt_E], [cur.N, cur.gt_N], color="#df2525", lw=1.5, ls=":", zorder=7)
        if hasattr(self, "_draw_stats"):
            self._draw_stats(ax, full)

    def _draw_stats(self,ax,p):
        xyz=p[["x","y","z"]].to_numpy(); lat,lon,h=ecef_to_geodetic(*xyz[0]); avlat,avlon,avh=ecef_to_geodetic(*xyz.mean(axis=0))
        if self.data["has_truth"]:
            e,n,u=p.de.dropna(),p.dn.dropna(),p.du.dropna(); rms_e=np.sqrt(np.mean(e*e));rms_n=np.sqrt(np.mean(n*n));rms_u=np.sqrt(np.mean(u*u));rms_2d=np.sqrt(np.mean(p.eh.dropna()**2))
            stat=f"ORI={lat:.8f}°N  {lon:.8f}°E  {h:.3f} m\nAVE={avlat:.8f}°N  {avlon:.8f}°E  {avh:.3f} m\nRMS E={rms_e:.3f} m  N={rms_n:.3f} m  U={rms_u:.3f} m  2D={rms_2d:.3f} m"
        else:
            stat=f"ORI={lat:.8f}°N  {lon:.8f}°E  {h:.3f} m\nAVE={avlat:.8f}°N  {avlon:.8f}°E  {avh:.3f} m\nMean RTK σ: E={p.sigE.mean():.3f} m  N={p.sigN.mean():.3f} m  U={p.sigU.mean():.3f} m  |  no truth loaded"
        ax.text(.995,.99,stat,transform=ax.transAxes,ha="right",va="top",fontsize=9,bbox=dict(facecolor="white",edgecolor="none",alpha=.78,pad=3))

    def _draw_errors(self,axes,p):
        if not self.data["has_truth"]:
            axes[0].text(.5,.5,"Ground truth not loaded — position errors unavailable",ha="center",va="center",transform=axes[0].transAxes)
            for ax in axes[1:]:ax.set_visible(False)
            return
        specs=[("de","ΔE (m)","#2878B5"),("dn","ΔN (m)","#E07A2D"),("du","ΔU (m)","#36965A")]
        kw=self._get_line_kwargs("ENU Error",default_ls="-",default_lw=.75)
        t_gap, de_gap, dn_gap, du_gap = insert_time_gaps(p.time, p.de, p.dn, p.du)
        for ax,(col,label,color),y_arr in zip(axes,specs,[de_gap, dn_gap, du_gap]):
            ax.plot(t_gap,y_arr,color=color,**kw);ax.axhline(0,color="#222",lw=.6);ax.set_ylabel(label);ax.grid(self.grid_on.get(),ls="--",alpha=.4)
            if hasattr(self, "_apply_robust_ylim"):
                self._apply_robust_ylim(ax, [p[col]], is_symmetric=True, min_span=1.0)
        axes[0].set_title("ENU Position Error Components",fontsize=12,weight="bold");axes[-1].set_xlabel("Time from start (s)")

    def _draw_sats(self,ax):
        if "_cached_sat_epoch" not in self.data:
            self.data["_cached_sat_epoch"] = diag._sat_epoch(self.data)
        e = self.data["_cached_sat_epoch"]
        if e.empty:return
        kw=self._get_line_kwargs("Satellites",default_ls="-",default_lw=1.0)
        t_gap, trk_g, used_g = insert_time_gaps(e.time, e.tracked, e.used)
        ax.plot(t_gap, trk_g, label="Tracked satellites", color="#2878B5", **kw)
        ax.plot(t_gap, used_g, label="Used satellites", color="#e07a2d", **kw)
        ax.set(xlabel="Time from start (s)",ylabel="Satellites",title="Tracked and Used Satellites");ax.legend()
        if hasattr(self, "_apply_robust_ylim"):
            self._apply_robust_ylim(ax, [e.tracked, e.used], non_negative=True, min_span=5.0, default_max=20.0)

    def _draw_dop(self,ax):
        if "_cached_dop" not in self.data:
            self.data["_cached_dop"] = diag._compute_dop(self.data["sat"])
        d = self.data["_cached_dop"]
        self.dop_frames[ax]=d
        if d.empty:ax.text(.5,.5,"Not enough satellite geometry for DOP",ha="center",va="center",transform=ax.transAxes);return
        kw=self._get_line_kwargs("DOP",default_ls="-",default_lw=.8)
        t_gap, pdop_g, hdop_g, vdop_g = insert_time_gaps(d.time, d.PDOP, d.HDOP, d.VDOP)
        for y_arr, c in zip([pdop_g, hdop_g, vdop_g], ["PDOP","HDOP","VDOP"]):
            ax.plot(t_gap, y_arr, label=c, **kw)
        if hasattr(self, "_apply_robust_ylim"):
            self._apply_robust_ylim(ax, [d.PDOP, d.HDOP, d.VDOP], non_negative=True, min_span=3.0, default_max=8.0)
        ax.set(xlabel="Time from start (s)",ylabel="DOP",title="DOP from satellite geometry");ax.legend()

    def _draw_sky(self,ax):
        s=self.data["sat"]
        if s.empty:return
        ax.set_theta_zero_location("N");ax.set_theta_direction(-1);ax.set_ylim(90,0);ax.set_yticks([0,30,60,90]);ax.set_yticklabels(["90°","60°","30°","0°"])
        for (prn,con),g in s[s.freq==1].groupby(["prn","constellation"]):
            g=self._display_sample(g.sort_values("time"),400)
            th=np.deg2rad(g.az);r=90-g.el
            ax.plot(th,r,lw=.6,alpha=.45,color=diag.COLORS.get(con,"#666"))
            k=int(np.nanargmax(g.el.to_numpy()))
            ax.text(th.iloc[k],r.iloc[k],prn,fontsize=6.5,color="#444",weight="bold")

        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0.0, vmax=15.0)
        sm = cm.ScalarMappable(norm=norm, cmap="turbo")
        sm.set_array([])
        fig_target = ax.figure if getattr(ax, "figure", None) is not None else getattr(self, "fig", None)
        if fig_target is not None:
            try:
                cb = fig_target.colorbar(sm, ax=ax, pad=0.08, shrink=0.75, fraction=0.046)
                cb.set_label("|Pseudorange Residual| (m)", fontsize=8)
                cb.ax.tick_params(labelsize=7)
            except Exception:
                pass

        self.sky_current=ax.scatter([],[],s=100,marker="o",facecolors="none",edgecolors=[],linewidths=2.2,zorder=12)
        labels=[ax.text(0,0,"",fontsize=7.5,weight="bold",color="#111",
                        bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="#d90429",lw=1.2,alpha=0.92),zorder=14) for _ in range(35)]
        self.sky_plots.append((ax,self.sky_current,labels))
        ax.set_title("Satellite Skyplot",pad=18)

    def _draw_cn0(self,ax):
        s=self.data["sat"];s=s[s.snr>0]
        if s.empty:return
        s=self._display_sample(s,60000)
        for f,g in s.groupby("freq"):ax.scatter(g.time,g.snr,s=3,alpha=.16,label=f"Frequency {f}")
        ax.set(xlabel="Time from start (s)",ylabel="C/N₀ (dB-Hz)",title="Signal Strength by Frequency");ax.legend()

    def _draw_elevation(self,ax):
        s=self.data["sat"];s=s[s.freq==1]
        if s.empty:return
        for _,g in s.groupby("prn"):
            g=self._display_sample(g,400)
            t_gap, el_gap = insert_time_gaps(g.time, g.el)
            ax.plot(t_gap, el_gap, lw=.5, alpha=.45)
        ax.set(xlabel="Time from start (s)",ylabel="Elevation (°)",title="Satellite Elevation vs Time")

    def _draw_residuals(self,ax):
        s=self.data["sat"]
        if s.empty:return
        s=self._display_sample(s,60000)
        g=s[(s.resp!=0)&(s.freq==1)];ax.scatter(g.time,g.resp,s=3,alpha=.16,label="Pseudorange residual")
        h=s[(s.resc!=0)&(s.freq==1)];ax.scatter(h.time,h.resc*100,s=3,alpha=.12,label="Carrier residual (cm)")
        if hasattr(self, "_apply_robust_ylim"):
            self._apply_robust_ylim(ax, [g.resp], is_symmetric=True, min_span=2.0)
        ax.set(xlabel="Time from start (s)",ylabel="Residual (m / cm)",title="RTKLIB Post-fit Residuals");ax.legend()

    def _draw_multipath(self, ax):
        s = self.data["sat"]
        if s.empty:
            ax.text(0.5, 0.5, "No .stat satellite records", ha="center", va="center", transform=ax.transAxes)
            return
        g = s[(s.freq == 1) & (s.resp != 0)]
        if g.empty:
            ax.text(0.5, 0.5, "No usable pseudorange residuals", ha="center", va="center", transform=ax.transAxes)
            return

        sampled = self._display_sample(g, 60000)
        ax.scatter(sampled.el, sampled.resp.abs(), s=4, alpha=0.15, color="#1f77b4", edgecolors="none", label="Observed |Residual|")

        if "_cached_mp_binned" not in self.data:
            bins = np.arange(0, 95, 5)
            abs_resp = g.resp.abs()
            el_cut = pd.cut(g.el, bins=bins)
            grouped = abs_resp.groupby(el_cut, observed=False)
            med = grouped.median().values
            p95 = grouped.quantile(0.95).values
            centers = 0.5 * (bins[:-1] + bins[1:])
            valid = ~np.isnan(med)
            self.data["_cached_mp_binned"] = (centers[valid], med[valid], p95[valid])

        centers, med, p95 = self.data["_cached_mp_binned"]
        if len(centers) > 0:
            ax.plot(centers, med, color="#d95f02", lw=2.0, marker="o", ms=3.5, label="Median (50%)", zorder=5)
            ax.plot(centers, p95, color="#e7298a", lw=1.8, ls="--", marker="s", ms=3.0, label="P95 Envelope", zorder=5)

        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 91, 15))
        ax.set_xticklabels([f"{x}°" for x in range(0, 91, 15)])

        if len(p95) > 0 and not np.isnan(np.nanmax(p95)):
            ymax = min(max(4.0, float(np.nanmax(p95)) * 1.35), 25.0)
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(bottom=0)

        ax.set_xlabel("Elevation (°)", fontsize=8)
        ax.set_ylabel("|Pseudorange residual| (m)", fontsize=8)
        ax.set_title("Pseudorange Residual vs Elevation (Multipath Diagnostic Proxy)", fontsize=9, weight="bold")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    @staticmethod
    def _display_sample(frame,limit):
        """Thin rendered points while keeping source data intact."""
        if len(frame)<=limit:return frame
        return frame.iloc[::max(1,int(np.ceil(len(frame)/limit)))]

    def _draw_slips(self,ax):
        s=self.data["sat"]
        if s.empty:ax.text(.5,.5,"No .stat satellite records",ha="center",va="center",transform=ax.transAxes);return
        g=s[s.slipc>0];prns=sorted(s.prn.unique());yi={prn:i for i,prn in enumerate(prns)}
        if not g.empty:ax.scatter(g.time,[yi[x] for x in g.prn],s=18,c=g.slipc,cmap="tab10",edgecolors="none")
        ax.set(xlabel="Time from start (s)",ylabel="Satellite PRN",title=f"RTKLIB slipc flags ({len(g)} marked records)")
        ticks=np.linspace(0,max(0,len(prns)-1),min(12,len(prns)),dtype=int);ax.set_yticks(ticks,[prns[i] for i in ticks])

    def _draw_differential_age(self, ax, p):
        if p is None or "age" not in p.columns or p.empty:
            ax.text(0.5, 0.5, "No Differential Age (age) in .pos file", ha="center", va="center", transform=ax.transAxes)
            return

        if "_cached_q_spans" not in self.data:
            p_tmp = p[["time", "q"]].copy()
            q_diff = p_tmp["q"].diff().ne(0)
            p_tmp["q_seg"] = q_diff.cumsum()
            q_spans = p_tmp.groupby("q_seg").agg(t_start=("time", "first"), t_end=("time", "last"), q=("q", "first")).reset_index()
            self.data["_cached_q_spans"] = q_spans
        else:
            q_spans = self.data["_cached_q_spans"]

        max_val = float(p.age.max()) if not p.empty else 10.0
        ymax = max(10.0, max_val * 1.15) if np.isfinite(max_val) else 10.0

        q_bg_colors = {
            1: ("#dcfce7", "Q=1: FIX (固定)"),
            2: ("#fef3c7", "Q=2: FLOAT (浮点)"),
            5: ("#fee2e2", "Q=5: SINGLE (单点)")
        }
        added_labels = set()
        for q_val, (col, lbl) in q_bg_colors.items():
            sub = q_spans[q_spans.q == q_val]
            if not sub.empty:
                xranges = [(float(row.t_start), max(0.1, float(row.t_end - row.t_start))) for row in sub.itertuples()]
                label = lbl if lbl not in added_labels else None
                ax.broken_barh(xranges, (0, ymax), facecolors=col, alpha=0.35, zorder=0, label=label)
                if label:
                    added_labels.add(lbl)

        kw = self._get_line_kwargs("Differential Age", default_ls="-", default_lw=1.0)
        ax.plot(p.time, p.age, color="#0B2B52", label="Age (s)", zorder=3, **kw)
        ax.axhline(2.0, color="#d97706", ls="--", lw=1.1, zorder=4, label="Warning (2.0s 预警)")
        ax.axhline(5.0, color="#dc2626", ls=":", lw=1.3, zorder=4, label="Critical (5.0s 超时)")

        ax.set_ylim(0, ymax)
        ax.set(xlabel="Date/time", ylabel="Differential Age (s)", title="Differential Age & Solution Status (差分时延与解状态)")
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)

    def _draw_satellite_visibility_gantt(self, ax, s):
        if s is None or s.empty:
            ax.text(0.5, 0.5, "No .stat satellite records", ha="center", va="center", transform=ax.transAxes)
            return

        s_l1 = s[s.freq == 1].sort_values(["prn", "time"]).copy()
        if s_l1.empty:
            s_l1 = s.sort_values(["prn", "time"]).copy()

        prefix_order = {'C': 1, 'E': 2, 'G': 3, 'J': 4, 'R': 5, 'S': 6}
        sorted_prns = sorted(
            s_l1["prn"].unique(),
            key=lambda p: (prefix_order.get(p[0] if p else "", 99), int(p[1:]) if p[1:].isdigit() else 0, p)
        )
        prn_to_y = {prn: i for i, prn in enumerate(sorted_prns)}
        self._gantt_prn_to_y = prn_to_y
        ax._gantt_prn_to_y = prn_to_y

        y_vals = [prn_to_y[p] for p in s_l1.prn]
        resp_vals = np.clip(s_l1.resp.abs().to_numpy(dtype=float), 0, 15.0) if "resp" in s_l1.columns else np.zeros(len(s_l1))

        # Scatter swimlane colored by |resp|
        sc = ax.scatter(s_l1.time, y_vals, c=resp_vals, cmap="turbo", s=9, vmin=0, vmax=15.0, alpha=0.85, edgecolors="none")

        # Overlay cycle slips
        slips = s[(s.slipc > 0) & (s.prn.isin(prn_to_y))] if "slipc" in s.columns else pd.DataFrame()
        if not slips.empty:
            slip_y = [prn_to_y[p] for p in slips.prn]
            ax.scatter(slips.time, slip_y, marker="x", color="black", s=18, lw=1.0, zorder=5, label=f"Cycle Slip (n={len(slips)})")

        # Constellation dividing dashed lines
        for i in range(1, len(sorted_prns)):
            if sorted_prns[i][0] != sorted_prns[i-1][0]:
                ax.axhline(i - 0.5, color="#cbd5e1", ls="--", lw=0.75, alpha=0.8)

        ax.set_yticks(range(len(sorted_prns)))
        ax.set_yticklabels(sorted_prns, fontsize=7 if len(sorted_prns) > 25 else 8)
        ax.set_ylim(-0.8, len(sorted_prns) - 0.2)

        first_p = self.data["pos"].iloc[0] if ("pos" in self.data and not self.data["pos"].empty) else None
        tow0 = float(first_p.tow) if first_p is not None and "tow" in first_p else 0.0
        ax.set_xlabel(f"Elapsed Time since TOW {tow0:.1f}s (s)")
        ax.set_ylabel("Satellite PRN")
        ax.set_title("卫星伪距残差与周跳泳道图 (Visibility & Slips)")

        fig_target = ax.figure if getattr(ax, "figure", None) is not None else getattr(self, "fig", None)
        if fig_target is not None:
            cb = fig_target.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
            cb.set_label("|Pseudorange Residual| (m)")
        if not slips.empty:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax.grid(self.grid_on.get(), alpha=0.25, ls=":")

    def _draw_multipath_skyplot(self, ax):
        s = self.data["sat"]
        if s is None or s.empty:
            ax.text(0.5, 0.5, "No Satellite Data", ha="center", va="center", transform=ax.transAxes)
            return
        g = s[(s.freq == 1) & (s.resp != 0)].copy()
        if g.empty:
            ax.text(0.5, 0.5, "No Pseudorange Residuals", ha="center", va="center", transform=ax.transAxes)
            return

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(90, 0)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"])

        g_draw = self._display_sample(g, 35000)
        th = np.deg2rad(g_draw.az.to_numpy())
        r = 90.0 - g_draw.el.to_numpy()
        abs_resp = g_draw.resp.abs().to_numpy()
        vmax = min(3.0, max(1.5, float(np.percentile(abs_resp, 98))))

        sc = ax.scatter(th, r, c=abs_resp, cmap="plasma", vmin=0, vmax=vmax, s=8, alpha=0.6, edgecolors="none", zorder=3)
        try:
            cb = ax.figure.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
            cb.set_label("|Pseudorange Residual| (m)", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        except Exception:
            pass

        for prn, sub in g.groupby("prn"):
            k = int(np.nanargmax(sub.el.to_numpy()))
            ax.text(np.deg2rad(sub.az.iloc[k]), 90.0 - sub.el.iloc[k], prn, fontsize=6.5, weight="bold", color="#1e293b", ha="center", zorder=6)

        ax.set_title("Multipath Skyplot Heatmap (|resp| in meters)", pad=16, fontsize=10.5, weight="bold")

    def _draw_per_prn_residuals(self, ax):
        s = self.data["sat"]
        if s is None or s.empty:
            ax.text(0.5, 0.5, "No Satellite Data", ha="center", va="center", transform=ax.transAxes)
            return

        if "_cached_prn_stats" not in self.data:
            s1 = s[(s.freq == 1) & (s.resp != 0)]
            if s1.empty:
                ax.text(0.5, 0.5, "No Pseudorange Residuals", ha="center", va="center", transform=ax.transAxes)
                return
            prefix_order = {'G': 1, 'C': 2, 'R': 3, 'E': 4, 'J': 5, 'S': 6}
            sorted_prns = sorted(s1['prn'].unique(), key=lambda p: (prefix_order.get(p[0] if p else '', 99), int(p[1:]) if p[1:].isdigit() else 0, p))

            stats = []
            for prn in sorted_prns:
                sub = s1[s1.prn == prn]
                resp = sub.resp.to_numpy()
                abs_resp = np.abs(resp)
                slips = int((sub.slipc > 0).sum())
                stats.append({
                    'prn': prn,
                    'con': sub.constellation.iloc[0],
                    'count': len(resp),
                    'rms': float(np.sqrt(np.mean(resp**2))),
                    'median': float(np.median(abs_resp)),
                    'p95': float(np.percentile(abs_resp, 95)),
                    'slips': slips
                })
            df = pd.DataFrame(stats)
            self.data["_cached_prn_stats"] = df
        else:
            df = self.data["_cached_prn_stats"]

        con_colors = {'GPS': '#2563eb', 'BeiDou': '#d97706', 'GLONASS': '#059669', 'Galileo': '#7c3aed', 'QZSS': '#db2777'}
        colors = [con_colors.get(c, '#64748b') for c in df.con]
        x = np.arange(len(df))

        edge_colors = ['#dc2626' if r > 3.0 else 'none' for r in df['rms']]
        ax.bar(x, df['rms'], width=0.6, color=colors, alpha=0.82, label='RMS Residual (m)', edgecolor=edge_colors, linewidth=1.5, zorder=2)
        ax.scatter(x, df['p95'], marker='v', color='#b91c1c', s=24, zorder=4, label='P95 |Residual| (m)')
        ax.scatter(x, df['median'], marker='_', color='#1e293b', s=38, zorder=5, label='Median |Residual|')

        ax.axhline(1.0, color='#f59e0b', ls='--', lw=1.0, alpha=0.75, label='Warning (1.0m)')
        ax.axhline(3.0, color='#dc2626', ls=':', lw=1.2, alpha=0.85, label='Severe (3.0m)')

        ax.set_xticks(x)
        ax.set_xticklabels(df['prn'], rotation=90, fontsize=7 if len(df) > 25 else 8)
        ax.set_ylabel('Pseudorange Residual (m)', fontsize=8.5)
        ax.set_title('Per-PRN Residual Profile & Satellite Contamination', fontsize=10.5, weight='bold')

        for i in range(1, len(df)):
            if df['con'].iloc[i] != df['con'].iloc[i-1]:
                ax.axvline(i - 0.5, color='#cbd5e1', ls='--', lw=0.75)

        twin = ax.twinx()
        self.cov_twin_axes[ax] = twin
        twin.plot(x, df['slips'], color='#e11d48', ls='--', marker='x', ms=4, lw=0.9, label='Cycle Slips')
        twin.set_ylabel('Cycle Slips Count', color='#e11d48', fontsize=8)
        twin.tick_params(axis='y', labelcolor='#e11d48', labelsize=7.5)
        twin.set_ylim(0, max(10, df['slips'].max() * 1.3))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7.5, ncol=3, framealpha=0.85)

    def _draw_covariance(self, ax, p):
        if self.data["has_truth"]:
            sde = p.sigE if "sigE" in p.columns else (p.sde if "sde" in p.columns else pd.Series(0, index=p.index))
            t_gap, de_gap, sde_gap = insert_time_gaps(p.time, p.de, sde)

            bound_pos = 3.0 * sde_gap
            bound_neg = -3.0 * sde_gap

            ax.plot(t_gap, de_gap, color="#1f77b4", lw=1.1, label="Actual East Error (m)")
            ax.plot(t_gap, bound_pos, color="#d62728", ls="--", lw=1.0, label="+3σ Filter Bound (3*sde)")
            ax.plot(t_gap, bound_neg, color="#d62728", ls="--", lw=1.0, label="-3σ Filter Bound (-3*sde)")

            if hasattr(self, "_apply_robust_ylim"):
                self._apply_robust_ylim(ax, [p.de, 3.0 * sde, -3.0 * sde], is_symmetric=True, min_span=2.0)
            else:
                ax.set_ylim(-15.0, 15.0)

            first = p.iloc[0]
            tow0 = float(first.tow) if "tow" in first else 0.0
            ax.set_xlabel(f"Elapsed Time since TOW {tow0:.1f}s (s)")
            ax.set_ylabel("East Error & ±3σ Bound (m)")
            ax.set_title("滤波协方差一致性检验 (Filter Covariance Consistency)")
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
            ax.grid(self.grid_on.get(), alpha=0.3, ls=":")
        else:
            sde = p.sigE if "sigE" in p.columns else p.sde
            sdn = p.sigN if "sigN" in p.columns else p.sdn
            sdu = p.sigU if "sigU" in p.columns else p.sdu
            t_gap, sde_gap, sdn_gap, sdu_gap = insert_time_gaps(p.time, sde, sdn, sdu)
            for s_arr, col, lbl in [(sde_gap, "#1f77b4", "3σ East"), (sdn_gap, "#ff7f0e", "3σ North"), (sdu_gap, "#2ca02c", "3σ Up")]:
                ax.plot(t_gap, 3.0 * s_arr, color=col, ls="--", lw=1.0, label=f"+{lbl}")
                ax.plot(t_gap, -3.0 * s_arr, color=col, ls=":", lw=1.0, label=f"-{lbl}")
            ax.set_xlabel("Time from start (s)")
            ax.set_ylabel("±3σ Filter Bound (m)")
            ax.set_title("滤波协方差一致性 (±3σ 边界)")
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
            ax.grid(self.grid_on.get(), alpha=0.3, ls=":")

    def _draw_nis(self,ax):
        ax.set_axis_off()
        ax.text(.5,.5,"NIS 不可用\n需要滤波前创新向量 v 和协方差 S；\n.pos/.stat 不含这两项，后验残差不能替代。",
                ha="center",va="center",transform=ax.transAxes,fontsize=8.5,wrap=True,color="#555")

    def _draw_dashboard(self):
        self.fig.clear(); p = self.data["pos"]; full = self.data["pos"]; s = self.data["sat"]
        self.plot_axes = []; self.time_cursors = []
        if not self.dashboard_modes:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Dashboard has no subplots.\nClick '↺ Reset' in the toolbar to restore default layout.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#666")
            ax.set_axis_off(); self.plot_axes = [ax]; self.ax = ax
            self.fig.suptitle("RTKLIB Solution Overview", fontsize=14, weight="bold")
            self.fig.tight_layout(); return

        if self.focused_dashboard_subplot is not None and 0 <= self.focused_dashboard_subplot < len(self.dashboard_modes):
            idx = self.focused_dashboard_subplot
            name = self.dashboard_modes[idx]
            ax = self.fig.add_subplot(111, projection="polar" if name == "Skyplot" else None)
            self.plot_axes = [ax]; self.ax = ax
            self._render_dashboard_item(ax, name, p, full, s, is_single=True)
            for spine in ax.spines.values():
                spine.set_edgecolor("#168aad"); spine.set_linewidth(1.8)
            ax.grid(self.grid_on.get(), alpha=.3)
            self.fig.suptitle(f"RTKLIB Solution Overview — [{name}] (Maximized)", fontsize=13, weight="bold")
            self.fig.tight_layout(); return

        n = len(self.dashboard_modes)
        if n == 1: rows, cols = 1, 1
        elif n == 2: rows, cols = 1, 2
        elif n == 3: rows, cols = 1, 3
        elif n == 4: rows, cols = 2, 2
        elif n <= 6: rows, cols = 2, 3
        elif n <= 8: rows, cols = 2, 4
        else: cols = 3; rows = int(np.ceil(n / 3))

        axs = self.fig.subplots(rows, cols)
        axs_flat = np.array(axs).ravel()
        for i in range(len(axs_flat)):
            ax = axs_flat[i]
            if i < n:
                name = self.dashboard_modes[i]
                self.plot_axes.append(ax)
                if i == 0: self.ax = ax
                self._render_dashboard_item(ax, name, p, full, s, is_single=False)
                is_selected = (i == self.selected_dashboard_index)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#168aad" if is_selected else "#b9c0c8")
                    spine.set_linewidth(1.8 if is_selected else 0.75)
                ax.grid(self.grid_on.get(), alpha=.3)
            else:
                ax.set_axis_off()

        self.fig.suptitle("RTKLIB Solution Overview", fontsize=14, weight="bold")
        self.fig.tight_layout()

    def _render_dashboard_item(self, ax, name, p, full, s, is_single=False):
        if name == "Gnd Trk":
            self.track_axes.append(ax)
            self._draw_track(ax, self._filtered(), full)
        elif name == "Trajectory Error Heatmap":
            self.track_axes.append(ax)
            self._draw_trajectory_error_heatmap(ax, p, full)
        elif name == "Position Error":
            if self.data["has_truth"]:
                t_gap, eh_gap, e3d_gap = insert_time_gaps(p.time, p.eh, p.e3d)
                ax.plot(t_gap, eh_gap, lw=.6, label="Horizontal")
                ax.plot(t_gap, e3d_gap, lw=.6, label="3D")
                self._apply_robust_ylim(ax, [p.eh, p.e3d], non_negative=True, min_span=2.0)
                ax.legend(fontsize=7 if not is_single else 9)
                ax.set(title="Position Error", xlabel="Time (s)", ylabel="m")
            else:
                ax.text(.5, .5, "No truth loaded", ha="center", transform=ax.transAxes)
                ax.set(title="Position Error")
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Solution Status":
            t_gap, q_g = insert_time_gaps(p.time, p.q)
            ax.step(t_gap, q_g, where="post", lw=.65)
            ax.set(title="Solution Status", xlabel="Time (s)", ylabel="Q")
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "AR Ratio":
            self._draw_ar_ratio(ax, p)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Differential Age":
            self._draw_differential_age(ax, p)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Satellites":
            self._draw_sats(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name in ("Elevation vs C/N0", "Elevation vs C/N₀"):
            ss = s.query("freq == 1 and snr > 0") if not s.empty else s
            if not ss.empty:
                ax.scatter(ss.el, ss.snr, s=2, alpha=.15)
            ax.set(title="Elevation vs C/N₀", xlabel="Elevation (°)", ylabel="dB-Hz")
        elif name == "Height":
            t_gap, u_gap = insert_time_gaps(p.time, p.U)
            ax.plot(t_gap, u_gap, lw=.7, label="RTK U")
            if self.data["has_truth"]:
                t_gt_gap, gt_u_gap = insert_time_gaps(p.time, p.gt_U)
                ax.plot(t_gt_gap, gt_u_gap, lw=.7, ls="--", label="Truth U")
            ax.set(title=name, ylabel="Up (m)"); ax.legend(fontsize=7)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "DOP":
            self._draw_dop(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Residuals":
            self._draw_residuals(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Cycle Slips":
            self._draw_slips(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Satellite Visibility Gantt":
            self._draw_satellite_visibility_gantt(ax, s)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Multipath Skyplot":
            self._draw_multipath_skyplot(ax)
        elif name == "Per-PRN Residuals":
            self._draw_per_prn_residuals(ax)
        elif name == "Covariance":
            self._draw_covariance(ax, p)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "False Fix Analysis":
            self._draw_false_fix(ax, p)
        elif name == "ENU Position":
            t_gap, e_gap, n_gap, u_gap = insert_time_gaps(p.time, p["E"], p["N"], p["U"])
            for c_val, col, c_name in zip([e_gap, n_gap, u_gap], ["#2878B5", "#E07A2D", "#36965A"], ["E", "N", "U"]):
                ax.plot(t_gap, c_val, lw=.65, label=c_name, color=col)
            ax.set(title=name, ylabel="Position (m)")
            ax.legend(fontsize=6 if not is_single else 8, ncol=3)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "ENU Error":
            if self.data["has_truth"]:
                t_gap, de_gap, dn_gap, du_gap = insert_time_gaps(p.time, p["de"], p["dn"], p["du"])
                for c_val, col, c_name in zip([de_gap, dn_gap, du_gap], ["#2878B5", "#E07A2D", "#36965A"], ["DE", "DN", "DU"]):
                    ax.plot(t_gap, c_val, lw=.65, label=c_name, color=col)
                ax.axhline(0, color="k", lw=.5)
                ax.legend(fontsize=6 if not is_single else 8, ncol=3)
            else:
                ax.text(.5, .5, "No truth loaded", ha="center", transform=ax.transAxes)
            ax.set(title=name, ylabel="Error (m)")
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name in ("C/N0", "C/N₀"):
            self._draw_cn0(ax)
            ax.set_title(name)
        elif name == "Elevation":
            self._draw_elevation(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Skyplot":
            if getattr(ax, "name", "") == "polar":
                self._draw_sky(ax)
            else:
                ax.text(.5, .5, "Skyplot requires polar projection\n(Double-click to maximize or view alone)", ha="center", va="center", transform=ax.transAxes)
                ax.set(title="Skyplot")
        elif name == "Multipath Proxy":
            self._draw_multipath(ax)
        elif name == "Innovation / NIS":
            self._draw_nis(ax)
        elif name == "Error CDF":
            self._draw_error_cdf(ax, p)
        elif name == "Error Histogram":
            self._draw_error_histogram([ax], p)
        elif name == "Constellation Summary":
            self._draw_constellation_summary([ax], s)
        elif name == "Residual Triad":
            self._draw_residual_triad(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "El vs SNR & Resp":
            self._draw_el_snr_resp(ax)
        elif name == "Top Outlier Satellites":
            self._draw_top_outliers(ax)
        elif name == "Directional DOP":
            self._draw_directional_dop(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Clock Bias & ISB":
            self._draw_clock_bias_isb(ax)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Body-Frame Error":
            self._draw_body_frame_error(ax, p)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        else:
            ax.text(.5, .5, f"{name}", ha="center", transform=ax.transAxes)

    def _draw_multi_compare(self):
        self.fig.clear(); self.plot_axes = []; self.time_cursors = []
        visible_sols = [s for s in self.multi_solutions if s["visible"].get()]
        if not visible_sols:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No solutions visible for comparison.\nClick '+ Add Solution (.pos)...' or check solutions in the left panel.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#666")
            ax.set_axis_off(); self.plot_axes = [ax]; self.ax = ax
            self.fig.suptitle("Multi-Solution Comparison Overview", fontsize=14, weight="bold")
            self.fig.tight_layout(); return

        if self.focused_compare_subplot is not None and self.focused_compare_subplot in (0, 1):
            ax = self.fig.add_subplot(111)
            self.plot_axes = [ax]; self.ax = ax
            if self.focused_compare_subplot == 0:
                self._draw_compare_track(ax, visible_sols)
                self.fig.suptitle("Ground Track Comparison (Maximized — Right-click to restore)", fontsize=13, weight="bold")
            else:
                self._draw_compare_error(ax, visible_sols)
                self.fig.suptitle("Horizontal Error Comparison (Maximized — Right-click to restore)", fontsize=13, weight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor("#168aad"); spine.set_linewidth(1.8)
            ax.grid(self.grid_on.get(), alpha=.3)
            self.fig.tight_layout(); return

        axs = self.fig.subplots(2, 1)
        self.plot_axes = list(axs); self.ax = axs[0]

        # Top: Track Overlay
        self._draw_compare_track(axs[0], visible_sols)
        is_sel_0 = (self.selected_compare_subplot == 0)
        for spine in axs[0].spines.values():
            spine.set_edgecolor("#168aad" if is_sel_0 else "#b9c0c8")
            spine.set_linewidth(1.8 if is_sel_0 else 0.75)
        axs[0].grid(self.grid_on.get(), alpha=.3)

        # Bottom: Error Timeseries Overlay
        self._draw_compare_error(axs[1], visible_sols)
        is_sel_1 = (self.selected_compare_subplot == 1)
        for spine in axs[1].spines.values():
            spine.set_edgecolor("#168aad" if is_sel_1 else "#b9c0c8")
            spine.set_linewidth(1.8 if is_sel_1 else 0.75)
        axs[1].grid(self.grid_on.get(), alpha=.3)

        self.fig.suptitle("Multi-Solution Comparison (Track & Horizontal Error)", fontsize=14, weight="bold")
        self.fig.tight_layout()

    def _draw_compare_track(self, ax, visible_sols):
        self.track_axes.append(ax)
        has_truth = any(s["data"]["has_truth"] for s in visible_sols)
        if has_truth:
            for s in visible_sols:
                if s["data"]["has_truth"]:
                    gt_e = s["data"]["pos"]["gt_E"]
                    gt_n = s["data"]["pos"]["gt_N"]
                    ax.plot(gt_e, gt_n, "k--", lw=1.1, label="Truth", zorder=3)
                    break
        for s in visible_sols:
            p = s["data"]["pos"]
            ax.plot(p.E, p.N, color=s["color"], lw=1.2, alpha=0.85, label=s["name"], zorder=4)
        ax.set(xlabel="East (m)", ylabel="North (m)", title="Ground Track Comparison (Horizontal Trajectory)")
        ax.axis("equal" if self.equal_scale.get() else "auto")
        ax.legend(fontsize=8, loc="best")

    def _draw_compare_error(self, ax, visible_sols):
        has_any_truth = False
        for s in visible_sols:
            p = s["data"]["pos"]
            if s["data"]["has_truth"] and "eh" in p:
                has_any_truth = True
                rmse = np.sqrt(np.mean(p.eh.dropna()**2))
        if has_any_truth:
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.text(0.5, 0.5, "Ground truth (.csv) required to plot position errors.\nClick 'Open Reference' in the left sidebar to load truth data.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666")
        ax.set(xlabel="Date/time", ylabel="Horizontal Error (m)", title="Horizontal Positioning Error vs Time")
        self._set_datetime_axis(ax)
        self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=0.7, alpha=0.65))

    def on_scrub(self,value):
        if self.data is None:return
        self.epoch.set(int(float(value)));self._schedule_cursor_update()

    def _update_cursor(self):
        if self.data is None:return
        p=self.data["pos"];i=min(max(0,int(self.slider.get())),len(p)-1);r=p.iloc[i]
        if i==self._last_cursor_index:return
        self._last_cursor_index=i
        self.epoch_label.configure(text=f"{i+1:,} / {len(p):,}")
        dt=gpst_to_local_datetime(r.week,r.tow)
        self.time_label.configure(text=f"{dt:%Y-%m-%d %H:%M:%S} CST   Q={int(r.q)}:{Q_LABELS.get(int(r.q),'?')}")
        err_probe = f" | 2D: {float(r.eh):.3f}m | 3D: {float(r.e3d):.3f}m" if (self.data["has_truth"] and "eh" in r and np.isfinite(r.eh)) else ""
        ratio_probe = f" | Ratio: {float(r.ratio):.1f}" if ("ratio" in r and np.isfinite(r.ratio)) else ""
        sat_probe = f" | Sats: {int(r.ns)}" if ("ns" in r and np.isfinite(r.ns)) else ""
        self.status.configure(text=f"Epoch {i+1}/{len(p)} ({dt:%H:%M:%S} CST) | Q={int(r.q)} ({Q_LABELS.get(int(r.q),'?')}){err_probe}{ratio_probe}{sat_probe} | {self.pos_path.name if self.pos_path else ''}")
        for cursor in self.time_cursors:cursor.set_xdata([r.time,r.time])
        current_sat=self._sat_sorted.iloc[0:0]
        if self._sat_tows.size:
            j=int(np.clip(np.searchsorted(self._sat_tows,r.tow),0,len(self._sat_tows)-1))
            if j and abs(self._sat_tows[j-1]-r.tow)<abs(self._sat_tows[j]-r.tow):j-=1
            sat_tow=self._sat_tows[j];lo=np.searchsorted(self._sat_tows,sat_tow-1e-7,"left");hi=np.searchsorted(self._sat_tows,sat_tow+1e-7,"right")
            current_sat=self._sat_sorted.iloc[lo:hi]
        for update in self.custom_updaters:update(i,r,current_sat)
        if self.vehicle_marker is not None:
            self.vehicle_marker.set_data([r.E],[r.N]);start=max(0,i-120);stride=max(1,(i-start+1)//300)
            self.vehicle_trail.set_data(self._track_e[start:i+1:stride],self._track_n[start:i+1:stride])
            self.vehicle_label.xy=(r.E,r.N);self.vehicle_label.set_text(f"  {i+1}/{len(p)}  Q={int(r.q)}:{Q_LABELS.get(int(r.q),'?')}")
        if self.data["has_truth"] and np.isfinite(r.get("gt_E",np.nan)):
            if "_truth_str_cache" not in self.data:
                self.data["_truth_str_cache"] = {}
            if i not in self.data["_truth_str_cache"]:
                rlat,rlon,rh=ecef_to_geodetic(r.x,r.y,r.z)
                ref=self.data["reference"];tow=float(r.tow)
                gt_xyz=[np.interp(tow,ref["GPS TOW (s)"],ref[c],left=np.nan,right=np.nan) for c in ["ECEF X (m)","ECEF Y (m)","ECEF Z (m)"]]
                glat,glon,gh=ecef_to_geodetic(*gt_xyz)
                de,dn,du=float(r.de),float(r.dn),float(r.du);eh=float(r.eh);e3d=float(r.e3d)
                self.data["_truth_str_cache"][i] = (f"RTK: {rlat:.8f}°N, {rlon:.8f}°E, h={rh:.3f} m     "
                    f"Truth: {glat:.8f}°N, {glon:.8f}°E, h={gh:.3f} m\n"
                    f"Error RTK−Truth: ΔE={de:+.3f} m, ΔN={dn:+.3f} m, ΔU={du:+.3f} m     "
                    f"Horizontal={eh:.3f} m, 3D={e3d:.3f} m")
            self.truth_info.configure(text=self.data["_truth_str_cache"][i])
            if self.truth_marker is not None:self.truth_marker.set_data([r.gt_E],[r.gt_N])
            if self.error_link is not None:self.error_link.set_data([r.E,r.gt_E],[r.N,r.gt_N])
        elif self.data is None or not self.data["has_truth"]:
            self.truth_info.configure(text="Ground truth not loaded — RTK/Truth coordinate differences are unavailable.")
        else:
            self.truth_info.configure(text="This epoch is outside the ground-truth time span; no comparison is available.")
        if self.sky_plots and not self.data["sat"].empty:
            cur = current_sat[(current_sat.freq == 1) & (current_sat.snr > 0) & (current_sat.el > 0)].drop_duplicates("prn")
            if not cur.empty:
                import matplotlib.pyplot as _plt
                turbo = _plt.cm.turbo
                az = cur.az.to_numpy()
                el = cur.el.to_numpy()
                prn = cur.prn.to_numpy()
                th = np.deg2rad(az)
                r_val = 90.0 - el
                offsets = np.c_[th, r_val]

                sat_colors = []
                sat_faces = []
                for _, sat_row in cur.iterrows():
                    resp_val = float(sat_row.resp) if ("resp" in sat_row and pd.notna(sat_row.resp)) else 0.0
                    abs_r = abs(resp_val)
                    c = turbo(np.clip(abs_r / 15.0, 0.0, 1.0))
                    sat_colors.append(c)
                    sat_faces.append([c[0], c[1], c[2], 0.22])

                dt = gpst_to_local_datetime(r.week, r.tow)
                for ax, points, labels in self.sky_plots:
                    points.set_offsets(offsets)
                    points.set_edgecolors(sat_colors)
                    points.set_facecolors(sat_faces)
                    points.set_linewidths(2.2)
                    for j in range(len(cur)):
                        if j >= len(labels):
                            label = ax.text(0, 0, "", fontsize=7.5, weight="bold", zorder=14)
                            label.set_animated(True)
                            labels.append(label)
                            self.dynamic_artists.append(label)
                        labels[j].set_position((th[j], r_val[j]))
                        labels[j].set_text(str(prn[j]))
                        labels[j].get_bbox_patch().set_edgecolor(sat_colors[j])
                        labels[j].get_bbox_patch().set_linewidth(1.3)
                        labels[j].set_visible(True)
                    for label in labels[len(cur):]:
                        label.set_visible(False)
                    ax.set_title(f"卫星天空图 · {dt:%Y-%m-%d %H:%M:%S} CST")
            else:
                for ax, points, labels in self.sky_plots:
                    points.set_offsets(np.empty((0, 2)))
                    for label in labels:
                        label.set_visible(False)
        if self.dynamic_artists:self._blit_dynamic()

    def toggle_play(self):
        self.playing=not self.playing
        self.play_btn.configure(text="■ Stop" if self.playing else "▶ Play")
        if self.playing:self._step_play()
        elif self.play_job is not None:
            self.root.after_cancel(self.play_job);self.play_job=None
    def _step_play(self):
        if not self.playing or self.data is None:return
        multiplier=int(self.speed.get().rstrip("x"));i=int(self.slider.get())+multiplier
        if i>=len(self.data["pos"]):i=0
        self.slider.set(i);self.play_job=self.root.after(100,self._step_play)

    def _apply_aspect(self):
        for ax in self.fig.axes:
            if getattr(ax,"name","")=="rectilinear":ax.set_aspect("equal" if self.equal_scale.get() and self.mode.get()=="Gnd Trk" else "auto",adjustable="box")
        self.canvas.draw_idle()

    def fit_view(self):
        for ax in self.fig.axes:
            limits=getattr(self,"_home_limits",{}).get(id(ax))
            if limits:ax.set_xlim(limits[0]);ax.set_ylim(limits[1])
            elif getattr(ax,"name","")=="polar":ax.set_ylim(90,0)
            else:ax.relim();ax.autoscale_view()
            if self.mode.get()=="Gnd Trk":ax.set_aspect("equal" if self.equal_scale.get() else "auto",adjustable="box")
        self.canvas.draw_idle()

    def on_scroll(self,event):
        ax=event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:return
        factor=.80 if event.button=="up" else 1.25
        mode=self.zoom_axis.get();xlim=ax.get_xlim();ylim=ax.get_ylim()
        if mode in ("Both axes","X axis"):
            rx=(xlim[1]-xlim[0])*factor;fx=(event.xdata-xlim[0])/(xlim[1]-xlim[0]);ax.set_xlim(event.xdata-rx*fx,event.xdata+rx*(1-fx))
        if mode in ("Both axes","Y axis"):
            if getattr(ax,"name","")=="polar":
                ry=(ylim[1]-ylim[0])*factor;fy=(event.ydata-ylim[0])/(ylim[1]-ylim[0]);ax.set_ylim(event.ydata-ry*fy,event.ydata+ry*(1-fy))
            else:
                p_mode = getattr(ax, "_panel_mode", "")
                is_non_neg = p_mode in ("AR Ratio", "DOP", "Directional DOP", "Satellites", "Residual Triad", "C/N0")
                is_zero_sym = p_mode in ("ENU Error", "Body-Frame Error", "Residuals", "Per-PRN Residuals")
                span = max(1e-9, ylim[1] - ylim[0])
                if is_zero_sym and abs(ylim[0] + ylim[1]) < 0.2 * span:
                    new_half = span * factor / 2.0
                    ax.set_ylim(-new_half, new_half)
                else:
                    ry = span * factor
                    fy = (event.ydata - ylim[0]) / span
                    new_y0 = event.ydata - ry * fy
                    new_y1 = event.ydata + ry * (1 - fy)
                    if is_non_neg:
                        new_y0 = max(0.0, new_y0)
                        if new_y1 <= new_y0:
                            new_y1 = new_y0 + 1.0
                    ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def on_mouse_down(self,event):
        if event.inaxes is None or getattr(self.toolbar,"mode",""):return
        if getattr(event, "dblclick", False) and event.inaxes in self.plot_axes:
            if self.mode.get() == "Dashboard":
                clicked_idx = self.plot_axes.index(event.inaxes)
                self.focused_dashboard_subplot = None if self.focused_dashboard_subplot == clicked_idx else clicked_idx
                self.redraw(); return
            elif self.mode.get() == "Multi-Compare":
                clicked_idx = self.plot_axes.index(event.inaxes)
                self.focused_compare_subplot = None if self.focused_compare_subplot == clicked_idx else clicked_idx
                self.redraw(); return
            elif self.mode.get() == "Custom Analysis":
                panel_index = self.selected_custom_panel
                self.focused_custom_panel = None if self.focused_custom_panel == panel_index else panel_index
                self.redraw(); return
        if self.mode.get()=="Dashboard" and event.inaxes in self.plot_axes:
            self.selected_dashboard_index=self.plot_axes.index(event.inaxes)
            for i,ax in enumerate(self.plot_axes):
                for spine in ax.spines.values():
                    spine.set_edgecolor("#168aad" if i==self.selected_dashboard_index else "#b9c0c8")
                    spine.set_linewidth(1.8 if i==self.selected_dashboard_index else .75)
            self.canvas.draw_idle()
        elif self.mode.get()=="Multi-Compare" and event.inaxes in self.plot_axes:
            self.selected_compare_subplot=self.plot_axes.index(event.inaxes)
            for i,ax in enumerate(self.plot_axes):
                for spine in ax.spines.values():
                    spine.set_edgecolor("#168aad" if i==self.selected_compare_subplot else "#b9c0c8")
                    spine.set_linewidth(1.8 if i==self.selected_compare_subplot else .75)
            self.canvas.draw_idle()
        elif self.mode.get()=="Custom Analysis" and event.inaxes in self.plot_axes:
            visible_index=self.plot_axes.index(event.inaxes)
            self.selected_custom_panel=(self.visible_panel_indices[visible_index]
                                         if visible_index<len(self.visible_panel_indices) else visible_index)
            for i,ax in enumerate(self.plot_axes):
                panel_index=self.visible_panel_indices[i] if i<len(self.visible_panel_indices) else i
                for spine in ax.spines.values():
                    spine.set_edgecolor("#168aad" if panel_index==self.selected_custom_panel else "#b9c0c8")
                    spine.set_linewidth(1.8 if panel_index==self.selected_custom_panel else .75)
            self.canvas.draw_idle()
        if event.key=="shift" and event.button==1 and event.inaxes in self.plot_axes and event.xdata is not None:
            self._shift_drag_ax=event.inaxes;self._shift_drag_start=float(event.xdata)
            self._shift_drag_patch=event.inaxes.axvspan(self._shift_drag_start,self._shift_drag_start,color="#e07a2d",alpha=.35,zorder=20)
            self.canvas.draw_idle();return
        if event.button==1:
            if event.inaxes in self.track_axes:
                idx,distance=self._nearest_track_epoch(event)
                if distance<=20:
                    self._drag_epoch=True;self.slider.set(idx);self._press_state=None
                    self.canvas.get_tk_widget().configure(cursor="hand2");return
        if event.button in (1,3):
            self._press_state={"ax":event.inaxes,"button":event.button,"x":event.x,"y":event.y,
                "xdata":event.xdata,"ydata":event.ydata,"xlim":event.inaxes.get_xlim(),
                "ylim":event.inaxes.get_ylim(),"moved":False}

    def on_mouse_move(self,event):
        if self._shift_drag_patch is not None and self._shift_drag_ax is not None and event.inaxes is self._shift_drag_ax and event.xdata is not None:
            x0=self._shift_drag_start;x1=float(event.xdata);lo,hi=sorted((x0,x1))
            self._shift_drag_patch.set_xy([[lo,0],[lo,1],[hi,1],[hi,0],[lo,0]])
            self.canvas.draw_idle();return
        if self._drag_epoch:
            if event.inaxes in self.track_axes:
                idx,_=self._nearest_track_epoch(event);self.slider.set(idx)
            return
        if self._press_state:
            press=self._press_state
            if abs(event.x-press["x"])>3 or abs(event.y-press["y"])>3:press["moved"]=True
            if not press["moved"]:return
            if press["button"]==1:
                ax=press["ax"];x0,y0=press["x"],press["y"];xlim0,ylim0=press["xlim"],press["ylim"]
                mode=self.zoom_axis.get();dx=-(event.x-x0)*(xlim0[1]-xlim0[0])/max(ax.bbox.width,1);dy=-(event.y-y0)*(ylim0[1]-ylim0[0])/max(ax.bbox.height,1)
                if mode in ("Both axes","X axis"):ax.set_xlim(xlim0[0]+dx,xlim0[1]+dx)
                if mode in ("Both axes","Y axis"):ax.set_ylim(ylim0[0]+dy,ylim0[1]+dy)
                self.canvas.draw_idle()
            return
        # If hover syncing is disabled, do not scrub time series
        if not self.follow_mouse.get():return
        hover_mode=self.mode.get()
        if self.mode.get()=="Custom Analysis" and event.inaxes in self.plot_axes:
            visible_index=self.plot_axes.index(event.inaxes)
            panel_index=self.visible_panel_indices[visible_index] if visible_index<len(self.visible_panel_indices) else visible_index
            hover_mode=self.custom_modes[panel_index]
        if (event.inaxes in self.plot_axes and event.xdata is not None and not getattr(self.toolbar,"mode","")
                and hover_mode not in (None,"Gnd Trk", "Skyplot", "Innovation / NIS")):
            times=self._pos_times
            idx=int(np.clip(np.searchsorted(times,event.xdata),0,len(times)-1))
            if idx and abs(times[idx-1]-event.xdata)<abs(times[idx]-event.xdata):idx-=1
            if idx!=int(self.slider.get()):
                self.slider.set(idx)

    def on_mouse_up(self,event):
        if self._shift_drag_patch is not None:
            x0=self._shift_drag_start;x1=float(event.xdata) if event.xdata is not None else x0
            try:self._shift_drag_patch.remove()
            except Exception:pass
            self._shift_drag_patch=None;self._shift_drag_ax=None;self._shift_drag_start=None
            lo,hi=sorted((x0,x1));times=self._pos_times
            if len(times)>0 and (hi-lo)>1e-4:
                idx_lo=int(np.clip(np.searchsorted(times,lo),0,len(times)-1))
                idx_hi=int(np.clip(np.searchsorted(times,hi),0,len(times)-1))
                if idx_hi>idx_lo:
                    number=len(self.segments)+1;color=self.segment_palette[(number-1)%len(self.segment_palette)]
                    self.segments.append({"start":idx_lo,"end":idx_hi,"name":f"Segment {number}","color":color})
                    self._refresh_segment_list();self._refresh_segment_overlays()
                    self._auto_save_segments()
                    st = float(self.data["pos"].iloc[idx_lo].tow)
                    et = float(self.data["pos"].iloc[idx_hi].tow)
                    d_res = diag.diagnose_anomaly_segment(self.data["pos"], self.data["sat"], st, et)
                    self.segment_status.configure(text=f"Shift-drag 标记 Segment {number}: 【{d_res['primary_cause']}】{d_res['summary']} | 建议: {d_res['recommendation']}")
            self.canvas.draw_idle();return
        if self._drag_epoch:
            self._drag_epoch=False;self.canvas.get_tk_widget().configure(cursor="");return
        press=self._press_state;self._press_state=None
        self.canvas.get_tk_widget().configure(cursor="")
        if not press:return
        ax=press["ax"]
        if not press["moved"]:
            if event.button == 3:
                target_ax = event.inaxes if (event.inaxes is not None and event.inaxes in self.plot_axes) else ax
                if target_ax in self.plot_axes:
                    self._show_subplot_context_menu(target_ax, event)
                    return
                else:
                    self._restore_axis(ax); return

            if self.click_to_maximize.get():
                if self.mode.get()=="Dashboard":
                    if event.button==1 and event.inaxes in self.plot_axes:
                        clicked_idx=self.plot_axes.index(event.inaxes)
                        if self.focused_dashboard_subplot!=clicked_idx:
                            self.focused_dashboard_subplot=clicked_idx
                            self.redraw()
                        return
                elif self.mode.get()=="Multi-Compare":
                    if event.button==1 and event.inaxes in self.plot_axes:
                        clicked_idx=self.plot_axes.index(event.inaxes)
                        if self.focused_compare_subplot!=clicked_idx:
                            self.focused_compare_subplot=clicked_idx
                            self.redraw()
                        return
                elif self.mode.get()=="Custom Analysis":
                    if event.button==1 and event.inaxes in self.plot_axes:
                        panel_index=self.selected_custom_panel
                        if self.focused_custom_panel!=panel_index:
                            self.focused_custom_panel=panel_index
                            self.redraw()
                        return

            if event.button==1 and event.inaxes is ax and event.xdata is not None and event.ydata is not None:
                if self.mode.get() not in ("Dashboard","Custom Analysis","Multi-Compare"):
                    self._zoom_at_click(ax,event.xdata,event.ydata,.72)
            return
        if event.button==3 and event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            mode=self.zoom_axis.get()
            if mode in ("Both axes","X axis") and press["xdata"] is not None and event.xdata!=press["xdata"]:ax.set_xlim(sorted([press["xdata"],event.xdata]))
            if mode in ("Both axes","Y axis") and press["ydata"] is not None and event.ydata!=press["ydata"]:ax.set_ylim(sorted([press["ydata"],event.ydata]))
            self.canvas.draw_idle()

    def _zoom_at_click(self,ax,x,y,factor):
        mode=self.zoom_axis.get();xl=ax.get_xlim();yl=ax.get_ylim()
        if mode in ("Both axes","X axis"):
            width=(xl[1]-xl[0])*factor;fraction=(x-xl[0])/(xl[1]-xl[0]);ax.set_xlim(x-width*fraction,x+width*(1-fraction))
        if mode in ("Both axes","Y axis"):
            if getattr(ax,"name","")=="polar":
                height=(yl[1]-yl[0])*factor;fraction=(y-yl[0])/(yl[1]-yl[0]);ax.set_ylim(y-height*fraction,y+height*(1-fraction))
            else:
                p_mode = getattr(ax, "_panel_mode", "")
                is_non_neg = p_mode in ("AR Ratio", "DOP", "Directional DOP", "Satellites", "Residual Triad", "C/N0")
                is_zero_sym = p_mode in ("ENU Error", "Body-Frame Error", "Residuals", "Per-PRN Residuals")
                span = max(1e-9, yl[1] - yl[0])
                if is_zero_sym and abs(yl[0] + yl[1]) < 0.2 * span:
                    new_half = span * factor / 2.0
                    ax.set_ylim(-new_half, new_half)
                else:
                    height = span * factor
                    fraction = (y - yl[0]) / span
                    new_y0 = y - height * fraction
                    new_y1 = y + height * (1 - fraction)
                    if is_non_neg:
                        new_y0 = max(0.0, new_y0)
                        if new_y1 <= new_y0:
                            new_y1 = new_y0 + 1.0
                    ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _restore_axis(self,ax):
        limits=getattr(self,"_home_limits",{}).get(id(ax))
        if limits:
            ax.set_xlim(limits[0]);ax.set_ylim(limits[1]);self.canvas.draw_idle()

    def _show_subplot_context_menu(self, ax, event):
        if not self.data:
            return
        menu = tk.Menu(self.root, tearoff=False)
        current_mode = self._get_subplot_mode_name(ax)
        is_maximized = self._is_subplot_maximized(ax)

        menu.add_command(label=f"📌 当前子图: {current_mode}", state="disabled")
        menu.add_separator()

        if is_maximized:
            menu.add_command(label="🗗 恢复多图显示 (Restore Grid)", command=lambda: self._toggle_maximize_subplot(ax))
        else:
            menu.add_command(label="🗖 放大当前子图 (Maximize Subplot)", command=lambda: self._toggle_maximize_subplot(ax))

        # Categorized Submenu for replacing subplots with 20 plot modes
        replace_menu = tk.Menu(menu, tearoff=False)
        for cat_name, modes in PLOT_CATEGORIES.items():
            cat_submenu = tk.Menu(replace_menu, tearoff=False)
            for m in modes:
                cat_submenu.add_command(label=MODE_BILINGUAL.get(m, m), command=lambda new_m=m: self._replace_subplot(ax, new_m))
            replace_menu.add_cascade(label=cat_name, menu=cat_submenu)
        menu.add_cascade(label="⟳ 替换此子图为 (Replace Subplot with…)", menu=replace_menu)

        menu.add_separator()
        menu.add_command(label="单独导出此子图 (300 DPI PNG)…", command=lambda: self._export_single_subplot(ax))
        menu.add_command(label="↺ 重置此子图视野 (Reset Zoom)", command=lambda: self._restore_axis(ax))
        is_robust = getattr(ax, "_robust_mode", getattr(self, "robust_view", type("V", (), {"get": lambda s: True})()).get())
        if hasattr(ax, "_robust_ylim") or hasattr(ax, "_full_ylim"):
            lbl = "🎯 切换为全量极值视图 (Full Range)" if is_robust else "🎯 切换为稳健主体视图 (95% Focus)"
            menu.add_command(label=lbl, command=lambda a=ax: self._toggle_subplot_robust_view(a))

        mode = self.mode.get()
        if mode == "Custom Analysis":
            idx = self.plot_axes.index(ax) if ax in self.plot_axes else self.selected_custom_panel
            panel_idx = self.focused_custom_panel if self.focused_custom_panel is not None else (
                self.visible_panel_indices[idx] if idx < len(self.visible_panel_indices) else idx
            )
            menu.add_separator()
            if panel_idx > 0 and len(self.custom_panels) > 1:
                menu.add_command(label="★ 设为当前主图 (Promote to Primary)", command=lambda: self._promote_custom_panel_to_primary(ax))
            if len(self.custom_panels) > 1:
                swap_m = tk.Menu(menu, tearoff=False)
                for other_idx, p in enumerate(self.custom_panels):
                    if other_idx != panel_idx:
                        swap_m.add_command(
                            label=f"槽位 {other_idx+1}: {p.get('mode', 'Empty')}",
                            command=lambda t=other_idx: self._swap_custom_panels(panel_idx, t)
                        )
                menu.add_cascade(label="⇄ 与其他子图互换 (Swap Position)", menu=swap_m)

            preset_m = tk.Menu(menu, tearoff=False)
            for p_name in ["Auto Grid (自适应等分)", "Focus Left (左主右副)", "Vertical Stack (纵向流)", "Focus Top (上主下副)"]:
                preset_m.add_radiobutton(label=p_name, variable=self.custom_preset, value=p_name, command=self._on_custom_preset_changed)
            menu.add_cascade(label="📐 切换布局预设 (Layout Presets)", menu=preset_m)

        if mode in ("Dashboard", "Custom Analysis"):
            menu.add_separator()
            cur_idx = self.selected_dashboard_index if mode == "Dashboard" else (
                self.focused_custom_panel if self.focused_custom_panel is not None else (
                    self.visible_panel_indices[self.plot_axes.index(ax)] if ax in self.plot_axes and self.plot_axes.index(ax) < len(self.visible_panel_indices) else self.selected_custom_panel
                )
            )
            add_after_m = tk.Menu(menu, tearoff=False)
            for cat_name, m_list in PLOT_CATEGORIES.items():
                cat_menu = tk.Menu(add_after_m, tearoff=False)
                for m in m_list:
                    cat_menu.add_command(
                        label=m,
                        command=lambda name=m: self._insert_or_append_subplot(name, cur_idx)
                    )
                add_after_m.add_cascade(label=cat_name, menu=cat_menu)
            menu.add_cascade(label="➕ 在此后插入新子图 (Insert Subplot After…)", menu=add_after_m)

            add_append_m = tk.Menu(menu, tearoff=False)
            for cat_name, m_list in PLOT_CATEGORIES.items():
                cat_menu = tk.Menu(add_append_m, tearoff=False)
                for m in m_list:
                    cat_menu.add_command(
                        label=m,
                        command=lambda name=m: self._insert_or_append_subplot(name, None)
                    )
                add_append_m.add_cascade(label=cat_name, menu=cat_menu)
            menu.add_cascade(label="➕ 追加新子图到末尾 (Append Subplot…)", menu=add_append_m)

            menu.add_separator()
            menu.add_command(label="🗑️ 删除此子图 (Delete Subplot)", command=lambda: self._delete_subplot(ax))

        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def _get_subplot_mode_name(self, ax):
        if ax not in self.plot_axes:
            return self.mode.get()
        mode = self.mode.get()
        if mode == "Dashboard":
            if self.focused_dashboard_subplot is not None:
                idx = self.focused_dashboard_subplot
            else:
                idx = self.plot_axes.index(ax)
            if 0 <= idx < len(self.dashboard_modes):
                return self.dashboard_modes[idx]
            return "Dashboard"
        elif mode == "Custom Analysis":
            idx = self.plot_axes.index(ax)
            panel_idx = self.focused_custom_panel if self.focused_custom_panel is not None else (
                self.visible_panel_indices[idx] if idx < len(self.visible_panel_indices) else idx
            )
            if 0 <= panel_idx < len(self.custom_panels):
                return self.custom_panels[panel_idx].get("mode") or "Empty Slot"
            return "Custom Analysis"
        elif mode == "Multi-Compare":
            idx = self.focused_compare_subplot if self.focused_compare_subplot is not None else self.plot_axes.index(ax)
            if idx == 0:
                return "Multi-Compare Trajectory"
            return "Multi-Compare Horizontal Error"
        else:
            return mode

    def _is_subplot_maximized(self, ax):
        mode = self.mode.get()
        if mode == "Dashboard":
            return self.focused_dashboard_subplot is not None
        elif mode == "Custom Analysis":
            return self.focused_custom_panel is not None
        elif mode == "Multi-Compare":
            return self.focused_compare_subplot is not None
        return False

    def _toggle_maximize_subplot(self, ax):
        mode = self.mode.get()
        if mode == "Dashboard":
            if self.focused_dashboard_subplot is not None:
                self.focused_dashboard_subplot = None
            else:
                if ax in self.plot_axes:
                    self.focused_dashboard_subplot = self.plot_axes.index(ax)
                elif self.dashboard_modes:
                    self.focused_dashboard_subplot = min(max(0, self.selected_dashboard_index), len(self.dashboard_modes) - 1)
            self.redraw()
        elif mode == "Custom Analysis":
            if self.focused_custom_panel is not None:
                self.focused_custom_panel = None
            else:
                if ax in self.plot_axes:
                    idx = self.plot_axes.index(ax)
                    self.focused_custom_panel = (
                        self.visible_panel_indices[idx] if idx < len(self.visible_panel_indices) else idx
                    )
                elif self.custom_panels:
                    self.focused_custom_panel = min(max(0, self.selected_custom_panel), len(self.custom_panels) - 1)
            self.redraw()
        elif mode == "Multi-Compare":
            if self.focused_compare_subplot is not None:
                self.focused_compare_subplot = None
            else:
                if ax in self.plot_axes:
                    self.focused_compare_subplot = self.plot_axes.index(ax)
                else:
                    self.focused_compare_subplot = self.selected_compare_subplot
            self.redraw()
        else:
            self.fit_view()

    def _replace_subplot(self, ax, new_mode):
        mode = self.mode.get()
        if mode == "Dashboard":
            if self.focused_dashboard_subplot is not None:
                idx = self.focused_dashboard_subplot
            elif ax in self.plot_axes:
                idx = self.plot_axes.index(ax)
            else:
                idx = self.selected_dashboard_index
            if 0 <= idx < len(self.dashboard_modes):
                self.dashboard_modes[idx] = new_mode
                self.selected_dashboard_index = idx
                self.segment_status.configure(text=f"子图 {idx+1} 已替换为: {new_mode}")
                self.redraw()
        elif mode == "Custom Analysis":
            if not self.custom_panels:
                self.custom_panels = self._initial_custom_panels([new_mode])
                self.custom_modes = [new_mode]
                self.selected_custom_panel = 0
                self.segment_status.configure(text=f"自定义分区 1 已设置为: {new_mode}")
                self.redraw()
                return
            if ax in self.plot_axes:
                idx = self.plot_axes.index(ax)
                panel_idx = self.focused_custom_panel if self.focused_custom_panel is not None else (
                    self.visible_panel_indices[idx] if idx < len(self.visible_panel_indices) else idx
                )
            else:
                panel_idx = self.selected_custom_panel
            if 0 <= panel_idx < len(self.custom_panels):
                self.custom_panels[panel_idx]["mode"] = new_mode
                self.custom_modes = [p["mode"] for p in self.custom_panels]
                self.selected_custom_panel = panel_idx
                self.segment_status.configure(text=f"自定义分区 {panel_idx+1} 已替换为: {new_mode}")
                self.redraw()
        else:
            self.set_mode(new_mode)

    def _delete_subplot(self, ax):
        mode = self.mode.get()
        if mode == "Dashboard":
            if self.focused_dashboard_subplot is not None:
                self.selected_dashboard_index = self.focused_dashboard_subplot
            elif ax in self.plot_axes:
                self.selected_dashboard_index = self.plot_axes.index(ax)
            self.delete_selected_subplot()
        elif mode == "Custom Analysis":
            if ax in self.plot_axes:
                idx = self.plot_axes.index(ax)
                self.selected_custom_panel = self.focused_custom_panel if self.focused_custom_panel is not None else (
                    self.visible_panel_indices[idx] if idx < len(self.visible_panel_indices) else idx
                )
            self.delete_selected_subplot()
        else:
            self.delete_selected_subplot()

    def _export_single_subplot(self, ax):
        if self.data is None:
            messagebox.showwarning("提示", "未加载任何解算数据。")
            return
        mode_name = self._get_subplot_mode_name(ax)
        clean_name = mode_name.lower().replace(" ", "_").replace("/", "_")
        path = filedialog.asksaveasfilename(
            title=f"单独导出子图 [{mode_name}]",
            defaultextension=".png",
            initialfile=f"subplot_{clean_name}.png",
            filetypes=[("PNG 图像 (*.png)", "*.png"), ("PDF 文档 (*.pdf)", "*.pdf"), ("SVG 矢量图 (*.svg)", "*.svg")]
        )
        if not path:
            return
        try:
            temp_fig = plt.Figure(figsize=(10, 6), dpi=300, facecolor="white")
            if self.mode.get() == "Multi-Compare":
                temp_ax = temp_fig.add_subplot(111)
                visible_sols = [s for s in self.multi_solutions if s["visible"].get()]
                idx = self.focused_compare_subplot if self.focused_compare_subplot is not None else (
                    self.plot_axes.index(ax) if ax in self.plot_axes else 0
                )
                if idx == 0:
                    self._draw_compare_track(temp_ax, visible_sols)
                else:
                    self._draw_compare_error(temp_ax, visible_sols)
                temp_ax.grid(self.grid_on.get(), alpha=0.3)
                temp_fig.tight_layout()
            else:
                pos_sub = self._filtered()
                sat_sub = self.data.get("sat")
                metrics = diag.compute_segment_metrics(pos_sub, sat_sub)
                self.export_render_mode(temp_fig, mode_name, pos_sub, sat_sub, metrics, with_hud=True)

            temp_fig.savefig(path, dpi=300 if path.lower().endswith(".png") else None, bbox_inches="tight")
            self.segment_status.configure(text=f"子图 [{mode_name}] 已成功导出至: {os.path.basename(path)}")
            messagebox.showinfo("导出成功", f"子图 [{mode_name}] 已成功以 300 DPI 导出至:\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", f"导出子图时发生异常:\n{str(exc)}")

    def _nearest_track_epoch(self,event):
        p=self.data["pos"]
        points=event.inaxes.transData.transform(np.c_[p.E.to_numpy(),p.N.to_numpy()])
        distance=np.hypot(points[:,0]-event.x,points[:,1]-event.y)
        index=int(np.argmin(distance))
        return index,float(distance[index])

    def _on_style_changed(self):
        scope = self.line_scope.get()
        style_dict = {
            "line_style": self.line_style.get(),
            "line_width": float(self.line_width.get()),
            "marker_style": self.marker_style.get(),
            "marker_size": float(self.marker_size.get()),
        }
        if scope == "Current Subplot":
            current_mode = self.mode.get()
            if current_mode == "Dashboard":
                current_mode = self.dashboard_modes[self.selected_dashboard_index] if self.dashboard_modes else "Dashboard"
            elif current_mode == "Custom Analysis":
                current_mode = self.custom_modes[self.selected_custom_panel] if self.custom_modes else "Custom Analysis"
            self.curve_styles[current_mode] = style_dict
        else:
            self.curve_styles["global"] = style_dict
        self.redraw()

    def _reset_curve_styles(self):
        self.curve_styles.clear()
        self.line_style.set("Solid (―)")
        self.line_width.set(1.2)
        self.marker_style.set("None")
        self.marker_size.set(3.0)
        self.redraw()

    def _get_line_kwargs(self, mode_name=None, default_ls="-", default_lw=1.0):
        style = self.curve_styles.get(mode_name) if mode_name else None
        if not style:
            style = self.curve_styles.get("global")

        ls_map = {
            "Solid (―)": "-",
            "Dashed (---)": "--",
            "Dotted (···)": ":",
            "Dash-Dot (-·-)": "-.",
            "Scatter Only": "None",
            "Line + Marker": "-"
        }
        marker_map = {
            "None": None,
            "Circle (o)": "o",
            "Square (s)": "s",
            "Point (.)": ".",
            "Cross (+)": "+"
        }

        ls_str = style["line_style"] if style and "line_style" in style else self.line_style.get()
        lw = float(style["line_width"]) if style and "line_width" in style else float(self.line_width.get())
        m_str = style["marker_style"] if style and "marker_style" in style else self.marker_style.get()
        ms = float(style["marker_size"]) if style and "marker_size" in style else float(self.marker_size.get())

        ls = ls_map.get(ls_str, default_ls)
        marker = marker_map.get(m_str, None)
        if ls_str == "Scatter Only" and marker is None:
            marker = "o"

        kw = {"ls": ls, "lw": lw}
        if marker:
            kw["marker"] = marker
            kw["ms"] = ms
        return kw

    def _on_basemap_changed(self, event=None):
        self.cached_basemap = None
        self.redraw()

    def _apply_basemap(self, ax, full):
        provider = self.basemap_provider.get()
        if provider == "None" or not hasattr(full, "lat") or not full.lat.notna().any():
            return
        try:
            cache_key = (provider, str(self.pos_path))
            if self.cached_basemap and self.cached_basemap[0] == cache_key:
                img, extent = self.cached_basemap[1], self.cached_basemap[2]
            else:
                lat_min, lat_max = float(full.lat.min()), float(full.lat.max())
                lon_min, lon_max = float(full.lon.min()), float(full.lon.max())
                origin_lat = float(full.lat.iloc[0])
                origin_lon = float(full.lon.iloc[0])
                origin_h = float(full.h.iloc[0]) if "h" in full else 0.0
                img, extent = mt.get_stitched_tiles_enu(
                    lat_min, lat_max, lon_min, lon_max,
                    origin_lat, origin_lon, origin_h,
                    provider=provider, max_tiles=16, margin=0.20
                )
                self.cached_basemap = (cache_key, img, extent)

            if img is not None and extent is not None:
                ax.imshow(img, extent=extent, origin="upper", zorder=1, alpha=self.basemap_alpha.get())
                ax.set_facecolor("#151515" if "Dark" in provider else "#f0f0f0")
        except Exception as e:
            print(f"Basemap notice: {e}")

    def _get_ground_truth_df(self) -> pd.DataFrame | None:
        """Extract or reconstruct aligned ground-truth dataframe with (week, tow, lat, lon, h)."""
        if self.data is None or not self.data.get("has_truth"):
            return None
        ref = self.data.get("reference")
        pos = self.data.get("pos")
        if pos is None or pos.empty:
            return None

        # 1. Primary: If reference dataframe loaded
        if isinstance(ref, pd.DataFrame) and not ref.empty:
            tow_col = "GPS TOW (s)" if "GPS TOW (s)" in ref.columns else ("tow" if "tow" in ref.columns else None)
            if tow_col:
                t_min = float(pos.tow.min()) - 15.0
                t_max = float(pos.tow.max()) + 15.0
                sub = ref[(ref[tow_col] >= t_min) & (ref[tow_col] <= t_max)].copy()
                if sub.empty:
                    sub = ref.copy()
            else:
                sub = ref.copy()

            lat_col = "Latitude (deg)" if "Latitude (deg)" in sub.columns else ("lat" if "lat" in sub.columns else None)
            lon_col = "Longitude (deg)" if "Longitude (deg)" in sub.columns else ("lon" if "lon" in sub.columns else None)
            if lat_col and lon_col:
                lat = sub[lat_col].to_numpy(dtype=float)
                lon = sub[lon_col].to_numpy(dtype=float)
                if "Ellipsoid Height (m)" in sub.columns:
                    h = sub["Ellipsoid Height (m)"].to_numpy(dtype=float)
                elif "h" in sub.columns:
                    h = sub["h"].to_numpy(dtype=float)
                elif all(c in sub.columns for c in ["ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"]):
                    _, _, h = diag._ecef_to_geodetic(
                        sub["ECEF X (m)"].to_numpy(dtype=float),
                        sub["ECEF Y (m)"].to_numpy(dtype=float),
                        sub["ECEF Z (m)"].to_numpy(dtype=float)
                    )
                else:
                    h = np.zeros(len(sub), dtype=float)

                week = sub["GPS Week"].to_numpy() if "GPS Week" in sub.columns else np.full(len(sub), int(pos.iloc[0].get("week", 2000)))
                tow = sub[tow_col].to_numpy(dtype=float) if tow_col else np.arange(len(sub), dtype=float)
                df = pd.DataFrame({"week": week, "tow": tow, "lat": lat, "lon": lon, "h": h, "q": np.zeros(len(sub), dtype=int)})
                return df.dropna(subset=["lat", "lon"])

        # 2. Fallback from pos ENU gt_E, gt_N, gt_U
        if "gt_E" in pos.columns and "gt_N" in pos.columns:
            valid_gt = pos.dropna(subset=["gt_E", "gt_N"]).copy()
            if not valid_gt.empty:
                x0, y0, z0 = pos.iloc[0].x, pos.iloc[0].y, pos.iloc[0].z
                lat0 = float(pos.iloc[0]["lat"]) if np.isfinite(pos.iloc[0]["lat"]) else 0.0
                lon0 = float(pos.iloc[0]["lon"]) if np.isfinite(pos.iloc[0]["lon"]) else 0.0
                p_rad, l_rad = np.deg2rad([lat0, lon0])
                rot = np.array([[-np.sin(l_rad), np.cos(l_rad), 0],
                                [-np.sin(p_rad)*np.cos(l_rad), -np.sin(p_rad)*np.sin(l_rad), np.cos(p_rad)],
                                [np.cos(p_rad)*np.cos(l_rad), np.cos(p_rad)*np.sin(l_rad), np.sin(p_rad)]])
                enu = np.column_stack([valid_gt.gt_E, valid_gt.gt_N, valid_gt.get("gt_U", np.zeros(len(valid_gt)))])
                d_xyz = enu @ rot
                gt_xyz = d_xyz + np.array([x0, y0, z0])
                gt_lat, gt_lon, gt_h = diag._ecef_to_geodetic(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2])
                df = pd.DataFrame({
                    "week": valid_gt.week.to_numpy(),
                    "tow": valid_gt.tow.to_numpy(),
                    "lat": gt_lat,
                    "lon": gt_lon,
                    "h": gt_h,
                    "q": np.zeros(len(valid_gt), dtype=int)
                })
                return df.dropna(subset=["lat", "lon"])

        return None

    def export_kml(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先打开 .pos 解算文件。")
            return
        base = self.pos_path.stem if self.pos_path else "rtklib_solution"
        path = filedialog.asksaveasfilename(
            title="导出 Google Earth 3D KML",
            defaultextension=".kml",
            initialfile=f"{base}_3d.kml",
            filetypes=[("Google Earth KML files", "*.kml"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            mt.export_kml_3d(self.data["pos"], path, segments=self.segments, pos_name=self.pos_path.name if self.pos_path else base)
            truth_df = self._get_ground_truth_df()
            truth_path = None
            if truth_df is not None and not truth_df.empty:
                p_obj = Path(path)
                stem = p_obj.stem
                truth_stem = stem[:-3] + "_truth_3d" if stem.endswith("_3d") else f"{stem}_truth_3d"
                truth_path = p_obj.with_name(f"{truth_stem}.kml")
                mt.export_truth_kml_3d(truth_df, truth_path, truth_name=f"{base} (Ground Truth Reference)")

            if truth_path and truth_path.exists():
                msg = (
                    f"Google Earth 3D KML 文件已成功导出（含独立真值文件）：\n"
                    f"1. 解算轨迹: {path}\n"
                    f"2. 真值轨迹: {truth_path}\n\n"
                    f"【提示】：真值文件采用白色高亮 3D 幕墙与同步时间轴，在 Google Earth 中同时加载即可并排对比真实轨迹与解算轨迹！\n\n"
                    f"是否立即在 Google Earth 中同时打开？"
                )
                if messagebox.askyesno("导出成功", msg):
                    self._open_in_google_earth([path, truth_path])
            else:
                msg = (
                    f"Google Earth 3D KML 文件已成功导出至：\n{path}\n\n"
                    f"【提示】：此文件包含 3D 高程拉伸与时间轴，专用于「Google Earth (谷歌地球)」！\n"
                    f"若在网页版「Google 我的地图 (My Maps)」中使用，请选择【My Maps 2D…】以防导入报错。\n\n"
                    f"是否立即在 Google Earth 中打开？"
                )
                if messagebox.askyesno("导出成功", msg):
                    self._open_in_google_earth([path])
        except Exception as exc:
            messagebox.showerror("导出 KML 失败", str(exc))

    def _open_in_google_earth(self, paths):
        import shutil, subprocess
        if isinstance(paths, (str, Path)):
            path_list = [Path(paths)]
        else:
            path_list = [Path(p) for p in paths]
        if not path_list:
            return False

        candidates = [
            r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe",
            r"C:\Program Files (x86)\Google\Google Earth Pro\client\googleearth.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Google Earth Pro\client\googleearth.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), r"Google\Google Earth Pro\client\googleearth.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", ""), r"Google\Google Earth Pro\client\googleearth.exe"),
        ]
        ge_exe = shutil.which("googleearth.exe") or shutil.which("googleearth")
        if not ge_exe:
            for c in candidates:
                if c and os.path.exists(c):
                    ge_exe = c
                    break
        if ge_exe:
            try:
                cmd = [ge_exe] + [str(p) for p in path_list]
                subprocess.Popen(cmd)
                return True
            except Exception:
                pass
        try:
            for p in path_list:
                os.startfile(str(p))
            return True
        except Exception:
            open_folder = messagebox.askyesno(
                "未检测到 Google Earth 客户端",
                f"系统未检测到「Google Earth (谷歌地球)」客户端安装路径，且 Windows 尚未为 .kml 文件关联默认打开程序。\n\n"
                f"已生成 KML 文件：\n" + "\n".join(str(p) for p in path_list) + "\n\n"
                f"是否立即在文件管理器中打开该文件所在文件夹？\n(可直接拖拽至 Google Earth Pro 或浏览器 Google Earth Web)"
            )
            if open_folder:
                try:
                    subprocess.run(["explorer", f"/select,{str(path_list[0])}"])
                except Exception:
                    pass
            return False

    def export_kml_2d(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先打开 .pos 解算文件。")
            return
        base = self.pos_path.stem if self.pos_path else "rtklib_solution"
        path = filedialog.asksaveasfilename(
            title="导出 Google My Maps 2D 兼容 KML (我的地图)",
            defaultextension=".kml",
            initialfile=f"{base}_2d_mymaps.kml",
            filetypes=[("Google My Maps KML files", "*.kml"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            mt.export_kml_2d_mymaps(self.data["pos"], path, pos_name=self.pos_path.name if self.pos_path else base)
            truth_df = self._get_ground_truth_df()
            truth_path = None
            if truth_df is not None and not truth_df.empty:
                p_obj = Path(path)
                stem = p_obj.stem
                truth_stem = stem[:-10] + "_truth_2d" if stem.endswith("_2d_mymaps") else f"{stem}_truth_2d"
                truth_path = p_obj.with_name(f"{truth_stem}.kml")
                mt.export_truth_kml_2d_mymaps(truth_df, truth_path, truth_name=f"{base} (Ground Truth Reference)")

            if truth_path and truth_path.exists():
                msg = (
                    f"Google My Maps 2D 兼容 KML 文件已成功导出（含独立真值文件）：\n"
                    f"1. 解算轨迹: {path}\n"
                    f"2. 真值轨迹: {truth_path}\n\n"
                    f"顶点数量均已控制在 8,000 点以内，现在上传到「Google 我的地图 (maps.google.com/d)」即可 100% 成功解析！"
                )
            else:
                msg = (
                    f"Google My Maps 2D 兼容 KML 文件已成功导出至：\n{path}\n\n"
                    f"该文件已做如下优化：\n"
                    f"1. 移除了 My Maps 不支持的 3D 拉伸与时序标签；\n"
                    f"2. 顶点数量自动抽稀控制在 8,000 点内（规避 My Maps 10,000 点上限）；\n"
                    f"3. 保持 FIX/FLOAT/SINGLE 状态彩色分段与起止点。\n\n"
                    f"现在上传到「Google 我的地图 (maps.google.com/d)」即可 100% 成功解析！"
                )
            messagebox.showinfo("导出成功", msg)
        except Exception as exc:
            messagebox.showerror("导出 2D KML 失败", str(exc))

    def export_geojson(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先打开 .pos 解算文件。")
            return
        base = self.pos_path.stem if self.pos_path else "rtklib_solution"
        path = filedialog.asksaveasfilename(
            title="导出标准 GIS GeoJSON",
            defaultextension=".geojson",
            initialfile=f"{base}.geojson",
            filetypes=[("GeoJSON files", "*.geojson"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            mt.export_geojson(self.data["pos"], path)
            messagebox.showinfo("导出成功", f"标准 GIS GeoJSON 要素集已成功导出至：\n{path}\n\n可直接拖入 QGIS 或 ArcGIS 加载分析。")
        except Exception as exc:
            messagebox.showerror("导出 GeoJSON 失败", str(exc))

    def _draw_track_export(self, ax, p, full):
        self._apply_basemap(ax, full)
        ax.set_title("Ground Track / ENU Position — colored by solution status", fontsize=11, weight="bold")
        if self.data["has_truth"]:
            gt_t, gt_e, gt_n = insert_time_gaps(full.time, full.gt_E, full.gt_N)
            ax.plot(gt_e, gt_n, color="#252525", lw=1.1, ls="--", alpha=0.75, label="Ground Truth", zorder=3)
        for q, g in p.groupby("q"):
            label = f"Q={q}: {Q_LABELS.get(int(q), 'Other')}"
            ax.scatter(g.E, g.N, s=7, color=Q_COLORS.get(int(q), "#666"), label=label, alpha=0.85, zorder=4)
        ax.scatter([p.E.iloc[0]], [p.N.iloc[0]], s=65, marker="o", facecolors="#28a745", edgecolors="white", linewidths=1.2, label="Start", zorder=6)
        ax.scatter([p.E.iloc[-1]], [p.N.iloc[-1]], s=65, marker="s", facecolors="#dc3545", edgecolors="white", linewidths=1.2, label="End", zorder=6)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal" if self.equal_scale.get() else "auto")
        ax.legend(loc="upper left", fontsize=8)

    def _draw_sky_export(self, ax, s):
        if s is None or s.empty:
            ax.text(0.5, 0.5, "No Satellite Data", ha="center", va="center", transform=ax.transAxes)
            return
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(90, 0)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"])
        for (prn, con), g in s[s.freq == 1].groupby(["prn", "constellation"]):
            g = self._display_sample(g.sort_values("time"), 400)
            th = np.deg2rad(g.az)
            r = 90 - g.el
            ax.plot(th, r, lw=0.75, alpha=0.6, color=diag.COLORS.get(con, "#666"))
            k = int(np.nanargmax(g.el.to_numpy()))
            ax.text(th.iloc[k], r.iloc[k], prn, fontsize=7, weight="bold")
        ax.set_title("Satellite Skyplot", pad=18, fontsize=12, weight="bold")

    def export_render_mode(self, fig, mode_name, pos_sub, sat_sub, metrics, with_hud=True):
        fig.clear()
        full = pos_sub
        p = pos_sub
        s = sat_sub if sat_sub is not None else self.data["sat"]

        if mode_name == "Dashboard":
            fig.set_size_inches(16, 9)
            axs = fig.subplots(2, 3)
            axs_flat = axs.ravel()
            for i, name in enumerate(["Gnd Trk", "Position Error", "Solution Status", "AR Ratio", "Satellites", "Elevation vs C/N0"]):
                ax = axs_flat[i]
                self._render_dashboard_item(ax, name, p, full, s, is_single=False)
                ax.grid(self.grid_on.get(), alpha=0.3)
            if with_hud:
                diag.render_hud_card(axs_flat[1], metrics, loc="upper right", font_size=8.0)
            fig.suptitle("RTKLIB Solution Dashboard", fontsize=15, weight="bold")
            fig.tight_layout()
            return

        if mode_name == "Skyplot":
            fig.set_size_inches(8, 8)
            ax = fig.add_subplot(111, projection="polar")
            self._draw_sky_export(ax, s)
            if with_hud:
                diag.render_hud_card(ax, metrics, loc="upper right", font_size=8.0)
            fig.tight_layout()
            return

        fig.set_size_inches(10, 6)
        if mode_name == "ENU Position":
            axs = fig.subplots(3, 1, sharex=True)
            self._draw_enu_positions(axs, p)
            self._set_datetime_axis(axs[-1])
            if with_hud:
                diag.render_hud_card(axs[0], metrics, loc="upper right", font_size=8.0)
            fig.tight_layout()
            return
        elif mode_name == "ENU Error":
            if not self.data["has_truth"]:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No Ground Truth Loaded — Position Errors Unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=11)
                return
            axs = fig.subplots(3, 1, sharex=True)
            self._draw_errors(axs, p)
            self._set_datetime_axis(axs[-1])
            if with_hud:
                diag.render_hud_card(axs[0], metrics, loc="upper right", font_size=8.0)
            fig.tight_layout()
            return

        ax = fig.add_subplot(111)
        if mode_name == "Gnd Trk":
            self._draw_track_export(ax, p, full)
        elif mode_name == "Trajectory Error Heatmap":
            self._draw_trajectory_error_heatmap(ax, p, full)
        elif mode_name == "Height":
            kw = self._get_line_kwargs("Height")
            t_gap, u_gap = insert_time_gaps(p.time, p.U)
            ax.plot(t_gap, u_gap, label="RTK U", **kw)
            if self.data["has_truth"]:
                t_gt_gap, gt_u_gap = insert_time_gaps(p.time, p.gt_U)
                ax.plot(t_gt_gap, gt_u_gap, ls="--", color="#cf4b43", lw=kw.get("lw", 1.2), label="Truth U")
            ax.set(title="Height (Up component)", xlabel="Date/time", ylabel="Up (m)")
            ax.legend(loc="best")
            self._set_datetime_axis(ax)
        elif mode_name == "Solution Status":
            t_gap, q_g = insert_time_gaps(p.time, p.q)
            ax.step(t_gap, q_g, where="post", lw=1.2, color="#0B2B52")
            ax.set(title="Solution Status (Q: 1=FIX, 2=FLOAT, 5=SINGLE)", xlabel="Date/time", ylabel="Q")
            self._set_datetime_axis(ax)
        elif mode_name == "AR Ratio":
            self._draw_ar_ratio(ax, p)
            self._set_datetime_axis(ax)
        elif mode_name == "Differential Age":
            self._draw_differential_age(ax, p)
            self._set_datetime_axis(ax)
        elif mode_name == "Satellites":
            self._draw_sats(ax)
            self._set_datetime_axis(ax)
        elif mode_name == "DOP":
            self._draw_dop(ax)
            self._set_datetime_axis(ax)
        elif mode_name in ("C/N0", "C/N₀"):
            self._draw_cn0(ax)
        elif mode_name == "Elevation":
            self._draw_elevation(ax)
            self._set_datetime_axis(ax)
        elif mode_name == "Residuals":
            self._draw_residuals(ax)
            self._set_datetime_axis(ax)
        elif mode_name == "Multipath Proxy":
            self._draw_multipath(ax)
        elif mode_name == "Cycle Slips":
            self._draw_slips(ax)
            self._set_datetime_axis(ax)
        elif mode_name == "Satellite Visibility Gantt":
            self._draw_satellite_visibility_gantt(ax, s)
            self._set_datetime_axis(ax)
        elif mode_name == "Multipath Skyplot":
            fig.set_size_inches(8, 8)
            fig.clear()
            ax = fig.add_subplot(111, projection="polar")
            self._draw_multipath_skyplot(ax)
            if with_hud and metrics and "fix_rate" in metrics:
                diag.render_hud_card(ax, metrics, loc="upper right", font_size=8.0)
            fig.tight_layout()
            return
        elif mode_name == "Per-PRN Residuals":
            fig.set_size_inches(13, 5)
            fig.clear()
            ax = fig.add_subplot(111)
            self._draw_per_prn_residuals(ax)
            if with_hud and metrics and "fix_rate" in metrics:
                diag.render_hud_card(ax, metrics, loc="upper right", font_size=8.0)
            fig.tight_layout()
            return
        elif mode_name == "Covariance":
            self._draw_covariance(ax, p)
            self._set_datetime_axis(ax)
        elif mode_name == "False Fix Analysis":
            self._draw_false_fix(ax, p)
        elif mode_name == "Error CDF":
            self._draw_error_cdf(ax, p)
        elif mode_name == "Error Histogram":
            fig.set_size_inches(13, 4.5)
            fig.clear()
            axs = fig.subplots(1, 3, sharey=True)
            self._draw_error_histogram(axs, p)
            if with_hud:
                diag.render_hud_card(axs[0], metrics, loc="upper right", font_size=8.0)
            fig.tight_layout()
            return
        elif mode_name == "Innovation / NIS":
            self._draw_nis(ax)
        elif mode_name in ("Elevation vs C/N0", "Elevation vs C/N₀"):
            ss = s.query("freq == 1 and snr > 0") if not s.empty else s
            if not ss.empty:
                ax.scatter(ss.el, ss.snr, s=2, alpha=.15)
            ax.set(title="Elevation vs C/N₀", xlabel="Elevation (°)", ylabel="dB-Hz")
        elif mode_name in ("Position Error", "Position error"):
            if self.data["has_truth"]:
                t_gap, eh_gap, e3d_gap = insert_time_gaps(p.time, p.eh, p.e3d)
                ax.plot(t_gap, eh_gap, lw=.8, label="Horizontal")
                ax.plot(t_gap, e3d_gap, lw=.8, label="3D")
                ax.legend(fontsize=8)
                ax.set(title="Position Error", xlabel="Time (s)", ylabel="Error (m)")
            else:
                ax.text(.5, .5, "No truth loaded", ha="center", transform=ax.transAxes)
                ax.set(title="Position Error")
            self._set_datetime_axis(ax)
        elif mode_name == "Residual Triad":
            self._draw_residual_triad(ax)
            self._set_datetime_axis(ax)
        elif mode_name == "El vs SNR & Resp":
            self._draw_el_snr_resp(ax)
        elif mode_name == "Top Outlier Satellites":
            self._draw_top_outliers(ax)
        elif mode_name == "Directional DOP":
            self._draw_directional_dop(ax)
            self._set_datetime_axis(ax)
        elif mode_name == "Body-Frame Error":
            self._draw_body_frame_error(ax, p)
            self._set_datetime_axis(ax)
        elif mode_name == "Clock Bias & ISB":
            self._draw_clock_bias_isb(ax)
            self._set_datetime_axis(ax)
        elif mode_name == "Constellation Summary":
            fig.set_size_inches(13, 4.5)
            fig.clear()
            axs = fig.subplots(1, 2)
            self._draw_constellation_summary(axs, s)
            if with_hud:
                diag.render_hud_card(axs[0], metrics, loc="upper right", font_size=8.0)
            fig.tight_layout()
            return

        ax.grid(self.grid_on.get(), alpha=0.3)
        if with_hud and metrics and "fix_rate" in metrics:
            diag.render_hud_card(ax, metrics, loc="upper right", font_size=8.5)
        fig.tight_layout()

    def open_batch_export_dialog(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先打开 .pos 解算文件后再执行批量导出。")
            return
        BatchExportDialog(self)

    def export_plot(self):
        if self.data is None:return
        path=filedialog.asksaveasfilename(title="Export current plot",defaultextension=".png",initialfile=f"{self.mode.get().lower().replace(' ','_')}.png",filetypes=[("PNG image","*.png"),("PDF document","*.pdf"),("SVG vector image","*.svg")])
        if not path:return
        try:self.fig.savefig(path,dpi=300 if path.lower().endswith(".png") else None,bbox_inches="tight");self.status.configure(text=f"Exported: {path}")
        except Exception as exc:messagebox.showerror("Export failed",str(exc))


class BatchExportDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("批量导出所选子图与数据切片")
        self.geometry("640x720")
        self.minsize(560, 620)
        self.transient(app.root)
        self.grab_set()

        # 1. Segment Selection
        top_frame = ttk.LabelFrame(self, text="1. 选择时间段 (Time Segment Range)", padding=(10, 8))
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.seg_options = ["全程全部历元 (Full Dataset)"]
        for seg in self.app.segments:
            self.seg_options.append(f"{seg['name']}: {self.app._epoch_datetime_label(seg['start'])} — {self.app._epoch_datetime_label(seg['end'])} CST")

        self.seg_var = tk.StringVar(value=self.seg_options[-1] if len(self.seg_options) > 1 else self.seg_options[0])
        ttk.Label(top_frame, text="时间区间:").pack(side=tk.LEFT, padx=(0, 6))
        seg_combo = ttk.Combobox(top_frame, textvariable=self.seg_var, values=self.seg_options, state="readonly", width=48)
        seg_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 2. Subplots to Export (Multi-select Checkboxes)
        mid_frame = ttk.LabelFrame(self, text="2. 选择要导出的子图 (Select Subplots to Export)", padding=(10, 8))
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        btn_row = ttk.Frame(mid_frame)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_row, text="全选 (Select All)", command=self._select_all).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="清空 (Clear)", command=self._clear_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="常用推荐 (Recommended 6)", command=self._select_recommended).pack(side=tk.LEFT, padx=4)

        chk_canvas = tk.Canvas(mid_frame, borderwidth=0, highlightthickness=0)
        chk_scroll = ttk.Scrollbar(mid_frame, orient=tk.VERTICAL, command=chk_canvas.yview)
        chk_inner = ttk.Frame(chk_canvas)
        chk_inner.bind("<Configure>", lambda e: chk_canvas.configure(scrollregion=chk_canvas.bbox("all")))
        chk_window = chk_canvas.create_window((0, 0), window=chk_inner, anchor="nw")
        chk_canvas.bind("<Configure>", lambda e: chk_canvas.itemconfig(chk_window, width=e.width))
        chk_canvas.configure(yscrollcommand=chk_scroll.set)
        chk_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chk_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.all_export_modes = ["Dashboard"] + list(MODES)
        self.chk_vars = {}
        for i, mode in enumerate(self.all_export_modes):
            v = tk.BooleanVar(value=(mode in ("Dashboard", "Gnd Trk", "ENU Position", "ENU Error", "AR Ratio", "Satellites")))
            self.chk_vars[mode] = v
            cb = ttk.Checkbutton(chk_inner, text=mode, variable=v)
            row = i // 2
            col = i % 2
            cb.grid(row=row, column=col, sticky="w", padx=12, pady=3)

        # 3. Export Options
        opt_frame = ttk.LabelFrame(self, text="3. 学术增强与导出选项 (Export & Academic Options)", padding=(10, 8))
        opt_frame.pack(fill=tk.X, padx=10, pady=5)

        self.var_hud = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="在图角绘制学术量化指标 HUD 卡片 (FIX率、RMS 3D、CEP50/95、平均卫星数)", variable=self.var_hud).pack(anchor="w", pady=2)

        self.var_dpi300 = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="学术汇报/论文级高清输出 (300 DPI, 自动优化线条与字号清晰度)", variable=self.var_dpi300).pack(anchor="w", pady=2)

        self.var_crop_pos = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="同时导出该时间段裁剪版 .pos 解算文件 (保留完整RTKLIB头信息与高精度坐标)", variable=self.var_crop_pos).pack(anchor="w", pady=2)

        self.var_kml = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="同时导出 Google Earth 3D 轨迹工程文件 (.kml，含真实高程挤出与分色)", variable=self.var_kml).pack(anchor="w", pady=2)

        self.var_geojson = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="同时导出标准 GIS 空间要素集 (.geojson，包含历元属性，可直接在 QGIS 分析)", variable=self.var_geojson).pack(anchor="w", pady=2)

        # 4. Output Path
        path_frame = ttk.LabelFrame(self, text="4. 导出存储目录 (Output Directory)", padding=(10, 8))
        path_frame.pack(fill=tk.X, padx=10, pady=5)

        default_dir = (self.app.pos_path.parent / "exports") if self.app.pos_path else Path.cwd() / "exports"
        self.out_dir_var = tk.StringVar(value=str(default_dir))
        entry = ttk.Entry(path_frame, textvariable=self.out_dir_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(path_frame, text="浏览… (Browse)", command=self._browse_dir).pack(side=tk.RIGHT)

        # 5. Bottom action bar
        act_frame = ttk.Frame(self, padding=(10, 10))
        act_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.progress = ttk.Progressbar(act_frame, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 6))
        self.status_label = ttk.Label(act_frame, text="就绪。点击下方按钮开始导出。", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(act_frame, text="关闭 (Close)", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        self.btn_export = ttk.Button(act_frame, text="🚀 开始批量导出 (Start Export)", command=self._do_export)
        self.btn_export.pack(side=tk.RIGHT, padx=4)

    def _select_all(self):
        for v in self.chk_vars.values(): v.set(True)

    def _clear_all(self):
        for v in self.chk_vars.values(): v.set(False)

    def _select_recommended(self):
        rec = {"Dashboard", "Gnd Trk", "ENU Position", "ENU Error", "AR Ratio", "Satellites"}
        for k, v in self.chk_vars.items(): v.set(k in rec)

    def _browse_dir(self):
        d = filedialog.askdirectory(title="选择导出目录", initialdir=self.out_dir_var.get())
        if d: self.out_dir_var.set(d)

    def _do_export(self):
        selected_modes = [m for m, v in self.chk_vars.items() if v.get()]
        if not selected_modes and not self.var_crop_pos.get():
            messagebox.showwarning("Warning", "请至少勾选一个要导出的子图或勾选导出裁剪版 .pos 文件。")
            return

        out_dir = Path(self.out_dir_var.get())
        out_dir.mkdir(parents=True, exist_ok=True)

        full_pos = self.app.data["pos"]
        seg_idx = self.seg_options.index(self.seg_var.get())
        if seg_idx == 0:
            seg_slug = "full"
            pos_sub = full_pos.copy()
            sat_sub = self.app.data["sat"]
            t_lo = float(full_pos.tow.iloc[0])
            t_hi = float(full_pos.tow.iloc[-1])
        else:
            seg = self.app.segments[seg_idx - 1]
            seg_slug = f"seg{seg_idx}"
            pos_sub = full_pos.iloc[seg["start"]:seg["end"]+1].copy()
            t0 = float(pos_sub.time.iloc[0])
            t1 = float(pos_sub.time.iloc[-1])
            sat_sub = self.app._segment_sat_records(t0, t1)
            t_lo = float(pos_sub.tow.iloc[0])
            t_hi = float(pos_sub.tow.iloc[-1])

        metrics = diag.compute_segment_metrics(pos_sub, sat_sub)
        base_name = self.app.pos_path.stem if self.app.pos_path else "rtklib_solution"

        self.btn_export.configure(state="disabled")
        total_tasks = (len(selected_modes)
                       + (1 if self.var_crop_pos.get() else 0)
                       + (1 if self.var_kml.get() else 0)
                       + (1 if self.var_geojson.get() else 0))
        self.progress["maximum"] = total_tasks
        current_step = 0
        exported_files = []

        if self.var_crop_pos.get() and self.app.pos_path and self.app.pos_path.exists():
            self.status_label.configure(text="正在导出裁剪版 .pos 解算文件...")
            self.update_idletasks()
            out_pos_file = out_dir / f"{base_name}_{seg_slug}.pos"
            try:
                cnt = diag.crop_and_export_pos(self.app.pos_path, out_pos_file, t_lo, t_hi)
                exported_files.append(out_pos_file.name)
            except Exception as e:
                print(f"Error exporting .pos: {e}")
            current_step += 1
            self.progress["value"] = current_step
            self.update_idletasks()

        if self.var_kml.get():
            self.status_label.configure(text="正在导出 Google Earth 3D KML 文件...")
            self.update_idletasks()
            out_kml_file = out_dir / f"{base_name}_{seg_slug}_3d.kml"
            try:
                mt.export_kml_3d(pos_sub, out_kml_file, segments=self.app.segments, pos_name=f"{base_name}_{seg_slug}")
                exported_files.append(out_kml_file.name)
            except Exception as e:
                print(f"Error exporting .kml: {e}")
            current_step += 1
            self.progress["value"] = current_step
            self.update_idletasks()

        if self.var_geojson.get():
            self.status_label.configure(text="正在导出标准 GIS GeoJSON 要素集...")
            self.update_idletasks()
            out_json_file = out_dir / f"{base_name}_{seg_slug}.geojson"
            try:
                mt.export_geojson(pos_sub, out_json_file)
                exported_files.append(out_json_file.name)
            except Exception as e:
                print(f"Error exporting .geojson: {e}")
            current_step += 1
            self.progress["value"] = current_step
            self.update_idletasks()

        dpi = 300 if self.var_dpi300.get() else 180
        for mode in selected_modes:
            self.status_label.configure(text=f"正在导出子图 [{mode}]...")
            self.update_idletasks()
            slug = mode.lower().replace(" ", "_").replace("/", "_")
            out_img = out_dir / f"{base_name}_{seg_slug}_{slug}.png"
            try:
                fig = plt.Figure(facecolor="white")
                self.app.export_render_mode(fig, mode, pos_sub, sat_sub, metrics, with_hud=self.var_hud.get())
                fig.savefig(out_img, dpi=dpi, bbox_inches="tight")
                exported_files.append(out_img.name)
            except Exception as e:
                print(f"Error exporting {mode}: {e}")
            current_step += 1
            self.progress["value"] = current_step
            self.update_idletasks()

        self.status_label.configure(text=f"导出完成！成功生成 {len(exported_files)} 个文件。")
        self.btn_export.configure(state="normal")

        msg = f"批量导出成功完成！\n共生成 {len(exported_files)} 个文件保存在：\n{out_dir}\n\n是否立即在文件管理器中打开该目录？"
        if messagebox.askyesno("导出成功", msg):
            try:
                os.startfile(out_dir)
            except Exception:
                pass


class DiagnosticReportDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("RTKLIB 诊断质量评估报告 (Solution Quality Report)")
        self.geometry("820x650")
        self.transient(app.root)

        rep = diag.compute_quality_report(app.data)
        self.rep = rep
        pos_name = app.pos_path.name if app.pos_path else "Solution"
        self.md_text = diag.generate_markdown_report(rep, pos_name)

        # Header card
        header = ttk.Frame(self, padding=(12, 10))
        header.pack(fill=tk.X)

        score = rep.get("score", 0)
        grade = rep.get("grade", "N/A")
        grade_color = "#28a745" if grade.startswith("A") else ("#d49200" if grade.startswith("B") else "#dc3545")

        lbl_score = tk.Label(header, text=f"综合评分：{score} / 100", font=("Microsoft YaHei", 14, "bold"), fg=grade_color)
        lbl_score.pack(side=tk.LEFT)

        lbl_grade = tk.Label(header, text=f"评级：{grade}", font=("Microsoft YaHei", 12, "bold"), fg="#333")
        lbl_grade.pack(side=tk.LEFT, padx=16)

        fix_r = rep.get("fix_rate", 0)
        lbl_fix = tk.Label(header, text=f"固定率 (FIX)：{fix_r:.1f}%", font=("Microsoft YaHei", 10), fg="#444")
        lbl_fix.pack(side=tk.LEFT, padx=8)

        if rep.get("has_truth"):
            m = rep.get("metrics", {})
            rmse2d = np.hypot(m.get("rms_e", 0), m.get("rms_n", 0))
            lbl_rmse = tk.Label(header, text=f"水平 RMSE：{rmse2d:.3f} m", font=("Microsoft YaHei", 10), fg="#444")
            lbl_rmse.pack(side=tk.LEFT, padx=8)

        # Scrolled Text for Markdown
        content_frame = ttk.Frame(self, padding=(10, 5))
        content_frame.pack(fill=tk.BOTH, expand=True)

        self.txt = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#fdfdfd", fg="#222")
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.txt.insert(tk.END, self.md_text)
        self.txt.configure(state="disabled")

        # Bottom buttons
        btn_bar = ttk.Frame(self, padding=(10, 8))
        btn_bar.pack(fill=tk.X)

        ttk.Button(btn_bar, text="📋 复制报告至剪贴板 (Copy)", command=self._copy_clipboard).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_bar, text="💾 导出报告 (.md)", command=self._save_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_bar, text="📊 导出答辩 PPTX (武大母版)", command=self.app.export_academic_pptx).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_bar, text="🌐 导出诊断 HTML", command=self.app.export_diagnostic_html).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_bar, text="关闭", command=self.destroy).pack(side=tk.RIGHT, padx=6)

    def _copy_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self.md_text)
        messagebox.showinfo("复制成功", "质量诊断报告已成功复制到剪贴板！", parent=self)

    def _save_file(self):
        fn = filedialog.asksaveasfilename(
            parent=self,
            title="保存质量诊断报告",
            defaultextension=".md",
            initialfile=f"rtklib_quality_report_{self.app.pos_path.stem if self.app.pos_path else 'pos'}.md",
            filetypes=[("Markdown files", "*.md"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not fn:
            return
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write(self.md_text)
            messagebox.showinfo("保存成功", f"报告已成功保存至：\n{fn}", parent=self)
        except Exception as e:
            messagebox.showerror("保存失败", str(e), parent=self)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--pos",type=Path,default=None);parser.add_argument("--reference",type=Path,default=None);parser.add_argument("--plot",choices=MODES+['Dashboard'],default="Gnd Trk",help="Initial view to show");args=parser.parse_args()
    root=tk.Tk();RTKPlotApp(root,args.pos,args.reference,args.plot);root.mainloop()

if __name__=="__main__":main()
