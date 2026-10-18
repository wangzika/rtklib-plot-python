"""Interactive RTKLIB-style plotter for .pos and optional .stat/reference files."""
from __future__ import annotations

import argparse
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

Q_LABELS = {1: "FIX", 2: "FLOAT", 3: "SBAS", 4: "DGPS", 5: "SINGLE", 6: "PPP"}
Q_COLORS = {1: "#07851d", 2: "#dc8a12", 3: "#5b83c4", 4: "#9b59b6", 5: "#777777", 6: "#17a2a4"}
MODES = ["Gnd Trk", "ENU Position", "Height", "ENU Error", "Solution Status",
         "AR Ratio", "Differential Age", "Satellites", "DOP", "Skyplot", "C/N0", "Elevation", "Residuals",
         "Multipath Proxy", "Cycle Slips", "Satellite Visibility Gantt", "Multipath Skyplot", "Per-PRN Residuals", "Covariance", "False Fix Analysis", "Innovation / NIS",
         "Error CDF", "Error Histogram", "Constellation Summary",
         "Env & Geometry Combined", "Residual Triad", "El vs SNR & Resp", "Top Outlier Satellites", "Directional DOP", "Clock Bias & ISB", "Body-Frame Error"]

NON_TIME_SERIES_MODES = (
    "Gnd Trk", "Skyplot", "Multipath Skyplot", "Multipath Proxy",
    "Per-PRN Residuals", "Innovation / NIS", "Error CDF",
    "Error Histogram", "Constellation Summary",
    "El vs SNR & Resp", "Top Outlier Satellites", "Env & Geometry Combined"
)

PLOT_CATEGORIES = {
    "故障诊断链路 (Fault Diagnostics)": [
        "Env & Geometry Combined", "Residual Triad", "El vs SNR & Resp", "Top Outlier Satellites", "Directional DOP", "Clock Bias & ISB", "Body-Frame Error"
    ],
    "轨迹与高程 (Trajectory & Position)": [
        "Gnd Trk", "ENU Position", "Height", "Body-Frame Error"
    ],
    "定位精度与误差 (Accuracy & Errors)": [
        "ENU Error", "False Fix Analysis", "Error CDF", "Error Histogram", "Covariance", "Residual Triad"
    ],
    "解算状态与可靠性 (Status & Reliability)": [
        "Solution Status", "AR Ratio", "Differential Age", "Innovation / NIS", "Clock Bias & ISB"
    ],
    "卫星星历与信号质量 (Satellites & Signal Quality)": [
        "Satellites", "DOP", "Directional DOP", "Skyplot", "C/N0", "Elevation", "Residuals", "Multipath Proxy", "Cycle Slips", "Satellite Visibility Gantt", "Multipath Skyplot", "Per-PRN Residuals", "Top Outlier Satellites", "El vs SNR & Resp", "Constellation Summary"
    ]
}

MODE_BILINGUAL = {
    "Env & Geometry Combined": "①+③ 环境与几何综合 (Env & Geometry)",
    "Gnd Trk": "地面轨迹 (Gnd Trk)",
    "ENU Position": "ENU位置 (ENU Pos)",
    "Height": "大地高程 (Height)",
    "ENU Error": "ENU误差 (ENU Err)",
    "Solution Status": "解状态 (Sol Status)",
    "AR Ratio": "Ratio比值 (AR Ratio)",
    "Differential Age": "差分时延与解状态 (Age & Q)",
    "Satellites": "卫星数量 (Satellites)",
    "DOP": "DOP精度因子 (DOP)",
    "Skyplot": "星空图 (Skyplot)",
    "C/N0": "载噪比 (C/N0)",
    "Elevation": "卫星高度角 (Elevation)",
    "Residuals": "伪距残差 (Residuals)",
    "Multipath Proxy": "多路径效应 (Multipath)",
    "Cycle Slips": "周跳探测 (Cycle Slips)",
    "Satellite Visibility Gantt": "卫星可见与周跳甘特图 (Gantt)",
    "Multipath Skyplot": "多路径星空热力图 (MP Skyplot)",
    "Per-PRN Residuals": "单星残差画像 (Per-PRN Res)",
    "Covariance": "方差协方差 (Covariance)",
    "False Fix Analysis": "假固定分析 (False Fix)",
    "Innovation / NIS": "新息/NIS (Innovation)",
    "Error CDF": "误差累计分布 (CDF)",
    "Error Histogram": "误差直方图 (Histogram)",
    "Constellation Summary": "各星座汇总 (Constellation)",
    "Residual Triad": "伪距残差三线 (Residual Triad)",
    "El vs SNR & Resp": "高度角-SNR-残差 (El vs SNR & Resp)",
    "Top Outlier Satellites": "粗差卫星贡献榜 (Top Outliers)",
    "Directional DOP": "方向DOP与扇区 (Directional DOP)",
    "Clock Bias & ISB": "接收机钟差与ISB (Clock & ISB)",
    "Body-Frame Error": "车体坐标系误差 (Body-Frame Err)",
}

DIAGNOSTIC_WORKSPACES = [
    {
        "id": 1,
        "name": "① 外部环境与多路径",
        "desc": "星空多路径热力图、高度角-信噪比-残差联合分布与伪距残差三线时序",
        "preset": "Focus Left (左主右副)",
        "modes": ["Multipath Skyplot", "El vs SNR & Resp", "Residual Triad"]
    },
    {
        "id": 2,
        "name": "② 单星污染与健康度",
        "desc": "全星座可见性/周跳泳道图、粗差卫星贡献度排行榜与载噪比时序",
        "preset": "Focus Top (上主下副)",
        "modes": ["Satellite Visibility Gantt", "Top Outlier Satellites", "C/N0"]
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
        "desc": "假固定点空间识别、状态协方差 ±3σ 检验与接收机钟差/系统间偏差",
        "preset": "Focus Left (左主右副)",
        "modes": ["False Fix Analysis", "Covariance", "Clock Bias & ISB"]
    },
    {
        "id": 5,
        "name": "⑤ 全局精度与轨迹复盘",
        "desc": "2D航迹着色图、车体坐标系(沿航向/跨航向/垂直)误差分解与误差CDF",
        "preset": "Focus Left (左主右副)",
        "modes": ["Gnd Trk", "Body-Frame Error", "Error CDF"]
    }
]


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
        self.false_fix_limit=tk.DoubleVar(value=0.20)
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
        self.mode_box=ttk.Combobox(row1,textvariable=self.mode,values=MODES+['Dashboard','Custom Analysis','Multi-Compare'],state="readonly",width=17)
        self.mode_box.pack(side=tk.LEFT,padx=4); self.mode_box.bind("<<ComboboxSelected>>",lambda _e:self.redraw())
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
        if cat in ("全部", "全部 (24)", "全部 (22)", "全部 (20)"):
            listed = list(MODES)
        elif cat in ("轨迹", "轨迹 (3)"):
            listed = [m for m in MODES if m in PLOT_CATEGORIES["轨迹与高程 (Trajectory & Position)"]]
        elif cat in ("精度", "精度 (5)"):
            listed = [m for m in MODES if m in PLOT_CATEGORIES["定位精度与误差 (Accuracy & Errors)"]]
        elif cat in ("状态", "状态 (4)", "状态 (3)"):
            listed = [m for m in MODES if m in PLOT_CATEGORIES["解算状态与可靠性 (Status & Reliability)"]]
        elif cat in ("卫星", "卫星 (12)", "卫星 (10)", "卫星 (9)"):
            listed = [m for m in MODES if m in PLOT_CATEGORIES["卫星星历与信号质量 (Satellites & Signal Quality)"]]
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
            ("全部", "全部 (24)"),
            ("轨迹", "轨迹 (3)"),
            ("精度", "精度 (5)"),
            ("状态", "状态 (4)"),
            ("卫星", "卫星 (12)"),
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
        self.plot_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, exportselection=False, width=28, height=8)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.plot_list.yview)
        self.plot_list.configure(yscrollcommand=list_scroll.set)
        self.plot_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.plot_list.bind("<ButtonPress-1>", self._begin_plot_drag, add="+")
        self.plot_list.bind("<B1-Motion>", lambda _e: "break", add="+")
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

        ttk.Checkbutton(card2_content, text="⏱ 时间轴联动 (Sync Time)", variable=self.sync_time_axes, command=self.redraw).pack(anchor="w", pady=(1, 2))

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
        self.segment_box = tk.Listbox(seg_frame, height=3, width=28, exportselection=False)
        self.segment_box.bind("<Double-Button-1>", self.on_segment_double_click)
        seg_scroll = ttk.Scrollbar(seg_frame, orient=tk.VERTICAL, command=self.segment_box.yview); self.segment_box.configure(yscrollcommand=seg_scroll.set)
        self.segment_box.pack(side=tk.LEFT, fill=tk.X, expand=True); seg_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(card4_content, text="✕ 删除所选区段 (Delete)", command=self.delete_segment).pack(fill=tk.X, pady=(2, 1))
        ttk.Button(card4_content, text="📷 导出区段多图 (Export)…", command=self.open_batch_export_dialog).pack(fill=tk.X, pady=(1, 1))

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
        self.status=ttk.Label(self.root,text="No solution loaded",anchor="w",relief=tk.SUNKEN); self.status.pack(side=tk.BOTTOM,fill=tk.X)
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
            messagebox.showerror("Could not read solution",str(exc)); return
        self.segments.clear();self.segment_start_idx=None;self._refresh_segment_list()
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

    def set_mode(self,name):self.mode.set(name); self.redraw()

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
            self.segment_box.itemconfig(i,fg=seg["color"])
        if len(self.segments) > 0 and hasattr(self, "_expand_segments_card") and callable(self._expand_segments_card):
            self._expand_segments_card()
        self._update_sidebar_scroll()

    def delete_segment(self):
        selected=list(self.segment_box.curselection())
        if not selected:return
        for i in reversed(selected):del self.segments[i]
        for i,seg in enumerate(self.segments):seg["name"]=f"Segment {i+1}"
        self._refresh_segment_list();self._refresh_segment_overlays()

    def clear_segments(self):
        self.segments.clear();self.segment_start_idx=None;self._refresh_segment_list()
        self.segment_status.configure(text="All marked segments cleared.");self._refresh_segment_overlays()

    def on_segment_double_click(self, event=None):
        if not hasattr(self, "segment_box"): return
        sel = self.segment_box.curselection()
        if not sel or not self.segments or self.data is None: return
        idx = sel[0]
        if idx >= len(self.segments): return
        seg = self.segments[idx]
        p = self.data["pos"]
        t0 = float(p.iloc[seg["start"]].time)
        t1 = float(p.iloc[seg["end"]].time)
        pad = max(2.0, (t1 - t0) * 0.08)

        # 1. Zoom timeline datetime axes
        for ax in self.datetime_axes:
            ax.set_xlim(t0 - pad, t1 + pad)

        # 2. Zoom ground track axes
        g = p.iloc[seg["start"]:seg["end"]+1]
        for ax in self.track_axes:
            e_min, e_max = float(g.E.min()), float(g.E.max())
            n_min, n_max = float(g.N.min()), float(g.N.max())
            de = max(3.0, (e_max - e_min) * 0.12)
            dn = max(3.0, (n_max - n_min) * 0.12)
            ax.set_xlim(e_min - de, e_max + de)
            ax.set_ylim(n_min - dn, n_max + dn)

        # 3. Position epoch slider to start of segment
        self.slider.set(seg["start"])
        self.canvas.draw_idle()
        st = float(p.iloc[seg["start"]].tow)
        et = float(p.iloc[seg["end"]].tow)
        d_res = diag.diagnose_anomaly_segment(p, self.data["sat"], st, et)
        self.segment_status.configure(text=f"🔍 聚焦 [{seg['name']}]: 【{d_res['primary_cause']}】{d_res['summary']} | 建议: {d_res['recommendation']}")

    def auto_detect_issues(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先加载 .pos 解算数据后再进行智能异常探测。")
            return
        detected = diag.detect_problematic_segments(self.data["pos"], error_limit=float(self.false_fix_limit.get()), min_duration=2.0)
        if not detected:
            messagebox.showinfo("智能诊断结果", "全文未探测到持续失锁（Float/Single）或3D误差超标时段，解算质量表现优异！")
            return
        new_count = 0
        for item in detected:
            if not any(abs(s["start"] - item["start"]) < 5 and abs(s["end"] - item["end"]) < 5 for s in self.segments):
                self.segments.append(item)
                new_count += 1
        self._refresh_segment_list()
        self._refresh_segment_overlays()
        if hasattr(self, "_expand_segments_card") and callable(self._expand_segments_card):
            self._expand_segments_card()
        messagebox.showinfo("智能诊断完成", f"共自动探测出 {len(detected)} 处异常/跳固定时段（新增标记 {new_count} 处）。\n\n提示：双击左侧列表中的任意时段即可快速缩放聚焦！")
        self.segment_status.configure(text=f"已自动标记 {len(detected)} 处异常时段。双击左侧时段列表可自动缩放聚焦。")

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
                        gap_y = 0.055
                        top, bottom, right_x, right_w = 0.92, 0.09, 0.67, 0.295
                        h = (top - bottom - (sub_total - 1) * gap_y) / sub_total
                        y = top - (sub_idx + 1) * h - sub_idx * gap_y
                        panels.append({"mode": name, "rect": (right_x, y, right_w, h)})
                    else:
                        r_cols = 2
                        r_rows = (sub_total + 1) // 2
                        r_row, r_col = divmod(sub_idx, r_cols)
                        gap_x, gap_y = 0.035, 0.045
                        top, bottom, right_x = 0.92, 0.09, 0.64
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
                        panels.append({"mode": name, "rect": (0.075, 0.09, 0.89, 0.83)})
                    else:
                        panels.append({"mode": name, "rect": (0.075, 0.53, 0.89, 0.39)})
                else:
                    sub_idx = page_idx - 1
                    sub_total = page_total - 1
                    gap_x = 0.045
                    left, right, bottom, h = 0.075, 0.965, 0.09, 0.36
                    sub_w = (right - left - (sub_total - 1) * gap_x) / sub_total
                    x = left + sub_idx * (sub_w + gap_x)
                    panels.append({"mode": name, "rect": (x, bottom, sub_w, h)})

        else: # auto_grid
            if n == 1:
                panels.append({"mode": modes[0], "rect": (0.075, 0.09, 0.89, 0.83)})
            elif n == 2:
                w = 0.425; gap_x = 0.06; left = 0.075
                panels.append({"mode": modes[0], "rect": (left, 0.09, w, 0.82)})
                panels.append({"mode": modes[1], "rect": (left + w + gap_x, 0.09, w, 0.82)})
            elif n == 3:
                w_top = 0.425; gap_x = 0.06; left = 0.075
                panels.append({"mode": modes[0], "rect": (left, 0.54, w_top, 0.38)})
                panels.append({"mode": modes[1], "rect": (left + w_top + gap_x, 0.54, w_top, 0.38)})
                panels.append({"mode": modes[2], "rect": (left, 0.09, 0.89, 0.37)})
            elif n == 4:
                w = 0.425; gap_x = 0.06; left = 0.075
                h = 0.38; gap_y = 0.07
                panels.append({"mode": modes[0], "rect": (left, 0.54, w, h)})
                panels.append({"mode": modes[1], "rect": (left + w + gap_x, 0.54, w, h)})
                panels.append({"mode": modes[2], "rect": (left, 0.09, w, h)})
                panels.append({"mode": modes[3], "rect": (left + w + gap_x, 0.09, w, h)})
            else:
                page_size = 6
                cols = 3
                rows = 2
                left, right, bottom, top = 0.065, 0.985, 0.09, 0.93
                gap_x, gap_y = 0.045, 0.065
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
            for m in modes_to_add:
                self.dashboard_modes.append(m)
            self.selected_dashboard_index = len(self.dashboard_modes) - 1
            self.focused_dashboard_subplot = None
            self.segment_status.configure(text=f"已向 Dashboard 追加 {len(modes_to_add)} 个子图，当前共 {len(self.dashboard_modes)} 个。")
            self.redraw()
        else:
            if mode != "Custom Analysis":
                self.mode.set("Custom Analysis")
            current_modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
            for m in modes_to_add:
                current_modes.append(m)
            self.custom_modes = current_modes
            self.custom_panels = self._initial_custom_panels(current_modes, self.custom_preset.get())
            self.selected_custom_panel = len(self.custom_panels) - 1
            self.focused_custom_panel = None
            self.segment_status.configure(text=f"已追加 {len(modes_to_add)} 个子图，当前共 {len(self.custom_panels)} 个。")
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
            if mode != "Custom Analysis":
                self.mode.set("Custom Analysis")
            current_modes = [p["mode"] for p in self.custom_panels if p.get("mode")]
            if insert_idx is not None and 0 <= insert_idx < len(current_modes):
                current_modes.insert(insert_idx + 1, mode_name)
                new_sel = insert_idx + 1
            else:
                current_modes.append(mode_name)
                new_sel = len(current_modes) - 1
            self.custom_modes = current_modes
            self.custom_panels = self._initial_custom_panels(current_modes, self.custom_preset.get())
            self.selected_custom_panel = new_sel
            self.focused_custom_panel = None
            self.segment_status.configure(text=f"已添加子图 [{mode_name}]，当前共 {len(self.custom_panels)} 个。")
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
        if mode in ("ENU Position","ENU Error","Error Histogram","Body-Frame Error"):
            sharex = (mode != "Error Histogram")
            axes=self.fig.subplots(3,1,sharex=sharex);self.plot_axes=list(axes);self.ax=axes[0]
        elif mode in ("Constellation Summary", "Directional DOP", "Clock Bias & ISB"):
            axes=self.fig.subplots(2,1,sharex=True if mode != "Constellation Summary" else False);self.plot_axes=list(axes);self.ax=axes[0]
        else:
            self.ax=self.fig.add_subplot(111,projection="polar" if mode in ("Skyplot", "Multipath Skyplot") else None);self.plot_axes=[self.ax]
        ax=self.ax
        if mode=="Gnd Trk":self._draw_track(ax,p,full)
        elif mode=="ENU Position":self._draw_enu_positions(self.plot_axes,p)
        elif mode=="Height":
            me=max(1,len(p)//120);ax.plot(p.time,p.U,label="RTK height U",color="#16853b",marker=".",ms=3,markevery=me)
            if self.data["has_truth"]:ax.plot(p.time,p.gt_U,label="Ground truth U",color="#252525",lw=.8,ls="--",marker=".",ms=2,markevery=me)
            ax.set(xlabel="Time from start (s)",ylabel="Up (m)",title="Height / Up Component");ax.legend()
        elif mode=="ENU Error":self._draw_errors(self.plot_axes,p)
        elif mode=="Solution Status":ax.step(p.time,p.q,where="post",lw=.8,marker=".",ms=2,markevery=max(1,len(p)//120));ax.set(xlabel="Time from start (s)",ylabel="Q",yticks=sorted(full.q.unique()),title="Solution Status")
        elif mode=="AR Ratio":ax.plot(p.time,p.ratio,lw=.75,marker=".",ms=3,markevery=max(1,len(p)//120));ax.axhline(3,color="red",ls="--",label="Reference threshold 3");ax.set(xlabel="Time from start (s)",ylabel="Ratio",title="Ambiguity Resolution Ratio");ax.legend()
        elif mode=="Differential Age":self._draw_differential_age(ax,p)
        elif mode=="Satellites":self._draw_sats(ax)
        elif mode=="DOP":self._draw_dop(ax)
        elif mode=="Skyplot":self._draw_sky(ax)
        elif mode=="C/N0":self._draw_cn0(ax)
        elif mode=="Elevation":self._draw_elevation(ax)
        elif mode=="Residuals":self._draw_residuals(ax)
        elif mode=="Multipath Proxy":self._draw_multipath(ax)
        elif mode=="Cycle Slips":self._draw_slips(ax)
        elif mode=="Satellite Visibility Gantt":self._draw_satellite_visibility_gantt(ax,s)
        elif mode=="Multipath Skyplot":self._draw_multipath_skyplot(ax)
        elif mode=="Per-PRN Residuals":self._draw_per_prn_residuals(ax)
        elif mode=="Covariance":self._draw_covariance(ax,p)
        elif mode=="False Fix Analysis":self._draw_false_fix(ax,p)
        elif mode=="Innovation / NIS":self._draw_nis(ax)
        elif mode=="Error CDF":self._draw_error_cdf(ax,p)
        elif mode=="Error Histogram":self._draw_error_histogram(self.plot_axes,p)
        elif mode=="Constellation Summary":self._draw_constellation_summary(self.plot_axes,s)
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
                self.plot_axes[1].legend(fontsize=8, ncol=3, loc="upper right")
            else:
                self._draw_directional_dop(ax)
        elif mode=="Clock Bias & ISB":
            clk = self.data.get("clk")
            if clk is not None and not clk.empty and len(self.plot_axes) >= 2:
                t0 = self.data["pos"].tow.iloc[0]
                clk_t = clk.tow - t0
                self.plot_axes[0].plot(clk_t, clk.clk_gps, color="#2878B5", lw=1.1, label="GPS 接收机钟差 (ns)")
                if "drift" in clk.columns and clk.drift.abs().max() > 0:
                    ax0_tw = self.plot_axes[0].twinx()
                    ax0_tw.plot(clk_t, clk.drift, color="#E07A2D", lw=0.9, ls="--", label="钟漂 (ns/s)")
                    ax0_tw.set_ylabel("钟漂 (ns/s)", color="#E07A2D")
                    self.cov_twin_axes[id(self.plot_axes[0])] = ax0_tw
                self.plot_axes[0].set(ylabel="钟差 (ns)", title="接收机钟差漂移与系统间偏差 ($CLK)")
                self.plot_axes[0].legend(fontsize=8, loc="upper left")
                if "isb_glo" in clk.columns and clk.isb_glo.abs().max() > 0:
                    self.plot_axes[1].plot(clk_t, clk.isb_glo, label="ISB GLONASS-GPS", color="#9b59b6", lw=1.0)
                if "isb_gal" in clk.columns and clk.isb_gal.abs().max() > 0:
                    self.plot_axes[1].plot(clk_t, clk.isb_gal, label="ISB Galileo-GPS", color="#36965A", lw=1.0)
                if "isb_bds" in clk.columns and clk.isb_bds.abs().max() > 0:
                    self.plot_axes[1].plot(clk_t, clk.isb_bds, label="ISB BDS-GPS", color="#d90429", lw=1.0)
                if "isb_qzs" in clk.columns and clk.isb_qzs.abs().max() > 0:
                    self.plot_axes[1].plot(clk_t, clk.isb_qzs, label="ISB QZSS-GPS", color="#E07A2D", lw=1.0)
                self.plot_axes[1].set(xlabel="相对时间 (s)", ylabel="ISB 系统间偏差 (m)")
                self.plot_axes[1].legend(fontsize=8, ncol=4, loc="upper right")
            else:
                self._draw_clock_bias_isb(ax)
        elif mode=="Body-Frame Error":
            if self.data.get("has_truth") and "along_err" in p.columns and len(self.plot_axes) >= 3:
                self.plot_axes[0].plot(p.time, p.along_err, color="#2878B5", lw=0.8, label="Along-track (沿航向/纵向)")
                self.plot_axes[0].axhline(0, color="k", lw=0.5, ls=":")
                self.plot_axes[0].set(ylabel="沿向误差 (m)", title="车体坐标系误差分解 (Along-track / Cross-track / Vertical)")
                self.plot_axes[0].legend(fontsize=8)
                self.plot_axes[1].plot(p.time, p.cross_err, color="#d90429", lw=0.8, label="Cross-track (跨航向/横向)")
                self.plot_axes[1].axhline(0, color="k", lw=0.5, ls=":")
                self.plot_axes[1].set(ylabel="横向误差 (m)")
                self.plot_axes[1].legend(fontsize=8)
                self.plot_axes[2].plot(p.time, p.vert_err, color="#36965A", lw=0.8, label="Vertical (垂直高程/天向)")
                self.plot_axes[2].axhline(0, color="k", lw=0.5, ls=":")
                self.plot_axes[2].set(xlabel="相对时间 (s)", ylabel="天向误差 (m)")
                self.plot_axes[2].legend(fontsize=8)
            else:
                self._draw_body_frame_error(ax, p)
        for plot_ax in self.plot_axes:plot_ax.grid(self.grid_on.get(),linestyle="--",alpha=.45)
        if mode not in NON_TIME_SERIES_MODES:
            for plot_ax in self.plot_axes:self.time_cursors.append(plot_ax.axvline(0,color="#333",ls="--",lw=.8,alpha=.7,zorder=10))
        if mode in ("ENU Position","ENU Error","Height","Solution Status","AR Ratio","Differential Age","Satellites","DOP","C/N0","Elevation","Residuals","Cycle Slips","Satellite Visibility Gantt","Covariance","False Fix Analysis","Residual Triad","Directional DOP","Clock Bias & ISB","Body-Frame Error"):
            for plot_ax in self.plot_axes:self._set_datetime_axis(plot_ax)
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
        if not self.segments or self.data is None:return
        p=self.data["pos"]
        for ax in self.datetime_axes:
            for seg in self.segments:
                t0=float(p.iloc[seg["start"]].time);t1=float(p.iloc[seg["end"]].time)
                patch=ax.axvspan(t0,t1,facecolor=seg["color"],alpha=.15,edgecolor=seg["color"],linewidth=.6,zorder=1.5)
                self.segment_artists.append(patch)
        for ax in self.track_axes:
            for seg in self.segments:
                g=p.iloc[seg["start"]:seg["end"]+1]
                line,=ax.plot(g.E,g.N,color=seg["color"],lw=3.0,alpha=.86,zorder=6,label=seg["name"])
                self.segment_track_lines.append(line);self.segment_artists.append(line)
                first=ax.scatter([g.E.iloc[0]],[g.N.iloc[0]],s=34,marker="o",facecolors="white",edgecolors=seg["color"],linewidths=1.5,zorder=8)
                last=ax.scatter([g.E.iloc[-1]],[g.N.iloc[-1]],s=34,marker="s",facecolors=seg["color"],edgecolors="white",linewidths=.7,zorder=8)
                self.segment_artists.extend((first,last))
            ax.legend(fontsize=6,ncol=3)
        for ax,points,labels in self.sky_plots:
            for seg in self.segments:
                for endpoint,marker in [(seg["start"],"o"),(seg["end"],"s")]:
                    sat=self._sat_records_at_tow(float(p.iloc[endpoint].tow))
                    sat=sat[sat.freq==1].drop_duplicates("prn")
                    if sat.empty:continue
                    collection=ax.scatter(np.deg2rad(sat.az),90-sat.el,s=24,marker=marker,
                        facecolors="none",edgecolors=seg["color"],linewidths=1.0,zorder=7)
                    self.segment_sky_artists.append(collection);self.segment_artists.append(collection)
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

    def _draw_false_fix(self,ax,p):
        if not self.data["has_truth"]:
            ax.text(.5,.5,"Ground truth is required to evaluate false FIX epochs",ha="center",va="center",transform=ax.transAxes);return
        limit=float(self.false_fix_limit.get());limit=max(0.001,limit)
        fixed=p[p.q==1];false=fixed[fixed.e3d>limit]
        colors={1:"#07851d",2:"#e3a008"}
        for q,g in p.groupby("q"):
            ax.scatter(g.time,g.e3d,s=13,marker="o",color=colors.get(int(q),"#888888"),alpha=.8,
                       edgecolors="none",label=f"Q={q} {Q_LABELS.get(int(q),'')}")
        ax.axhline(limit,color="#c22",ls="--",lw=1.1,label=f"3D error limit {limit:g} m")
        over=p[p.e3d>limit]
        if not over.empty:ax.scatter(over.time,over.e3d,s=34,marker="x",linewidths=1.3,color="#d00000",zorder=6,label="3D error above limit")
        cap=max(limit*1.5,float(p.e3d.quantile(.995))*1.1)
        clipped=int((p.e3d>cap).sum())
        if clipped:ax.text(.99,.98,f"{clipped} epochs above display cap",ha="right",va="top",transform=ax.transAxes,fontsize=8,color="#a22")
        ax.set_ylim(0,cap);ax.set(xlabel="Date/time",ylabel="3D position error (m)",title="3D Error by Solution Status (points; red × exceeds limit)");ax.legend(fontsize=8,ncol=2)

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
        triad = diag.compute_residual_triad(self.data)
        if triad.empty:
            ax.text(0.5, 0.5, "无有效伪距残差数据 (需 .stat 文件)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("伪距残差统计三线时序")
            return
        ax.plot(triad.time, triad.max_resp, color="#d90429", lw=1.1, label="第一大残差 (Max |resp|)")
        ax.plot(triad.time, triad.sec_resp, color="#00b4d8", lw=1.0, label="第二大残差 (2nd Max)")
        ax.plot(triad.time, triad.rms_resp, color="#0B2B52", lw=1.2, ls="--", label="全场RMS (RMS |resp|)")
        ax.set(xlabel="时间 (s)", ylabel="伪距残差 (m)", title="伪距残差统计三线时序 (Max / 2nd Max / RMS)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_el_snr_resp(self, ax):
        sat = self.data.get("sat")
        if sat is None or sat.empty:
            ax.text(0.5, 0.5, "无卫星观测数据 (需 .stat 文件)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("高度角-载噪比-残差联合分布")
            return
        s1 = sat[(sat.freq == 1) & (sat.snr > 0) & (sat.resp != 0)]
        if s1.empty:
            s1 = sat[(sat.freq == 1) & (sat.snr > 0)]
        if not s1.empty:
            s1_sub = s1.iloc[::max(1, len(s1)//20000)]
            resp_clip = np.clip(s1_sub.resp.abs(), 0, 8) if "resp" in s1_sub.columns else 0
            ax.scatter(s1_sub.el, s1_sub.snr, c=resp_clip, cmap="turbo", s=6, alpha=0.45, edgecolors="none")
            el_grid = np.linspace(5, 90, 100)
            ax.plot(el_grid, 28.0 + 22.0 * np.sin(np.deg2rad(el_grid)), "k--", lw=1.5, label="标称信噪比包络")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, 90)
        ax.set(xlabel="高度角 (°)", ylabel="载噪比 (dB-Hz)", title="高度角 - 载噪比 - 残差联合分布")
        ax.grid(self.grid_on.get(), alpha=0.3)

    def _draw_top_outliers(self, ax):
        df_rank = diag.compute_satellite_health_rankings(self.data)
        if df_rank.empty:
            ax.text(0.5, 0.5, "无卫星残差统计数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("粗差卫星贡献度排行榜")
            return
        top = df_rank.head(10).iloc[::-1]
        constell_colors = {"G": "#2878B5", "R": "#9b59b6", "E": "#36965A", "C": "#d90429", "J": "#E07A2D"}
        bar_colors = [constell_colors.get(prn[0], "#777777") for prn in top.prn]
        bars = ax.barh(top.prn, top.p95_resp, color=bar_colors, height=0.6, alpha=0.85)
        for b, (_, row) in zip(bars, top.iterrows()):
            max_c = row.get("max_share_count", row.get("max_resp_count", 0))
            slips = row.get("total_slips", row.get("slip_count", 0))
            ax.text(b.get_width() + 0.05, b.get_y() + b.get_height()/2,
                    f"P95: {row.p95_resp:.2f}m | 最大次: {int(max_c)} | 周跳: {int(slips)}",
                    va="center", ha="left", fontsize=7.5, color="#333333")
        ax.set(xlabel="P95 伪距残差 (m)", ylabel="卫星 PRN", title="粗差卫星贡献度排行榜 (Top Outliers)")
        ax.set_xlim(0, max(float(top.p95_resp.max()) * 1.55, 3.0))
        ax.grid(self.grid_on.get(), alpha=0.3, axis="x")

    def _draw_directional_dop(self, ax):
        dop_df = diag.compute_directional_dop(self.data)
        if dop_df.empty:
            ax.text(0.5, 0.5, "无法计算方向 DOP (需 .stat 卫星仰角方位角)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("方向 DOP 几何精度因子")
            return
        ax.plot(dop_df.time, dop_df.EDOP, color="#2878B5", lw=1.1, label="EDOP (东向)")
        ax.plot(dop_df.time, dop_df.NDOP, color="#E07A2D", lw=1.1, label="NDOP (北向)")
        ax.plot(dop_df.time, dop_df.VDOP, color="#36965A", lw=1.1, label="VDOP (天向)")
        if "HDOP" in dop_df.columns:
            ax.plot(dop_df.time, dop_df.HDOP, color="#666666", lw=0.9, ls="--", label="HDOP (水平)")
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
        ax.plot(p.time, p.along_err, label="沿航向误差 (Along-track)", color="#2878B5", lw=0.9)
        ax.plot(p.time, p.cross_err, label="跨航向误差 (Cross-track)", color="#d90429", lw=0.9)
        ax.plot(p.time, p.vert_err, label="垂直高程误差 (Vertical)", color="#36965A", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set(xlabel="时间 (s)", ylabel="误差 (m)", title="车体坐标系误差分解 (Along-track / Cross-track / Vertical)")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.grid(self.grid_on.get(), alpha=0.3)

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
            sc1 = ax_sky.scatter(theta, r, c=np.clip(s_sub.resp.abs(), 0, 8), cmap="turbo", s=8, alpha=0.6, edgecolors="none")
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

        # 2. Left-Bottom: El vs SNR & Resp
        ax_el = self.fig.add_subplot(gs_left[1])
        s_snr = sat[(sat.freq == 1) & (sat.snr > 0)]
        if not s_snr.empty:
            s_sub2 = s_snr.iloc[::max(1, len(s_snr) // 15000)]
            sc2 = ax_el.scatter(s_sub2.el, s_sub2.snr, c=np.clip(s_sub2.resp.abs(), 0, 8), cmap="turbo", s=6, alpha=0.45, edgecolors="none")
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
            ax_dop.plot(dop_df.time, dop_df.EDOP, color='#2878B5', lw=1.1, label='EDOP (东向跨街)')
            ax_dop.plot(dop_df.time, dop_df.NDOP, color='#E07A2D', lw=1.1, label='NDOP (北向沿街)')
            ax_dop.plot(dop_df.time, dop_df.VDOP, color='#36965A', lw=1.1, label='VDOP (天向高程)')
            ax_dop.plot(dop_df.time, dop_df.PDOP, color='#111111', lw=1.2, ls=':', label='PDOP (空间几何)')
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
            ax_tri.plot(triad.time, triad.max_resp, color='#d90429', lw=1.1, label='第一大残差 (Max |resp|)')
            ax_tri.plot(triad.time, triad.sec_resp, color='#00b4d8', lw=0.95, label='第二大残差 (2nd Max)')
            ax_tri.plot(triad.time, triad.rms_resp, color='#0B2B52', lw=1.2, ls='--', label='历元均方根 (RMS)')
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
        for i in self.visible_panel_indices:
            panel=self.custom_panels[i]
            name=panel["mode"]
            rect=(.065,.09,.92,.82) if self.focused_custom_panel is not None else panel["rect"]
            is_time_series = name not in NON_TIME_SERIES_MODES
            share = base_time_ax if (self.sync_time_axes.get() and is_time_series and self.focused_custom_panel is None) else None
            ax=self.fig.add_axes(rect,projection="polar" if name in ("Skyplot", "Multipath Skyplot") else None, sharex=share)
            ax._panel_mode = name
            if self.sync_time_axes.get() and is_time_series and base_time_ax is None and self.focused_custom_panel is None:
                base_time_ax = ax
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
                if self.data["has_truth"]:ax.plot(full.gt_E,full.gt_N,"k--",lw=.7,label="Truth")
                for q,g in p.groupby("q"):ax.scatter(g.E,g.N,s=2,color=Q_COLORS.get(int(q),"#777"),label=Q_LABELS.get(int(q),str(q)))
                ax.set(xlabel="E (m)",ylabel="N (m)",title=name);ax.legend(fontsize=6,ncol=2);ax.set_aspect("equal" if self.equal_scale.get() else "auto")
            elif name=="ENU Position":
                for c,col in zip(["E","N","U"],["#2878B5","#E07A2D","#36965A"]):ax.plot(p.time,p[c],lw=.65,marker=".",ms=2,markevery=max(1,len(p)//50),label=c,color=col)
                ax.set(title=name,ylabel="Position (m)");ax.legend(fontsize=6,ncol=3)
            elif name=="Height":
                ax.plot(p.time,p.U,lw=.7,marker=".",ms=2,markevery=max(1,len(p)//50),label="RTK U")
                if self.data["has_truth"]:ax.plot(p.time,p.gt_U,lw=.7,ls="--",label="Truth U")
                ax.set(title=name,ylabel="Up (m)");ax.legend(fontsize=6)
            elif name=="ENU Error":
                if self.data["has_truth"]:
                    for c,col in zip(["de","dn","du"],["#2878B5","#E07A2D","#36965A"]):ax.plot(p.time,p[c],lw=.65,label=c.upper(),color=col)
                    ax.axhline(0,color="k",lw=.5);ax.legend(fontsize=6,ncol=3)
                else:ax.text(.5,.5,"No truth",ha="center",transform=ax.transAxes)
                ax.set(title=name,ylabel="Error (m)")
            elif name=="Solution Status":ax.step(p.time,p.q,where="post",lw=.65,marker=".",ms=2,markevery=max(1,len(p)//50));ax.set(title=name,ylabel="Q")
            elif name=="AR Ratio":ax.plot(p.time,p.ratio,lw=.65,marker=".",ms=2,markevery=max(1,len(p)//50));ax.axhline(3,color="r",ls="--");ax.set(title=name,ylabel="Ratio")
            elif name=="Differential Age":self._draw_differential_age(ax,p)
            elif name=="Satellites":self._draw_sats(ax)
            elif name=="DOP":self._draw_dop(ax)
            elif name=="Skyplot":self._draw_sky(ax)
            elif name=="C/N0":self._draw_cn0(ax);ax.set_title(name)
            elif name=="Elevation":self._draw_elevation(ax)
            elif name=="Residuals":self._draw_residuals(ax)
            elif name=="Multipath Proxy":self._draw_multipath(ax)
            elif name=="Cycle Slips":self._draw_slips(ax)
            elif name=="Satellite Visibility Gantt":self._draw_satellite_visibility_gantt(ax,s)
            elif name=="Multipath Skyplot":self._draw_multipath_skyplot(ax)
            elif name=="Per-PRN Residuals":self._draw_per_prn_residuals(ax)
            elif name=="Covariance":self._draw_covariance(ax,p)
            elif name=="False Fix Analysis":self._draw_false_fix(ax,p)
            elif name=="Innovation / NIS":self._draw_nis(ax)
            elif name=="Error CDF":self._draw_error_cdf(ax,p)
            elif name=="Error Histogram":self._draw_error_histogram([ax],p)
            elif name=="Constellation Summary":self._draw_constellation_summary([ax],s)
            elif name=="Residual Triad":self._draw_residual_triad(ax)
            elif name=="Env & Geometry Combined":self._draw_residual_triad(ax)
            elif name=="El vs SNR & Resp":self._draw_el_snr_resp(ax)
            elif name=="Top Outlier Satellites":self._draw_top_outliers(ax)
            elif name=="Directional DOP":self._draw_directional_dop(ax)
            elif name=="Clock Bias & ISB":self._draw_clock_bias_isb(ax)
            elif name=="Body-Frame Error":self._draw_body_frame_error(ax,p)
            ax.grid(self.grid_on.get(),alpha=.28)
            if is_time_series:
                self._set_datetime_axis(ax);ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis="x",labelsize=7,labelrotation=0)
                self.time_cursors.append(ax.axvline(0,color="#333",ls="--",lw=.7,alpha=.65))
            self._register_custom_dynamic(name,ax,p,full)
        for ax in self.plot_axes:
            box=ax.get_position()
            p_mode=getattr(ax, "_panel_mode", "")
            is_ts=p_mode not in NON_TIME_SERIES_MODES
            has_ts_panel_below=any(other is not ax and getattr(other, "_panel_mode", "") not in NON_TIME_SERIES_MODES
                and other.get_position().y1<box.y0-.005
                and min(box.x1,other.get_position().x1)-max(box.x0,other.get_position().x0)>.2*min(box.width,other.get_position().width)
                for other in self.plot_axes)
            if has_ts_panel_below and is_ts and self.sync_time_axes.get():ax.tick_params(axis="x",labelbottom=False);ax.set_xlabel("")
        self.fig.suptitle("Custom RTKLIB Analysis",fontsize=13,weight="bold")

    def _register_custom_dynamic(self,name,ax,p,full):
        """Add a current-epoch highlight to every plot in a custom grid."""
        def point(color="#111",marker="o",size=7,target=None):
            target=target if target is not None else ax
            return target.plot([],[],linestyle="None",marker=marker,ms=size,mfc=color,mec="white",mew=.7,zorder=12)[0]
        artists=[]
        if name=="Gnd Trk":
            trail,=ax.plot([],[],color="#ed3030",lw=1.4,zorder=8);artists=[trail,point("#ed3030","o",9)]
            if self.data["has_truth"]:artists.extend([point("#ffd92f","*",11),ax.plot([],[],color="#d22",ls=":",lw=1,zorder=9)[0]])
        elif name=="ENU Position":artists=[point(c) for c in ["#2878B5","#E07A2D","#36965A"]]
        elif name=="Height":
            artists=[point("#16853b")]
            if self.data["has_truth"]:artists.append(point("#252525"))
        elif name=="ENU Error" and self.data["has_truth"]:artists=[point(c) for c in ["#2878B5","#E07A2D","#36965A"]]
        elif name=="Solution Status":artists=[point("#333","o",8)]
        elif name=="AR Ratio":artists=[point("#333","o",8)]
        elif name=="Differential Age":artists=[point("#dc2626","o",8)]
        elif name=="Satellites":artists=[point("#2878B5"),point("#E07A2D")]
        elif name=="DOP":artists=[point(c) for c in ["#2878B5","#E07A2D","#36965A"]]
        elif name=="C/N0":artists=[ax.scatter([],[],s=24,edgecolors="#222",linewidths=.5,zorder=11) for _ in range(2)]
        elif name=="Elevation":artists=[ax.scatter([],[],s=18,marker="o",facecolors="none",edgecolors="#222",zorder=11)]
        elif name=="Residuals":artists=[ax.scatter([],[],s=24,marker="o",facecolors="none",edgecolors=c,zorder=11) for c in ["#2878B5","#E07A2D"]]
        elif name=="Multipath Proxy":artists=[ax.scatter([],[],s=24,marker="o",facecolors="none",edgecolors="#222",zorder=11)]
        elif name=="Cycle Slips":artists=[ax.scatter([],[],s=32,marker="x",color="#d00000",zorder=11)]
        elif name=="Satellite Visibility Gantt":artists=[ax.axvline(np.nan,color="#dc2626",ls="--",lw=1.0,alpha=.8,zorder=10)]
        elif name=="Multipath Skyplot":
            artists=[ax.scatter([],[],s=60,marker="*",edgecolors="#fff",linewidths=0.8,color="#dc2626",zorder=12)]
        elif name=="Per-PRN Residuals":
            artists=[point("#dc2626","o",7)]
        elif name=="Covariance":
            artists=[point(c) for c in ["#2878B5","#E07A2D","#36965A"]]
            if ax in self.cov_twin_axes:artists.append(point("#aa3377",target=self.cov_twin_axes[ax]))
        elif name=="False Fix Analysis":artists=[point("#07851d","o",9),point("#e3a008","o",9),point("#d00000","x",10)]
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
            if name=="Gnd Trk":
                start=max(0,i-120);stride=max(1,(i-start+1)//300)
                artists[0].set_data(track_e[start:i+1:stride],track_n[start:i+1:stride])
                if (i-start)%stride!=0:artists[0].set_data(np.r_[track_e[start:i+1:stride],track_e[i]],np.r_[track_n[start:i+1:stride],track_n[i]])
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
            elif name=="AR Ratio":set_point(artists[0],x,r.ratio,visible and np.isfinite(r.ratio))
            elif name=="Differential Age":
                age_val=float(getattr(r,"age",np.nan))
                set_point(artists[0],x,age_val,visible and np.isfinite(age_val))
            elif name=="Satellites" and sat_epoch is not None and not sat_epoch.empty:
                j=int(np.clip(np.searchsorted(sat_epoch.time.to_numpy(),x),0,len(sat_epoch)-1));row=sat_epoch.iloc[j]
                set_point(artists[0],x,row.tracked,visible);set_point(artists[1],x,row.used,visible)
            elif name=="DOP" and dop is not None and not dop.empty:
                j=int(np.clip(np.searchsorted(dop.time.to_numpy(),x),0,len(dop)-1));row=dop.iloc[j]
                for art,col in zip(artists,["PDOP","HDOP","VDOP"]):set_point(art,x,row[col],visible and np.isfinite(row[col]))
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
                for art,col in zip(artists[:3],["sigE","sigN","sigU"]):set_point(art,x,r[col],visible and r[col]>0)
                if self.data["has_truth"] and len(artists)>3:
                    ratio=abs(r.de)/(1.96*max(float(r.sigE),1e-12));set_point(artists[3],x,ratio,visible and np.isfinite(ratio))
            elif name=="False Fix Analysis" and self.data["has_truth"]:
                above=r.e3d>float(self.false_fix_limit.get())
                set_point(artists[0],x,r.e3d,visible and int(r.q)==1 and not above)
                set_point(artists[1],x,r.e3d,visible and int(r.q)==2 and not above)
                set_point(artists[2],x,r.e3d,visible and above)
            elif name=="Innovation / NIS":
                dt=gpst_to_local_datetime(r.week,r.tow);artists[0].set_text(f"当前历元：{dt:%Y-%m-%d %H:%M:%S} CST · Q={int(r.q)}")

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
        for ax,(col,label,color) in zip(axes,specs):
            ax.plot(p.time,p[col],color=color,**kw)
            ax.set_ylabel(label);ax.set_title(f"{label.split()[0]} component",loc="left",fontsize=9);ax.grid(self.grid_on.get(),ls="--",alpha=.4)
        axes[0].set_title("ENU Position Components",loc="center",fontsize=12,weight="bold")
        axes[-1].set_xlabel("Time from start (s)")

    def _draw_track(self,ax,p,full):
        self.track_axes.append(ax)
        self._track_e=full.E.to_numpy();self._track_n=full.N.to_numpy()
        self._apply_basemap(ax, full)
        ax.set_title("Ground Track / ENU Position — colored by solution status")
        if self.data["has_truth"]:
            ax.plot(full.gt_E,full.gt_N,color="#252525",lw=1.05,ls="--",alpha=.72,label="Ground truth")
        for q,g in p.groupby("q"):
            label=f"Q={q}: {Q_LABELS.get(q,'Other')}"; ax.plot(g.E,g.N,".",ms=3,color=Q_COLORS.get(q,"#666"),label=label,alpha=.9)
        ax.set_xlabel("East (m)");ax.set_ylabel("North (m)");ax.set_aspect("equal" if self.equal_scale.get() else "auto",adjustable="box");ax.legend(loc="upper left",fontsize=8)
        idx=min(int(self.slider.get()),len(full)-1); cur=full.iloc[idx]
        stride=max(1,(idx+1)//2500)
        self.vehicle_trail,=ax.plot(self._track_e[:idx+1:stride],self._track_n[:idx+1:stride],color="#ed3030",lw=1.8,alpha=.9,zorder=5)
        self.vehicle_marker,=ax.plot([cur.E],[cur.N],marker=(3,0,0),ms=17,mfc="#ff2727",mec="black",mew=1.0,linestyle="None",zorder=8)
        self.vehicle_label=ax.annotate(f"  {idx+1}/{len(full)}  Q={int(cur.q)}:{Q_LABELS.get(int(cur.q),'?')}",(cur.E,cur.N),xytext=(9,8),textcoords="offset points",fontsize=8,weight="bold",bbox=dict(boxstyle="round,pad=.25",fc="white",ec="#888",alpha=.9),zorder=9)
        if self.data["has_truth"] and np.isfinite(cur.gt_E) and np.isfinite(cur.gt_N):
            self.truth_marker,=ax.plot([cur.gt_E],[cur.gt_N],marker="*",ms=14,mfc="#ffd92f",mec="#222",mew=.8,linestyle="None",zorder=9,label="Current truth")
            self.error_link,=ax.plot([cur.E,cur.gt_E],[cur.N,cur.gt_N],color="#df2525",lw=1.5,ls=":",zorder=7)
        self._draw_stats(ax,full)

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
        for ax,(col,label,color) in zip(axes,specs):
            ax.plot(p.time,p[col],color=color,**kw);ax.axhline(0,color="#222",lw=.6);ax.set_ylabel(label);ax.grid(self.grid_on.get(),ls="--",alpha=.4)
        axes[0].set_title("ENU Position Error Components",fontsize=12,weight="bold");axes[-1].set_xlabel("Time from start (s)")

    def _draw_sats(self,ax):
        if "_cached_sat_epoch" not in self.data:
            self.data["_cached_sat_epoch"] = diag._sat_epoch(self.data)
        e = self.data["_cached_sat_epoch"]
        if e.empty:return
        kw=self._get_line_kwargs("Satellites",default_ls="-",default_lw=1.0)
        ax.plot(e.time,e.tracked,label="Tracked satellites",color="#2878B5",**kw);ax.plot(e.time,e.used,label="Used satellites",color="#e07a2d",**kw);ax.set(xlabel="Time from start (s)",ylabel="Satellites",title="Tracked and Used Satellites");ax.legend()

    def _draw_dop(self,ax):
        if "_cached_dop" not in self.data:
            self.data["_cached_dop"] = diag._compute_dop(self.data["sat"])
        d = self.data["_cached_dop"]
        self.dop_frames[ax]=d
        if d.empty:ax.text(.5,.5,"Not enough satellite geometry for DOP",ha="center",va="center",transform=ax.transAxes);return
        kw=self._get_line_kwargs("DOP",default_ls="-",default_lw=.8)
        for c in ["PDOP","HDOP","VDOP"]:ax.plot(d.time,d[c],label=c,**kw)
        ax.set(xlabel="Time from start (s)",ylabel="DOP",title="DOP from satellite geometry");ax.legend()

    def _draw_sky(self,ax):
        s=self.data["sat"]
        if s.empty:return
        ax.set_theta_zero_location("N");ax.set_theta_direction(-1);ax.set_ylim(90,0);ax.set_yticks([0,30,60,90]);ax.set_yticklabels(["90°","60°","30°","0°"])
        for (prn,con),g in s[s.freq==1].groupby(["prn","constellation"]):
            g=self._display_sample(g.sort_values("time"),400);th=np.deg2rad(g.az);r=90-g.el;ax.plot(th,r,lw=.55,alpha=.55,color=diag.COLORS.get(con,"#666"));k=int(np.nanargmax(g.el.to_numpy()));ax.text(th.iloc[k],r.iloc[k],prn,fontsize=6)
        self.sky_current=ax.scatter([],[],s=65,marker="*",edgecolors="#222",linewidths=.5,zorder=8)
        labels=[]
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
            g=self._display_sample(g,400);ax.plot(g.time,g.el,lw=.5,alpha=.45)
        ax.set(xlabel="Time from start (s)",ylabel="Elevation (°)",title="Satellite Elevation vs Time")

    def _draw_residuals(self,ax):
        s=self.data["sat"]
        if s.empty:return
        s=self._display_sample(s,60000)
        g=s[(s.resp!=0)&(s.freq==1)];ax.scatter(g.time,g.resp,s=3,alpha=.16,label="Pseudorange residual")
        h=s[(s.resc!=0)&(s.freq==1)];ax.scatter(h.time,h.resc*100,s=3,alpha=.12,label="Carrier residual (cm)")
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

        if "_cached_gantt" not in self.data:
            s_l1 = s[s.freq == 1].sort_values(["prn", "time"]).copy()
            if s_l1.empty:
                s_l1 = s.sort_values(["prn", "time"]).copy()
            prefix_order = {'G': 1, 'C': 2, 'R': 3, 'E': 4, 'J': 5, 'S': 6}
            sorted_prns = sorted(
                s_l1["prn"].unique(),
                key=lambda p: (prefix_order.get(p[0] if p else "", 99), int(p[1:]) if p[1:].isdigit() else 0, p)
            )
            prn_to_y = {prn: i for i, prn in enumerate(sorted_prns)}

            s_l1["dt"] = s_l1.groupby("prn")["time"].diff()
            dt_med = s_l1["dt"].median()
            time_gap = s_l1["dt"] > (dt_med * 2.5 if (dt_med and dt_med > 0) else 2.0)
            vsat_diff = s_l1.groupby("prn")["vsat"].diff().ne(0)
            s_l1["seg_id"] = (vsat_diff | time_gap).cumsum()
            grouped = s_l1.groupby(["prn", "seg_id"]).agg(
                t_start=("time", "first"),
                t_end=("time", "last"),
                vsat=("vsat", "first")
            ).reset_index()

            slips = s[s.slipc > 0][["time", "prn", "slipc"]]
            self.data["_cached_gantt"] = (grouped, sorted_prns, prn_to_y, slips)
        else:
            grouped, sorted_prns, prn_to_y, slips = self.data["_cached_gantt"]

        constellation_colors = {
            'G': '#2563eb', # GPS (Blue)
            'C': '#d97706', # BDS (Amber/Orange)
            'R': '#059669', # GLONASS (Green)
            'E': '#7c3aed', # Galileo (Purple)
            'J': '#db2777', # QZSS (Pink)
        }

        v1 = grouped[grouped.vsat == 1]
        if not v1.empty:
            y1 = [prn_to_y[p] for p in v1.prn]
            c1 = [constellation_colors.get(p[0] if p else '', '#334155') for p in v1.prn]
            x0 = v1.t_start.to_numpy(dtype=float)
            x1 = np.maximum(v1.t_end.to_numpy(dtype=float), x0 + 0.1)
            ax.hlines(y1, x0, x1, colors=c1, lw=3.8, alpha=0.88, zorder=2)

        v0 = grouped[grouped.vsat == 0]
        if not v0.empty:
            y0 = [prn_to_y[p] for p in v0.prn]
            x0 = v0.t_start.to_numpy(dtype=float)
            x1 = np.maximum(v0.t_end.to_numpy(dtype=float), x0 + 0.1)
            ax.hlines(y0, x0, x1, colors='#cbd5e1', lw=1.8, alpha=0.65, zorder=1)

        if not slips.empty:
            valid_slips = slips[slips.prn.isin(prn_to_y)]
            if not valid_slips.empty:
                slip_y = [prn_to_y[p] for p in valid_slips.prn]
                ax.scatter(valid_slips.time, slip_y, marker="|", color="#dc2626", s=50, lw=1.4, zorder=5)

        for i in range(1, len(sorted_prns)):
            if sorted_prns[i][0] != sorted_prns[i-1][0]:
                ax.axhline(i - 0.5, color="#cbd5e1", ls="--", lw=0.75, alpha=0.8)

        ax.set_yticks(range(len(sorted_prns)))
        ax.set_yticklabels(sorted_prns, fontsize=7 if len(sorted_prns) > 25 else 8)
        ax.set_ylim(-0.8, len(sorted_prns) - 0.2)
        ax.set(xlabel="Date/time", ylabel="Satellite PRN", title=f"Satellite Visibility & Cycle Slip Gantt ({len(sorted_prns)} Sats, {len(slips)} Slips)")

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#2563eb", lw=3.5, label="Tracked & Used (vsat=1)"),
            Line2D([0], [0], color="#cbd5e1", lw=2, label="Tracked Only (vsat=0)"),
            Line2D([0], [0], marker="|", color="#dc2626", ls="None", markersize=10, markeredgewidth=1.6, label="Cycle Slip (周跳)")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.88)

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

    def _draw_covariance(self,ax,p):
        for c,col in [("sigE","#2878B5"),("sigN","#E07A2D"),("sigU","#36965A")]:ax.plot(p.time,p[c],label=f"σ{c[-1]}",lw=.75,color=col)
        # Keep labels short enough for a half-width custom panel.  The full
        # covariance/coverage definition remains in the chart analysis text.
        ax.set_yscale("log");ax.set(xlabel="Time from start (s)",ylabel="1σ uncertainty (m)",title="Position uncertainty (RTK covariance)");ax.legend(loc="upper left",fontsize=8)
        if self.data["has_truth"]:
            ratio=np.abs(p.de)/np.maximum(1.96*p.sigE,1e-12);cap=max(1.0,float(ratio.quantile(.995)));twin=ax.twinx();self.cov_twin_axes[ax]=twin;twin.plot(p.time,ratio.clip(upper=cap),color="#aa3377",alpha=.55,lw=.65,label="|ΔE| / 1.96σE, capped P99.5");twin.axhline(1,color="#aa3377",ls=":",lw=.8);twin.set_ylim(0,cap*1.05);twin.set_ylabel("Error / 95% bound")

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
        elif name == "Position Error":
            if self.data["has_truth"]:
                ax.plot(p.time, p.eh, lw=.6, label="Horizontal")
                ax.plot(p.time, p.e3d, lw=.6, label="3D")
                ax.legend(fontsize=7 if not is_single else 9)
                ax.set(title="Position Error", xlabel="Time (s)", ylabel="m")
            else:
                ax.text(.5, .5, "No truth loaded", ha="center", transform=ax.transAxes)
                ax.set(title="Position Error")
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "Solution Status":
            ax.step(p.time, p.q, where="post", lw=.65)
            ax.set(title="Solution Status", xlabel="Time (s)", ylabel="Q")
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "AR Ratio":
            ax.plot(p.time, p.ratio, lw=.5)
            ax.axhline(3, color="red", ls="--")
            ax.set(title="AR Ratio", xlabel="Time (s)")
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
            ax.plot(p.time, p.U, lw=.7, label="RTK U")
            if self.data["has_truth"]: ax.plot(p.time, p.gt_U, lw=.7, ls="--", label="Truth U")
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
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "ENU Position":
            for c, col in zip(["E", "N", "U"], ["#2878B5", "#E07A2D", "#36965A"]):
                ax.plot(p.time, p[c], lw=.65, label=c, color=col)
            ax.set(title=name, ylabel="Position (m)")
            ax.legend(fontsize=6 if not is_single else 8, ncol=3)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "ENU Error":
            if self.data["has_truth"]:
                for c, col in zip(["de", "dn", "du"], ["#2878B5", "#E07A2D", "#36965A"]):
                    ax.plot(p.time, p[c], lw=.65, label=c.upper(), color=col)
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
            cur=current_sat[current_sat.freq==1].drop_duplicates("prn")
            offsets=np.c_[np.deg2rad(cur.az.to_numpy()),90-cur.el.to_numpy()]
            colors=[diag.COLORS.get(c,"#555555") for c in cur.constellation]
            for ax,points,labels in self.sky_plots:
                points.set_offsets(offsets);points.set_color(colors)
                az=cur.az.to_numpy();el=cur.el.to_numpy();prn=cur.prn.to_numpy()
                for j in range(len(cur)):
                    if j>=len(labels):
                        label=ax.text(0,0,"",fontsize=7,weight="bold",zorder=10);label.set_animated(True);labels.append(label);self.dynamic_artists.append(label)
                    labels[j].set_position((np.deg2rad(az[j]),90-el[j]));labels[j].set_text(str(prn[j]));labels[j].set_visible(True)
                for label in labels[len(cur):]:label.set_visible(False)
                dt=gpst_to_local_datetime(r.week,r.tow)
                ax.set_title(f"卫星天空图 · {dt:%Y-%m-%d %H:%M:%S} CST")
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
                # Keep y=0 fixed while zooming, matching the origin-centered
                # scaling users expect for signed position/error components.
                ax.set_ylim(ylim[0]*factor,ylim[1]*factor)
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
            else:ax.set_ylim(yl[0]*factor,yl[1]*factor)
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
            msg = (
                f"Google Earth 3D KML 文件已成功导出至：\n{path}\n\n"
                f"【提示】：此文件包含 3D 高程拉伸与时间轴，专用于「Google Earth (谷歌地球)」！\n"
                f"若在网页版「Google 我的地图 (My Maps)」中使用，请选择【My Maps 2D…】以防导入报错。\n\n"
                f"是否立即在 Google Earth 中打开？"
            )
            if messagebox.askyesno("导出成功", msg):
                try:
                    os.startfile(path)
                except Exception:
                    pass
        except Exception as exc:
            messagebox.showerror("导出 KML 失败", str(exc))

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
            ax.plot(full.gt_E, full.gt_N, color="#252525", lw=1.1, ls="--", alpha=0.75, label="Ground Truth", zorder=3)
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
        elif mode_name == "Height":
            kw = self._get_line_kwargs("Height")
            ax.plot(p.time, p.U, label="RTK U", **kw)
            if self.data["has_truth"]:
                ax.plot(p.time, p.gt_U, ls="--", color="#cf4b43", lw=kw.get("lw", 1.2), label="Truth U")
            ax.set(title="Height (Up component)", xlabel="Date/time", ylabel="Up (m)")
            ax.legend(loc="best")
            self._set_datetime_axis(ax)
        elif mode_name == "Solution Status":
            ax.step(p.time, p.q, where="post", lw=1.2, color="#0B2B52")
            ax.set(title="Solution Status (Q: 1=FIX, 2=FLOAT, 5=SINGLE)", xlabel="Date/time", ylabel="Q")
            self._set_datetime_axis(ax)
        elif mode_name == "AR Ratio":
            kw = self._get_line_kwargs("AR Ratio")
            ax.plot(p.time, p.ratio, color="#0B2B52", **kw)
            ax.axhline(3, color="red", ls="--", lw=1.2, label="Threshold = 3.0")
            ax.set(title="Ambiguity Resolution (AR) Ratio", xlabel="Date/time", ylabel="Ratio")
            ax.legend(loc="best")
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
            self._set_datetime_axis(ax)
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
                ax.plot(p.time, p.eh, lw=.8, label="Horizontal")
                ax.plot(p.time, p.e3d, lw=.8, label="3D")
                ax.legend(fontsize=8)
                ax.set(title="Position Error", xlabel="Time (s)", ylabel="Error (m)")
            else:
                ax.text(.5, .5, "No truth loaded", ha="center", transform=ax.transAxes)
                ax.set(title="Position Error")
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
