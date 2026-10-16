"""Interactive RTKLIB-style plotter for .pos and optional .stat/reference files."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import rtklib_diagnostics as diag

Q_LABELS = {1: "FIX", 2: "FLOAT", 3: "SBAS", 4: "DGPS", 5: "SINGLE", 6: "PPP"}
Q_COLORS = {1: "#07851d", 2: "#dc8a12", 3: "#5b83c4", 4: "#9b59b6", 5: "#777777", 6: "#17a2a4"}
MODES = ["Gnd Trk", "ENU Position", "Height", "ENU Error", "Solution Status",
         "AR Ratio", "Satellites", "DOP", "Skyplot", "C/N0", "Elevation", "Residuals",
         "Multipath Proxy", "Cycle Slips", "Covariance", "False Fix Analysis", "Innovation / NIS",
         "Error CDF", "Error Histogram", "Constellation Summary"]


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
        filem.add_command(label="Export Current Plot…",command=self.export_plot,accelerator="Ctrl+S")
        filem.add_command(label="Batch Export Subplots…",command=self.open_batch_export_dialog,accelerator="Ctrl+E")
        filem.add_separator(); filem.add_command(label="Exit",command=self.root.destroy); bar.add_cascade(label="File",menu=filem)
        edit=tk.Menu(bar,tearoff=False); edit.add_command(label="Save Plot As…",command=self.export_plot); edit.add_command(label="Reset View",command=self.redraw); bar.add_cascade(label="Edit",menu=edit)
        view=tk.Menu(bar,tearoff=False)
        view.add_checkbutton(label="Grid",variable=self.grid_on,command=self.redraw)
        view.add_checkbutton(label="Hover scrubs timeline",variable=self.follow_mouse)
        view.add_checkbutton(label="Click to maximize / restore",variable=self.click_to_maximize)
        view.add_separator()
        view.add_command(label="Delete selected subplot",command=self.delete_selected_subplot,accelerator="Del")
        view.add_command(label="Reset subplots layout",command=self.reset_subplots_layout)
        view.add_command(label="Fit to Data",command=self.redraw); bar.add_cascade(label="View",menu=view)
        windows=tk.Menu(bar,tearoff=False)
        windows.add_command(label="Dashboard",command=lambda:self.set_mode("Dashboard"))
        windows.add_command(label="Multi-Compare",command=lambda:self.set_mode("Multi-Compare"))
        bar.add_cascade(label="Windows",menu=windows)
        helpm=tk.Menu(bar,tearoff=False); helpm.add_command(label="About",command=lambda:messagebox.showinfo("About RTKLIB Plot - Python","Interactive RTKLIB result viewer\nReads .pos, optional .pos.stat and reference CSV.")); bar.add_cascade(label="Help",menu=helpm)
        self.root.config(menu=bar)

    def _build_controls(self):
        row=ttk.Frame(self.root,padding=(8,3)); row.pack(side=tk.TOP,fill=tk.X)
        ttk.Button(row,text="Open…",command=self.open_file).pack(side=tk.LEFT,padx=(0,8))
        ttk.Label(row,text="Plot:").pack(side=tk.LEFT)
        self.mode_box=ttk.Combobox(row,textvariable=self.mode,values=MODES+['Dashboard','Custom Analysis','Multi-Compare'],state="readonly",width=18); self.mode_box.pack(side=tk.LEFT,padx=5); self.mode_box.bind("<<ComboboxSelected>>",lambda _e:self.redraw())
        ttk.Label(row,text="Solution:").pack(side=tk.LEFT,padx=(10,0))
        self.qbox=ttk.Combobox(row,textvariable=self.qfilter,values=["All Q"]+[f"Q={q}: {name}" for q,name in Q_LABELS.items()],state="readonly",width=13); self.qbox.pack(side=tk.LEFT,padx=5); self.qbox.bind("<<ComboboxSelected>>",lambda _e:self.redraw())
        ttk.Checkbutton(row,text="Grid",variable=self.grid_on,command=self.redraw).pack(side=tk.LEFT,padx=8)

        nav=ttk.Frame(self.root,padding=(8,2));nav.pack(side=tk.TOP,fill=tk.X)
        ttk.Label(nav,text="Drag: pan · Right-drag: zoom · Wheel Y at 0:").pack(side=tk.LEFT,padx=(0,4))
        self.zoom_box=ttk.Combobox(nav,textvariable=self.zoom_axis,values=["Both axes","X axis","Y axis"],state="readonly",width=11);self.zoom_box.pack(side=tk.LEFT,padx=4)
        ttk.Checkbutton(nav,text="Equal scale",variable=self.equal_scale,command=self._apply_aspect).pack(side=tk.LEFT,padx=3)
        ttk.Button(nav,text="Fit",command=self.fit_view).pack(side=tk.LEFT,padx=3)
        ttk.Label(nav,text="3D error limit (m):").pack(side=tk.LEFT,padx=(6,0))
        self.false_fix_box=ttk.Spinbox(nav,from_=0.01,to=10.0,increment=0.05,textvariable=self.false_fix_limit,width=5,command=self.redraw);self.false_fix_box.pack(side=tk.LEFT,padx=3)
        self.false_fix_box.bind("<Return>",lambda _e:self.redraw());self.false_fix_box.bind("<FocusOut>",lambda _e:self.redraw())
        ttk.Label(nav,text="Speed:").pack(side=tk.LEFT,padx=(7,0))
        self.speed_box=ttk.Combobox(nav,textvariable=self.speed,values=["1x","10x","50x"],state="readonly",width=5);self.speed_box.pack(side=tk.LEFT,padx=3)
        ttk.Checkbutton(nav,text="Hover scrubs timeline",variable=self.follow_mouse).pack(side=tk.LEFT,padx=(5,0))
        ttk.Checkbutton(nav,text="Click to Maximize",variable=self.click_to_maximize).pack(side=tk.LEFT,padx=(5,0))
        ttk.Button(nav,text="🗑️ Del Subplot",command=self.delete_selected_subplot).pack(side=tk.LEFT,padx=(4,0))
        ttk.Button(nav,text="↺ Reset",command=self.reset_subplots_layout).pack(side=tk.LEFT,padx=(2,0))
        self.epoch_toggle_btn=ttk.Button(nav,text="Hide Epoch slider",command=self._toggle_epoch_slider);self.epoch_toggle_btn.pack(side=tk.LEFT,padx=(5,0))
        ttk.Button(nav,text="◀ Page",command=lambda:self._change_custom_page(-1)).pack(side=tk.LEFT,padx=(10,1))
        self.custom_page_label=ttk.Label(nav,text="Page 1 / 1",anchor="center",width=9);self.custom_page_label.pack(side=tk.LEFT,padx=1)
        ttk.Button(nav,text="Page ▶",command=lambda:self._change_custom_page(1)).pack(side=tk.LEFT,padx=1)
        ttk.Button(nav,text="📸 Batch Export…",command=self.open_batch_export_dialog).pack(side=tk.RIGHT,padx=4)
        ttk.Button(nav,text="Export Image…",command=self.export_plot).pack(side=tk.RIGHT,padx=4)
        self.play_btn=ttk.Button(nav,text="▶ Play",command=self.toggle_play); self.play_btn.pack(side=tk.RIGHT,padx=4)
        marks=ttk.Frame(self.root,padding=(8,2));marks.pack(side=tk.TOP,fill=tk.X)
        ttk.Button(marks,text="Set start",command=self.mark_segment_start).pack(side=tk.LEFT,padx=(0,3))
        ttk.Button(marks,text="Set end & add",command=self.add_segment).pack(side=tk.LEFT,padx=3)
        ttk.Button(marks,text="Clear",command=self.clear_segments).pack(side=tk.LEFT,padx=3)
        ttk.Button(marks,text="🔍 Auto Detect Issues",command=self.auto_detect_issues).pack(side=tk.LEFT,padx=4)
        ttk.Button(marks,text="📋 Diagnostic Report",command=self.open_report_dialog).pack(side=tk.LEFT,padx=4)
        ttk.Button(marks,text="📸 Export Subplots…",command=self.open_batch_export_dialog).pack(side=tk.LEFT,padx=4)
        self.segment_status=ttk.Label(marks,text="Shift+Drag to mark segment · Double-click segment to zoom · Double-click subplot to maximize.",anchor="w");self.segment_status.pack(side=tk.LEFT,padx=8,fill=tk.X,expand=True)

    def _build_plot(self):
        self.fig=plt.Figure(figsize=(10,6),dpi=100,facecolor="white"); self.ax=self.fig.add_subplot(111)
        self.main_pane=ttk.Panedwindow(self.root,orient=tk.HORIZONTAL);self.main_pane.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        self.sidebar=ttk.Frame(self.main_pane,padding=5);self.plot_frame=ttk.Frame(self.main_pane)
        self.main_pane.add(self.sidebar,weight=0);self.main_pane.add(self.plot_frame,weight=1)
        
        # Standard sidebar (Available plots, Custom grid, Segments)
        self.standard_sidebar=ttk.Frame(self.sidebar); self.standard_sidebar.pack(fill=tk.BOTH,expand=True)
        ttk.Label(self.standard_sidebar,text="Available plots",font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(0,4))
        list_frame=ttk.Frame(self.standard_sidebar);list_frame.pack(fill=tk.BOTH,expand=True)
        self.plot_list=tk.Listbox(list_frame,selectmode=tk.EXTENDED,exportselection=False,width=23,height=8)
        list_scroll=ttk.Scrollbar(list_frame,orient=tk.VERTICAL,command=self.plot_list.yview);self.plot_list.configure(yscrollcommand=list_scroll.set)
        self.plot_list.pack(side=tk.LEFT,fill=tk.BOTH,expand=True);list_scroll.pack(side=tk.RIGHT,fill=tk.Y)
        self.plot_list.bind("<ButtonPress-1>",self._begin_plot_drag,add="+")
        # Keep Tk's Listbox class binding from extending the selection across
        # every row while a plot is being dragged to the canvas.
        self.plot_list.bind("<B1-Motion>",lambda _e:"break",add="+")
        # Tk's implicit mouse grab sends release to the drag-origin widget.
        self.plot_list.bind("<ButtonRelease-1>",self._drop_plot_on_panel,add="+")
        self.root.bind_all("<ButtonRelease-1>",self._drop_plot_on_panel,add="+")
        for name in MODES:self.plot_list.insert(tk.END,name)
        ttk.Button(self.standard_sidebar,text="Show selected plots",command=self.show_selected).pack(fill=tk.X,pady=(4,1))
        ttk.Button(self.standard_sidebar,text="Select all",command=lambda:self.plot_list.select_set(0,tk.END)).pack(fill=tk.X,pady=1)
        ttk.Button(self.standard_sidebar,text="Clear",command=lambda:self.plot_list.selection_clear(0,tk.END)).pack(fill=tk.X,pady=1)

        # Line & Curve Style Controls
        ttk.Separator(self.standard_sidebar,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)
        style_frame=ttk.LabelFrame(self.standard_sidebar,text="Line & Curve Styles",padding=4)
        style_frame.pack(fill=tk.X,pady=(1,4))
        r_s1=ttk.Frame(style_frame);r_s1.pack(fill=tk.X,pady=1)
        ttk.Label(r_s1,text="Target:").pack(side=tk.LEFT)
        ttk.Combobox(r_s1,textvariable=self.line_scope,values=["All Subplots","Current Subplot"],state="readonly",width=13).pack(side=tk.RIGHT,fill=tk.X,expand=True)
        r_s2=ttk.Frame(style_frame);r_s2.pack(fill=tk.X,pady=1)
        ttk.Label(r_s2,text="Line:").pack(side=tk.LEFT)
        c_ls=ttk.Combobox(r_s2,textvariable=self.line_style,values=["Solid (―)","Dashed (---)","Dotted (···)","Dash-Dot (-·-)","Scatter Only","Line + Marker"],state="readonly",width=13)
        c_ls.pack(side=tk.RIGHT,fill=tk.X,expand=True);c_ls.bind("<<ComboboxSelected>>",lambda _e:self._on_style_changed())
        r_s3=ttk.Frame(style_frame);r_s3.pack(fill=tk.X,pady=1)
        ttk.Label(r_s3,text="Width:").pack(side=tk.LEFT)
        sp_w=ttk.Spinbox(r_s3,from_=0.5,to=5.0,increment=0.25,textvariable=self.line_width,width=4,command=self._on_style_changed)
        sp_w.pack(side=tk.LEFT,padx=2);sp_w.bind("<Return>",lambda _e:self._on_style_changed())
        ttk.Label(r_s3,text="Marker:").pack(side=tk.LEFT,padx=(2,0))
        c_m=ttk.Combobox(r_s3,textvariable=self.marker_style,values=["None","Circle (o)","Square (s)","Point (.)","Cross (+)"],state="readonly",width=8)
        c_m.pack(side=tk.RIGHT,fill=tk.X,expand=True);c_m.bind("<<ComboboxSelected>>",lambda _e:self._on_style_changed())
        r_s4=ttk.Frame(style_frame);r_s4.pack(fill=tk.X,pady=(2,1))
        ttk.Button(r_s4,text="Reset",command=self._reset_curve_styles).pack(side=tk.LEFT,fill=tk.X,expand=True,padx=1)
        ttk.Button(r_s4,text="Apply",command=self._on_style_changed).pack(side=tk.RIGHT,fill=tk.X,expand=True,padx=1)

        ttk.Separator(self.standard_sidebar,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)
        ttk.Label(self.standard_sidebar,text="Marked time segments",font=("Segoe UI",9,"bold")).pack(anchor="w")
        seg_frame=ttk.Frame(self.standard_sidebar);seg_frame.pack(fill=tk.X,pady=(2,0))
        self.segment_box=tk.Listbox(seg_frame,height=3,width=23,exportselection=False)
        self.segment_box.bind("<Double-Button-1>",self.on_segment_double_click)
        seg_scroll=ttk.Scrollbar(seg_frame,orient=tk.VERTICAL,command=self.segment_box.yview);self.segment_box.configure(yscrollcommand=seg_scroll.set)
        self.segment_box.pack(side=tk.LEFT,fill=tk.X,expand=True);seg_scroll.pack(side=tk.RIGHT,fill=tk.Y)
        ttk.Button(self.standard_sidebar,text="Delete selected segment",command=self.delete_segment).pack(fill=tk.X,pady=(2,0))
        ttk.Button(self.standard_sidebar,text="📸 Export Segment Subplots…",command=self.open_batch_export_dialog).pack(fill=tk.X,pady=(2,0))

        ttk.Separator(self.standard_sidebar,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)
        ttk.Label(self.standard_sidebar,text="Custom layout · click a panel first",font=("Segoe UI",9,"bold")).pack(anchor="w")
        ttk.Button(self.standard_sidebar,text="Split left / right",command=lambda:self._split_custom_panel("vertical")).pack(fill=tk.X,pady=1)
        ttk.Button(self.standard_sidebar,text="Split top / bottom",command=lambda:self._split_custom_panel("horizontal")).pack(fill=tk.X,pady=1)
        ttk.Button(self.standard_sidebar,text="Put selected plot in panel",command=self._assign_selected_plot).pack(fill=tk.X,pady=1)
        ttk.Button(self.standard_sidebar,text="Show selected panel only",command=self._focus_custom_panel).pack(fill=tk.X,pady=1)
        ttk.Button(self.standard_sidebar,text="Clear selected panel",command=self._clear_custom_panel).pack(fill=tk.X,pady=1)
        ttk.Button(self.standard_sidebar,text="Repack panels",command=self._reset_custom_grid).pack(fill=tk.X,pady=1)

        # Multi-compare sidebar (Solution overlay manager)
        self.compare_sidebar=ttk.Frame(self.sidebar)
        ttk.Label(self.compare_sidebar,text="Multi-Compare",font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(0,4))
        ttk.Label(self.compare_sidebar,text="Overlay 2–4 solutions side-by-side with ground truth.",wraplength=165,justify=tk.LEFT).pack(anchor="w",pady=(0,6))
        ttk.Button(self.compare_sidebar,text="+ Add Solution (.pos)…",command=self.add_compare_solution).pack(fill=tk.X,pady=2)
        ttk.Button(self.compare_sidebar,text="Open Reference (.csv)…",command=self.open_reference_file).pack(fill=tk.X,pady=2)
        ttk.Separator(self.compare_sidebar,orient=tk.HORIZONTAL).pack(fill=tk.X,pady=6)
        ttk.Label(self.compare_sidebar,text="Loaded Solutions:",font=("Segoe UI",9,"bold")).pack(anchor="w")
        self.compare_list_container=ttk.Frame(self.compare_sidebar);self.compare_list_container.pack(fill=tk.BOTH,expand=True,pady=(4,6))
        ttk.Button(self.compare_sidebar,text="Clear Extra Solutions",command=self.clear_compare_solutions).pack(fill=tk.X,pady=2)

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
        self.segment_status.configure(text=f"Added Segment {number}: {self._epoch_datetime_label(lo)} — {self._epoch_datetime_label(hi)} CST")
        self._refresh_segment_overlays()

    def _refresh_segment_list(self):
        if not hasattr(self,"segment_box"):return
        self.segment_box.delete(0,tk.END)
        for i,seg in enumerate(self.segments):
            self.segment_box.insert(tk.END,f"{seg['name']}: {self._epoch_datetime_label(seg['start'])} – {self._epoch_datetime_label(seg['end'])}")
            self.segment_box.itemconfig(i,fg=seg["color"])

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
        self.segment_status.configure(text=f"🔍 已自动缩放并聚焦至时段: {seg['name']} ({self._epoch_datetime_label(seg['start'])} — {self._epoch_datetime_label(seg['end'])} CST)")

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
        messagebox.showinfo("智能诊断完成", f"共自动探测出 {len(detected)} 处异常/跳固定时段（新增标记 {new_count} 处）。\n\n提示：双击左侧列表中的任意时段即可快速缩放聚焦！")
        self.segment_status.configure(text=f"已自动标记 {len(detected)} 处异常时段。双击左侧时段列表可自动缩放聚焦。")

    def open_report_dialog(self):
        if self.data is None:
            messagebox.showwarning("Warning", "请先加载 .pos 解算数据后再生成诊断报告。")
            return
        DiagnosticReportDialog(self)

    def show_selected(self):
        if self.data is None:return
        selected=[MODES[i] for i in self.plot_list.curselection()]
        if not selected:return
        self.custom_modes=selected;self.custom_panels=self._initial_custom_panels(selected);self.custom_page=0;self.selected_custom_panel=0;self.focused_custom_panel=None
        self.mode.set("Custom Analysis");self.redraw()

    @staticmethod
    def _initial_custom_panels(modes):
        n=len(modes);page_size=6;cols=1 if n==1 else 2;rows=3
        left,right,bottom,top=.075,.985,.105,.91;gap_x=.07;gap_y=.075
        width=(right-left-gap_x*(cols-1))/cols;height=(top-bottom-gap_y*(rows-1))/rows
        panels=[]
        for i,name in enumerate(modes):
            row,col=divmod(i%page_size,cols);x=left+col*(width+gap_x);y=top-(row+1)*height-row*gap_y
            panels.append({"mode":name,"rect":(x,y,width,height)})
        return panels

    def _change_custom_page(self,step):
        if self.mode.get()!="Custom Analysis" or not self.custom_panels:return
        pages=max(1,(len(self.custom_panels)+self.custom_page_size-1)//self.custom_page_size)
        self.custom_page=(self.custom_page+step)%pages
        self.selected_custom_panel=min(self.custom_page*self.custom_page_size,len(self.custom_panels)-1)
        self.focused_custom_panel=None;self.redraw()

    def _begin_plot_drag(self,event):
        index=self.plot_list.nearest(event.y)
        if 0<=index<len(MODES) and self.plot_list.bbox(index):self._drag_plot_name=MODES[index]

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
        name=MODES[selected[-1]]
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
        if self.mode.get()!="Custom Analysis":return
        modes=[p["mode"] for p in self.custom_panels if p["mode"]]
        if not modes:modes=[MODES[i] for i in self.plot_list.curselection()]
        if not modes:return
        self.custom_modes=modes;self.custom_panels=self._initial_custom_panels(modes);self.custom_page=0;self.selected_custom_panel=0;self.focused_custom_panel=None;self.redraw()

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
                self.custom_panels=self._initial_custom_panels(modes)
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
        if mode=="Multi-Compare":
            self._toggle_sidebar_mode("multi_compare")
            self._draw_multi_compare();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        else:
            self._toggle_sidebar_mode("standard")
        if mode=="Dashboard":
            self._draw_dashboard();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        if mode=="Custom Analysis":
            self._draw_custom_grid();self._apply_segment_overlays();self._register_dynamic_artists();self.canvas.draw_idle();self._update_cursor();self.update_analysis();return
        if mode in ("ENU Position","ENU Error","Error Histogram"):
            sharex = (mode != "Error Histogram")
            axes=self.fig.subplots(3,1,sharex=sharex);self.plot_axes=list(axes);self.ax=axes[0]
        elif mode=="Constellation Summary":
            axes=self.fig.subplots(1,2);self.plot_axes=list(axes);self.ax=axes[0]
        else:
            self.ax=self.fig.add_subplot(111,projection="polar" if mode=="Skyplot" else None);self.plot_axes=[self.ax]
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
        elif mode=="Satellites":self._draw_sats(ax)
        elif mode=="DOP":self._draw_dop(ax)
        elif mode=="Skyplot":self._draw_sky(ax)
        elif mode=="C/N0":self._draw_cn0(ax)
        elif mode=="Elevation":self._draw_elevation(ax)
        elif mode=="Residuals":self._draw_residuals(ax)
        elif mode=="Multipath Proxy":self._draw_multipath(ax)
        elif mode=="Cycle Slips":self._draw_slips(ax)
        elif mode=="Covariance":self._draw_covariance(ax,p)
        elif mode=="False Fix Analysis":self._draw_false_fix(ax,p)
        elif mode=="Innovation / NIS":self._draw_nis(ax)
        elif mode=="Error CDF":self._draw_error_cdf(ax,p)
        elif mode=="Error Histogram":self._draw_error_histogram(self.plot_axes,p)
        elif mode=="Constellation Summary":self._draw_constellation_summary(self.plot_axes,s)
        for plot_ax in self.plot_axes:plot_ax.grid(self.grid_on.get(),linestyle="--",alpha=.45)
        if mode not in ("Gnd Trk","Skyplot","Innovation / NIS","Error CDF","Error Histogram","Constellation Summary"):
            for plot_ax in self.plot_axes:self.time_cursors.append(plot_ax.axvline(0,color="#333",ls="--",lw=.8,alpha=.7,zorder=10))
        if mode in ("ENU Position","ENU Error","Height","Solution Status","AR Ratio","Satellites","DOP","C/N0","Elevation","Residuals","Multipath Proxy","Cycle Slips","Covariance","False Fix Analysis"):
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
        counts = s1.groupby(["time", "constellation"]).prn.nunique().unstack(fill_value=0)
        valid_snr = s1[s1.snr > 0]
        con_cols = [c for c in counts.columns if c in diag.COLORS] or list(counts.columns)

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
        for i in self.visible_panel_indices:
            panel=self.custom_panels[i]
            name=panel["mode"]
            rect=(.065,.09,.92,.82) if self.focused_custom_panel is not None else panel["rect"]
            ax=self.fig.add_axes(rect,projection="polar" if name=="Skyplot" else None);self.plot_axes.append(ax)
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
            elif name=="Satellites":self._draw_sats(ax)
            elif name=="DOP":self._draw_dop(ax)
            elif name=="Skyplot":self._draw_sky(ax)
            elif name=="C/N0":self._draw_cn0(ax);ax.set_title(name)
            elif name=="Elevation":self._draw_elevation(ax)
            elif name=="Residuals":self._draw_residuals(ax)
            elif name=="Multipath Proxy":self._draw_multipath(ax)
            elif name=="Cycle Slips":self._draw_slips(ax)
            elif name=="Covariance":self._draw_covariance(ax,p)
            elif name=="False Fix Analysis":self._draw_false_fix(ax,p)
            elif name=="Innovation / NIS":self._draw_nis(ax)
            elif name=="Error CDF":self._draw_error_cdf(ax,p)
            elif name=="Error Histogram":self._draw_error_histogram([ax],p)
            elif name=="Constellation Summary":self._draw_constellation_summary([ax],s)
            ax.grid(self.grid_on.get(),alpha=.28)
            if name not in ("Gnd Trk","Skyplot","Innovation / NIS","Error CDF","Error Histogram","Constellation Summary"):
                self._set_datetime_axis(ax);ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis="x",labelsize=7,labelrotation=0)
                self.time_cursors.append(ax.axvline(0,color="#333",ls="--",lw=.7,alpha=.65))
            self._register_custom_dynamic(name,ax,p,full)
        for ax in self.plot_axes:
            box=ax.get_position()
            has_panel_below=any(other is not ax and other.get_position().y1<box.y0-.005
                and min(box.x1,other.get_position().x1)-max(box.x0,other.get_position().x0)>.2*min(box.width,other.get_position().width)
                for other in self.plot_axes)
            if has_panel_below:ax.tick_params(axis="x",labelbottom=False);ax.set_xlabel("")
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
        elif name=="Satellites":artists=[point("#2878B5"),point("#E07A2D")]
        elif name=="DOP":artists=[point(c) for c in ["#2878B5","#E07A2D","#36965A"]]
        elif name=="C/N0":artists=[ax.scatter([],[],s=24,edgecolors="#222",linewidths=.5,zorder=11) for _ in range(2)]
        elif name=="Elevation":artists=[ax.scatter([],[],s=18,marker="o",facecolors="none",edgecolors="#222",zorder=11)]
        elif name=="Residuals":artists=[ax.scatter([],[],s=24,marker="o",facecolors="none",edgecolors=c,zorder=11) for c in ["#2878B5","#E07A2D"]]
        elif name=="Multipath Proxy":artists=[ax.scatter([],[],s=24,marker="o",facecolors="none",edgecolors="#222",zorder=11)]
        elif name=="Cycle Slips":artists=[ax.scatter([],[],s=32,marker="x",color="#d00000",zorder=11)]
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
        e=diag._sat_epoch(self.data)
        if e.empty:return
        kw=self._get_line_kwargs("Satellites",default_ls="-",default_lw=1.0)
        ax.plot(e.time,e.tracked,label="Tracked satellites",color="#2878B5",**kw);ax.plot(e.time,e.used,label="Used satellites",color="#e07a2d",**kw);ax.set(xlabel="Time from start (s)",ylabel="Satellites",title="Tracked and Used Satellites");ax.legend()

    def _draw_dop(self,ax):
        d=diag._compute_dop(self.data["sat"])
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

    def _draw_multipath(self,ax):
        s=self.data["sat"];g=s[(s.freq==1)&(s.resp!=0)]
        if g.empty:ax.text(.5,.5,"No usable pseudorange residuals",ha="center",va="center",transform=ax.transAxes);return
        g=self._display_sample(g,60000)
        ax.scatter(g.el,g.resp.abs(),s=4,alpha=.14);ax.set(xlabel="Elevation (°)",ylabel="|Pseudorange post-fit residual| (m)",title="Multipath diagnostic proxy (not a direct MP combination)")

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
        elif name == "Covariance":
            self._draw_covariance(ax, p)
            self._set_datetime_axis(ax)
            self.time_cursors.append(ax.axvline(0, color="#333", ls="--", lw=.7, alpha=.65))
        elif name == "False Fix Analysis":
            self._draw_false_fix(ax, p)
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
            rlat,rlon,rh=ecef_to_geodetic(r.x,r.y,r.z)
            ref=self.data["reference"];tow=float(r.tow)
            gt_xyz=[np.interp(tow,ref["GPS TOW (s)"],ref[c],left=np.nan,right=np.nan) for c in ["ECEF X (m)","ECEF Y (m)","ECEF Z (m)"]]
            glat,glon,gh=ecef_to_geodetic(*gt_xyz)
            de,dn,du=float(r.de),float(r.dn),float(r.du);eh=float(r.eh);e3d=float(r.e3d)
            self.truth_info.configure(text=(f"RTK: {rlat:.8f}°N, {rlon:.8f}°E, h={rh:.3f} m     "
                f"Truth: {glat:.8f}°N, {glon:.8f}°E, h={gh:.3f} m\n"
                f"Error RTK−Truth: ΔE={de:+.3f} m, ΔN={dn:+.3f} m, ΔU={du:+.3f} m     "
                f"Horizontal={eh:.3f} m, 3D={e3d:.3f} m"))
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
                    self.segment_status.configure(text=f"Shift-drag added Segment {number}: {self._epoch_datetime_label(idx_lo)} — {self._epoch_datetime_label(idx_hi)} CST")
            self.canvas.draw_idle();return
        if self._drag_epoch:
            self._drag_epoch=False;self.canvas.get_tk_widget().configure(cursor="");return
        press=self._press_state;self._press_state=None
        self.canvas.get_tk_widget().configure(cursor="")
        if not press:return
        ax=press["ax"]
        if not press["moved"]:
            if self.click_to_maximize.get():
                if self.mode.get()=="Dashboard":
                    if event.button==1 and event.inaxes in self.plot_axes:
                        clicked_idx=self.plot_axes.index(event.inaxes)
                        if self.focused_dashboard_subplot!=clicked_idx:
                            self.focused_dashboard_subplot=clicked_idx
                            self.redraw()
                        return
                    elif event.button==3 and self.focused_dashboard_subplot is not None:
                        self.focused_dashboard_subplot=None
                        self.redraw()
                        return
                    elif event.button==3:
                        self._restore_axis(ax);return
                elif self.mode.get()=="Multi-Compare":
                    if event.button==1 and event.inaxes in self.plot_axes:
                        clicked_idx=self.plot_axes.index(event.inaxes)
                        if self.focused_compare_subplot!=clicked_idx:
                            self.focused_compare_subplot=clicked_idx
                            self.redraw()
                        return
                    elif event.button==3 and self.focused_compare_subplot is not None:
                        self.focused_compare_subplot=None
                        self.redraw()
                        return
                    elif event.button==3:
                        self._restore_axis(ax);return
                elif self.mode.get()=="Custom Analysis":
                    if event.button==1 and event.inaxes in self.plot_axes:
                        panel_index=self.selected_custom_panel
                        if self.focused_custom_panel!=panel_index:
                            self.focused_custom_panel=panel_index
                            self.redraw()
                        return
                    elif event.button==3 and self.focused_custom_panel is not None:
                        self.focused_custom_panel=None
                        self.redraw()
                        return
                    elif event.button==3:
                        self._restore_axis(ax);return

            if event.button==1 and event.inaxes is ax and event.xdata is not None and event.ydata is not None:
                if self.mode.get() not in ("Dashboard","Custom Analysis","Multi-Compare"):
                    self._zoom_at_click(ax,event.xdata,event.ydata,.72)
            elif event.button==3:
                self._restore_axis(ax)
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

    def _draw_track_export(self, ax, p, full):
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
        if with_hud:
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
        self.title("📸 批量导出所选子图与数据切片")
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
        total_tasks = len(selected_modes) + (1 if self.var_crop_pos.get() else 0)
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
