import time
import sys
from pathlib import Path
import tkinter as tk
import numpy as np

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rtklib_plot_gui import RTKPlotApp

pos_file = Path(r"D:\BaiduNetdiskDownload\GNSS_RTKLIB_Experiments\results_pos\6_RTK_Multi_L1L2.pos")
stat_file = Path(r"D:\BaiduNetdiskDownload\GNSS_RTKLIB_Experiments\results_pos\6_RTK_Multi_L1L2.pos.stat")

root = tk.Tk()
root.withdraw()
app = RTKPlotApp(root)

results = {}

print("=" * 60)
print("1. 数据加载与预处理测试 (10,603 历元)")
t0 = time.perf_counter()
app.load_solution(pos_file, stat=stat_file)
t_load = (time.perf_counter() - t0) * 1000
n_epochs = len(app.data["pos"])
n_sats = len(app.data["sat"])
results["数据加载耗时"] = f"{t_load:.1f} ms"
print(f"-> 加载完成: {t_load:.1f} ms ({n_epochs:,} 历元, {n_sats:,} 条卫星记录)")

print("\n" + "=" * 60)
print("2. 自定义多图网格绘制性能 (9 个子图配置, 视口 6 个)")
modes_9 = ["ENU Position", "Gnd Trk", "Height", "ENU Error", "Solution Status", "AR Ratio", "Satellites", "DOP", "Covariance"]
app.custom_modes = modes_9
app.custom_panels = app._initial_custom_panels(modes_9, "Auto Grid (自适应等分)")
app.mode.set("Custom Analysis")

times_draw = []
for i in range(3):
    t0 = time.perf_counter()
    app.redraw()
    app.canvas.draw()
    dt = (time.perf_counter() - t0) * 1000
    times_draw.append(dt)
avg_draw = np.mean(times_draw)
results["全量重绘 (redraw + canvas.draw)"] = f"{avg_draw:.1f} ms"
print(f"-> 全量重绘耗时: 平均 {avg_draw:.1f} ms (单次: {[round(x,1) for x in times_draw]})")

# 子图分别绘制耗时分析
print("\n--- 各子图独立绘制耗时分解 ---")
p = app._filtered()
full = app.data["pos"]
s = app.data["sat"]

for m in ["Gnd Trk", "ENU Position", "Height", "ENU Error", "Solution Status", "AR Ratio", "Satellites", "DOP", "Covariance"]:
    fig_sub = app.fig
    ax_sub = fig_sub.add_subplot(111)
    t0 = time.perf_counter()
    if m == "Gnd Trk":
        app._draw_track(ax_sub, p, full)
    elif m == "ENU Position":
        app._draw_enu_positions([ax_sub], p)
    elif m == "Height":
        ax_sub.plot(p.time, p.U)
    elif m == "ENU Error":
        app._draw_errors([ax_sub], p)
    elif m == "Solution Status":
        ax_sub.step(p.time, p.q)
    elif m == "AR Ratio":
        ax_sub.plot(p.time, p.ratio)
    elif m == "Satellites":
        app._draw_sats(ax_sub)
    elif m == "DOP":
        app._draw_dop(ax_sub)
    elif m == "Covariance":
        app._draw_covariance(ax_sub, p)
    dt_sub = (time.perf_counter() - t0) * 1000
    fig_sub.delaxes(ax_sub)
    print(f"   - {m:<18}: {dt_sub:6.2f} ms")

print("\n" + "=" * 60)
print("3. 时间轴游标与 Blit 响应性能 (Scrubbing / Hover)")
times_cursor = []
for ep in range(100, 300, 10):
    app.slider.set(ep)
    t0 = time.perf_counter()
    app._update_cursor()
    times_cursor.append((time.perf_counter() - t0) * 1000)
avg_cursor = np.mean(times_cursor)
results["游标单次刷新 (_update_cursor + blit)"] = f"{avg_cursor:.2f} ms"
print(f"-> 游标单次响应耗时: 平均 {avg_cursor:.2f} ms")

print("\n" + "=" * 60)
print("4. 分页切换性能 (Page 1 -> Page 2)")
times_page = []
for _ in range(4):
    t0 = time.perf_counter()
    app._change_custom_page(1)
    app.canvas.draw()
    times_page.append((time.perf_counter() - t0) * 1000)
avg_page = np.mean(times_page)
results["分页切换 (_change_custom_page + draw)"] = f"{avg_page:.1f} ms"
print(f"-> 分页切换耗时: 平均 {avg_page:.1f} ms")

print("\n" + "=" * 60)
print("5. 排版预设切换性能 (Preset Changing)")
times_preset = []
for pset in ["Focus Left (左主右副)", "Vertical Stack (纵向流)", "Focus Top (上主下副)", "Auto Grid (自适应等分)"]:
    app.custom_preset.set(pset)
    t0 = time.perf_counter()
    app._on_custom_preset_changed()
    app.canvas.draw()
    times_preset.append((time.perf_counter() - t0) * 1000)
avg_preset = np.mean(times_preset)
results["排版预设切换 (Preset Switch)"] = f"{avg_preset:.1f} ms"
print(f"-> 预设切换耗时: 平均 {avg_preset:.1f} ms")

print("\n" + "=" * 60)
print("6. 单子图放大与还原 (Maximize / Restore)")
ax = app.plot_axes[0]
t0 = time.perf_counter()
app._toggle_maximize_subplot(ax)
app.canvas.draw()
t_max = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
app._toggle_maximize_subplot(ax)
app.canvas.draw()
t_restore = (time.perf_counter() - t0) * 1000
results["单子图最大化"] = f"{t_max:.1f} ms"
results["子图还原网格"] = f"{t_restore:.1f} ms"
print(f"-> 满屏放大: {t_max:.1f} ms, 还原网格: {t_restore:.1f} ms")

print("\n" + "=" * 60)
print("7. 追加子图与移除子图 (Add / Delete Subplot)")
t0 = time.perf_counter()
app.add_selected_plot("Elevation")
app.canvas.draw()
t_add = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
app.delete_selected_subplot()
app.canvas.draw()
t_del = (time.perf_counter() - t0) * 1000
results["追加子图 (Add Subplot)"] = f"{t_add:.1f} ms"
results["删除子图 (Del Subplot)"] = f"{t_del:.1f} ms"
print(f"-> 追加子图: {t_add:.1f} ms, 删除子图: {t_del:.1f} ms")

print("\n" + "=" * 60)
print("8. 文本分析统计更新 (update_analysis)")
times_analysis = []
for _ in range(5):
    t0 = time.perf_counter()
    app.update_analysis()
    times_analysis.append((time.perf_counter() - t0) * 1000)
avg_analysis = np.mean(times_analysis)
results["文本分析更新 (update_analysis)"] = f"{avg_analysis:.2f} ms"
print(f"-> 文本分析统计: 平均 {avg_analysis:.2f} ms")

print("\n" + "=" * 60)
print("汇总统计表:")
for k, v in results.items():
    print(f"  {k:<35}: {v}")

root.destroy()
