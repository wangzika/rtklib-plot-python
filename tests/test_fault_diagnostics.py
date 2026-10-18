"""
Unit & integration verification test for RTKLIB Fault Diagnosis Pipeline and Workspaces.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import rtklib_diagnostics as diag
import rtklib_plot_gui as gui
import rtklib_export_report as rep

pos_file = Path(r"d:\BaiduNetdiskDownload\Tokyo_Data\Tokyo_Data\Odaiba\rover_trimble.pos")
stat_file = Path(r"d:\BaiduNetdiskDownload\Tokyo_Data\Tokyo_Data\Odaiba\rover_trimble.pos.stat")
ref_file = Path(r"d:\BaiduNetdiskDownload\Tokyo_Data\Tokyo_Data\Odaiba\reference.csv")

print("1. Testing load_data with truth, stat, and clk...")
data = diag.load_data(pos_file, reference_path=ref_file, stat_path=stat_file)
print(f"   Loaded: pos={len(data['pos'])}, sat={len(data['sat'])}, clk={len(data.get('clk', []))}")
assert len(data['pos']) > 0, "pos data empty"
assert len(data['sat']) > 0, "sat data empty"
assert "clk" in data and not data["clk"].empty, "clk data empty"
assert "along_err" in data["pos"].columns, "along_err missing in pos"
assert "cross_err" in data["pos"].columns, "cross_err missing in pos"

print("2. Testing 5 Workspaces configurations...")
assert len(gui.DIAGNOSTIC_WORKSPACES) == 5, "Must have 5 workspaces"
for ws in gui.DIAGNOSTIC_WORKSPACES:
    print(f"   Workspace {ws['id']}: {ws['name']} -> {ws['modes']}")
    for m in ws["modes"]:
        assert m in gui.MODES, f"Mode {m} not in MODES!"

print("3. Testing diagnose_anomaly_segment...")
st = float(data['pos'].tow.iloc[100])
et = float(data['pos'].tow.iloc[500])
diag_res = diag.diagnose_anomaly_segment(data['pos'], data['sat'], st, et)
print(f"   Diagnosis result: {diag_res['primary_cause']} | {diag_res['recommendation']}")
assert "primary_cause" in diag_res
assert "recommendation" in diag_res

print("4. Testing drawing methods for new modes...")
fig, ax = plt.subplots(figsize=(6, 4))

class MockApp:
    def __init__(self, data):
        self.data = data
        self.grid_on = type("Grid", (), {"get": lambda s: True})()

mock = MockApp(data)

# Test 1: Residual Triad
gui.RTKPlotApp._draw_residual_triad(mock, ax)
print("   _draw_residual_triad: OK")

# Test 2: El vs SNR & Resp
ax.clear()
gui.RTKPlotApp._draw_el_snr_resp(mock, ax)
print("   _draw_el_snr_resp: OK")

# Test 3: Top Outliers
ax.clear()
gui.RTKPlotApp._draw_top_outliers(mock, ax)
print("   _draw_top_outliers: OK")

# Test 4: Directional DOP
ax.clear()
gui.RTKPlotApp._draw_directional_dop(mock, ax)
print("   _draw_directional_dop: OK")

# Test 5: Clock Bias & ISB
ax.clear()
gui.RTKPlotApp._draw_clock_bias_isb(mock, ax)
print("   _draw_clock_bias_isb: OK")

# Test 6: Body-Frame Error
ax.clear()
gui.RTKPlotApp._draw_body_frame_error(mock, ax, data['pos'])
print("   _draw_body_frame_error: OK")

plt.close(fig)

print("5. Testing GUI instantiation and Workspace switching...")
import tkinter as tk
root = tk.Tk()
root.withdraw()  # headless
app = gui.RTKPlotApp(root, pos=pos_file, reference=ref_file)
print("   App initialized with solution:", app.pos_path.name)

for idx in range(5):
    app.switch_workspace(idx)
    ws = gui.DIAGNOSTIC_WORKSPACES[idx]
    print(f"   Switched to Workspace {idx+1} [{ws['name']}]: OK (mode={app.mode.get()}, panels={len(app.custom_panels)})")

print("6. Testing Env & Geometry Combined mode in GUI...")
app.set_mode("Env & Geometry Combined")
assert len(app.plot_axes) == 5, f"Expected 5 plot axes in combined mode, got {len(app.plot_axes)}"
print("   Env & Geometry Combined in GUI: OK (5 subplots rendered)")

root.destroy()

print("7. Testing plot_combined_environment_and_geometry...")
test_img = ROOT.parent / "test_combined_verify.png"
fig_comb = rep.plot_combined_environment_and_geometry(data, test_img)
assert test_img.exists() and test_img.stat().st_size > 10000, "Combined image generation failed"
print(f"   plot_combined_environment_and_geometry: OK ({test_img.stat().st_size:,} bytes)")
test_img.unlink(missing_ok=True)

print("8. Testing academic PPTX generation with combined slide...")
pptx_out = ROOT.parent / "test_diagnostic_verify.pptx"
rep.generate_academic_pptx(data, pptx_out)
assert pptx_out.exists() and pptx_out.stat().st_size > 50000, "PPTX export failed"
print(f"   generate_academic_pptx: OK ({pptx_out.stat().st_size:,} bytes)")
pptx_out.unlink(missing_ok=True)

print("9. Testing HTML report generation with combined figure...")
html_out = ROOT.parent / "test_diagnostic_verify.html"
rep.generate_diagnostic_html(data, html_out)
assert html_out.exists() and html_out.stat().st_size > 50000, "HTML export failed"
print(f"   generate_diagnostic_html: OK ({html_out.stat().st_size:,} bytes)")
html_out.unlink(missing_ok=True)

print("\n>>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")
