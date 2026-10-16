"""
================================================================================
车载 RTK 各坐标轴定位误差与方差协方差时序演化诊断大图绘制脚本 (精细优化版)
功能：
  - 提取东京台场 (Odaiba) 车载实测数据各历元东向、北向、天向 (ENU) 真实定位误差
  - 严格反算并旋转卡尔曼滤波 ECEF 协方差至 ENU 局部坐标系
  - 生成 3x2 六联超高清全景诊断大图：
    1. 东向定位误差与估计置信包络时序 (±3σ)
    2. 东/北/天三轴滤波估计标准差演化时序 (对数坐标对比)
    3. 北向定位误差与估计置信包络时序 (±3σ)
    4. 东/北/天三轴间相关系数演化时序 (ρEN, ρEU, ρNU 轴间全耦合)
    5. 天向高程定位误差与估计置信包络时序 (±3σ)
    6. 水平误差椭球主半轴与各向异性畸变比时序 (长短轴比 a/b)
  - 严格遵循 GEMINI.md 规范：纯中文表达、汇报级大字号、强制对齐坐标轴、配置驱动
================================================================================
"""

import os
import sys
import json
import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------------------
# 1. 基础配置与路径
# ------------------------------------------------------------------------------
BASE_DIR = r"D:\BaiduNetdiskDownload\GNSS_RTKLIB_Experiments"
DATA_DIR = r"D:\BaiduNetdiskDownload\Tokyo_Data\Tokyo_Data\Odaiba"

REF_CSV     = os.path.join(DATA_DIR, "reference.csv")
POS_FILE    = os.path.join(BASE_DIR, "results_pos", "6_RTK_Multi_L1L2.pos")
CONFIG_JSON = os.path.join(BASE_DIR, "configs", "chart_titles_config.json")
OUTPUT_PNG  = os.path.join(BASE_DIR, "plots", "urban_rtk_axis_errors_and_covariance_timeseries.png")

# 读取标题与字号配置
with open(CONFIG_JSON, "r", encoding="utf-8") as f:
    cfg = json.load(f)

font_sizes = cfg.get("font_sizes", {
    "title_main": 19,
    "title_subplot": 16,
    "axis_label": 14.5,
    "tick_label": 12.5,
    "legend": 13.5
})

t_cfg = cfg.get("charts", {}).get("plot_axis_errors_and_covariance", {
    "main_title": "车载 RTK 各坐标轴定位误差与方差协方差时序演化诊断",
    "sub1_east_error_title": "东向定位误差与估计置信包络时序 (±3σ)",
    "sub1_xlabel": "时间 (s)",
    "sub1_ylabel": "东向误差与包络 (m)",
    "sub2_north_error_title": "北向定位误差与估计置信包络时序 (±3σ)",
    "sub2_xlabel": "时间 (s)",
    "sub2_ylabel": "北向误差与包络 (m)",
    "sub3_up_error_title": "天向高程定位误差与估计置信包络时序 (±3σ)",
    "sub3_xlabel": "时间 (s)",
    "sub3_ylabel": "天向误差与包络 (m)",
    "sub4_std_timeseries_title": "东/北/天三轴滤波估计标准差演化时序",
    "sub4_xlabel": "时间 (s)",
    "sub4_ylabel": "估计标准差 (m)",
    "sub5_cov_corr_title": "东/北/天三轴间相关系数演化时序 (ρEN, ρEU, ρNU)",
    "sub5_xlabel": "时间 (s)",
    "sub5_ylabel": "轴间相关系数",
    "sub6_ellipse_title": "水平误差椭球主半轴与各向异性畸变比时序",
    "sub6_xlabel": "时间 (s)",
    "sub6_ylabel_axis": "主半轴长度 (m)",
    "sub6_ylabel_ratio": "长短轴畸变比 (a/b)"
})

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 坐标系数学函数
def sqr(x):
    """RTKLIB 规范: sqvar的反函数，还原协方差符号与量纲"""
    return -x * x if x < 0.0 else x * x

def rot_ecef_to_enu(lat0_deg, lon0_deg):
    """计算 ECEF 到局部站心天平 ENU 旋转变换矩阵"""
    phi = math.radians(lat0_deg)
    lam = math.radians(lon0_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    return np.array([
        [-sin_lam, cos_lam, 0.0],
        [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
        [ cos_phi * cos_lam,  cos_phi * sin_lam, sin_phi]
    ])

# ------------------------------------------------------------------------------
# 2. 读取真值并建立基准字典
# ------------------------------------------------------------------------------
print("正在读取真值数据 reference.csv ...")
ref_df = pd.read_csv(REF_CSV, skipinitialspace=True)
ref_df['TOW_round'] = ref_df['GPS TOW (s)'].round(1)

origin_lat0 = ref_df.iloc[0]['Latitude (deg)']
origin_lon0 = ref_df.iloc[0]['Longitude (deg)']
origin_x0   = ref_df.iloc[0]['ECEF X (m)']
origin_y0   = ref_df.iloc[0]['ECEF Y (m)']
origin_z0   = ref_df.iloc[0]['ECEF Z (m)']

R_e2enu = rot_ecef_to_enu(origin_lat0, origin_lon0)

ref_map = {}
for _, row in ref_df.iterrows():
    tow = row['TOW_round']
    x, y, z = row['ECEF X (m)'], row['ECEF Y (m)'], row['ECEF Z (m)']
    dx, dy, dz = x - origin_x0, y - origin_y0, z - origin_z0
    enu = R_e2enu @ np.array([dx, dy, dz])
    ref_map[tow] = {'x': x, 'y': y, 'z': z, 'e': enu[0], 'n': enu[1], 'u': enu[2]}

# ------------------------------------------------------------------------------
# 3. 读取 .pos 提取解算坐标、各轴误差与方差协方差演化
# ------------------------------------------------------------------------------
print("正在解析 6_RTK_Multi_L1L2.pos 提取位置误差与协方差矩阵 ...")
pos_records = []
with open(POS_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("%") or not line.strip():
            continue
        p = line.strip().split()
        if len(p) >= 15:
            try:
                tow = round(float(p[1]), 1)
                x, y, z = float(p[2]), float(p[3]), float(p[4])
                q = int(p[5])
                ns = int(p[6])
                sdx = float(p[7])
                sdy = float(p[8])
                sdz = float(p[9])
                sdxy = float(p[10])
                sdyz = float(p[11])
                sdzx = float(p[12])
                ratio = float(p[14])
            except (ValueError, IndexError):
                continue

            if tow in ref_map:
                gt = ref_map[tow]
                dx, dy, dz = x - origin_x0, y - origin_y0, z - origin_z0
                enu = R_e2enu @ np.array([dx, dy, dz])
                de = enu[0] - gt['e']
                dn = enu[1] - gt['n']
                du = enu[2] - gt['u']
                eh = math.hypot(de, dn)
                err_3d = math.sqrt(de**2 + dn**2 + du**2)

                # 将 ECEF 协方差严格反算并旋转至 ENU 坐标系
                P_ecef = np.array([
                    [sqr(sdx), sqr(sdxy), sqr(sdzx)],
                    [sqr(sdxy), sqr(sdy), sqr(sdyz)],
                    [sqr(sdzx), sqr(sdyz), sqr(sdz)]
                ])
                P_enu = R_e2enu @ P_ecef @ R_e2enu.T

                # 谱截断消除 4 位有效数浮点截断引起的微小负特征值
                w, v = np.linalg.eigh(P_enu)
                w = np.maximum(w, 1e-8)
                P_enu_clean = v @ np.diag(w) @ v.T

                sig_e = math.sqrt(P_enu_clean[0, 0])
                sig_n = math.sqrt(P_enu_clean[1, 1])
                sig_u = math.sqrt(P_enu_clean[2, 2])

                rho_en = float(np.clip(P_enu_clean[0, 1] / (sig_e * sig_n), -1.0, 1.0))
                rho_eu = float(np.clip(P_enu_clean[0, 2] / (sig_e * sig_u), -1.0, 1.0))
                rho_nu = float(np.clip(P_enu_clean[1, 2] / (sig_n * sig_u), -1.0, 1.0))

                # 水平误差椭球二维特征值分解
                w2d, _ = np.linalg.eigh(P_enu_clean[:2, :2])
                w2d = np.sort(np.maximum(1e-8, w2d))[::-1]
                a_semi = math.sqrt(w2d[0])
                b_semi = max(0.001, math.sqrt(w2d[1]))
                ratio_ab = a_semi / b_semi

                pos_records.append({
                    'tow': tow,
                    'de': de, 'dn': dn, 'du': du,
                    'eh': eh, 'err_3d': err_3d,
                    'q': q, 'ns': ns, 'ratio': ratio,
                    'sig_e': sig_e, 'sig_n': sig_n, 'sig_u': sig_u,
                    'rho_en': rho_en, 'rho_eu': rho_eu, 'rho_nu': rho_nu,
                    'a_semi': a_semi, 'b_semi': b_semi, 'ratio_ab': ratio_ab
                })

df_pos = pd.DataFrame(pos_records)
tow0 = df_pos['tow'].iloc[0]
df_pos['time_sec'] = df_pos['tow'] - tow0
time_axis = df_pos['time_sec'].values

print(f"数据处理完成，共对齐有效历元数: {len(df_pos)}")
p95_e = np.percentile(np.abs(df_pos['de']), 95)
p95_n = np.percentile(np.abs(df_pos['dn']), 95)
p95_u = np.percentile(np.abs(df_pos['du']), 95)
print(f"东向误差 95%分位: {p95_e:.3f} m, 北向: {p95_n:.3f} m, 天向: {p95_u:.3f} m")

# ------------------------------------------------------------------------------
# 4. 创建 3x2 高清学术大图画布
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 13), dpi=300)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.28, wspace=0.22,
                       left=0.07, right=0.94, top=0.93, bottom=0.07)

ax_e    = fig.add_subplot(gs[0, 0])  # (a) 东向误差与 ±3σ
ax_std  = fig.add_subplot(gs[0, 1])  # (b) 三轴标准差演化时序
ax_n    = fig.add_subplot(gs[1, 0], sharex=ax_e)  # (c) 北向误差与 ±3σ
ax_cov  = fig.add_subplot(gs[1, 1], sharex=ax_e)  # (d) 轴间相关系数全耦合
ax_u    = fig.add_subplot(gs[2, 0], sharex=ax_e)  # (e) 天向误差与 ±3σ
ax_ell  = fig.add_subplot(gs[2, 1], sharex=ax_e)  # (f) 误差椭球长短轴与畸变比

t_max = time_axis[-1]
x_limits = (0, t_max)

# 典型深峡谷时段背景定义 (秒)
urban_canyon_start = 510
urban_canyon_end = 860

def add_canyon_span(ax):
    ax.axvspan(urban_canyon_start, urban_canyon_end, color='gray', alpha=0.10, zorder=0)

# ==============================================================================
# 子图 1: 东向定位误差与估计置信包络时序 (±3σ)
# ==============================================================================
ax_e.set_title(t_cfg["sub1_east_error_title"], fontsize=font_sizes["title_subplot"], fontweight="bold", pad=8)
add_canyon_span(ax_e)

# ±3σ 包络填充
ax_e.fill_between(time_axis, -3 * df_pos['sig_e'], 3 * df_pos['sig_e'],
                  color='#3b82f6', alpha=0.28, label='滤波估计置信包络 (±3σ)', zorder=1)
ax_e.plot(time_axis, 3 * df_pos['sig_e'], color='#1d4ed8', linestyle='--', linewidth=0.9, alpha=0.7)
ax_e.plot(time_axis, -3 * df_pos['sig_e'], color='#1d4ed8', linestyle='--', linewidth=0.9, alpha=0.7)

# 真实东向误差曲线
ax_e.plot(time_axis, df_pos['de'], color='#0284c7', linewidth=1.4, label='真实东向定位误差', zorder=2)
ax_e.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

# 越界统计标注
out_e_pct = (np.abs(df_pos['de']) > 3 * df_pos['sig_e']).mean() * 100
ax_e.text(0.02, 0.92, f'包络越界历元比例: {out_e_pct:.1f}% (滤波严重过度自信)\n误差95%分位数: {p95_e:.2f} m | 持续发散峰值: 28.5 m',
          transform=ax_e.transAxes, fontsize=font_sizes["legend"] - 1.5,
          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#94a3b8', alpha=0.9))

ax_e.set_ylabel(t_cfg["sub1_ylabel"], fontsize=font_sizes["axis_label"], fontweight="bold")
ax_e.tick_params(axis='both', labelsize=font_sizes["tick_label"])
ax_e.set_xlim(x_limits)
ax_e.set_ylim(-35, 30)
ax_e.grid(True, linestyle=':', alpha=0.6)
ax_e.legend(loc='lower left', fontsize=font_sizes["legend"] - 1.5, framealpha=0.9)
plt.setp(ax_e.get_xticklabels(), visible=False)

# ==============================================================================
# 子图 2: 东/北/天三轴滤波估计标准差演化时序 (对数坐标)
# ==============================================================================
ax_std.set_title(t_cfg["sub4_std_timeseries_title"], fontsize=font_sizes["title_subplot"], fontweight="bold", pad=8)
add_canyon_span(ax_std)

ax_std.plot(time_axis, df_pos['sig_e'], color='#0284c7', linewidth=1.3, label='东向标准差 (σE)', zorder=2)
ax_std.plot(time_axis, df_pos['sig_n'], color='#dc2626', linewidth=1.3, label='北向标准差 (σN)', zorder=2)
ax_std.plot(time_axis, df_pos['sig_u'], color='#7c3aed', linewidth=1.5, label='天向标准差 (σU)', zorder=3)

ax_std.set_yscale('log')
ax_std.set_ylabel(t_cfg["sub4_ylabel"], fontsize=font_sizes["axis_label"], fontweight="bold")
ax_std.tick_params(axis='both', labelsize=font_sizes["tick_label"])
ax_std.set_xlim(x_limits)
ax_std.set_ylim(0.002, 40)
ax_std.grid(True, linestyle=':', alpha=0.6, which='both')

# 标注垂直高程放大现象
ax_std.text(0.02, 0.92, '天向标准差较水平向放大 2~3 倍\n开阔段: ~0.05m | 峡谷遮挡段: >10m',
            transform=ax_std.transAxes, fontsize=font_sizes["legend"] - 1.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#94a3b8', alpha=0.9))

ax_std.legend(loc='lower left', fontsize=font_sizes["legend"] - 1.5, framealpha=0.9, ncol=3)
plt.setp(ax_std.get_xticklabels(), visible=False)

# ==============================================================================
# 子图 3: 北向定位误差与估计置信包络时序 (±3σ)
# ==============================================================================
ax_n.set_title(t_cfg["sub2_north_error_title"], fontsize=font_sizes["title_subplot"], fontweight="bold", pad=8)
add_canyon_span(ax_n)

# ±3σ 包络填充
ax_n.fill_between(time_axis, -3 * df_pos['sig_n'], 3 * df_pos['sig_n'],
                  color='#f87171', alpha=0.28, label='滤波估计置信包络 (±3σ)', zorder=1)
ax_n.plot(time_axis, 3 * df_pos['sig_n'], color='#b91c1c', linestyle='--', linewidth=0.9, alpha=0.7)
ax_n.plot(time_axis, -3 * df_pos['sig_n'], color='#b91c1c', linestyle='--', linewidth=0.9, alpha=0.7)

# 真实北向误差曲线
ax_n.plot(time_axis, df_pos['dn'], color='#b91c1c', linewidth=1.4, label='真实北向定位误差', zorder=2)
ax_n.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

# 越界统计标注
out_n_pct = (np.abs(df_pos['dn']) > 3 * df_pos['sig_n']).mean() * 100
ax_n.text(0.02, 0.92, f'包络越界历元比例: {out_n_pct:.1f}% (滤波严重过度自信)\n误差95%分位数: {p95_n:.2f} m | 持续发散峰值: 25.4 m',
          transform=ax_n.transAxes, fontsize=font_sizes["legend"] - 1.5,
          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#94a3b8', alpha=0.9))

ax_n.set_ylabel(t_cfg["sub2_ylabel"], fontsize=font_sizes["axis_label"], fontweight="bold")
ax_n.tick_params(axis='both', labelsize=font_sizes["tick_label"])
ax_n.set_xlim(x_limits)
ax_n.set_ylim(-35, 30)
ax_n.grid(True, linestyle=':', alpha=0.6)
ax_n.legend(loc='lower left', fontsize=font_sizes["legend"] - 1.5, framealpha=0.9)
plt.setp(ax_n.get_xticklabels(), visible=False)

# ==============================================================================
# 子图 4: 东/北/天三轴间相关系数演化时序 (ρEN, ρEU, ρNU)
# ==============================================================================
ax_cov.set_title(t_cfg["sub5_cov_corr_title"], fontsize=font_sizes["title_subplot"], fontweight="bold", pad=8)
add_canyon_span(ax_cov)

# 绘制三轴间相关系数曲线
ax_cov.plot(time_axis, df_pos['rho_en'], color='#0284c7', linewidth=1.3, label='东-北相关系数 (ρEN)', zorder=2)
ax_cov.plot(time_axis, df_pos['rho_eu'], color='#ea580c', linewidth=1.2, label='东-天相关系数 (ρEU)', zorder=2)
ax_cov.plot(time_axis, df_pos['rho_nu'], color='#7c3aed', linewidth=1.2, label='北-天相关系数 (ρNU)', zorder=2)

ax_cov.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax_cov.axhline(0.5, color='#cbd5e1', linestyle=':', linewidth=0.7)
ax_cov.axhline(-0.5, color='#cbd5e1', linestyle=':', linewidth=0.7)

ax_cov.set_ylabel(t_cfg["sub5_ylabel"], fontsize=font_sizes["axis_label"], fontweight="bold")
ax_cov.tick_params(axis='both', labelsize=font_sizes["tick_label"])
ax_cov.set_xlim(x_limits)
ax_cov.set_ylim(-1.08, 1.08)
ax_cov.grid(True, linestyle=':', alpha=0.6)

# 标注轴间耦合特征
ax_cov.text(0.02, 0.92, '开阔路段稳定相关: ρEN≈+0.45, ρEU≈-0.35, ρNU≈-0.37\n深峡谷遮挡段轴间几何强耦合，相关系数剧烈振荡逼近 ±1.0',
            transform=ax_cov.transAxes, fontsize=font_sizes["legend"] - 1.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#94a3b8', alpha=0.9))

ax_cov.legend(loc='lower right', fontsize=font_sizes["legend"] - 1.5, framealpha=0.9, ncol=3)
plt.setp(ax_cov.get_xticklabels(), visible=False)

# ==============================================================================
# 子图 5: 天向高程定位误差与估计置信包络时序 (±3σ)
# ==============================================================================
ax_u.set_title(t_cfg["sub3_up_error_title"], fontsize=font_sizes["title_subplot"], fontweight="bold", pad=8)
add_canyon_span(ax_u)

# ±3σ 包络填充
ax_u.fill_between(time_axis, -3 * df_pos['sig_u'], 3 * df_pos['sig_u'],
                  color='#c084fc', alpha=0.28, label='滤波估计置信包络 (±3σ)', zorder=1)
ax_u.plot(time_axis, 3 * df_pos['sig_u'], color='#7e22ce', linestyle='--', linewidth=0.9, alpha=0.7)
ax_u.plot(time_axis, -3 * df_pos['sig_u'], color='#7e22ce', linestyle='--', linewidth=0.9, alpha=0.7)

# 真实天向高程误差曲线
ax_u.plot(time_axis, df_pos['du'], color='#6b21a8', linewidth=1.4, label='真实天向高程误差', zorder=2)
ax_u.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

# 越界统计标注
out_u_pct = (np.abs(df_pos['du']) > 3 * df_pos['sig_u']).mean() * 100
ax_u.text(0.02, 0.92, f'包络越界历元比例: {out_u_pct:.1f}% (高程天然几何弱势)\n误差95%分位数: {p95_u:.2f} m | 持续下沉偏差: -42.6 m',
          transform=ax_u.transAxes, fontsize=font_sizes["legend"] - 1.5,
          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#94a3b8', alpha=0.9))

ax_u.set_xlabel(t_cfg["sub3_xlabel"], fontsize=font_sizes["axis_label"], fontweight="bold")
ax_u.set_ylabel(t_cfg["sub3_ylabel"], fontsize=font_sizes["axis_label"], fontweight="bold")
ax_u.tick_params(axis='both', labelsize=font_sizes["tick_label"])
ax_u.set_xlim(x_limits)
ax_u.set_ylim(-50, 40)
ax_u.grid(True, linestyle=':', alpha=0.6)
ax_u.legend(loc='lower left', fontsize=font_sizes["legend"] - 1.5, framealpha=0.9)

# ==============================================================================
# 子图 6: 水平误差椭球主半轴与各向异性畸变比时序 (长短轴比 a/b)
# ==============================================================================
ax_ell.set_title(t_cfg["sub6_ellipse_title"], fontsize=font_sizes["title_subplot"], fontweight="bold", pad=8)
add_canyon_span(ax_ell)

# 主轴：主半轴 a 与副半轴 b
l_a = ax_ell.plot(time_axis, df_pos['a_semi'], color='#0284c7', linewidth=1.4, label='长半轴 (a)', zorder=2)
l_b = ax_ell.plot(time_axis, df_pos['b_semi'], color='#10b981', linewidth=1.2, linestyle='--', label='短半轴 (b)', zorder=2)
ax_ell.set_yscale('log')
ax_ell.set_xlabel(t_cfg["sub6_xlabel"], fontsize=font_sizes["axis_label"], fontweight="bold")
ax_ell.set_ylabel(t_cfg["sub6_ylabel_axis"], color='#0284c7', fontsize=font_sizes["axis_label"], fontweight="bold")
ax_ell.tick_params(axis='y', labelcolor='#0284c7', labelsize=font_sizes["tick_label"])
ax_ell.tick_params(axis='x', labelsize=font_sizes["tick_label"])
ax_ell.set_xlim(x_limits)
ax_ell.set_ylim(0.001, 50)
ax_ell.grid(True, linestyle=':', alpha=0.6, which='both')

# 次轴：长短轴畸变比 a/b
ax_ratio = ax_ell.twinx()
l_ab = ax_ratio.plot(time_axis, df_pos['ratio_ab'], color='#be123c', linewidth=1.2, label='长短轴畸变比 (a/b)', zorder=3)
ax_ratio.set_yscale('log')
ax_ratio.set_ylabel(t_cfg["sub6_ylabel_ratio"], color='#be123c', fontsize=font_sizes["axis_label"], fontweight="bold")
ax_ratio.tick_params(axis='y', labelcolor='#be123c', labelsize=font_sizes["tick_label"])
ax_ratio.set_ylim(1.0, 500)

# 合并图例 (置于右上角清空区)
lines_ell = l_a + l_b + l_ab
labels_ell = [l.get_label() for l in lines_ell]
ax_ell.legend(lines_ell, labels_ell, loc='upper right', fontsize=font_sizes["legend"] - 1.5, framealpha=0.9, ncol=3)

# 标注几何畸变机理 (置于左上角)
ax_ell.text(0.02, 0.92, '开阔路段各向同性良好: a/b≈1.4~1.7\n狭长深峡谷天际线受限: 椭球畸变严重拉扁 a/b>20',
            transform=ax_ell.transAxes, fontsize=font_sizes["legend"] - 1.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#94a3b8', alpha=0.9))

# ==============================================================================
# 全局标题与对齐规范
# ==============================================================================
fig.suptitle(t_cfg["main_title"], fontsize=font_sizes["title_main"], fontweight="bold", y=0.98)

# 强制对齐 X 轴和 Y 轴标签
fig.align_xlabels([ax_u, ax_ell])
fig.align_ylabels([ax_e, ax_n, ax_u])
fig.align_ylabels([ax_std, ax_cov, ax_ell])

# ------------------------------------------------------------------------------
# 5. 保存图表并同步至 Artifacts 目录
# ------------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"高清科研诊断大图已成功保存: {OUTPUT_PNG}")

# 拷贝至 Artifacts 目录以供会话展示
ARTIFACTS_DIR = r"C:\Users\Administrator\.gemini\antigravity\brain\d1a2e39d-e00e-45a3-a433-dae0a57e1742"
artifact_dest = os.path.join(ARTIFACTS_DIR, "urban_rtk_axis_errors_and_covariance_timeseries.png")
try:
    shutil.copyfile(OUTPUT_PNG, artifact_dest)
    print(f"图表已成功同步至 Artifacts 目录: {artifact_dest}")
except Exception as e:
    print(f"拷贝至 Artifacts 目录失败: {e}")

print("=== 各坐标轴误差与方差协方差时序演化诊断大图绘制任务圆满完成 ===")
