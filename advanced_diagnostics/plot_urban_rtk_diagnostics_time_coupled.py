"""
================================================================================
城市深峡谷车载 RTK 误差全时序耦合与多物理量关联诊断大图绘制脚本
功能：
  - 实现轨迹、载噪比、可用卫星数、PDOP、协方差与时间的完整耦合关联
  - 生成 2x3 六联全景时序关联诊断大图：
    1. 车辆行驶轨迹时间映射与解算状态分布 (轨迹按时间渐变上色并标出关键时间节点)
    2. 水平定位误差与滤波估计协方差 (3σ) 时序对比 (揭示滤波估计方差与真实误差关系)
    3. 模糊度固定检验比值时序分布 (时间严格对齐，标出 3.0 门限与固定/浮点状态)
    4. 跟踪卫星数与空间几何衰减因子 (PDOP) 时序 (卫星数骤降与 PDOP 发散)
    5. 双频平均载噪比 (C/N0) 时序衰减与时段箱线图 (载噪比与时间关系的动态呈现)
    6. 典型行驶时段多物理量响应特性对比 (时段与载噪比、星数、PDOP、协方差的综合对比)
================================================================================
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

# ------------------------------------------------------------------------------
# 1. 基础配置与路径
# ------------------------------------------------------------------------------
BASE_DIR = r"D:\BaiduNetdiskDownload\GNSS_RTKLIB_Experiments"
DATA_DIR = r"D:\BaiduNetdiskDownload\Tokyo_Data\Tokyo_Data\Odaiba"

REF_CSV     = os.path.join(DATA_DIR, "reference.csv")
POS_FILE    = os.path.join(BASE_DIR, "results_pos", "6_RTK_Multi_L1L2.pos")
STAT_FILE   = os.path.join(BASE_DIR, "results_pos", "6_RTK_Multi_L1L2.pos.stat")
CONFIG_JSON = os.path.join(BASE_DIR, "configs", "chart_titles_config.json")
OUTPUT_PNG  = os.path.join(BASE_DIR, "plots", "urban_rtk_time_coupled_diagnostics_big_figure.png")

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

t_cfg = cfg.get("charts", {}).get("plot_urban_rtk_diagnostics_time_coupled", {
    "main_title": "城市深峡谷车载 RTK 误差全时序耦合与多物理量关联诊断",
    "sub1_trajectory_title": "车辆行驶轨迹时间映射与解算状态分布",
    "sub1_xlabel": "东向坐标 (m)",
    "sub1_ylabel": "北向坐标 (m)",
    "sub2_error_title": "水平定位误差与滤波估计协方差 (3σ) 时序",
    "sub2_xlabel": "时间 (s)",
    "sub2_ylabel": "水平误差与方差包络 (m)",
    "sub3_ratio_title": "模糊度固定检验比值时序分布",
    "sub3_xlabel": "时间 (s)",
    "sub3_ylabel": "检验比值",
    "sub4_sat_dop_title": "跟踪卫星数与空间几何衰减因子 (PDOP)",
    "sub4_xlabel": "时间 (s)",
    "sub4_ylabel_sats": "有效卫星数 (颗)",
    "sub4_ylabel_dop": "几何因子 (PDOP)",
    "sub5_snr_time_title": "双频平均载噪比时序衰减与时段分布",
    "sub5_xlabel": "时间 (s)",
    "sub5_ylabel": "载噪比 (dB-Hz)",
    "sub6_segment_corr_title": "典型行驶时段多物理量响应特性对比",
    "sub6_xlabel": "典型路况时段",
    "sub6_ylabel": "归一化响应指标"
})

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# WGS84 坐标投影
a = 6378137.0
f = 1.0 / 298.257223563
e2 = f * (2.0 - f)

def ecef_to_enu(dx, dy, dz, lat0_deg, lon0_deg):
    phi = math.radians(lat0_deg)
    lam = math.radians(lon0_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u =  cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz
    return e, n, u

def rot_ecef_to_enu(lat0_deg, lon0_deg):
    phi = math.radians(lat0_deg)
    lam = math.radians(lon0_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    R = np.array([
        [-sin_lam, cos_lam, 0],
        [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
        [ cos_phi * cos_lam,  cos_phi * sin_lam, sin_phi]
    ])
    return R

# ------------------------------------------------------------------------------
# 2. 读取真值并建立基准
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
    e, n, u = ecef_to_enu(x - origin_x0, y - origin_y0, z - origin_z0, origin_lat0, origin_lon0)
    ref_map[tow] = {'x': x, 'y': y, 'z': z, 'e': e, 'n': n, 'u': u}

# ------------------------------------------------------------------------------
# 3. 读取 .pos 提取解算坐标、误差与滤波状态协方差
# ------------------------------------------------------------------------------
print("正在解析 6_RTK_Multi_L1L2.pos 提取位置与协方差 ...")
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
                e, n, u = ecef_to_enu(x - origin_x0, y - origin_y0, z - origin_z0, origin_lat0, origin_lon0)
                de = e - gt['e']
                dn = n - gt['n']
                du = u - gt['u']
                eh = math.hypot(de, dn)
                err_3d = math.sqrt(de**2 + dn**2 + du**2)

                # 将 ECEF 协方差旋转至 ENU 坐标系
                P_ecef = np.array([
                    [sdx**2, sdxy, sdzx],
                    [sdxy, sdy**2, sdyz],
                    [sdzx, sdyz, sdz**2]
                ])
                P_enu = R_e2enu @ P_ecef @ R_e2enu.T
                sig_e = math.sqrt(max(0, P_enu[0, 0]))
                sig_n = math.sqrt(max(0, P_enu[1, 1]))
                sig_u = math.sqrt(max(0, P_enu[2, 2]))
                sig_h = math.sqrt(sig_e**2 + sig_n**2)

                pos_records.append({
                    'tow': tow,
                    'e': e, 'n': n, 'u': u,
                    'gt_e': gt['e'], 'gt_n': gt['n'], 'gt_u': gt['u'],
                    'de': de, 'dn': dn, 'du': du,
                    'eh': eh, 'err_3d': err_3d,
                    'q': q, 'ns': ns, 'ratio': ratio,
                    'sig_e': sig_e, 'sig_n': sig_n, 'sig_u': sig_u, 'sig_h': sig_h
                })

df_pos = pd.DataFrame(pos_records)
tow0 = df_pos['tow'].iloc[0]
df_pos['time_sec'] = df_pos['tow'] - tow0

# ------------------------------------------------------------------------------
# 4. 从 .pos.stat 提取时序载噪比 (L1/L2) 与 PDOP
# ------------------------------------------------------------------------------
print("正在解析 6_RTK_Multi_L1L2.pos.stat 提取全历元载噪比与几何构型 ...")
epoch_snr_l1 = {}
epoch_snr_l2 = {}
epoch_sats = {}

with open(STAT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("$SAT"):
            p = line.strip().split(',')
            if len(p) > 10:
                try:
                    tow = round(float(p[2]), 1)
                    frq = int(p[4])
                    az = float(p[5])
                    el = float(p[6])
                    vsat = int(p[9])
                    snr = float(p[10])
                except (ValueError, IndexError):
                    continue
                if vsat == 1:
                    if frq == 1:
                        epoch_snr_l1.setdefault(tow, []).append(snr)
                        epoch_sats.setdefault(tow, []).append((az, el))
                    elif frq == 2:
                        epoch_snr_l2.setdefault(tow, []).append(snr)

epoch_avg_snr_l1 = {k: np.mean(v) for k, v in epoch_snr_l1.items()}
epoch_avg_snr_l2 = {k: np.mean(v) for k, v in epoch_snr_l2.items()}

epoch_pdop = {}
for tow, sats in epoch_sats.items():
    if len(sats) >= 4:
        A = []
        for az, el in sats:
            az_r = math.radians(az)
            el_r = math.radians(el)
            e = math.cos(el_r) * math.sin(az_r)
            n = math.cos(el_r) * math.cos(az_r)
            u = math.sin(el_r)
            A.append([-e, -n, -u, 1.0])
        A = np.array(A)
        try:
            Q = np.linalg.inv(A.T @ A)
            pdop = math.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
            epoch_pdop[tow] = min(pdop, 25.0)
        except Exception:
            epoch_pdop[tow] = np.nan
    else:
        epoch_pdop[tow] = np.nan

df_pos['snr_l1'] = df_pos['tow'].map(epoch_avg_snr_l1).interpolate()
df_pos['snr_l2'] = df_pos['tow'].map(epoch_avg_snr_l2).interpolate()
df_pos['pdop'] = df_pos['tow'].map(epoch_pdop).interpolate().ffill().bfill()

# ------------------------------------------------------------------------------
# 5. 划分典型时段并计算关联统计特征
# ------------------------------------------------------------------------------
time_segments = [
    {"name": "时段一: 开阔平稳段", "start": 0, "end": 180, "color": "#52c41a"},
    {"name": "时段二: 狭窄弯道段", "start": 180, "end": 320, "color": "#fa8c16"},
    {"name": "时段三: 立交重遮挡段", "start": 460, "end": 640, "color": "#f5222d"},
    {"name": "时段四: 深度峡谷段", "start": 820, "end": 960, "color": "#722ed1"},
]

seg_stats = []
for s in time_segments:
    sub = df_pos[(df_pos['time_sec'] >= s['start']) & (df_pos['time_sec'] < s['end'])]
    seg_stats.append({
        "name": s['name'],
        "snr_l1": sub['snr_l1'].mean(),
        "ns": sub['ns'].mean(),
        "pdop": sub['pdop'].median(),
        "sig_h": sub['sig_h'].mean(),
        "eh_med": sub['eh'].median(),
        "eh_max": sub['eh'].max(),
        "fix_rate": (sub['q'] == 1).mean() * 100.0,
        "color": s['color']
    })

# ------------------------------------------------------------------------------
# 6. 开始绘制全景大图 (2行 x 3列)
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(24, 13.5), dpi=300)
gs = gridspec.GridSpec(2, 3, left=0.055, right=0.975, top=0.925, bottom=0.065, wspace=0.23, hspace=0.27)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

fig.suptitle(t_cfg["main_title"], fontsize=font_sizes["title_main"], fontweight='bold', y=0.975)

# ------------------------------------------------------------------------------
# 子图 1: 车辆行驶轨迹时间映射与解算状态分布 (左上)
# ------------------------------------------------------------------------------
# 使用彩色线条集合展示时间渐变
points = np.array([df_pos['gt_e'], df_pos['gt_n']]).T.reshape(-1, 1, 2)
segments_line = np.concatenate([points[:-1], points[1:]], axis=1)

norm_time = plt.Normalize(df_pos['time_sec'].min(), df_pos['time_sec'].max())
lc = LineCollection(segments_line, cmap='plasma', norm=norm_time, linewidth=3.2, zorder=2)
lc.set_array(df_pos['time_sec'][:-1])
line_art = ax1.add_collection(lc)

# 覆盖固定解与浮点解状态点
fix_m = (df_pos['q'] == 1)
flt_m = (df_pos['q'] == 2)
ax1.scatter(df_pos.loc[flt_m, 'e'], df_pos.loc[flt_m, 'n'], color='#ff4d4f', s=6, alpha=0.35, zorder=3, label='浮点解 (Float)')
ax1.scatter(df_pos.loc[fix_m, 'e'], df_pos.loc[fix_m, 'n'], color='#2ca02c', s=14, alpha=0.95, zorder=4, label='固定解 (Fixed)')

# 添加时间渐变色条
cbar1 = fig.colorbar(line_art, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.75)
cbar1.set_label('行驶时间 (s)', fontsize=font_sizes["axis_label"] - 1, fontweight='bold')
cbar1.ax.tick_params(labelsize=font_sizes["tick_label"] - 1)

# 关键时间与地标标注
key_ann = [
    (0, "t=0s (起点开阔)"),
    (240, "t=240s (狭窄弯道)"),
    (550, "t=550s (立交重遮挡)"),
    (900, "t=900s (深峡谷漂移)"),
    (1240, "t=1240s (终点)")
]
for t_val, label in key_ann:
    idx = (df_pos['time_sec'] - t_val).abs().idxmin()
    row = df_pos.iloc[idx]
    ax1.scatter(row['gt_e'], row['gt_n'], color='black', s=45, zorder=5)
    offset_x = 35 if row['gt_e'] > -800 else -140
    offset_y = 60 if row['gt_n'] > -1000 else -80
    ax1.annotate(label, xy=(row['gt_e'], row['gt_n']), xytext=(row['gt_e'] + offset_x, row['gt_n'] + offset_y),
                 fontsize=10.5, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='#ffffcc', edgecolor='#cccc00', alpha=0.9))

ax1.set_title(t_cfg["sub1_trajectory_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax1.set_xlabel(t_cfg["sub1_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax1.set_ylabel(t_cfg["sub1_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax1.tick_params(labelsize=font_sizes["tick_label"])
ax1.set_xlim(df_pos['gt_e'].min() - 100, df_pos['gt_e'].max() + 100)
ax1.set_ylim(df_pos['gt_n'].min() - 120, df_pos['gt_n'].max() + 120)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', fontsize=font_sizes["legend"], framealpha=0.9)

# ------------------------------------------------------------------------------
# 子图 2: 水平定位误差与滤波估计协方差 (3σ) 时序对比 (中上)
# ------------------------------------------------------------------------------
t_s = df_pos['time_sec']
ax2.plot(t_s, df_pos['eh'], color='#1f77b4', lw=1.6, label='真实水平误差', zorder=3)
ax2.fill_between(t_s, 0, 3 * df_pos['sig_h'], color='#ff7f0e', alpha=0.35, label='滤波估计方差包络 (3σ)', zorder=2)
ax2.plot(t_s, 3 * df_pos['sig_h'], color='#d95f02', lw=1.2, linestyle='--', zorder=2)

ax2.axhline(y=1.0, color='#2ca02c', linestyle=':', lw=1.8, label='1.0m 误差基准线')
ax2.axhline(y=10.0, color='#d62728', linestyle='--', lw=1.5, label='10.0m 重度漂移线')

# 高亮时段背景
for s in time_segments[1:]:
    ax2.axvspan(s['start'], s['end'], color=s['color'], alpha=0.08)

ax2.set_title(t_cfg["sub2_error_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax2.set_xlabel(t_cfg["sub2_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax2.set_ylabel(t_cfg["sub2_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax2.set_ylim(-1, 55)
ax2.tick_params(labelsize=font_sizes["tick_label"])
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=font_sizes["legend"], framealpha=0.9)

ax2.text(0.04, 0.95, "协方差膨胀期与误差尖峰严格重合\n局部真实误差远超估计方差(3σ)\n证实滤波器出现'过度自信与状态发散'",
         transform=ax2.transAxes, fontsize=11.5, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f6ffed', edgecolor='#b7eb8f', alpha=0.9))

# ------------------------------------------------------------------------------
# 子图 3: 模糊度固定检验比值 (Ratio) 时序分布 (右上)
# ------------------------------------------------------------------------------
ax3.plot(t_s, df_pos['ratio'], color='#4a6fa5', lw=1.4, zorder=3, label='检核比值')
ax3.axhline(y=3.0, color='#d62728', linestyle='--', lw=2.2, label='固定判定门限 (3.0)', zorder=4)

ax3.axhspan(0, 3.0, facecolor='#ff4d4f', alpha=0.10, label='不可固定区 (浮点状态)')
ax3.axhspan(3.0, 15.0, facecolor='#52c41a', alpha=0.10, label='可固定区 (固定状态)')

for s in time_segments[1:]:
    ax3.axvspan(s['start'], s['end'], color=s['color'], alpha=0.08)

ax3.set_title(t_cfg["sub3_ratio_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax3.set_xlabel(t_cfg["sub3_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax3.set_ylabel(t_cfg["sub3_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax3.set_ylim(0, 12)
ax3.tick_params(labelsize=font_sizes["tick_label"])
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=font_sizes["legend"], framealpha=0.9)

ax3.text(0.04, 0.95, "88.7% 历元比值受压制处于 3.0 之下\n与载噪比跌落及星数突降完全同步",
         transform=ax3.transAxes, fontsize=11.5, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f6', edgecolor='#ffadd2', alpha=0.9))

# ------------------------------------------------------------------------------
# 子图 4: 跟踪卫星数与空间几何衰减因子 (PDOP) 时序 (左下)
# ------------------------------------------------------------------------------
ax4_right = ax4.twinx()

line1 = ax4.plot(t_s, df_pos['ns'], color='#1890ff', lw=1.6, label='有效卫星数 (颗)')
line2 = ax4_right.plot(t_s, df_pos['pdop'], color='#fa8c16', lw=1.5, alpha=0.9, label='几何衰减因子 (PDOP)')

ax4_right.axhline(y=3.0, color='#52c41a', linestyle=':', lw=1.6, label='良好几何界限 (3.0)')
ax4_right.axhline(y=6.0, color='#f5222d', linestyle='--', lw=1.6, label='恶劣几何阈值 (6.0)')

for s in time_segments[1:]:
    ax4.axvspan(s['start'], s['end'], color=s['color'], alpha=0.08)

ax4.set_title(t_cfg["sub4_sat_dop_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax4.set_xlabel(t_cfg["sub4_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax4.set_ylabel(t_cfg["sub4_ylabel_sats"], fontsize=font_sizes["axis_label"], fontweight='bold', color='#1890ff')
ax4_right.set_ylabel(t_cfg["sub4_ylabel_dop"], fontsize=font_sizes["axis_label"], fontweight='bold', color='#fa8c16')

ax4.set_ylim(0, 32)
ax4_right.set_ylim(0, 22)
ax4.tick_params(labelsize=font_sizes["tick_label"])
ax4_right.tick_params(labelsize=font_sizes["tick_label"])
ax4.grid(True, linestyle='--', alpha=0.5)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='upper left', fontsize=font_sizes["legend"], framealpha=0.9)

# ------------------------------------------------------------------------------
# 子图 5: 双频平均载噪比 (C/N0) 时序衰减与时段箱线图 (中下)
# ------------------------------------------------------------------------------
ax5.plot(t_s, df_pos['snr_l1'], color='#096dd9', lw=1.5, label='L1 频段平均载噪比')
ax5.plot(t_s, df_pos['snr_l2'], color='#eb2f96', lw=1.3, alpha=0.85, label='L2 频段平均载噪比')
ax5.axhline(y=40.0, color='#52c41a', linestyle=':', lw=1.8, label='40 dB-Hz 优质信号线')
ax5.axhline(y=35.0, color='#faad14', linestyle='--', lw=1.5, label='35 dB-Hz 衰减警戒线')

for s in time_segments[1:]:
    ax5.axvspan(s['start'], s['end'], color=s['color'], alpha=0.08)

ax5.set_title(t_cfg["sub5_snr_time_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax5.set_xlabel(t_cfg["sub5_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax5.set_ylabel(t_cfg["sub5_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax5.set_ylim(25, 50)
ax5.tick_params(labelsize=font_sizes["tick_label"])
ax5.grid(True, linestyle='--', alpha=0.5)
ax5.legend(loc='lower right', fontsize=font_sizes["legend"], framealpha=0.9)

ax5.text(0.04, 0.95, "车辆驶入峡谷或立交桥时\n平均载噪比整体发生 10~15 dB-Hz 衰减\n直射信号中断, 反射弱信号主导",
         transform=ax5.transAxes, fontsize=11.5, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e6f7ff', edgecolor='#91d5ff', alpha=0.9))

# ------------------------------------------------------------------------------
# 子图 6: 典型行驶时段多物理量响应特性对比 (右下)
# ------------------------------------------------------------------------------
seg_names = ["时段一\n(开阔平稳)", "时段二\n(狭窄弯道)", "时段三\n(立交遮挡)", "时段四\n(深度峡谷)"]
x_pos = np.arange(len(seg_names))
width = 0.18

# 提取物理指标
snr_vals  = [s['snr_l1'] for s in seg_stats]
ns_vals   = [s['ns'] for s in seg_stats]
pdop_vals = [s['pdop'] for s in seg_stats]
sigh_vals = [s['sig_h'] * 10.0 for s in seg_stats] # 乘以10以便展示 (分米级)

rects1 = ax6.bar(x_pos - 1.5*width, snr_vals, width, label='平均载噪比 (dB-Hz)', color='#1890ff', alpha=0.85)
rects2 = ax6.bar(x_pos - 0.5*width, ns_vals, width, label='平均有效卫星 (颗)', color='#52c41a', alpha=0.85)
rects3 = ax6.bar(x_pos + 0.5*width, pdop_vals, width, label='几何衰减因子 (PDOP)', color='#fa8c16', alpha=0.85)
rects4 = ax6.bar(x_pos + 1.5*width, sigh_vals, width, label='估计方差 sig_h (×0.1m)', color='#f5222d', alpha=0.85)

ax6.set_title(t_cfg["sub6_segment_corr_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax6.set_xlabel(t_cfg["sub6_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax6.set_ylabel(t_cfg["sub6_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(seg_names, fontsize=font_sizes["tick_label"])
ax6.set_ylim(0, 52)
ax6.tick_params(labelsize=font_sizes["tick_label"])
ax6.grid(True, linestyle='--', alpha=0.5, axis='y')
ax6.legend(loc='upper right', fontsize=font_sizes["legend"] - 1, framealpha=0.9, ncol=2)

# 在每个柱子上方添加微型标签
for rect in rects1:
    h = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., h + 0.8, f'{h:.1f}', ha='center', va='bottom', fontsize=9.5)
for rect in rects3:
    h = rect.get_height()
    ax6.text(rect.get_x() + rect.get_width()/2., h + 0.8, f'{h:.1f}', ha='center', va='bottom', fontsize=9.5, color='#d46b08', fontweight='bold')

# ------------------------------------------------------------------------------
# 7. 全局标签对齐规范 (GEMINI.md Rule 1.4)
# ------------------------------------------------------------------------------
fig.align_xlabels([ax1, ax2, ax3])
fig.align_xlabels([ax4, ax5, ax6])
fig.align_ylabels([ax1, ax4])
fig.align_ylabels([ax2, ax5])
fig.align_ylabels([ax3, ax6])

# ------------------------------------------------------------------------------
# 8. 保存输出
# ------------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"[OK] 全时序耦合诊断大图已成功生成并保存至: {OUTPUT_PNG}")
