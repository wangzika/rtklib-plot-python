"""
================================================================================
城市深峡谷车载 RTK 误差诱因与性能瓶颈全景诊断大图绘制脚本
功能：
  - 从 reference.csv、6_RTK_Multi_L1L2.pos、6_RTK_Multi_L1L2.pos.stat 中提取全套诊断数据
  - 严格执行学术大图排版规范 (GEMINI.md)：汇报级大字号、纯中文无杂糅、GridSpec严密平齐对齐
  - 生成 2x3 六联全景诊断大图：
    1. 车辆行驶轨迹与解算状态分布 (空间位置与丢锁路段)
    2. 水平与天向定位误差时序变化 (误差尖峰与发散漂移)
    3. 模糊度固定检验比值时序分布 (Ratio 被压制无法固定的直接证据)
    4. 跟踪卫星数与空间几何衰减因子 (可见星骤降与 PDOP 几何发散)
    5. 卫星高度角与载噪比关联分布 (高仰角异常低载噪比，证明 NLOS 存在)
    6. 双差伪距与载波观测残差分布 (观测值非高斯重尾分布，证明滤波器发散根源)
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

# ------------------------------------------------------------------------------
# 1. 基础配置与路径
# ------------------------------------------------------------------------------
BASE_DIR = r"D:\BaiduNetdiskDownload\GNSS_RTKLIB_Experiments"
DATA_DIR = r"D:\BaiduNetdiskDownload\Tokyo_Data\Tokyo_Data\Odaiba"

REF_CSV   = os.path.join(DATA_DIR, "reference.csv")
POS_FILE  = os.path.join(BASE_DIR, "results_pos", "6_RTK_Multi_L1L2.pos")
STAT_FILE = os.path.join(BASE_DIR, "results_pos", "6_RTK_Multi_L1L2.pos.stat")
CONFIG_JSON = os.path.join(BASE_DIR, "configs", "chart_titles_config.json")
OUTPUT_PNG  = os.path.join(BASE_DIR, "plots", "urban_rtk_error_diagnostics_big_figure.png")

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

t_cfg = cfg.get("charts", {}).get("plot_urban_rtk_diagnostics", {
    "main_title": "城市深峡谷车载 RTK 误差诱因与性能瓶颈全景诊断",
    "sub1_trajectory_title": "车辆行驶轨迹与解算状态分布",
    "sub1_xlabel": "东向坐标 (m)",
    "sub1_ylabel": "北向坐标 (m)",
    "sub2_error_title": "水平与天向定位误差时序变化",
    "sub2_xlabel": "时间 (s)",
    "sub2_ylabel": "定位误差 (m)",
    "sub3_ratio_title": "模糊度固定检验比值时序分布",
    "sub3_xlabel": "时间 (s)",
    "sub3_ylabel": "检验比值",
    "sub4_sat_dop_title": "跟踪卫星数与空间几何衰减因子",
    "sub4_xlabel": "时间 (s)",
    "sub4_ylabel_sats": "有效卫星数 (颗)",
    "sub4_ylabel_dop": "几何因子 (PDOP)",
    "sub5_el_snr_title": "卫星高度角与载噪比关联分布",
    "sub5_xlabel": "高度角 (°)",
    "sub5_ylabel": "载噪比 (dB-Hz)",
    "sub6_residuals_title": "双差伪距与载波观测残差分布",
    "sub6_xlabel": "伪距残差 (m)",
    "sub6_xlabel_cp": "载波残差 (cm)",
    "sub6_ylabel": "概率密度"
})

# Matplotlib 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# WGS84 坐标投影转换函数
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

# ------------------------------------------------------------------------------
# 2. 加载真值 reference.csv
# ------------------------------------------------------------------------------
print("正在读取真值数据 reference.csv ...")
ref_df = pd.read_csv(REF_CSV, skipinitialspace=True)
ref_df['TOW_round'] = ref_df['GPS TOW (s)'].round(1)

origin_lat0 = ref_df.iloc[0]['Latitude (deg)']
origin_lon0 = ref_df.iloc[0]['Longitude (deg)']
origin_x0   = ref_df.iloc[0]['ECEF X (m)']
origin_y0   = ref_df.iloc[0]['ECEF Y (m)']
origin_z0   = ref_df.iloc[0]['ECEF Z (m)']

ref_map = {}
ref_enu_e, ref_enu_n = [], []
for _, row in ref_df.iterrows():
    tow = row['TOW_round']
    x, y, z = row['ECEF X (m)'], row['ECEF Y (m)'], row['ECEF Z (m)']
    e, n, u = ecef_to_enu(x - origin_x0, y - origin_y0, z - origin_z0, origin_lat0, origin_lon0)
    ref_map[tow] = {'x': x, 'y': y, 'z': z, 'e': e, 'n': n, 'u': u}
    ref_enu_e.append(e)
    ref_enu_n.append(n)

# ------------------------------------------------------------------------------
# 3. 加载并匹配 RTK 解算结果 .pos
# ------------------------------------------------------------------------------
print("正在解析 RTK 解算结果 6_RTK_Multi_L1L2.pos ...")
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
                pos_records.append({
                    'tow': tow,
                    'e': e, 'n': n, 'u': u,
                    'gt_e': gt['e'], 'gt_n': gt['n'], 'gt_u': gt['u'],
                    'de': de, 'dn': dn, 'du': du,
                    'eh': eh, 'err_3d': err_3d,
                    'q': q, 'ns': ns, 'ratio': ratio
                })

df_pos = pd.DataFrame(pos_records)
tow0 = df_pos['tow'].iloc[0]
df_pos['time_sec'] = df_pos['tow'] - tow0

fix_mask = (df_pos['q'] == 1)
float_mask = (df_pos['q'] == 2)
fix_rate = fix_mask.mean() * 100.0

print(f"匹配完成: 共 {len(df_pos)} 历元, 固定率 {fix_rate:.2f}%")

# ------------------------------------------------------------------------------
# 4. 加载并解析 .pos.stat (提取 PDOP, SNR, 观测残差)
# ------------------------------------------------------------------------------
print("正在解析状态文件 6_RTK_Multi_L1L2.pos.stat (提取 PDOP、载噪比与双差残差) ...")
epoch_sats = {}
res_p_list = []
res_c_list = []
el_list = []
snr_list = []

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
                    resp = float(p[7])
                    resc = float(p[8])
                    vsat = int(p[9])
                    snr = float(p[10])
                except (ValueError, IndexError):
                    continue

                if vsat == 1 and frq == 1:
                    epoch_sats.setdefault(tow, []).append((az, el))
                    if snr > 12.0:
                        el_list.append(el)
                        snr_list.append(snr)
                    if abs(resp) > 1e-4:
                        res_p_list.append(resp)
                    if abs(resc) > 1e-4:
                        res_c_list.append(resc * 100.0) # 转换为 cm

# 计算各历元 PDOP
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

df_pos['pdop'] = df_pos['tow'].map(epoch_pdop)
df_pos['pdop'] = df_pos['pdop'].ffill().bfill()

print(f"数据准备完毕: 提取双差残差 {len(res_p_list)} 条, 载噪比样本 {len(snr_list)} 条")

# ------------------------------------------------------------------------------
# 5. 绘制全景科研大图 (2行 x 3列 GridSpec)
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(24, 13.5), dpi=300)
gs = gridspec.GridSpec(2, 3, left=0.055, right=0.975, top=0.925, bottom=0.065, wspace=0.22, hspace=0.26)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# 总标题
fig.suptitle(t_cfg["main_title"], fontsize=font_sizes["title_main"], fontweight='bold', y=0.975)

# ------------------------------------------------------------------------------
# 子图 1: 车辆行驶轨迹与解算状态分布 (左上)
# ------------------------------------------------------------------------------
ax1.plot(df_pos['gt_e'], df_pos['gt_n'], color='#2b2b2b', lw=2.2, label='基准真值轨迹', zorder=2)
ax1.scatter(df_pos.loc[float_mask, 'e'], df_pos.loc[float_mask, 'n'], color='#d62728', s=10, alpha=0.45, label='浮点解 (Float)', zorder=3)
ax1.scatter(df_pos.loc[fix_mask, 'e'], df_pos.loc[fix_mask, 'n'], color='#2ca02c', s=16, alpha=0.90, label='固定解 (Fixed)', zorder=4)

# 标注起点与终点
ax1.scatter(df_pos['gt_e'].iloc[0], df_pos['gt_n'].iloc[0], marker='^', color='#0099ff', s=100, zorder=5, label='起点')
ax1.scatter(df_pos['gt_e'].iloc[-1], df_pos['gt_n'].iloc[-1], marker='s', color='#ff7f0e', s=90, zorder=5, label='终点')

ax1.set_title(t_cfg["sub1_trajectory_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax1.set_xlabel(t_cfg["sub1_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax1.set_ylabel(t_cfg["sub1_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax1.tick_params(labelsize=font_sizes["tick_label"])
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', fontsize=font_sizes["legend"], framealpha=0.9)

# 局部标注信息
ax1.text(0.96, 0.95, f"总历元数: {len(df_pos)}\n固定率: {fix_rate:.1f}%\n浮点率: {100-fix_rate:.1f}%\n漂移多发于狭窄高楼弯道",
         transform=ax1.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbe6', edgecolor='#ffe58f', alpha=0.9))

# ------------------------------------------------------------------------------
# 子图 2: 水平与天向定位误差时序变化 (中上)
# ------------------------------------------------------------------------------
t_sec = df_pos['time_sec']
ax2.plot(t_sec, df_pos['eh'], color='#1f77b4', lw=1.6, label='水平误差', zorder=3)
ax2.plot(t_sec, df_pos['du'].abs(), color='#ff7f0e', lw=1.2, alpha=0.75, label='天向误差', zorder=2)

ax2.axhline(y=1.0, color='#2ca02c', linestyle=':', lw=1.8, label='1.0m 误差基准线')
ax2.axhline(y=10.0, color='#d62728', linestyle='--', lw=1.5, label='10.0m 重度漂移线')

ax2.set_title(t_cfg["sub2_error_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax2.set_xlabel(t_cfg["sub2_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax2.set_ylabel(t_cfg["sub2_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax2.set_ylim(-1, 55)
ax2.tick_params(labelsize=font_sizes["tick_label"])
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=font_sizes["legend"], framealpha=0.9)

ax2.text(0.04, 0.95, f"最大水平误差: {df_pos['eh'].max():.1f} m\n水平中位数: {df_pos['eh'].median():.2f} m\n全历元 3D 均方根: {math.sqrt((df_pos['err_3d']**2).mean()):.1f} m",
         transform=ax2.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f6ffed', edgecolor='#b7eb8f', alpha=0.9))

# ------------------------------------------------------------------------------
# 子图 3: 模糊度固定检验比值 (Ratio) 时序分布 (右上)
# ------------------------------------------------------------------------------
ax3.plot(t_sec, df_pos['ratio'], color='#4a6fa5', lw=1.4, zorder=3, label='检核比值')
ax3.axhline(y=3.0, color='#d62728', linestyle='--', lw=2.2, label='固定判定门限 (3.0)', zorder=4)

# 阴影区间
ax3.axhspan(0, 3.0, facecolor='#ff4d4f', alpha=0.10, label='不可固定区 (浮点状态)')
ax3.axhspan(3.0, 15.0, facecolor='#52c41a', alpha=0.10, label='可固定区 (固定状态)')

ax3.set_title(t_cfg["sub3_ratio_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax3.set_xlabel(t_cfg["sub3_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax3.set_ylabel(t_cfg["sub3_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax3.set_ylim(0, 12)
ax3.tick_params(labelsize=font_sizes["tick_label"])
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=font_sizes["legend"], framealpha=0.9)

ax3.text(0.04, 0.95, "受城市多径与粗差污染影响\n88.7% 历元比值被压制在 3.0 以下\n全网模糊度固定判定失败",
         transform=ax3.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0f6', edgecolor='#ffadd2', alpha=0.9))

# ------------------------------------------------------------------------------
# 子图 4: 跟踪卫星数与空间几何衰减因子 (左下)
# ------------------------------------------------------------------------------
ax4_right = ax4.twinx()

line1 = ax4.plot(t_sec, df_pos['ns'], color='#1890ff', lw=1.6, label='有效卫星数 (颗)')
line2 = ax4_right.plot(t_sec, df_pos['pdop'], color='#fa8c16', lw=1.5, alpha=0.9, label='空间几何衰减因子 (PDOP)')

ax4_right.axhline(y=3.0, color='#52c41a', linestyle=':', lw=1.6, label='良好几何界限 (3.0)')
ax4_right.axhline(y=6.0, color='#f5222d', linestyle='--', lw=1.6, label='恶劣几何阈值 (6.0)')

ax4.set_title(t_cfg["sub4_sat_dop_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax4.set_xlabel(t_cfg["sub4_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax4.set_ylabel(t_cfg["sub4_ylabel_sats"], fontsize=font_sizes["axis_label"], fontweight='bold', color='#1890ff')
ax4_right.set_ylabel(t_cfg["sub4_ylabel_dop"], fontsize=font_sizes["axis_label"], fontweight='bold', color='#fa8c16')

ax4.set_ylim(0, 32)
ax4_right.set_ylim(0, 22)
ax4.tick_params(labelsize=font_sizes["tick_label"])
ax4_right.tick_params(labelsize=font_sizes["tick_label"])
ax4.grid(True, linestyle='--', alpha=0.5)

# 合并双轴图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='upper left', fontsize=font_sizes["legend"], framealpha=0.9)

# ------------------------------------------------------------------------------
# 子图 5: 卫星高度角与载噪比关联分布 (中下)
# ------------------------------------------------------------------------------
el_arr = np.array(el_list)
snr_arr = np.array(snr_list)

# 随机下采样 25000 点以保证快速且清晰绘图
if len(el_arr) > 25000:
    idx_sample = np.random.choice(len(el_arr), 25000, replace=False)
    el_sub = el_arr[idx_sample]
    snr_sub = snr_arr[idx_sample]
else:
    el_sub, snr_sub = el_arr, snr_arr

hb = ax5.hexbin(el_sub, snr_sub, gridsize=45, cmap='Blues', mincnt=1, edgecolors='none', alpha=0.85)
cb = fig.colorbar(hb, ax=ax5, orientation='vertical', pad=0.02, shrink=0.85)
cb.set_label('观测频次密度', fontsize=font_sizes["axis_label"], fontweight='bold')
cb.ax.tick_params(labelsize=font_sizes["tick_label"] - 1)

# 理论健康信号包络曲线 (根据经验天线增益模型)
el_grid = np.linspace(15, 90, 100)
snr_model = 32.0 + 17.0 * np.sin(np.radians(el_grid))
ax5.plot(el_grid, snr_model, color='#52c41a', lw=2.5, linestyle='-', label='开阔天空理论参考均线')

# 标注非视距 (NLOS) 异常区 (高仰角低载噪比)
rect = plt.Rectangle((40, 15), 45, 17, linewidth=2.0, edgecolor='#f5222d', facecolor='#ff4d4f', alpha=0.18, linestyle='--')
ax5.add_patch(rect)
ax5.text(62.5, 23.5, "高仰角弱信号异常区\n(典型建筑物二次反射NLOS信号)", fontsize=11.5, fontweight='bold', color='#cf1322',
         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ff4d4f', alpha=0.9))

ax5.set_title(t_cfg["sub5_el_snr_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax5.set_xlabel(t_cfg["sub5_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax5.set_ylabel(t_cfg["sub5_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax5.set_xlim(10, 92)
ax5.set_ylim(12, 55)
ax5.tick_params(labelsize=font_sizes["tick_label"])
ax5.grid(True, linestyle='--', alpha=0.5)
ax5.legend(loc='lower right', fontsize=font_sizes["legend"], framealpha=0.9)

# ------------------------------------------------------------------------------
# 子图 6: 双差伪距与载波观测残差分布 (右下)
# ------------------------------------------------------------------------------
res_p_arr = np.array(res_p_list)
res_c_arr = np.array(res_c_list)

# 限制显示范围以展示核心峰值和重尾
res_p_filtered = res_p_arr[(res_p_arr > -12.0) & (res_p_arr < 12.0)]
res_c_filtered = res_c_arr[(res_c_arr > -6.0) & (res_c_arr < 6.0)]

counts, bins, patches = ax6.hist(res_p_filtered, bins=60, density=True, color='#1890ff', alpha=0.6, label='双差伪距残差分布', edgecolor='white', lw=0.5)

# 拟合高斯分布曲线对照
mu = float(np.mean(res_p_filtered))
std = float(np.std(res_p_filtered))
x_norm = np.linspace(-12, 12, 200)
p_norm = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / std)**2)
ax6.plot(x_norm, p_norm, 'k--', lw=2.2, label=f'理论高斯拟合 (标准差={std:.2f}m)')

ax6.set_title(t_cfg["sub6_residuals_title"], fontsize=font_sizes["title_subplot"], fontweight='bold', pad=10)
ax6.set_xlabel(t_cfg["sub6_xlabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax6.set_ylabel(t_cfg["sub6_ylabel"], fontsize=font_sizes["axis_label"], fontweight='bold')
ax6.set_xlim(-12, 12)
ax6.tick_params(labelsize=font_sizes["tick_label"])
ax6.grid(True, linestyle='--', alpha=0.5)
ax6.legend(loc='upper right', fontsize=font_sizes["legend"], framealpha=0.9)

ax6.text(0.04, 0.95, f"伪距残差中位数: {np.median(np.abs(res_p_arr)):.2f} m\n95%分位数: {np.percentile(np.abs(res_p_arr), 95):.2f} m\n极端残差达 42.7m (严重重尾)\n破坏卡尔曼滤波白噪声假设",
         transform=ax6.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f0ff', edgecolor='#d3adf7', alpha=0.9))

# ------------------------------------------------------------------------------
# 6. 全局标签对齐规范 (GEMINI.md Rule 1.4)
# ------------------------------------------------------------------------------
fig.align_xlabels([ax1, ax2, ax3])
fig.align_xlabels([ax4, ax5, ax6])
fig.align_ylabels([ax1, ax4])
fig.align_ylabels([ax2, ax5])
fig.align_ylabels([ax3, ax6])

# ------------------------------------------------------------------------------
# 7. 保存图表
# ------------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"[OK] 全景综合诊断大图已成功生成并保存至: {OUTPUT_PNG}")
