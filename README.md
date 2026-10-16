# RTKLIB Plot Python (RTKLIB 结果可视化与多维科研诊断平台)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**RTKLIB Plot Python** 是一个专为 GNSS 高精度定位解算设计的开源 Python 可视化与深度诊断平台。它不仅全面涵盖了官方 C 版 `rtkplot.exe` 的经典视图（轨迹、高程、卫星天空图、残差、DOP 等），更针对科研论文与工程调试深度集成了 **真值对比、卡尔曼滤波协方差检验、城市弱纹理/多路径时空耦合诊断、假固定（False Fix）智能排查、动画轨迹回放与自定义多屏拖拽拼图**。

---

## 🌟 核心特性

- 🖥️ **现代化交互式 GUI (`rtklib_plot_gui.py`)**：
  - **时空联动游标 (Epoch Sync)**：拖拽车辆标记或时间轴，所有子图（轨迹、天空图、时序、残差）同步联动，基于 Matplotlib Blitting 实现无延迟高刷。
  - **动态轨迹回放 (Playback)**：支持 1x / 10x / 50x 倍速动画模拟车辆运动与卫星星空图轨迹演变。
  - **自由多屏网格布局 (Custom Analysis Grid)**：支持多图自由勾选组合，支持任意水平/垂直分屏（Split）、单面板放大（Focus）及拖拽排版。
  - **时间段多色彩标记 (Time Segments)**：支持自由添加感兴趣区域（开阔路段、林荫道、立交桥、隧道），并在所有视图中同步色块对比。
  - **假固定排查 (False Fix Analysis)**：智能对比真值与解算状态，当模糊度固定但误差超出设定门限（如 0.2m）时自动预警并定位。
- 📊 **批量诊断分析引擎 (`rtklib_diagnostics.py`)**：
  - 无需图形界面的批处理命令行工具，一键生成 **15 项标准化科研诊断图表** 与 6-in-1 综合质检看板（Dashboard）。
  - 支持直接导出高质量出版级矢量图（PDF、SVG）及高分辨率位图（PNG，支持自定义 DPI）。
- 📐 **严谨的测绘与数学推导**：
  - 自动从 RTKLIB `.pos` 反算 ECEF 协方差矩阵（处理非对角线 `sqvar` 符号保持）。
  - 严格坐标系与协方差投影变换（ECEF $\rightarrow$ 局部站心 ENU），保留 3 轴置信包络线（$1.96\sigma$）。
  - 内置 GPS-UTC 历次闰秒变迁表，支持将 GPST 精准转换为东八区（CST）本地标准时间。
  - 支持高精度真值（`reference.csv`）时间戳自动插值对齐与 RMSE / CEP50 / 2DRMS 统计。

---

## 📂 项目结构

```text
rtklib-plot-python/
├── rtklib_plot_gui.py         # 交互式 GUI 分析平台（主程序）
├── rtklib_diagnostics.py      # 15 项科研诊断分析引擎与批处理导出
├── requirements.txt           # Python 依赖项
├── docs/                      # 详细技术与使用手册
│   ├── USER_GUIDE.md          # 交互式 GUI 详细使用说明
│   └── TECHNICAL_MANUAL.md    # 核心算法、协方差转换与数学原理手册
├── advanced_diagnostics/      # 复杂场景（如城市峡谷多路径耦合）专项分析脚本
│   ├── plot_urban_rtk_diagnostics_big_figure.py
│   ├── plot_urban_rtk_diagnostics_time_coupled.py
│   └── plot_axis_errors_and_covariance.py
├── examples/                  # 示例与测试数据
│   ├── demo.pos               # RTKLIB 解算位置示例文件
│   └── demo.pos.stat          # 卫星观测与残差状态示例文件
└── LICENSE                    # MIT 开源许可证
```

---

## 🚀 快速上手

### 1. 安装环境依赖

```bash
git clone git@github.com:wangzika/rtklib-plot-python.git
cd rtklib-plot-python

pip install -r requirements.txt
```

### 2. 启动交互式分析平台 (GUI)

直接启动工具（自动载入内置示例数据）：
```bash
python rtklib_plot_gui.py
```

或指定您自己的 `.pos` 解算结果与高精度真值 `reference.csv`：
```bash
python rtklib_plot_gui.py --pos "path/to/solution.pos" --reference "path/to/reference.csv"
```
*(注：如果同一目录下存在同名的 `<solution>.pos.stat`，脚本会自动识别并载入单星观测残差与天空图数据。)*

### 3. 无界面批量导出 15 项科研诊断图

```bash
# 导出全部 15 张标准图及 Dashboard 到 plots_15_diagnostics 目录
python rtklib_diagnostics.py --plots all --dpi 200

# 仅导出指定图（如第 1, 2, 3, 8 张），并输出为 PDF 和 PNG
python rtklib_diagnostics.py --plots 1,2,3,8 --format png,pdf
```

---

## 📈 支持的 15 项标准诊断清单

| 编号 | 诊断图类别 | 说明与包含指标 |
| :---: | :--- | :--- |
| **01** | `ENU Position` | 水平二维轨迹、高程时序、解状态（Fix/Float/Single）背景色带 |
| **02** | `ENU Error` | E/N/U 各轴向误差、$\mu \pm 1.96\sigma$ 经验区间与滤波理论包络线对比 |
| **03** | `Horizontal / 3D Error` | 水平与 3D 误差时间序列、经验累积分布函数（CDF）、RMSE 标注 |
| **04** | `Solution Status` | 解状态时间序列跳变图与各解算类型百分比柱状图 |
| **05** | `AR Ratio` | 模糊度固定检验 Ratio 时序与直方图（中位数、>3 固定门限占比） |
| **06** | `Satellites` | 可见星数 vs 参与解算星数、各卫星系统（GPS/BDS/GAL/GLO/QZSS）堆叠图 |
| **07** | `DOP` | 由空间视线几何现场计算的 PDOP / HDOP / VDOP 变化曲线与分布 |
| **08** | `Skyplot` | 极坐标天空图（0~90° 仰角与 0~360° 方位角），各卫星轨迹与 PRN 标注 |
| **09** | `C/N0` | 各频段（L1/L2 等）与各系统载噪比时序散点图及统计箱线图 |
| **10** | `Elevation` | 卫星高度角时序、高度角与 C/N0 关系、高度角与伪距残差关联分布 |
| **11** | `Multipath Proxy` | 伪距残差按高度角分组（10° Bin）的 RMS 曲线（多路径效应代理分析） |
| **12** | `Residuals` | 伪距残差（m）与载波相位残差（cm）时序分布与高斯拟合直方图 |
| **13** | `Cycle Slips` | 周跳事件时空分布散点与历元累计周跳阶跃线 |
| **14** | `Covariance` | 滤波协方差标准差对数时序、E-N 相关系数时序、1.96$\sigma$ 真实误差覆盖率 |
| **15** | `Innovation / NIS` | 滤波状态检验规范说明 |
| **看板**| `Dashboard` | 6 合 1 核心科研质检综合看板 |

---

## 🤝 贡献与许可

本项目采用 [MIT 许可证](LICENSE)。欢迎提交 Issue 与 Pull Request 共同改进！
