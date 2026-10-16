# RTKLIB Python 绘图工具完整手册

本手册介绍 `rtklib_plot_gui.py` 交互式桌面程序和 `rtklib_diagnostics.py` 批量出图程序：各图的含义、计算定义、界面操作、数据格式，以及 Windows、macOS 和 Linux 下的安装方式。

## 1. 工具概览

工具读取 RTKLIB `.pos` 解算文件，可选读取同名 `.pos.stat` 卫星状态文件和真值 CSV。它包括两种使用方式：

1. **交互式 GUI**：浏览轨迹、逐历元查看、播放、筛选解算状态、组合图表、标记区间并导出当前视图。
2. **批量诊断出图**：一次生成 Plot 01–15 和 Dashboard，支持 PNG、PDF、SVG，适用于报告和归档。

程序源码位于 `scripts/`：

| 文件 | 用途 |
| --- | --- |
| `scripts/rtklib_plot_gui.py` | Tk 图形界面、交互与当前视图导出 |
| `scripts/rtklib_diagnostics.py` | 数据读取、指标计算、15 张静态诊断图和 Dashboard |
| `requirements-rtklib-plot.txt` | Python 依赖 |

建议 Python 3.10 或更新版本。此工具的图形界面使用 Tkinter 和 Matplotlib TkAgg；批量出图不需要图形界面。当前开发环境使用 Python 3.11.9、NumPy 2.4.6、pandas 3.0.5 和 Matplotlib 3.11.2。日期转换按 GPST 与 UTC 之间的闰秒偏移计算，再显示为固定 UTC+08:00；当前内置偏移表为 2017 年以来的 18 秒，若将来 UTC 再增加闰秒，需要更新程序中的表。

## 2. 各图说明

### 2.1 批量诊断图 Plot 01–15

| 编号 | 图名 | 内容与用途 |
| --- | --- | --- |
| 01 | ENU 轨迹与高程 | 水平 ENU 轨迹、真值轨迹和 RTK/真值 Up 分量随时间变化；高程图背景按解算状态着色。 |
| 02 | ENU 位置误差与 95% 区间 | 分别显示 ΔE、ΔN、ΔU。标出各分量平均偏差、经验 `均值 ± 1.96×样本标准差` 区间，并叠加 `.pos` 协方差给出的 `±1.96σ` 参考包络。需要真值。 |
| 03 | 水平、三维误差与经验 CDF | 同时显示水平误差和三维误差时间序列，以及两者的经验累积分布；标题包含 RMSE。需要真值。 |
| 04 | 解算状态与历元占比 | RTKLIB Q 状态随时间变化，并汇总 FIX、FLOAT 等状态所占比例。 |
| 05 | 模糊度固定 Ratio | 显示 AR ratio 时间序列和分布；图中的 3 是便于观察的参考线，不等于固定解正确性的保证。 |
| 06 | 跟踪、使用卫星数与星座组成 | 按历元统计跟踪 PRN 数和 `vsat=1` 的使用 PRN 数；另按 GPS、GLONASS、Galileo、BeiDou、QZSS 等星座显示 L1 有效卫星组成。 |
| 07 | DOP | 从 `.stat` 的 L1、`vsat=1` 卫星方位角和高度角重新计算 PDOP、HDOP、VDOP，并显示 PDOP 分布。它是几何重算值，不是 `.pos` 中直接记录的 DOP。 |
| 08 | 卫星天空图 | 以极坐标显示方位角、高度角和各 PRN 的观测轨迹；颜色表示星座。 |
| 09 | C/N₀ 信号质量 | 显示各频率 C/N₀ 时间序列，并按星座和频率比较分布。 |
| 10 | 卫星高度角与观测质量 | 显示 L1 高度角时间序列、高度角与 C/N₀ 关系，以及高度角与伪距后验残差关系。 |
| 11 | 伪距残差诊断 | 显示 L1 伪距后验残差时间序列和按高度角分箱的 RMS。此量只能作为多路径/遮挡诊断代理，**不是**独立的多路径组合观测值。 |
| 12 | `$SAT` 观测残差 | 分别查看伪距和载波后验残差时间序列及其直方图；载波残差直方图按厘米显示。 |
| 13 | 周跳标记 | 显示 `.stat` 中 `slipc>0` 的记录，并汇总涉及卫星数和累计标记数。 |
| 14 | 协方差与误差覆盖 | 显示 ENU 估计标准差、E-N 相关系数，以及东向真值误差相对于 `1.96σE` 的覆盖情况。 |
| 15 | 创新量 / NIS 数据说明 | 当前 `.pos/.stat` 不包含滤波创新向量 `v` 和对应协方差 `S`，所以只显示不可计算的原因，不生成伪造的 NIS 曲线。 |

### 2.2 交互式 GUI 图表

| 图表 | 说明 |
| --- | --- |
| Gnd Trk | 水平 ENU 轨迹，按 Q 状态区分 FIX/FLOAT；叠加真值轨迹、当前历元小车标记、真值点及误差连线。 |
| ENU Position | E、N、U 三个共享时间轴的子图，查看相对 ENU 位置分量变化。它描述运动轨迹，不是误差。 |
| Height | RTK Up 分量与真值 Up 分量的对比。保留单独的高度图入口。 |
| ENU Error | ΔE、ΔN、ΔU 三个共享时间轴子图，需匹配真值。 |
| Solution Status | Q 状态时间序列和状态比例摘要。 |
| AR Ratio | Ratio 时间序列；参考线 3 不是正确固定率。 |
| Satellites | 跟踪和参与解算的卫星数时间序列。 |
| DOP | 由卫星几何重算的 DOP 时间序列。 |
| Skyplot | 卫星天空轨迹；当前历元可见卫星随播放或滑块变化。 |
| C/N₀ | 各频率载噪比时间序列。 |
| Elevation | L1 卫星高度角时间序列。 |
| Residuals | 伪距和载波后验残差时间序列。 |
| Multipath Proxy | 按高度角查看绝对伪距后验残差，仅作多路径/遮挡代理指标。 |
| Cycle Slips | `.stat` 周跳标志记录。 |
| Covariance | 左轴为 RTKLIB 协方差传播得到的 E/N/U 位置 `1σ` 估计不确定度；有真值时右轴对照 `|ΔE|/(1.96σE)`。估计不确定度不是实测定位误差。 |
| False Fix Analysis | 3D 误差点图：FIX 绿点、FLOAT 黄点；超过所设 3D 误差门限的历元标红色叉号，并报告经验错误 FIX 率。需要真值。 |
| Innovation / NIS | 解释 `.pos/.stat` 缺少创新向量 `v` 和协方差 `S`，因而无法计算 `NIS=vᵀS⁻¹v`。 |

## 3. 指标定义、坐标与适用范围

真值数据按 GPST 时间与 `.pos` 历元匹配，并在真值采样时刻之间线性插值 ECEF 坐标。ENU 原点取真值 CSV 的第一行位置。定义：

```text
ΔE = E_RTK − E_truth
ΔN = N_RTK − N_truth
ΔU = U_RTK − U_truth
Horizontal error = √(ΔE² + ΔN²)
3D error         = √(ΔE² + ΔN² + ΔU²)
RMSE(x)          = √mean(x²)
```

经验 CDF 将误差样本排序后，以 `序号 / 样本数` 作为累计概率。ENU 误差是有符号的；RMSE 同时反映偏差和随机散布。

DOP 使用设计矩阵每行 `Aᵢ=[cos(el)sin(az), cos(el)cos(az), sin(el), 1]`，由 `(AᵀA)⁻¹` 的位置协因数部分计算 PDOP、HDOP 和 VDOP。卫星不足四颗或矩阵不可逆时，该历元不参与计算。DOP 仅描述几何，不包含观测噪声、多路径和遮挡误差。

错误 FIX 率是一个基于外部真值和应用门限的经验指标：

```text
错误 FIX 率 = count(Q=FIX 且 3D error > limit) / count(Q=FIX)
```

它不是 RTKLIB 内部置信度，也不是只看 `Q=FIX` 或 AR ratio 就能推导出的概率。结果会受真值精度、时间同步、参考坐标和门限选择影响。

图 02 中的经验区间基于误差样本的均值和样本标准差；协方差包络基于解算器给出的估计协方差。只有误差分布近似正态且协方差校准良好时，`±1.96σ` 才可近似解释为 95% 区间。两者含义不同。

## 4. 输入文件

### `.pos`

使用 RTKLIB 标准位置结果文本格式。脚本跳过 `%` 开头的表头行，读取各历元的 GPS week、TOW、ECEF X/Y/Z、Q、卫星数、位置标准差/协方差字段、age 和 ratio。不要删除或重排列；坐标单位应为米，时间必须是 GPS 周和周内秒。

Q 值采用 RTKLIB 常见约定：`1=FIX`、`2=FLOAT`、`3=SBAS`、`4=DGPS`、`5=SINGLE`、`6=PPP`。具体输出模式可能不包含所有状态。

### `.pos.stat`

默认按 `<pos 文件名>.pos.stat` 自动查找；也可在 GUI 打开时一并读取同名文件。程序使用 `$SAT` 记录解析 PRN、频率编号、方位角、高度角、伪距/载波后验残差、`vsat`、SNR、模糊度状态和 `slipc`。如果没有 `.stat`，位置/误差图仍可用，但卫星、DOP、C/N₀、残差和周跳诊断会缺少数据。

### 真值 CSV

需要以下列名（大小写及空格应保持一致）：

```text
GPS TOW (s)
Latitude (deg)
Longitude (deg)
ECEF X (m)
ECEF Y (m)
ECEF Z (m)
```

ECEF 列必须使用米。匹配按 `GPS TOW (s)` 完成；当前数据加载逻辑假设 `.pos` 和 CSV 对应同一 GPS 周。跨周数据应先确保两者时间范围在同一周内，或者扩展输入处理后再分析。没有真值时程序不会计算 ENU 误差、误差 CDF 或错误 FIX 率。

## 5. 环境安装

### 5.1 通用准备

取得项目目录后，进入包含 `scripts/` 和 `requirements-rtklib-plot.txt` 的 `GNSS_RTKLIB_Experiments` 目录。推荐使用独立虚拟环境，避免和其他 Python 项目的依赖互相影响。

依赖为 NumPy、pandas 和 Matplotlib。`pandas>=2.2` 是因为脚本使用了 pandas 2.2 加入的分组接口参数。GUI 还需要可用的 Tkinter/Tk。Matplotlib TkAgg 依赖 Tk 绑定；Windows 和 macOS 的 Python 安装有时包含它，有些 Linux 或 Homebrew 安装则需要单独安装。可用 `python -m tkinter` 检查 Tk 是否可用。详见 [Python Tkinter 文档](https://docs.python.org/3/library/tkinter.html) 与 [Matplotlib 安装文档](https://matplotlib.org/stable/install/index.html)。

### 5.2 Windows 10/11

安装 Python 3.10+。使用 python.org 安装程序时保留 Tcl/Tk 支持；若系统已有 Python，建议使用 `py -3.11` 显式选择解释器。Python 安装器/管理器说明见 [Python for Windows](https://docs.python.org/3/using/windows.html)。

在 PowerShell 中运行：

```powershell
cd D:\path\to\GNSS_RTKLIB_Experiments
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-rtklib-plot.txt
python -m tkinter
```

如果 PowerShell 不允许激活脚本，可以不激活虚拟环境，直接调用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rtklib-plot.txt
.\.venv\Scripts\python.exe -m tkinter
```

### 5.3 macOS

可使用 python.org 的 Python 安装包，或 Homebrew Python。Homebrew Python 某些版本将 Tkinter 分为单独组件；安装与 Python 主次版本匹配的 `python-tk`，例如 Python 3.13 对应 `python-tk@3.13`（参见 [Homebrew Formula](https://formulae.brew.sh/formula/python-tk%403.13)）。

```bash
cd /path/to/GNSS_RTKLIB_Experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rtklib-plot.txt
python -m tkinter
```

如果 `python -m tkinter` 报错，先安装与你的 Python 版本匹配的 Tcl/Tk 绑定，再重新创建虚拟环境。

### 5.4 Linux（Ubuntu/Debian 示例）

安装 Python、虚拟环境和 Tk 组件：

```bash
sudo apt update
sudo apt install python3 python3-venv python3-tk
```

进入项目目录并安装 Python 依赖：

```bash
cd /path/to/GNSS_RTKLIB_Experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rtklib-plot.txt
python -m tkinter
```

其他发行版使用其对应的 Python Tk 包管理包名。交互 GUI 需要桌面会话/X11/Wayland 显示；无显示服务器的 SSH 或服务器环境请使用批量出图脚本。

## 6. 启动与使用

以下命令均假设当前目录是 `GNSS_RTKLIB_Experiments`，且虚拟环境已激活。

### 启动交互界面

```bash
python scripts/rtklib_plot_gui.py
```

指定解算文件和真值文件，并选择启动时显示的图：

```bash
python scripts/rtklib_plot_gui.py \
  --pos "/path/to/session.pos" \
  --reference "/path/to/reference.csv" \
  --plot "False Fix Analysis"
```

Windows PowerShell 示例：

```powershell
python .\scripts\rtklib_plot_gui.py `
  --pos "D:\data\session.pos" `
  --reference "D:\data\reference.csv" `
  --plot "Gnd Trk"
```

`--plot` 可选值包括 `Gnd Trk`、`ENU Position`、`Height`、`ENU Error`、`Solution Status`、`AR Ratio`、`Satellites`、`DOP`、`Skyplot`、`C/N0`、`Elevation`、`Residuals`、`Multipath Proxy`、`Cycle Slips`、`Covariance`、`False Fix Analysis`、`Innovation / NIS` 和 `Dashboard`。不指定时默认打开 `Gnd Trk`。

### 界面操作

| 操作 | 效果 |
| --- | --- |
| `Open…` | 打开 `.pos`，自动查找同名 `.pos.stat` 和附近的 `reference.csv`；也可手动选真值文件。 |
| `Plot` 下拉框 | 切换单图，`Windows → Dashboard` 打开总览。 |
| `Solution` 下拉框 | 仅显示全部解，或筛选 FIX/FLOAT 等 Q 状态。 |
| 左侧图表列表 + `Show selected plots` | 多选图表组合成自定义网格。每个子图随当前历元显示对应位置/观测高亮。 |
| 底部历元滑块 | 移动当前历元；显示当前日期时间、Q 状态及真值对比。 |
| `▶ Play`、`Speed` | 连续播放历元，可选 1x、10x、50x。天空图同步更新当前卫星。 |
| 在时间序列上移动鼠标 | 跳至横坐标最近历元；显示同步竖直游标。 |
| 在 Ground Track 轨迹线上按住左键拖动 | 移动当前历元小车标记。点击轨迹附近开始拖动，松开结束。 |
| 左键拖动图表其他区域 | 平移视图。 |
| 右键拖动 | 框选放大。 |
| 鼠标滚轮 | 缩放横轴、纵轴或双轴；普通图纵轴缩放以 0 为中心。 |
| `Fit` | 将视图恢复到数据范围；`Equal scale` 调整轨迹轴比例。 |
| `Export Image…` / `Ctrl+S` | 导出当前单图、Dashboard 或自定义网格为 PNG、PDF、SVG。 |

### 标记时间段

1. 用历元滑块、播放或轨迹拖动，将小车移动到分段起点。
2. 点击 `Set segment start at current epoch`。
3. 移动到分段终点，点击 `Set end & add segment`。
4. 分段出现在左侧 `Marked time segments` 列表。可选择并删除单段，或点击 `Clear segments` 清除全部。

各分段以不同颜色显示：时间序列图上为背景色带，Ground Track 上为加粗轨迹并标出起点/终点，Skyplot 上标出时间段两端卫星位置；Dashboard 的高度角/C/N₀散点也会叠加该时间段的观测。分段会在当前 `.pos` 文件的不同视图间保留；打开另一份 `.pos` 后会清空。

## 7. 批量出图与导出

生成全部 15 张诊断图和 Dashboard：

```bash
python scripts/rtklib_diagnostics.py
```

指定输入、输出目录及格式：

```bash
python scripts/rtklib_diagnostics.py \
  --pos "/path/to/session.pos" \
  --stat "/path/to/session.pos.stat" \
  --reference "/path/to/reference.csv" \
  --output "/path/to/diagnostics" \
  --plots all --format png,pdf --dpi 300
```

只运行指定图号，或禁用真值：

```bash
python scripts/rtklib_diagnostics.py --pos "/path/to/session.pos" --plots 1,3,6,14
python scripts/rtklib_diagnostics.py --pos "/path/to/session.pos" --reference none
```

| 参数 | 默认/说明 |
| --- | --- |
| `--pos` | 默认项目的示例 `.pos` 文件。 |
| `--stat` | 可选 `.pos.stat`；默认按 `<pos>.stat` 自动查找。 |
| `--reference` | 真值 CSV；`none` 表示不使用真值。 |
| `--output` | 输出目录，默认 `plots_15_diagnostics/`。 |
| `--plots` | `all` 或逗号分隔图号，如 `1,2,3`。Dashboard 仍会生成。 |
| `--format` | `png`、`pdf`、`svg`，可以组合，如 `png,pdf`。 |
| `--dpi` | PNG 分辨率，默认 180。 |

## 8. 常见问题

### `No module named tkinter` 或 GUI 无法启动

运行 `python -m tkinter`。若没有弹出 Tk 测试窗口，请安装与你运行脚本的 Python 解释器匹配的 Tk 组件。Linux 通常为 `python3-tk`；Homebrew Python 通常需要对应版本的 `python-tk@版本`。确认测试窗口正常后，再启动 GUI。

### `No module named numpy/pandas/matplotlib`

先激活本项目的 `.venv`，然后用同一个解释器运行：

```bash
python -m pip install -r requirements-rtklib-plot.txt
```

使用 `python -m pip` 可避免把依赖装到另一个 Python 环境。

### 图表提示没有真值或错误图不可用

检查 CSV 是否包含本手册列出的字段、列名是否完全一致、ECEF 单位是否为米，以及 GPS TOW 是否覆盖 `.pos` 数据时段。没有真值时，轨迹和状态图仍可查看，但误差图、误差 CDF、协方差误差覆盖和错误 FIX 率无法计算。

### 卫星图、C/N₀、DOP 或残差图为空

检查 `.pos.stat` 是否与 `.pos` 对应、文件是否包含 `$SAT` 记录，以及运行时加载的是否是预期 `.stat` 路径。DOP 还要求某历元至少有四颗可用 L1 卫星且几何矩阵可逆。

### Linux 服务器没有图形桌面

Tk 界面无法在没有显示服务器的纯 headless 会话中打开。可运行 `rtklib_diagnostics.py` 批量保存 PNG/PDF/SVG，或在具有桌面显示的本机/远程桌面中运行 GUI。

### 日期时间或坐标看起来不匹配

时间轴按 GPS 周/TOW 转为中国标准时间（UTC+08:00），真值插值用 GPS TOW。检查 `.pos` 与 CSV 的 GPS 时间基准、周次和覆盖范围。文件跨 GPS 周或真值来自不同日期时，不要只依赖 TOW 的数值相近来匹配。

## 9. 已知限制

- NIS 不能从当前 `.pos/.stat` 反推；若研究需要 NIS，需修改/配置 RTKLIB 滤波器输出滤波前创新 `v` 和创新协方差 `S`，再用相同观测重新解算。
- `.stat` 中 `resp` 是后验伪距残差，不等价于独立的多路径组合；相关图仅用于诊断线索。
- DOP 是本工具根据卫星方向几何重新计算的值，其筛星条件和 RTKLIB 内部定位实现可能不同。
- 名义 95% 协方差包络依赖误差分布与协方差模型假设；应同时看经验误差统计。
- 错误 FIX 率强依赖真值质量、时间对齐和用户设置的 3D 误差门限；不可直接解释为所有场景通用的错误概率。
- 分段标记只保存在当前程序会话中，不会自动写入 `.pos` 或单独的项目配置文件。

## 10. 参考资料

- [Python Tkinter 文档](https://docs.python.org/3/library/tkinter.html)
- [Matplotlib 安装指南](https://matplotlib.org/stable/install/index.html)
- [Matplotlib 后端说明](https://matplotlib.org/stable/users/explain/figure/backends.html)
- [pandas 安装与入门](https://pandas.pydata.org/docs/getting_started/install.html)
- [Homebrew Python Tk 组件示例（Python 3.13）](https://formulae.brew.sh/formula/python-tk%403.13)

## 11. 界面效果图

以下截图来自 RTKLIB Plot - Python 的实际界面。图中包含真实轨迹数据和交互控件，用于说明各图的显示方式；不同数据集的曲线范围、统计值和卫星数量会有所不同。

### 11.1 轨迹与位置

#### 地面轨迹（Gnd Trk）

显示 ENU 平面轨迹、真值轨迹、FIX/FLOAT 状态颜色、当前历元小车标记和已标记时间段。误差统计依赖已加载且时间匹配的真值。

<p align="center"><img src="docs/rtklib_plot_screenshots/01_ground_track.png" alt="Ground Track 地面轨迹图" width="900"></p>

#### ENU 位置分量

将东、北、天三个位置分量分成子图，便于观察运动趋势。背景色带对应左侧标记的时间段，当前历元线会随 Epoch 滑块同步移动。

<p align="center"><img src="docs/rtklib_plot_screenshots/02_enu_position.png" alt="ENU Position 位置分量图" width="760"></p>

#### 高度 / Up 分量

比较 RTK 高程（Up）和真值高程。保留单独的 Height 图，便于检查垂向变化和异常跳变。

<p align="center"><img src="docs/rtklib_plot_screenshots/03_height.png" alt="Height 高度图" width="700"></p>

### 11.2 误差与解算状态

#### ENU 位置误差

分别显示 ΔE、ΔN、ΔU。零线用于区分误差方向，图下方汇总偏差、RMSE、绝对误差分位数等统计量。

<p align="center"><img src="docs/rtklib_plot_screenshots/04_enu_error.png" alt="ENU Error 位置误差分量图" width="700"></p>

#### 解算状态（Solution Status）

显示每个历元的 RTK 解算状态。可据此查看 FIX 与 FLOAT 的切换、状态持续时间，并结合其他图中的同色时间段定位异常。

<p align="center"><img src="docs/rtklib_plot_screenshots/05_solution_status.png" alt="Solution Status 解算状态图" width="700"></p>

#### 模糊度固定比值（AR Ratio）

显示 AR ratio 时间序列及参考阈值。阈值仅是模糊度检验指标，不代表该历元的位置误差一定较小；应与真值误差一起判断。

<p align="center"><img src="docs/rtklib_plot_screenshots/06_ar_ratio.png" alt="AR Ratio 模糊度固定比值图" width="700"></p>

### 11.3 卫星与观测诊断

#### 卫星数量（Satellites）

比较跟踪到的卫星数和参与解算的卫星数。图下方可查看有效卫星记录及系统组成统计。

<p align="center"><img src="docs/rtklib_plot_screenshots/07_satellites.png" alt="Satellites 卫星数量图" width="700"></p>

#### DOP 几何精度因子

显示 PDOP、HDOP、VDOP 随时间的变化。DOP 描述卫星几何对测量误差的放大程度，不包含观测噪声和多路径误差。

<p align="center"><img src="docs/rtklib_plot_screenshots/08_dop.png" alt="DOP 几何精度因子图" width="700"></p>

#### 卫星天空图（Skyplot）

以方位角和高度角展示卫星在天空中的分布及其随时间变化的轨迹；卫星颜色区分星座。拖动 Epoch 滑块时，卫星分布会随历元更新。

<p align="center"><img src="docs/rtklib_plot_screenshots/09_skyplot.png" alt="Skyplot 卫星天空图" width="700"></p>

#### 载噪比（C/N₀）

显示不同信号频率的 C/N₀ 观测散点，可用于检查信号强弱变化及遮挡时段。C/N₀ 是信号质量线索，不能单独证明误差由多路径导致。

<p align="center"><img src="docs/rtklib_plot_screenshots/10_cn0.png" alt="C/N0 载噪比图" width="700"></p>

#### 卫星高度角（Elevation）

显示各卫星高度角随时间的变化，可与 C/N₀ 和位置误差联动查看低高度角卫星的观测表现。

<p align="center"><img src="docs/rtklib_plot_screenshots/11_elevation.png" alt="Elevation 卫星高度角图" width="700"></p>

#### 后验残差（Residuals）

显示 `.stat` 中的伪距残差和载波残差。后验残差可用于解算后诊断，但不等价于滤波器更新前的创新量。

<p align="center"><img src="docs/rtklib_plot_screenshots/12_residuals.png" alt="Residuals 后验残差图" width="700"></p>

#### 多路径诊断代理量（Multipath Proxy）

根据可用的后验伪距残差展示诊断线索。它不是标准双频多路径组合，不能作为独立 MP 测量值解释。

<p align="center"><img src="docs/rtklib_plot_screenshots/13_multipath_proxy.png" alt="Multipath Proxy 多路径诊断代理图" width="700"></p>

#### 周跳标记（Cycle Slips）

将 `.stat` 中标记的卫星周跳记录按 PRN 和时间显示，适合检查周跳集中出现的卫星与时段。

<p align="center"><img src="docs/rtklib_plot_screenshots/14_cycle_slips.png" alt="Cycle Slips 周跳图" width="700"></p>

### 11.4 协方差、错误 FIX 与综合视图

#### 位置不确定度（Covariance）

显示 `.pos` 协方差推导的 E/N/U 方向 1σ 标准差，并与真值误差的名义 95% 包络覆盖情况对照。协方差反映解算器估计的不确定度，不是真实误差本身。

<p align="center"><img src="docs/rtklib_plot_screenshots/15_covariance.png" alt="Covariance 位置不确定度图" width="700"></p>

#### 基于真值误差的错误 FIX 检查（False Fix Analysis）

将历元 3D 位置误差按 FIX/FLOAT 状态着色，FIX 用绿色点、FLOAT 用黄色点，超过用户设置的 3D 误差限值时以红色叉号标出。图下方统计的是在当前真值和门限定义下的经验错误 FIX 比例，不是 RTKLIB 内部完整性告警。

<p align="center"><img src="docs/rtklib_plot_screenshots/16_false_fix.png" alt="False Fix Analysis 错误 FIX 检查图" width="900"></p>

#### Dashboard 总览

在同一视图中汇总轨迹、位置误差、状态、AR ratio、卫星数量和观测质量等关键信息。拖动 Epoch 时各子图同步显示当前历元。

<p align="center"><img src="docs/rtklib_plot_screenshots/17_dashboard.png" alt="Dashboard RTKLIB 解算总览" width="900"></p>

#### 自定义组合分析

在左侧选择任意图表组合并生成分析网格。组合图共享当前历元和分段标记，便于把某个轨迹区间与误差、状态及卫星诊断指标放在一起比较。

<p align="center"><img src="docs/rtklib_plot_screenshots/18_custom_analysis.png" alt="Custom Analysis 自定义组合分析" width="900"></p>
