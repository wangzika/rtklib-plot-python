# RTKLIB Python 结果绘图与综合诊断

本工具读取 RTKLIB 的 `.pos` 解算结果，并在有对应文件时读取 `.pos.stat` 和真值 CSV。功能参考 RTKLIB Plot 的结果浏览思路，同时增加误差统计、卫星观测诊断、Dashboard 和可选的图片导出格式。

## 交互式桌面界面

```powershell
python rtklib_plot_gui.py
```

程序启动后可打开 `.pos`，自动尝试读取同名 `.pos.stat` 并查找 `reference.csv`；若未自动检测到 `.stat`，系统会弹窗提示是否手动选择，也可以随时通过菜单 `File -> Open Satellite Status (.stat)...` 独立载入。

### 核心功能与交互增强
- **子图点击放大与右键还原**：
  - 勾选导航栏的 **`[☑ Click to Maximize]`** 后，鼠标左键单击任一子图（支持 Dashboard 与 Custom Analysis）即可**将其全屏放大**；鼠标右键单击即**恢复多图网格**。
  - 取消勾选时，点击仅作为选中子图与普通平移缩放，彻底杜绝操作冲突。
- **子图删除与自适应排版 (Subplot Deletion & Repack)**：
  - 在 Dashboard 或 Custom Analysis 模式下，单击选中某个子图（其边框高亮为蓝色），点击导航栏 **`🗑️ Del Subplot`** 按钮（或直接按键盘 **`Delete`** 键），即可移除该子图；
  - 剩余子图会自动重新计算网格并紧凑自适应铺满画布；
  - 点击 **`↺ Reset`** 按钮可一键恢复标准完整布局。
- **鼠标悬停时间轴联动开关 (Hover Scrubs Timeline)**：
  - 导航栏及 `View` 菜单提供 **`[☑ Hover scrubs timeline]`** 开关；
  - 开启时移动鼠标即可平滑追踪历元与各视图纵向游标；关闭后可在子图上随意移动鼠标、平移观察，不会意外改变当前选中历元。
- **时间段多色彩标记与对比**：
  - 在轨迹线上按住左键拖动可移动当前历元的小车标记；
  - 点击“Set segment start at current epoch”设定分段起点，移动历元后点击“Set end & add segment”保存该时间段；左侧分段列表可删除单段或清空全部分段；
  - 已标记区间会以对应颜色显示在时间序列图中，并在轨迹图和天空图上同步高亮。
- **假固定智能排查 (False Fix Analysis)**：
  - 使用点图展示 3D 误差：FIX 为绿色、FLOAT 为黄色，超过用户设定的 3D 误差门限时叠加红色叉号；错误 FIX 率以超过门限的 FIX 历元数除以 FIX 总历元数计算。
- **视图操作与导出**：
  - 滚轮缩放支持选定横轴、纵轴或双轴（Y 轴滚轮缩放以 0 为锚点）；按住左键拖动平移视图，右键拖动框选缩放；
  - `Fit` 恢复全景范围，`Equal scale` 切换轨迹轴等比例显示；`Export Image…` (Ctrl+S) 支持一键将当前图表导出为出版级高分辨率 PNG、PDF 或 SVG。

## 快速开始

在 `D:\BaiduNetdiskDownload` 下运行：

```powershell
python .\GNSS_RTKLIB_Experiments\scripts\rtklib_diagnostics.py
```

默认读取项目中的 `6_RTK_Multi_L1L2.pos`、对应 `.pos.stat` 和 Odaiba `reference.csv`，把图保存到 `GNSS_RTKLIB_Experiments\plots_15_diagnostics`。

指定自己的解算文件并导出 PNG 和 PDF：

```powershell
python .\GNSS_RTKLIB_Experiments\scripts\rtklib_diagnostics.py `
  --pos "D:\data\session.pos" `
  --stat "D:\data\session.pos.stat" `
  --reference "D:\data\reference.csv" `
  --output "D:\data\diagnostics" `
  --format png,pdf --dpi 300
```

也支持 `--format svg`；`--plots 1,2,3` 可只生成指定图。没有真值时可用 `--reference none`，此时位置误差图会标注不可计算。`.stat` 默认按 `<pos 文件名>.stat` 查找。需要 `numpy`、`pandas` 和 `matplotlib`。

## 输出图

运行 `--plots all` 会生成 15 张诊断图及 `dashboard`：位置/高程、ENU 误差、水平与三维误差 CDF、解算状态、AR Ratio、跟踪与使用卫星数、DOP、天空图、C/N₀、高度角、伪距残差诊断、伪距与载波残差、周跳标志、协方差与误差覆盖、创新/NIS 数据可用性说明，以及综合 Dashboard。

主要定义：

- `ΔE, ΔN, ΔU` 为 RTK 位置减去同一 GPST 时刻插值的真值位置；水平和三维误差分别为 `√(ΔE²+ΔN²)` 与 `√(ΔE²+ΔN²+ΔU²)`。
- 经验 CDF 由排序误差样本的秩除以样本数计算。
- DOP 按 L1 `vsat=1` 的不同卫星方位/高度角建立视线几何矩阵，计算 `(AᵀA)⁻¹` 的位置子矩阵迹。它是由观测几何重算的 DOP，不是 RTKLIB 文件直接给出的 DOP 字段。
- RTKLIB `.stat` 的 `resp` 用于伪距后验残差分析。Plot 11 仅把它作为多路径诊断代理量，不能等同于独立的代码多路径组合。
- `.pos/.stat` 不含滤波创新向量 `v` 和创新协方差 `S`，因此无法可靠计算 `NIS=vᵀS⁻¹v`；Plot 15 会说明这一数据限制。


