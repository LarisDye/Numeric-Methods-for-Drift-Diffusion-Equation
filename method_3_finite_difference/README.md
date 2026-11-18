# 方法三　有限差分网格实验

用 R 调度同一份单线程 Rcpp 求解器，比较显式 Euler、隐式 Euler、Crank–Nicolson 三种时间格式的终态误差与耗时。物理设置和解析基准见[共同实验方案](../guisheng-dai/numeric-experiment-plan.md)。图表解读见 [Word 实验报告](report/方法三_有限差分网格实验.docx)。

## 复现实验

在本目录执行以下命令。求解需要 R、Rcpp 和支持 C++11 的编译器。若缺少 Rcpp，在 R 中运行 `install.packages("Rcpp")`。绘图需要带 Matplotlib 的 Python，以及宋体和 Times New Roman 字体。

```sh
Rscript --vanilla tests/test_solver.R
Rscript --vanilla scripts/run_experiments.R
python scripts/plot_results.py
```

实验脚本默认更新 `results/` 中的同名数据文件，也可将新结果写入单独目录，例如 `Rscript --vanilla scripts/run_experiments.R /tmp/fd-repeat`。绘图脚本读取默认目录的汇总表，生成 [网格精度与耗时图](results/figures/grid_comparison.png)。

## 算例和实现

区间为 [0, 1]，两端固定为零。方程为 `u_t = a u_x + D u_xx`，取 `a = 0.2`、`D = 0.05`、`T = 0.5`。初值为 `f(x) = exp(-2x) sin(πx)`。实验脚本用 `f(x) exp[-(0.05π² + 0.2)T]` 独立评估终态。

- [求解核心](src/solver.cpp) 使用相同的中心空间差分。显式法在两个数组间交替推进。隐式 Euler 与 Crank–Nicolson 对固定的三对角矩阵作一次 Thomas 分解，随后各步复用。两端均始终为零。
- [R 接口](R/solver.R) 负责输入检查和调用。初始网格值由调用方传入。下面的例子使用 64 个小区间和 256 个时间步，返回包括两端在内的 65 个终态值。

```r
source("R/solver.R")
load_fd_solver(".")
x <- (0:64)/64
f <- exp(-2*x) * sin(pi*x)
f[c(1, length(f))] <- 0
u <- solve_fd("crank_nicolson", f, a = 0.2, D = 0.05, T = 0.5, n = 256L)
```

## 网格和计时设计

空间小区间数依次为 16、32、64、128，对应时间步数为 16、64、256、1024，各组保持 `D Δt / h² = 0.4`。这样三种格式使用相同的空间与时间预算。每次空间步长减半，时间步长变为原来的四分之一，观察阶衡量这条联合加密路径上的整体误差。

时间统计在 C++ 单调时钟内完成。每次求解重新复制初值、分配工作区、完成必要的三对角分解和全部时间步，并释放临时数组。R 调用与结果包装、输入检查、三个固定模板系数的计算、编译、初值函数采样、真值计算和文件读写均在计时之外。各配置先暖机，再用试测时间选择每批求解次数，目标约为 10 毫秒。9 轮测量中按固定随机种子交错排列各配置；每轮记录整批时间除以求解次数，以这 9 个单次均值的中位数和四分位数描述耗时。

## 数值记录

| 文件 | 用途 |
|---|---|
| [grid_summary.csv](results/grid_summary.csv) | 12 组主实验。RMSE 和最大绝对误差统计各自网格的全部内部点。另列相同 15 个位置的 `common15_rmse`、相邻网格的观察阶和毫秒计时。 |
| [terminal_profiles.csv](results/terminal_profiles.csv) | 各网格终态的坐标、预测、解析值和带符号误差，可重新计算误差或绘制曲线。 |
| [timings.csv](results/timings.csv) | 108 条原始批计时，含轮次、测量顺序、每批次数、整批毫秒数及平均每次毫秒数。 |
| [time_refinement.csv](results/time_refinement.csv) | 仅改变时间步数的对照。在固定 64 段的空间系统上用矩阵指数计算时间精确参照，逐点分解空间与时间误差。`space_rmse`、`time_rmse` 是两组带符号误差各自的均方根，`opposite_sign_nodes` 是二者符号相反的内部点数。 |
| [environment.txt](results/environment.txt) | 本机硬件、软件版本、精度和计时设定。 |

本轮联合加密时三种格式的终态 RMSE 均约缩小为上一级的四分之一。固定 64 段后，仅将步数由 256 增到 512，显式 Euler 的纯时间 RMSE 由 4.94×10⁻⁵ 降到 2.47×10⁻⁵，而相对连续真值的终态 RMSE 由 1.58×10⁻⁵ 增至 1.95×10⁻⁵。空间误差与时间误差在 63 个内部点中的 56 个方向相反，随着时间误差缩小，原来的抵消程度也改变。

## 隐式格式使用较大时间步的补充检查

运行 `Rscript --vanilla scripts/check_large_steps.R` 可重建[大步长精度对照表](results/large_step_check.csv)。[脚本](scripts/check_large_steps.R)固定原基准和 64 段网格，用 63 个内部点计算终态 RMSE 和最大绝对误差，并单列端点误差。隐式 Euler 与 CN 各用 4、8、16、32、64、256 步，显式 Euler 以 256 步作对照。

CN 在 4 步时的终态 RMSE 为 1.39×10⁻⁵，显式 Euler 在 256 步时为 1.58×10⁻⁵。同样取 4 步时，隐式 Euler 为 3.05×10⁻³。算法排名取决于步长和算例，本项只测精度。

## 工作分工与工具使用

本部分负责人提出比较三种有限差分格式，并通过改变网格粗细观察精度与耗时的实验安排。在讨论中进一步追问了显式 Euler 的误差表现，以及初值光滑度对结果的影响，由此形成时间误差分解与后续跨初值比较的研究问题。物理参数和边界条件依据团队共同讨论的讲义确定。

实现过程中使用 Codex 辅助生成和修改求解器、实验与测试代码，并协助调试、分析和图表文稿整理。求解正确性通过单独的[测试脚本](tests/test_solver.R)、解析解、独立矩阵对照、网格加密和固定网格的时间参照核验。本机测得的时间随处理器、编译器和当时负载变化。
