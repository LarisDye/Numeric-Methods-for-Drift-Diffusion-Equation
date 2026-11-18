# 2026-11-18 补充实验归档

本次归档保留原始数据、运行配置、环境与编译信息、源码快照、哈希和绘图代码。
保存归档时未重新执行性能实验，也未改写原始计时。

## 反向随机游走

目录：[`walk-cost-20261118/`](walk-cost-20261118/README.md)。

- 固定线程数分别改变时间步数 n、空间网格 N，并保留联合细化的实验记录。
- 固定求解任务，比较真正关闭 OpenMP 的 C++ 串行版与 1、2、4、8 线程版本。
- 18 个配置、162 条批量计时、221 项通过的正确性检查；原始随机流也已保存。
- 入口：`scripts/06_benchmark_walk_cost.R`；绘图：`scripts/07_plot_walk_cost.py`。

## Gaussian 直接卷积

目录：[`gaussian-cost-20261118/`](gaussian-cost-20261118/README.txt)。

- 固定问题与两线程，改变覆盖倍数 c，分别记录输入准备、复用输入求解及完整调用耗时。
- 对 N=512、8192 的整条曲线比较 1、2、4、8 线程，按同一求解器的单线程耗时计算加速比。
- 14 个配置、378 条批量计时、581 项通过的正确性检查。
- `raw/accuracy_internal.csv` 保存与原报告一致的内部节点 RMSE；原全节点 RMSE 和原始计时同时保留。
- 入口：`scripts/08_benchmark_gaussian_cost.R`；误差口径补充：`scripts/08a_gaussian_internal_rmse.R`；绘图：`scripts/09_plot_gaussian_cost.py`。

E1、E2 与 E3 精度展示复用已归档的 `openmp-final-20261117/` 数据，未当作本次新计时实验。
E3 的绘图输入为 `openmp-final-20261117/summary/kernel.csv`，本次演示不增加 E4。

## 重绘与构建

从 `guisheng-dai/` 运行（依赖与完整实验设置见两个实验目录的说明）：

```powershell
python scripts/07_plot_walk_cost.py walk-cost-20261118
python scripts/09_plot_gaussian_cost.py gaussian-cost-20261118
python presentation/report/build_slides.py
python presentation/scripts/build_pptx_from_pdf.py
```

如需重新实测，应使用新的 run-id，保留本次记录。演示稿图表位于
`presentation/report/figures/`，当前 PDF/PPTX 与生成动画的脚本一并保存。
PowerPoint 第 11 页的动画已实际放映验证；PDF 与 TeX 仍保留静态预留区。

`archive-20261118.json` 记录本次保存的两个实验目录、实验与绘图脚本、
演示稿和图形的 SHA-256。Git 属性对这些路径禁用自动换行转换，以保留字节哈希。
编译缓存和本地临时备份保留在本机，不纳入提交。
