# zhengxuan_gu 的实验内容

本目录收录本人负责的有限差分、PINN 及更换初始函数的对比实验。实验顺序为三种差分格式的网格比较、PINN 基线与改进训练比较，最后在四种初值下进行跨方法对照。

三部分的实验过程、结果与审阅补充汇总在[合并审阅版 Word](report/漂移扩散方程实验报告（有限差分与PINN合并审阅版）.docx)。讨论阶段的[实验讲义](source/漂移扩散方程实验讲义.md)提供物理参数、初边值条件及局部峰形式，各模块 README 记录实际采用的实验方案。

## 方法三：有限差分

比较显式 Euler、隐式 Euler 和 Crank–Nicolson，考察网格加密对精度与耗时的影响，并通过固定空间网格的时间步对照分析误差。

- [实现、复现与工具使用说明](method_3_finite_difference/README.md)
- [Word 实验说明](method_3_finite_difference/report/方法三_有限差分网格实验.docx)
- [网格精度与耗时图](method_3_finite_difference/results/figures/grid_comparison.png)
- [原始结果与汇总](method_3_finite_difference/results/)

运行前进入 `method_3_finite_difference/`，按该目录的 README 安装依赖并执行测试与实验。物理参数、初边值条件和解析基准沿用团队的[共同实验方案](../guisheng-dai/numeric-experiment-plan.md)。

## 方法四：PINN

先观察原始 Adam 训练的误差和初边值约束，再比较继续 Adam 与 Adam 后接 L-BFGS。软约束和硬约束均使用三个初始化种子，并另做边界损失权重实验。

- [实现、对照设计与复现说明](method_4_pinn/README.md)
- [Word 实验说明](method_4_pinn/report/方法四_PINN训练与精度实验.docx)
- [优化器比较图](method_4_pinn/results/figures/optimizer_comparison.png)
- [原始结果与训练模型](method_4_pinn/results/)

## 更换初始函数后的跨方法比较

保留原光滑基准，新增中央窄峰、近左边界窄峰和中央有限光滑峰。每个初值使用相同的评价位置和时刻，比较三种差分格式与硬约束 PINN，并保存独立参考解、误差、计时和内部负值检查。

- [实验设计、代码复用与复现说明](initial_function_comparison/README.md)
- [Word 实验说明](initial_function_comparison/report/更换初始函数后的漂移扩散求解比较.docx)
- [终态精度与成本表](initial_function_comparison/results/terminal_comparison.csv)
- [原始结果与训练模型](initial_function_comparison/results/)

## 依赖与运行顺序

实验使用 R 4.4.2、Rcpp 1.1.1 和支持 C++11 的编译器。Python 环境为 Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1、pandas 2.2.2、PyTorch 2.10.0、Matplotlib 3.9.2 和 pytest 7.4.4。Rcpp 可在 R 中用 `install.packages("Rcpp")` 安装。Python 依赖可用 `python -m pip install numpy scipy pandas torch matplotlib pytest` 安装，版本和计时环境另见各模块的运行记录。

在本目录先运行现有测试和一个小规模训练示例。

```sh
Rscript --vanilla method_3_finite_difference/tests/test_solver.R
python -m pytest -q method_4_pinn/tests initial_function_comparison/tests
python method_4_pinn/scripts/run_experiments.py --seeds 7 --adam-steps 5 --lbfgs-iterations 3 --output /tmp/pinn-smoke
```

复现完整结果时，在本目录依次执行以下命令。脚本会更新各模块 `results/` 中的实验记录和图表，Word 保存本轮归档结果。复现报告字体需要宋体和 Times New Roman。

```sh
Rscript --vanilla method_3_finite_difference/scripts/run_experiments.R
Rscript --vanilla method_3_finite_difference/scripts/check_large_steps.R
python method_3_finite_difference/scripts/plot_results.py
python method_4_pinn/scripts/run_experiments.py
python method_4_pinn/scripts/plot_results.py
python initial_function_comparison/scripts/make_reference.py
python initial_function_comparison/scripts/run_pinn.py
Rscript --vanilla initial_function_comparison/scripts/run_fd.R
python initial_function_comparison/scripts/compare.py
python initial_function_comparison/scripts/plot_results.py
```

方法四完整实验在本机约 65 秒，更换初值的 12 次 PINN 训练、评价与保存共约 82 秒，全部流程通常需几分钟，具体随机器和负载变化。更换初值的程序通过相对位置调用同级的方法三和方法四目录。

## 工作分工与工具使用

本人提出方法内比较、跨方法比较和更换初值的实验顺序，讨论网格粗细、训练改进、函数光滑度与算法表现的关系。实现过程中使用 Codex 辅助代码生成与修改、调试测试、结果分析及图表文稿整理。具体使用和复用内容见各模块 README。

验证包括解析解与独立矩阵对照、自动微分和约束检查、参考解的两套独立计算、训练模型重载，以及从保存预测重新计算误差。原始预测、训练设置、随机种子和计时记录随代码保存。
