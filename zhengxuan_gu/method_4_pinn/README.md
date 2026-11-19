# 方法四　PINN 的初边值约束与训练精度

求解一维漂移扩散方程 `u_t = 0.2 u_x + 0.05 u_xx`，区间为 [0, 1]，终止时间 0.5，两端固定为 0，初值为 `exp(-2x) sin(πx)`。问题来自[团队讲义](../source/漂移扩散方程实验讲义.md)。[Word 实验报告](report/方法四_PINN训练与精度实验.docx)包含原始结果、误差诊断、改进方案和实验图表。

## 如何运行

在本目录使用已安装 NumPy、PyTorch、Matplotlib 和 pytest 的 Python 环境。完整实验在本机运行约 65 秒，包含主实验和权重消融。本机实验用 Python 3.12、PyTorch 2.10，精确版本和硬件见[运行记录](results/experiment.json)。

新建 Python 环境后，可运行 `python -m pip install numpy torch matplotlib pytest` 安装依赖。

```bash
python -m pytest -q tests
python scripts/run_experiments.py
python scripts/plot_results.py
```

完整入口先运行 18 个主实验配置，再测试软约束的三个边界权重。数据默认更新到 `results/`。也可将一次新训练放到独立目录，避免混合不同设置下的图表和数据。下面的小预算命令用于检查流程，三种初始化的正式统计与绘图应采用默认配置。

```bash
python scripts/run_experiments.py --seeds 7 --adam-steps 5 --lbfgs-iterations 3 --output /tmp/pinn-smoke
```

可直接加载已保存的种子 7 硬约束精修模型，查询任意域内坐标。这里的输入依次为物理位置 x 和物理时间 t。

```python
import torch
from src.pinn import load_checkpoint

model = load_checkpoint("results/hard_seed7.pt")
coords = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
with torch.no_grad():
    print(model(coords).item())
```

## 代码与对照设计

- [求解核心](src/pinn.py)定义初值、网络、配点、自动微分、损失、优化与独立评估。网络使用两个宽度为 32 的 tanh 隐藏层，以双精度在单线程 CPU 上计算。自动微分作用于物理坐标和完整的预测函数，包含时间归一化的链式法则。
- [实验入口](scripts/run_experiments.py)固定 1024 个内部配点、128 个初值点、每端 64 个边界时间点。训练点种子为 2026，初始化种子为 7、17、27，固定物理问题和训练点。基线均用 Adam、学习率 0.001，训练 2000 步。
- 软约束将 PDE 残差、初值误差、边界误差三项均方损失相加，基础权重均为 1。硬约束输出为 `f(x) + (t/T) 4x(1-x) F(x,t/T)`，以公式固定初值和边界，训练时优化 PDE 残差。
- 同一基线完成后，复制网络权重做 L-BFGS。该分支采用全批量强 Wolfe 线搜索，最多进行 400 次迭代。原网络保留 Adam 的优化器状态和累计步数，继续训练与 L-BFGS 实际额外全批量梯度次数相同的步数。三种方案均在各自预算终点评分，解析解只用于训练后核验。
- 在软约束上另将边界权重取 0.1、1、10，各用三个初始化；初值权重固定为 1，全部观察第 2000 步。权重 1 直接使用同次主实验基线。

## 指标和计时

终态 RMSE 与最大绝对误差使用方法三 64 段网格上的 63 个内部点。时空 RMSE 使用 256 个空间区间、40 个时间区间，实际统计 255 个内部空间点和 40 个正时间点。用 257 个独立随机位置检查初值，用每端的 201 个均匀时刻加 257 个随机时刻检查边界，用独立抽取的 512 个内部配点测 PDE 残差均方根。独立抽样种子 102026。

`train_seconds` 是两阶段累计优化耗时，含损失计算、反向传播、更新和损失标量记录。两条分支共用对应的基线时间。准备时间包含模型、配点、优化器的建立和一次不更新参数的导数预热，L-BFGS 分支另加其优化器建立时间。独立评价、检查点保存和实验中复制基线以建立对照的操作位于优化计时之外。一次查询向网络送入终态的 65 个点。先预热 3 次，再取 7 次前向计算时间的中位数。`total_seconds` 汇总准备、训练及一次查询。训练结果按三个初始化汇总，每个初始化的训练耗时是一次测量。

## 实验结果

三次初始化下，硬约束基线的终态 RMSE 中位数为 5.51×10⁻⁴。继续 Adam 的中位数为 3.93×10⁻⁴，换用 L-BFGS 后为 1.58×10⁻⁵。三个种子的精修结果分别为 1.34×10⁻⁵、4.27×10⁻⁵、1.58×10⁻⁵。优化器图保留逐个种子的配对结果。

[种子 7 的终态曲线与逐点误差](results/figures/baseline_terminal.png) · [三个初始化的优化器对照](results/figures/optimizer_comparison.png)

## 原始结果

| 文件 | 内容 |
|---|---|
| [metrics.csv](results/metrics.csv) | 约束、种子、训练方案、额外梯度次数、L-BFGS 迭代次数与终止原因、全部误差和计时。 |
| [boundary_weights.csv](results/boundary_weights.csv) | 单因素权重实验的 9 组误差。 |
| [terminal_seed7.csv](results/terminal_seed7.csv) | 种子 7、两种约束和三种训练方案在终态 65 个节点上的预测、真值与逐点绝对误差。 |
| [spacetime_predictions.npz](results/spacetime_predictions.npz) | 完整时空网格的 x、t、exact，以及 18 种预测。示例键名为 `hard_7_adam_then_lbfgs`，数组形状为 (41,257)。 |
| [loss_history.csv.gz](results/loss_history.csv.gz) | 各轮损失分项。Adam 给定已完成更新数。L-BFGS 区分线搜索试探 `lbfgs_trial` 和最后接受的模型 `lbfgs_final`。 |
| [query_timings.csv](results/query_timings.csv) | 每个模型 7 次终态查询的原始时间。 |
| [hard_seed7.pt](results/hard_seed7.pt) | 预先指定的种子 7 硬约束精修模型，含模型配置、网络权重及物理参数。 |
| [experiment.json](results/experiment.json) | 训练和评估设置、环境以及完整运行耗时。 |

## 工作分工与工具使用

本部分负责人明确了先展示原始 PINN 的误差、分析原因、再比较改进效果的实验顺序，并提出同时报告精度与耗时。经讨论采用 Adam 后接 L-BFGS 的改进方案，进一步设置继续 Adam 的对照，检查改善是否超出了单纯增加训练量的效果。

核心和训练流程参考了前一期同项目实现，沿用 [PyTorch](https://pytorch.org/) 的自动微分及优化器。本轮重新训练并生成全部数值结果。使用 Codex 辅助生成和修改核心及实验代码、细化对照、调试测试和整理分析文稿。结果通过解析残差、梯度核验、约束测试、检查点重载，以及从保存预测重新计算误差核对。

报告提及的优化器研究见 Rathore 等人发表于 ICML 2024 的 [Challenges in Training PINNs: A Loss Landscape Perspective](https://proceedings.mlr.press/v235/rathore24a.html)。
