# Numeric Methods for the Drift-Diffusion Equation

本项目比较 [idea.md](idea.md) 中的四种漂移扩散方程求解方法。项目仓库：[LarisDye/Numeric-Methods-for-Drift-Diffusion-Equation](https://github.com/LarisDye/Numeric-Methods-for-Drift-Diffusion-Equation)。

## 分工

- **Guisheng Dai**：负责前两种解法——反向吸收随机游走（Monte Carlo）和 Gaussian 宽核直接卷积，包括 R 参考实现、Rcpp/RcppArmadillo 核心、OpenMP 并行、边界处理、正确性验证、精度与性能实验，以及对应报告和演示稿。全部工作独立放在 [`guisheng-dai/`](guisheng-dai/)。
- **zhengxuan_gu**：负责后两种解法——有限差分法和物理信息神经网络（PINN），提出三种差分格式的网格精度与耗时对比、PINN 训练改进对比及更换初值的实验安排。个人成果统一放在 [`zhengxuan_gu/`](zhengxuan_gu/)，目前包含有限差分部分；PINN 与后续对比另行提交。

`idea.md` 是共同的项目设想。两部分工作应统一物理问题、参数、初边值、查询位置和误差定义，合作者可在自己的独立目录中继续开发。

## Guisheng Dai 的交付

- [实现与复现说明](guisheng-dai/README.md)
- [实验方案](guisheng-dai/numeric-experiment-plan.md)
- [实验报告 PDF](guisheng-dai/report/report.pdf)
- [可编辑演示稿](guisheng-dai/report/slides.pptx) · [演示稿 PDF](guisheng-dai/report/slides.pdf)
- [最终 OpenMP 实验记录](guisheng-dai/results/openmp-final-20261117/) · [测试日志](guisheng-dai/results/tests/)

运行前先进入该部分的工作目录：

```powershell
Set-Location guisheng-dai
```

随后按照该目录中的 README 安装依赖、编译和运行。R 层串行组织实验；C++ 求解器使用 OpenMP，正式实验默认 2 线程，性能比较 1/2/4 线程。若为 R 算法增加并行，则采用 R 多进程，不叠加嵌套并行。

## zhengxuan_gu 的交付：方法三

使用 R 调度单线程 Rcpp 求解器，比较显式 Euler、隐式 Euler 和 Crank–Nicolson。保存网格加密、重复计时、空间与时间误差分解，以及隐式格式的大时间步精度检查。

- [实现、复现与工具使用说明](zhengxuan_gu/method_3_finite_difference/README.md)
- [Word 实验说明](zhengxuan_gu/method_3_finite_difference/report/方法三_有限差分网格实验.docx)
- [网格精度与耗时图](zhengxuan_gu/method_3_finite_difference/results/figures/grid_comparison.png) · [原始结果与汇总](zhengxuan_gu/method_3_finite_difference/results/)
- [求解器测试](zhengxuan_gu/method_3_finite_difference/tests/test_solver.R)

运行前先进入 `zhengxuan_gu/method_3_finite_difference/`，按该目录的 README 安装依赖并执行测试与实验。

## 本次整理

将 Guisheng Dai 的代码、文档、报告及原始数据封装到独立目录，保留随机状态与复现入口，并将旧批次放入 `guisheng-dai/results/archive/`。更新内容与验证记录见 [CHANGELOG.md](CHANGELOG.md)。缓存、临时文件及编译中间产物不纳入 Git。
