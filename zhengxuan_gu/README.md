# zhengxuan_gu 的实验内容

本目录收录本人负责的有限差分与 PINN 实验内容。目前已提交方法三；方法四和更换初值的对比实验将在后续加入本目录。

## 方法三：有限差分

比较显式 Euler、隐式 Euler 和 Crank–Nicolson，考察网格加密对精度与耗时的影响，并通过固定空间网格的时间步对照分析误差。

- [实现、复现与工具使用说明](method_3_finite_difference/README.md)
- [Word 实验说明](method_3_finite_difference/report/方法三_有限差分网格实验.docx)
- [网格精度与耗时图](method_3_finite_difference/results/figures/grid_comparison.png)
- [原始结果与汇总](method_3_finite_difference/results/)

运行前进入 `method_3_finite_difference/`，按该目录的 README 安装依赖并执行测试与实验。物理参数、初边值条件和解析基准沿用团队的[共同实验方案](../guisheng-dai/numeric-experiment-plan.md)。
