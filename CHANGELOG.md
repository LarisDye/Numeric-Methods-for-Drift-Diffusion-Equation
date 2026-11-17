# 更新记录

## 2026-11-17

- 在根目录 README 明确分工：Guisheng Dai 负责反向随机游走和 Gaussian 直接卷积；repository 合作者负责有限差分和 PINN。
- 将 Guisheng Dai 的实现、实验方案、复现脚本、报告与演示稿、种子和原始结果整理到 `guisheng-dai/`。
- C++ 随机游走和 RcppArmadillo 宽核卷积使用 OpenMP；R 层串行组织，避免嵌套并行。保留吸收边界、奇延拓、固定逻辑块 CMRG 和确定性归约。
- 收录最终 OpenMP 实验批次及其汇总，历史 PSOCK、预跑和早期 smoke 结果放入 `results/archive/`。保留 7 页实验报告、12 页演示稿 PDF 和含原生图表的 PPTX。
- 调整目录迁移后的文档链接和运行说明，排除缓存、临时构建与 LaTeX 中间文件。
- 验证：23 项串行 E0、7 项 OpenMP 集成核验及迁移后的串行 smoke 流程；检查最终结果和源码清单。原实验数据不因目录整理而重新计时。
- 一并提交工作区已有的 `idea.md` 拼写修正：`Dirft` → `Drift`。
