# 临时实验报告

- `report.pdf`：7 页实测报告。
- `slides.pdf`：12 页演示稿，直接使用 ZIP 中的 CleanEasy Beamer 主题编译。
- `slides.pptx`：相同内容的可编辑 PowerPoint 对应稿，按主题风格重建，含 6 张原生图表和嵌入的 Excel 数据。
- `figures/`：报告使用的科学图形，PDF 向量版与 PNG 预览。
- `report.tex`、`slides.tex`：修改文稿的入口；`template/` 为原始主题与许可。
- `summary.json`、`chart_data.json`、`slide_content.json`：本轮文稿的数据和内容快照。

两份 PDF 已逐页核对。PPTX 的文本和图表可以编辑，已核对其包结构、几何和嵌入数据；本机 PowerPoint 自动化不可用，没有宣称在 PowerPoint 内打开验证。需要固定版式时使用 `slides.pdf`。

完整实验代码、种子、原始 CSV 与运行说明见 [本部分的 README](../README.md) 和 [最终实验记录](../results/openmp-final-20261117/)，项目分工见 [仓库根目录 README](../../README.md)。本版 C++ 核心使用 OpenMP，R 层串行组织实验；已更新全部计时与性能结论。本报告不包含姓名、学号、分工或正式提交信息。
