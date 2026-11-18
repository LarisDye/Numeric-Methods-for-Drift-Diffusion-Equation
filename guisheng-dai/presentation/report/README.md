# 当前演示稿与编译入口

当前人工维护的演示稿是本目录的 `slides.tex`，即项目根目录下的
`guisheng-dai/presentation/report/slides.tex`。`guisheng-dai/report/slides.tex`
是早期自动生成稿，两者不是同步副本。

在项目根目录运行：

```powershell
python .\guisheng-dai\presentation\report\build_slides.py
```

该脚本只用 XeLaTeX 编译本目录现有源码两遍，不生成或覆盖 TeX，也不需要修改 PowerShell 执行策略。
中间文件存放在 `.build-slides/`，与编辑器自动编译隔离；成功后才更新 `slides.pdf`。
也可在本目录运行 `latexmk slides.tex`。不要用 `04_build_documents.py`
重新生成此目录；生成器已对此路径加上保护。

Positron 中查看源码时请核对完整路径。若标签里的内容与磁盘不同，先另存未保存的修改，
再关闭并重新打开该文件；不要直接用旧缓冲区覆盖磁盘文件。

## 演示文件

- `report.pdf`：7 页实测报告。
- `slides.pdf`：由当前 `slides.tex` 编译的演示稿，使用 CleanEasy Beamer 主题。
- `slides.pptx`：当前 `slides.pdf` 的 25 页 PowerPoint 版本，第 11 页预留区已嵌入反向随机游走动画。
- `figures/`：报告使用的科学图形，PDF 向量版与 PNG 预览。
- `report.tex`、`slides.tex`：修改文稿的入口；`template/` 为原始主题与许可。
- `summary.json`、`chart_data.json`、`slide_content.json`：本轮文稿的数据和内容快照。

需要当前演示稿的固定版式时使用本目录 `slides.pdf`。

## PowerPoint 与动画

`slides.pptx` 保留 PDF 的版式，静态内容使用 400 dpi 页面图，正文与公式不能在
PowerPoint 中逐字编辑；文字修改仍以 `slides.tex` 为准。动画是独立、内嵌的 GIF，
无需外部文件，进入第 11 页放映时自动循环（41 帧，约 14 秒）。
在 PowerPoint 中选中第 11 页，按 `Shift+F5` 可直接播放；按 `Esc` 退出。
已在本机 PowerPoint 验证正常打开与动画帧变化。

修改 TeX 并编译 PDF 后，在项目根目录运行以下命令更新 PPTX：

```powershell
python .\guisheng-dai\presentation\scripts\build_pptx_from_pdf.py
```

导出脚本读取当前 PDF，不改动 TeX 或 PDF，不启动实时编译。它自动定位动画预留区，
覆盖 PPTX 前备份旧版本至 `backups/`，并将验证记录保存在 `.build-pptx/pptx-validation.json`。
若需重新生成动画，先运行：

```powershell
python .\guisheng-dai\presentation\scripts\export_backward_walk_gif.py
```

动画复用原有反向游走模拟代码与固定种子，显示回溯、触边吸收与存活路径。

## Gaussian 卷积精度与开销图

演示稿的核宽精度数据复用 `results/openmp-final-20261117/summary/kernel.csv`
中的 E3；不展示 E4 网格实验。核宽开销和整条曲线的并行性能另行实测。

在 `guisheng-dai/` 下运行以下命令，可建立新实验记录并重绘图片：

```powershell
Rscript --vanilla scripts/08_benchmark_gaussian_cost.R --run-id gaussian-cost-NEW
python scripts/09_plot_gaussian_cost.py gaussian-cost-NEW
python presentation/report/build_slides.py
```

- 核宽实验：固定主问题、`N=512`、两线程，改变 `c=1,2,3,4,6,8`，求解全部 513 个节点。
- 并行实验：固定 `c=6`，分别求解 `N=512` 和 `N=8192` 的整条曲线，比较 1、2、4、8 线程。
- 计时先预热，再进行 9 轮批量测量；保存原始记录、中位数和四分位区间。
  输入准备、已准备输入的求解、完整的准备加求解分别独立计时。
  编译、预热、正确性核验和写盘不计入；批次内发生的内存分配与垃圾回收属于实际调用成本。
- 加速比以同一 `solve_gaussian` 程序的单线程求解中位耗时为基线。
  并行实验只计求解调用；输入准备不计入该加速比。BLAS 保持单线程。
- 绘图脚本从 CSV 生成 `figures/gaussian_kernel_tradeoff.pdf` 和
  `figures/gaussian_curve_parallel.pdf`，同时输出 PNG 预览；不要直接修改图片。

完整实验代码、种子、原始 CSV 与运行说明见 [项目说明](../../README.md)、
[原报告实验记录](../../results/openmp-final-20261117/) 和
[本次卷积计时说明](../../results/gaussian-cost-20261118/README.txt)。
本版 C++ 核心使用 OpenMP，R 层串行组织实验；本报告不包含姓名、学号、分工或正式提交信息。
