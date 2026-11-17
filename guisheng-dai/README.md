# Guisheng Dai：随机游走与 Gaussian 直接卷积

本目录由 **Guisheng Dai** 负责，对应共同 [idea.md](../idea.md) 中的前两种解法。按 [实验方案](numeric-experiment-plan.md) 实现反向吸收随机游走和 RcppArmadillo Gaussian 宽核直接卷积；有限差分与 PINN 由 repository 合作者负责，见 [根目录分工](../README.md)。R 参考算法与实验组织串行执行；C++ 核心使用 OpenMP 多线程。小规模验证串行，正式精度实验使用两个 OpenMP 线程，性能比较一、二、四线程。若以后为 R 参考算法增加并行，应使用 R 多进程，避免嵌套并行。现有 PDF 和演示稿仍是临时报告，不添加姓名、学号或正式提交信息。

交付文件位于 `report/`：

- `report.pdf`：7 页中文实测报告。
- `slides.pdf`：12 页 CleanEasy Beamer 模板演示稿，固定版式。
- `slides.pptx`：12 页可编辑 PowerPoint 对应稿，含 6 个原生图表及嵌入数据工作簿。
- `report.tex`、`slides.tex`：可修改的 LaTeX 源文件；`template/` 保留用户 ZIP 中的主题与许可。

## 本轮实际环境与结果

Windows 11，Intel Core i7-11800H（8 物理核 / 16 逻辑核），31.74 GiB RAM；R 4.6.1、Rcpp 1.1.2、RcppArmadillo 15.6.0-1（Armadillo 15.6.0）、Rtools45 / GCC 14.3.0。全部使用双精度，`_OPENMP=201511`；实际 OpenMP team 大小为 1/2/4。BLAS 单线程、Armadillo 内部隐式 OpenMP 关闭，仅显式外层循环并行。

23 项串行 E0 测试和 7 项 OpenMP 集成核验通过。正式 E1—E6 记录在 `results/openmp-final-20261117/`，包括 30 批独立随机重复、预热后的 7 次性能计时、配置、完整 L'Ecuyer-CMRG 随机状态、37,156 行逐点预测、539 条计时、16,512 行路径块统计。实验阶段耗时见该目录 `runtime_seconds.txt`，不含首次编译、图表和文稿编译；最终整理文件的时间也不计作算法性能。

性能结论与四线程加速比见报告及 [performance.csv](results/openmp-final-20261117/summary/performance.csv)。计时可能受调度、CPU 频率和系统负载影响，个别加速比略高于线程数不代表一般超线性保证。原 PSOCK、预跑及早期 smoke 批次作为历史记录保留在 [results/archive/](results/archive/)，当前报告仅使用 OpenMP 最终批次。

## 依赖与复现

需要 R、匹配的 Windows Rtools、Rcpp 与 RcppArmadillo。若尚未安装包，在 R 中执行 `install.packages(c("Rcpp", "RcppArmadillo"))`。实际版本记录在实验目录的 `environment.txt`。

所有相对路径均以本目录 `guisheng-dai/` 为工作目录。先从仓库根目录执行 `Set-Location guisheng-dai`。本机 Rscript 不在 PATH 中，以下使用实际路径；其他机器可改为其 Rscript 路径或直接使用 PATH 中的 `Rscript`：

```powershell
$taskRscript = 'C:/Program Files/R/R-4.6.1/bin/Rscript.exe'
$env:LC_ALL = 'en_US.UTF-8'
$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_MAX_ACTIVE_LEVELS = '1'
$env:OMP_WAIT_POLICY = 'PASSIVE'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $taskRscript --vanilla scripts/00_setup.R
& $taskRscript --vanilla scripts/01_tests.R
& $taskRscript --vanilla scripts/01_openmp_tests.R
& $taskRscript --vanilla scripts/02_run.R --mode smoke --experiment all --threads 1 --run-id NEW-SMOKE
& $taskRscript --vanilla scripts/02_run.R --mode full --experiment all --threads 2 --run-id NEW-FULL
```

`smoke` 强制 `--threads 1`，重复数缩减，只检查流程。正式 `full` 的 E5 内部比较 `p=1,2,4` 个 OpenMP 线程，其余实验使用 `--threads` 指定的线程数。旧 `--workers` 参数会拒绝并提示改用 `--threads`。每个 run-id 对应一次运行，已有记录不会被覆盖；建议用 `--experiment all` 一次执行全部实验。若单独选择某实验，使用独立 run-id；脚本不实现断点追加。

主要 Python 依赖的本机版本列于 `requirements.txt`；只使用 Python 绘图、汇总与排版，正式数值核心在 Rcpp / RcppArmadillo 中：

```powershell
python scripts/03_summarise_plot.py openmp-final-20261117
python scripts/04_build_documents.py
```

第一个命令只读取已保存结果，重建汇总 CSV 和图形，不重新求解。第二个命令从本轮 OpenMP 汇总生成文稿源文件，数值结论动态读取 `report/summary.json`。未来更换实验批次时仍应核对环境、问题设置及正文解释。

PDF 构建需要 XeLaTeX（本机 TeX Live 2024，中文使用 Windows 字体），PPTX 构建还需要 Pandoc 和 openpyxl：

```powershell
./report/build_documents.ps1
```

若本机 PowerShell 的脚本执行策略阻止 `.ps1`，可直接运行以下等价构建命令，无需修改系统策略：

```powershell
python scripts/04_build_documents.py
Push-Location report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error slides.tex
xelatex -interaction=nonstopmode -halt-on-error slides.tex
Pop-Location
python scripts/05_build_pptx.py
```

## 实现与计时说明

- `src/random_walk.cpp`：`walk_blocks_openmp` 对查询×固定逻辑块使用 OpenMP `parallel` / `for schedule(static)`。每块私有 CMRG 状态，线程内不调用 R API。返回全部路径的收益一、二阶统计和工作量；保留 `walk_block_cpp` 供串行 R 随机流对照。
- `src/gaussian_convolution.cpp`：指数变换、奇延拓与宽核窗口内积；`gaussian_openmp_arma` 用 OpenMP 分配独立输出点，共享完整只读填充数组，窗口内采用 Armadillo 内积。不使用三点卷积、FFT 或时间步进，截断核不重新归一化。
- `R/random_walk.R`：R 参考路径、小矩阵离散期望与正式路径块归约。
- `R/gaussian_convolution.R`：串行 R 参考滑窗及 C++ 调用，保留端点置零前残差。
- `R/rng.R`：CMRG 随机状态绑定逻辑任务，改变并行度不改变任务随机数。
- `scripts/02_run.R`：实验扫描、原始记录与计时。短任务批量执行，原始 `batch_A` 保留。相同输入的计时批量不当作独立随机重复。

`t_solve` 是整次求解器调用耗时，包含 R 任务打包、R/C++ 调用、OpenMP fork/join、实际计算、包装和固定块序归约，不称为纯 C++ 核耗时。`t_input` 是初值或填充/权重准备；没有 IPC，`t_prepare_call=0`。`t_reused_new_input` 为准备加求解，使用已初始化运行时。`t_openmp_init` 是首次请求该线程数并核验的包装调用的一次观测，可能复用已有运行时线程；`t_once` 再加此观测，并非全新 R 进程的端到端冷启动。首次编译、随机流分配、真值计算、核验、绘图与 CSV 写出不属于求解计时。E4 的单次时间不用于性能结论。

`results/openmp-final-20261117/source_hashes.csv` 标识实验后保存的源码快照，路径相对于本目录；后续仅整理目录及文档，数值核心保持原样。`openmp_info.rds` 与 `raw/initialization.csv` 保存编译和实际线程信息。`environment.txt` 中 Git HEAD 是实验运行时的基础提交，包含本目录的提交以仓库 Git 历史为准。

## 复用与 AI 声明

本次使用 AI 辅助实验方案、核心实现、测试、统计和文稿。验证包括 R 与 C++ 同流比较、手算路径/窗口、解析解、独立矩阵一二阶矩、谱参照加密及串并行一致性。核心算法不是调用现成 PDE 求解器。

OpenMP 写法参考课程仓库 `computer-skill/openmp-basics.md` 与 `homework/homework-03/starter-code/image-convolution-openmp.cpp` 的 Rcpp 插件、循环调度和线程安全要求；CMRG 递推按 R 的 `src/main/RNG.c` 实现，并逐块与 R 的随机路径核对。R 的 `parallel::nextRNGStream` 仅分配随机流，不启动工作进程。

模板复用用户提供的 CleanEasy BeamerTheme，原作者 Jose Paulo Marchezi，许可保留在 `report/template/LICENSE`。实验图表由本轮原始记录生成。PPTX 对应稿按模板颜色、衬线字体层次和页眉页脚重建，包含可编辑文本与原生图表；模板原生版本为 `slides.pdf`。本机 PowerPoint 自动化不可用，PPTX 使用包结构、几何和嵌入数据检查，PDF 使用逐页视觉检查。
