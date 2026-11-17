"""Build PDF report and Beamer presentation sources from actual measurements."""
from pathlib import Path
import argparse, json
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary",type=Path,default=root/"report/summary.json")
parser.add_argument("--report-dir",type=Path,default=root/"report")
parser.add_argument("--run-id",help="Override the measured run identifier recorded in summary.json")
parser.add_argument("--date",default="2026-11-17",help="Document date (YYYY-MM-DD, Beijing date)")
args=parser.parse_args();report=args.report_dir;report.mkdir(parents=True,exist_ok=True)
S=json.loads(args.summary.read_text(encoding="utf-8"))
def sci(x):
 a,b=f"{x:.2e}".split("e")
 return a+r"\times10^{"+str(int(b))+"}"
def fill(text,values):
 for k,v in values.items():text=text.replace("@@"+k+"@@",str(v))
 return text
def sel(rows,**kw):return next(r for r in rows if all(r.get(k)==v for k,v in kw.items()))
P=S["performance"];K=S["kernel"];M=S["mc"];C=S["cost"];G=S["grid"]
pm=sel(P,method="MC",M=65536,p=4);pc=sel(P,method="Gaussian",N=8192,p=4)
cm=sel(C,method="MC",M=16384);cc=sel(C,method="Gaussian",c=2)
small1=sel(P,method="Gaussian",N=512,p=1);small4=sel(P,method="Gaussian",N=512,p=4)
run_id=args.run_id or S.get("run_id")
if not run_id:raise ValueError("summary.json must identify run_id, or pass --run-id")
if S.get("backend")!="OpenMP":raise ValueError("Rebuild measurements and summary with the OpenMP backend first")
init={p:float(sel(P,method="MC",M=65536,p=p)["t_openmp_init"]) for p in [2,4]}
date=args.date;date_zh="{}年{}月{}日".format(*map(int,date.split("-")))
k1=sel(K,c=1);k4=sel(K,c=4);k6=sel(K,c=6);k10=sel(K,c=10)
small_comparison="高于" if small4["median"]>small1["median"] else "低于"
small_observation=f'本次 $N=512$ 卷积四线程耗时{small_comparison}单线程；小任务仍须实测线程开销。'
cost_ratio_once=cm["once"]/cc["once"]
V={"RUNTIME":f'{S["runtime"]:.1f}',"MCSPEED":f'{pm["speedup"]:.2f}',"CVSPEED":f'{pc["speedup"]:.2f}',
"RATIO":f'{cm["warm_new_input"]/cc["warm_new_input"]:.1f}',
"DATE":date,"DATEZH":date_zh,"RUNID":run_id.replace("_",r"\_"),"TESTS":S["tests"],"OMPTESTS":S["openmp_tests"],"TIMINGROWS":S["raw_timing_rows"],
"MCEFF":f'{100*pm["efficiency"]:.1f}',"CVEFF":f'{100*pc["efficiency"]:.1f}',
"SMALL1":f'{small1["median"]*1000:.3f}',"SMALL4":f'{small4["median"]*1000:.3f}',"SMALLOBS":small_observation,
"INIT2":f'{init[2]*1000:.3f}',"INIT4":f'{init[4]*1000:.3f}',
"CMRMSE":sci(cm["RMSE"]),"CCRMSE":sci(cc["RMSE"]),
"CMNEW":f'{cm["warm_new_input"]*1000:.3f}',"CCNEW":f'{cc["warm_new_input"]*1000:.3f}',
"CMONCE":f'{cm["once"]*1000:.3f}',"CCONCE":f'{cc["once"]*1000:.3f}',"ONCERATIO":f'{cost_ratio_once:.1f}',
"BIAS8":sci(abs(sel(M,N=8,M=16384)["bias"])),"RMSE8":sci(sel(M,N=8,M=16384)["RMSE_true"]),
"BIAS16":sci(abs(sel(M,N=16,M=4096)["bias"])),"BIAS32":sci(abs(sel(M,N=32,M=4096)["bias"])),
"K1ERR":sci(k1["RMSE"]),"K4ERR":sci(k4["RMSE"]),"K6ERR":sci(k6["RMSE"]),
"REFDIFF":sci(k10["trunc_rmse"]),"BOUNDARY":sci(max(r["boundary"] for r in K)),
"MINORDER":f'{min(r["order"] for r in G if r["order"] is not None):.2f}',
"MAXORDER":f'{max(r["order"] for r in G if r["order"] is not None):.2f}'}
V["SUPERIDEAL"]="个别 $S_p$ 略高于 $p$ 可由有限次数计时、频率及调度波动引起，不构成超线性加速保证。"
rows=[]
for N in [8,16]:
 for m in [256,4096,16384]:
  r=sel(M,N=N,M=m)
  rows.append(f'{N} & {m} & $'+sci(r["RMSE_discrete"])+'$ & $'+sci(r["RMSE_true"])+'$ & $'+sci(r["SD"])+r"$ \\")
V["MCROWS"]="\n".join(rows)
V["KERNELROWS"]="\n".join(f'{r["c"]} & {r["W"]} & $'+sci(r["RMSE"])+f'$ & {r["median"]*1000:.3f} & {r["warm_new_input"]*1000:.3f} '+r"\\" for r in K if r["c"]<=6)
V["GRIDROWS"]="\n".join(str(r["N"])+' & $'+sci(r["RMSE"])+'$ & '+("--" if r["order"] is None else f'{r["order"]:.3f}')+r" \\" for r in G)
chosen=[sel(P,method="MC",M=65536,p=p) for p in [1,2,4]]+[sel(P,method="Gaussian",N=8192,p=p) for p in [1,2,4]]
V["PERFROWS"]="\n".join(f'{("MC" if r["method"]=="MC" else "卷积")} & {r["p"]} & {r["median"]*1000:.2f} & [{r["Q1"]*1000:.2f}, {r["Q3"]*1000:.2f}] & {r["speedup"]:.2f} & {100*r["efficiency"]:.1f}\\% '+r"\\" for r in chosen)
V["COSTROWS"]="\n".join((f'MC, $M={int(r["M"])}$' if r["method"]=="MC" else f'Gaussian, $c={int(r["c"])}$')+' & $'+sci(r["RMSE"])+f'$ & {r["warm_new_input"]*1000:.3f} & {r["once"]*1000:.1f} '+r"\\" for r in C)
tex=r"""\documentclass[UTF8,10.5pt,a4paper]{ctexart}
\usepackage[margin=1.7cm,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,xcolor,fancyhdr,hyperref,enumitem}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,pdftitle={漂移扩散方程数值实验临时报告},pdfauthor={}}
\definecolor{deepblue}{HTML}{0D2659}
\ctexset{section={format=\Large\bfseries\color{deepblue}},subsection={format=\normalsize\bfseries}}
\setlength{\parindent}{2em}\setlength{\parskip}{.35em}
\setlist{nosep,leftmargin=2em}\setlength{\tabcolsep}{6pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{漂移扩散方程数值实验}\fancyhead[R]{临时报告 · @@DATE@@}\fancyfoot[C]{\thepage}
\newcommand{\fig}[2]{\begin{center}\includegraphics[width=#2\linewidth]{figures/#1.pdf}\end{center}}
\begin{document}
\begin{center}{\LARGE\bfseries 漂移扩散方程的数值实验}\\[.5em]
{\large 反向随机游走与 Gaussian 宽核直接卷积}\\[.5em]
Rcpp、RcppArmadillo 与 C++ OpenMP 并行计算
\end{center}
\noindent\textbf{摘要}\quad
在同一零 Dirichlet 边界问题上，实现逐步吸收的反向随机游走和经过指数变换、奇延拓的 Gaussian 直接卷积。完成 @@TESTS@@ 项串行 E0 测试及另行 @@OMPTESTS@@ 项 OpenMP 集成核验；正式实验采用两线程，并以一、二、四线程比较性能。R 层串行组织实验与参考计算；C++ 求解器采用 OpenMP 并行。随机游走同时报告抽样误差与离散偏差，卷积分别考察核截断和网格求积误差。本文的性能及精度结论均来自本轮 OpenMP 原始记录。

\section{问题、基准与实验环境}
研究
\[
u_t=a u_x+D u_{xx},\quad 0<x<1;\qquad
u(0,t)=u(1,t)=0,\quad u(x,0)=f(x).
\]
固定 $a=0.2,D=0.05,T=0.5$，令 $\beta=a/(2D)=2$、$\gamma=a^2/(4D)=0.2$。主初值和解析解为
\[
f(x)=e^{-2x}\sin(\pi x),\qquad
u_\star(x,t)=e^{-2x-(0.05\pi^2+0.2)t}\sin(\pi x).
\]
该解满足初边值条件及 PDE，可独立核验求解器。网格有 $N$ 段、$N+1$ 个节点，$h=1/N$。内部 RMSE 与最大误差单独计算，边界不混入内部指标。主基准只衰减振幅，不用于说明一般信号的峰移动。
\begin{center}\small
\begin{tabular}{ll}\toprule
项目 & 本轮设置\\\midrule
硬件 & Intel Core i7-11800H，8 物理核 / 16 逻辑核，31.74 GiB RAM\\
系统 & Windows 11 x64（build 26200）\\
计算软件 & R 4.6.1，Rcpp 1.1.2，RcppArmadillo 15.6.0-1\\
编译与数值库 & GCC 14.3.0（Rtools45），Armadillo 15.6.0，默认 BLAS\\
并行 & C++ OpenMP，正式精度实验 $p=2$，性能 $p=1,2,4$ 线程\\
随机性 & L'Ecuyer-CMRG，主种子 20261116，完整任务状态已保存\\
精度与重复 & 双精度，MC 每配置 30 批，性能预热后 7 次\\\bottomrule
\end{tabular}
\end{center}
正式 E1--E6 从已有编译缓存执行共约 \textbf{@@RUNTIME@@ 秒}，包含预热、批量测量、首次线程请求核验和实验阶段写出，不包含首次编译、绘图与文稿编译。R 层不创建工作进程；BLAS 设为单线程，避免嵌套并行。该耗时仅代表本机本次运行。
\newpage
\section{方法一：反向随机游走与正确性}
时间离散为 $n$ 步、$\tau=T/n$，令
\[
r=D\tau/h^2,\quad b=a\tau/(2h),\quad
p_-=r-b,\quad p_0=1-2r,\quad p_+=r+b.
\]
检查三个概率非负且和为 1。路径从查询节点出发，以 $p_-,p_0,p_+$ 左移、停留、右移；每一步检查端点，触边立即吸收，收益记零。存活到最后才取终点初值：
\[
Y=f(X_n)\mathbf1_{\{\text{未触边}\}},\qquad
\widehat u=\frac1M\sum_{k=1}^M Y_k.
\]
被吸收路径仍占分母。$+a u_x$ 下后向路径偏向右，是读取初值的方向。C++ 一次接收全部逻辑路径块，用 OpenMP 分配块；每块独立 RNG 状态，线程内部不调用 R API。
\subsection{区分抽样与离散误差}
小规模参照为内部转移矩阵 $Q$，下、主、上对角依次为 $p_-,p_0,p_+$；端点概率流失，不归一化边缘行。计算
\[
U_{\rm disc}=Q^nf_{\rm int},\qquad
v=Q^n(f_{\rm int}^2)-(Q^nf_{\rm int})^2,
\]
则 $\mathbb E(\widehat u-u_\star)^2=(U_{\rm disc}-u_\star)^2+v/M$。
每块返回贡献和、平方和、路径数、存活数和实际步数。样本方差用分母 $M-1$，标准误为 $\widehat s_Y/\sqrt M$。30 批 RMSE 先对平方误差取平均再开方，未用 30 批平均后的误差冒充单次预算精度。
\subsection{串行小规模测试}
@@TESTS@@ 项 E0 测试全部通过，日志见 \texttt{results/tests/e0\_tests.csv}。小规模测试固定一线程。
\begin{center}\small
\begin{tabular}{p{.27\linewidth}p{.61\linewidth}}\toprule
测试 & 核验内容与结果\\\midrule
已知一步期望 & $N=4,\tau=0.1$，初值 $[0,1,0,0,0]$，内部期望 $[0.84,0.04,0]$\\
强制路径 & 首次左右触边收益 0、步数 1；停留路径步数正确\\
R 与 Rcpp 随机流 & 同状态产生相同统计，误差低于 $10^{-12}$\\
MC 统计检查 & 10 批 $M=4096$ 总均值在离散期望的约四个标准误内\\
宽核与边界 & 手算宽核、奇延拓、有限核正弦恒等式及原始边界残差通过\\
输入与退化情形 & 零初值、$T=0$、非法概率、越界索引及非零边界处理通过\\\bottomrule
\end{tabular}
\end{center}
R 参考路径用于正确性核验，正式计算用 Rcpp；矩阵参照与解析计算在性能计时之外。该矩阵参考不是本项目的卷积算法。另完成 @@OMPTESTS@@ 项专门 OpenMP 集成核验，包括实际线程数、串并行位级一致性、CMRG 与 R 路径及输入只读检查，与本组串行 E0 分开记录。
\newpage
\section{随机游走实验：路径数与网格}
E1 采用 $(N,n)=(8,8),(16,32)$，$x=1/8$，均有 $r=0.2$，各扫描四档 $M$、独立 30 批，正式按 16 个逻辑块并行。E2 复用共同配置，仅新增 $(32,128),M=4096$。
\fig{mc_error}{1}
\begin{center}\small
\begin{tabular}{rrrrr}\toprule
$N$ & $M$ & 对离散期望 RMSE & 对解析解 RMSE & 30 批标准差\\\midrule
@@MCROWS@@
\bottomrule\end{tabular}
\end{center}
粗网格 $N=8$ 的离散偏差为 $@@BIAS8@@$，$M=16384$ 时总 RMSE 为 $@@RMSE8@@$。细网格偏差较小，但有限 30 批的估计会波动，不能要求每一档都精确符合理论斜率。
\fig{mc_grid}{.88}
$N=8,16,32$ 的离散偏差分别约 $@@BIAS8@@$、$@@BIAS16@@$、$@@BIAS32@@$；固定 $M=4096$ 的总 RMSE 同时受抽样噪声影响，噪声可以遮住网格改善。
\newpage
\section{方法二：正确边界的 Gaussian 直接卷积}
先消去漂移：
\[
v=e^{\beta x+\gamma t}u,\qquad g=e^{\beta x}f,\qquad v_t=Dv_{xx}.
\]
对 $g$ 作关于两个端点的奇延拓，形成周期 2 的 $\widetilde g$，则
\[
u(x,T)=e^{-\beta x-\gamma T}
\int_{\mathbb R}\frac{e^{-z^2/(4DT)}}{\sqrt{4\pi DT}}\widetilde g(x-z)\,dz.
\]
奇延拓与对称热核使两端贡献成对抵消。这里的周期性属于辅助填充，物理边界仍为零；直接对原始初值补零、普通镜像或周期化不能替代这一步。

对整数索引 $m$，令 $q=((m\bmod 2N)+2N)\bmod2N$，
\[
g_{\rm ext}(m)=
\begin{cases}g_q,&0\le q\le N,\\-g_{2N-q},&N<q<2N.\end{cases}
\]
该规则处理负下标和超过一个区间的窗口。核权重与一次终态求和为
\[
w_\ell=\frac{h}{\sqrt{4\pi DT}}e^{-(\ell h)^2/(4DT)},\qquad
U_j=e^{-\beta x_j-\gamma T}\sum_{\ell=-K}^K w_\ell g_{\rm ext}(j-\ell).
\]
这是一维图像滤波式宽窗口内积，不使用三点核、FFT 或时间推进。权重含 $h$，截断后不再归一化。有限窗口近似完整边界积分表示，其误差须实测。
\subsection{为什么采用这个核长度}
Gaussian 核的方差为 $2DT$，标准差 $s_T=\sqrt{2DT}$。覆盖 $c$ 个标准差对应
\[
K=\left\lceil c s_T/h\right\rceil,\qquad W=2K+1.
\]
向上取整保证物理半径至少为 $c s_T$；奇数长度包含中心和左右各 $K$ 点，保留对称抵消。$c=1,2,3,4,6$ 的连续双尾质量约为 $0.3173,0.04550,0.002700,6.33\times10^{-5},1.97\times10^{-9}$。它们只提示截断规模，未包含离散求积误差。
\subsection{实现与并行}
\begin{enumerate}
\item RcppArmadillo 准备变换初值、完整奇延拓、Gaussian 权重和指数因子。
\item 每点用 \texttt{arma::dot(padded.subvec(...),weights)}，输入只读、输出独立。
\item C++ 用 \texttt{omp parallel for schedule(static)} 按独立输出点并行，线程共享完整只读填充输入，不在分界处重新补零。
\item 记录端点置零前残差，再将输出端点置为精确零。
\end{enumerate}
全场成本 $O(NW)$，共享工作数组 $O(N+K)$；只查询 $q$ 点时求和成本 $O(qW)$。主实验最大置零前边界残差为 $@@BOUNDARY@@$。
\newpage
\section{卷积实验：核大小与网格求积}
E3 固定 $N=512$，仅改变覆盖倍数；以同网格 $c=8$ 结果诊断截断，另用 $c=10$ 检查参考。两参考之差的内部 RMSE 为 $@@REFDIFF@@$。
\fig{kernel}{1}
\begin{center}\small
\begin{tabular}{rrrrr}\toprule
$c$ & $W$ & 内部 RMSE & 求解中位/ms & 预热后新输入/ms\\\midrule
@@KERNELROWS@@
\bottomrule\end{tabular}
\end{center}
从 $c=1$ 到 $c=4$，RMSE 从 $@@K1ERR@@$ 降至 $@@K4ERR@@$；$c=6$ 达到 $@@K6ERR@@$。加宽核增加内积工作；完整时间还包含本地准备、R/C++ 调用和 OpenMP fork/join，短调用可能有测量波动。

E4 固定物理覆盖 $c=8$，随网格更新 $K$。使用 $f_{\rm poly}=e^{-2x}x(1-x)$，避免单一正弦初值过早达到舍入平台。其参照为
\[
u_{\rm poly}=e^{-2x-0.2T}
\sum_{k\ {\rm odd}}\frac{8}{\pi^3k^3}e^{-D(k\pi)^2T}\sin(k\pi x).
\]
最高奇数波数 31 与 63 的参照在本次节点上差异低于 $10^{-12}$。
\par\noindent
\begin{minipage}{.60\linewidth}\includegraphics[width=\linewidth]{figures/grid.pdf}\end{minipage}
\begin{minipage}{.38\linewidth}\centering\small
\begin{tabular}{rrr}\toprule
$N$ & RMSE & 观测阶\\\midrule
@@GRIDROWS@@
\bottomrule\end{tabular}
\end{minipage}

本输入的观测阶为 @@MINORDER@@--@@MAXORDER@@，属于该输入及设置下的结果，不能推广为任意初值的统一收敛阶。E3 与 E4 分开控制核半径和网格。
\newpage
\section{串行、并行与计算开销}
R 层仅串行组织实验，两个 C++ 求解器用 OpenMP 并行循环。MC 每点固定 16 个逻辑路径块，L'Ecuyer-CMRG 状态绑定逻辑块；线程各自更新私有 RNG，完成后按块编号固定顺序归约。卷积按独立输出点并行，输入共享只读，每窗口保持原顺序求和。改变线程数不改变样本或求和顺序，串并行预测和 MC 计数核对一致。每个性能配置预热后计时 7 次，短任务批量测量，线程数顺序按轮次交错。
\fig{parallel_time}{1}
\begin{center}\small
\begin{tabular}{lrrrrr}\toprule
任务 & $p$ & 中位/ms & [Q1,Q3]/ms & 加速比 & 效率\\\midrule
@@PERFROWS@@
\bottomrule\end{tabular}
\end{center}
上表 MC 取 $N=64,n=256,M=65536$；卷积取 $N=8192,c=4,W=14657$。本次四线程中位时间之比为 \textbf{@@MCSPEED@@} 和 \textbf{@@CVSPEED@@}，效率为 @@MCEFF@@\% 和 @@CVEFF@@\%。@@SUPERIDEAL@@
\fig{speedup}{.87}
$N=512$ 卷积单线程 @@SMALL1@@ ms、四线程 @@SMALL4@@ ms。$t_{\rm solve}$ 含 R/C++ 调用、fork/join、调用内准备、计算、包装与归约；$t_{\rm prepare\_call}=0$，无 IPC。首次请求两、四线程并核验的 R 包装调用分别观测 @@INIT2@@ ms、@@INIT4@@ ms，含检查且可能复用 libgomp 既有线程。$t_{\rm once}$ 是该单次观测加 $t_{\rm input}+t_{\rm solve}$，不是新 R 进程端到端冷启动或纯 C++ 核心耗时。随机流分配、编译、独立预热、真值与绘图在计时外。
\newpage
\section{共同任务、结论与复现}
E6 固定主问题和共同 8 点 $x=(1,3,\ldots,15)/16$、两线程。MC 使用 $N=16,n=32$，每点 $M$ 条路径、每配置 30 批；卷积使用 $N=512$，仅计算共同 8 个窗口，每配置计时 7 次。
\fig{cost}{.92}
\begin{center}\small
\begin{tabular}{lrrr}\toprule
配置 & 共同点 RMSE & 预热后新输入/ms & 加首次请求观测/ms\\\midrule
@@COSTROWS@@
\bottomrule\end{tabular}
\end{center}
MC 的 $M=16384$ 与卷积的 $c=2$ 观测 RMSE 分别为 $@@CMRMSE@@$、$@@CCRMSE@@$，预热后的新输入成本为 @@CMNEW@@ ms、@@CCNEW@@ ms，相差约 \textbf{@@RATIO@@ 倍}。加本次首次请求核验观测后为 @@CMONCE@@ ms、@@CCONCE@@ ms，成本比为 @@ONCERATIO@@。该结论仅针对本次一维、常系数、齐次边界任务。

\textbf{结论与局限。} 增加路径数降低 MC 抽样误差，细网格降低离散偏差；直接卷积在本任务更易达到高精度，宽核成本与核长相关。OpenMP 收益依赖规模、线程开销和硬件。有限 30 批不保证逐次误差达标。本轮只覆盖两种方法，未验证一般变系数、非零边界和高维，亦未进行正式课程提交。

\textbf{复现。} 先按 README 编译并运行串行 E0 与 OpenMP 集成测试，再在根目录运行：
\begin{flushleft}\small\ttfamily
Rscript scripts/02\_run.R --mode full --experiment all --threads 2 --run-id NEW\\
python scripts/03\_summarise\_plot.py NEW
\end{flushleft}
\noindent\texttt{results/@@RUNID@@/} 保留配置、完整 RNG 状态、@@TIMINGROWS@@ 条原始计时和源文件哈希。首次安装编译另需时间。

{\footnotesize
\textbf{资料与 AI 使用。} 数学与实验口径见 \texttt{numeric-experiment-plan.md}，并行方式依照 \texttt{idea.md} 及课程 OpenMP 讲义。AI 辅助方案、实现、统计与文稿，通过串行参考、解析真值、@@TESTS@@ 项串行 E0 及另 @@OMPTESTS@@ 项 OpenMP 集成核验。所有图表来自本轮原始记录。
\begin{enumerate}[leftmargin=1.5em]
\item RcppArmadillo：\url{https://github.com/RcppCore/RcppArmadillo}
\item R parallel：\url{https://stat.ethz.ch/R-manual/R-devel/library/parallel/html/RngStream.html}
\item 课程仓库：\texttt{computer-skill/openmp-basics.md}，Rcpp 与 OpenMP、图像卷积练习。
\item 反射法：Masayuki Yano, ESC384 Lecture Notes, Lecture 7，2021。
\item 演示主题：用户提供的 CleanEasy BeamerTheme，Jose Paulo Marchezi，保留模板许可。
\end{enumerate}}
\end{document}
"""
(report/"report.tex").write_text(fill(tex,V),encoding="utf-8")
slides=r"""\documentclass[UTF8,aspectratio=169,10pt]{ctexbeamer}
\usepackage{etoolbox,tikz,booktabs,amsmath,amssymb,graphicx}
\makeatletter\def\input@path{{template/theme/}}\makeatother
\usetheme{CleanEasy}\AtBeginSection[]{}
\title[漂移扩散方程数值实验]{漂移扩散方程数值实验}
\subtitle{反向随机游走与 Gaussian 宽核直接卷积\\临时实验报告}
\author{}\institute{}\date[]{@@DATEZH@@}
\hypersetup{pdfauthor={},pdftitle={漂移扩散方程数值实验演示}}
\newcommand{\pic}[1]{\begin{center}\includegraphics[width=.92\linewidth,height=5.1cm,keepaspectratio]{figures/#1.pdf}\end{center}}
\begin{document}
\begin{frame}[plain]\titlepage\end{frame}
\begin{frame}{模型与比较任务}
\[
u_t=0.2u_x+0.05u_{xx},\quad 0<x<1,\quad T=0.5
\]
\[
u(0,t)=u(1,t)=0,\qquad f=e^{-2x}\sin(\pi x)
\]
\[
u_\star=e^{-2x-(0.05\pi^2+0.2)t}\sin(\pi x)
\]
\begin{itemize}
\item 内部 RMSE 与边界残差分别记录。
\item R 层串行组织实验；C++ OpenMP 使用一、二、四线程。
\item 30 批独立随机重复，性能预热后重复 7 次。
\end{itemize}\end{frame}
\begin{frame}{反向随机游走}
\[
p_\pm=D\tau/h^2\pm a\tau/(2h),\qquad p_0=1-2D\tau/h^2
\]
\[
Y=f(X_n)\mathbf1_{\{\text{未触边}\}},\qquad \widehat u=M^{-1}\sum Y_k
\]
\begin{itemize}
\item 从查询点出发，逐步吸收，触边贡献为零。
\item 分母包含所有路径，每查询点固定 16 个逻辑块。
\item OpenMP 分配路径块，私有随机流，按块序归约。
\end{itemize}
\begin{exampleblock}{正确性}@@TESTS@@ 项一线程 E0 测试通过，包含同流、吸收、手算宽核及解析核验；另通过 @@OMPTESTS@@ 项 OpenMP 集成核验。\end{exampleblock}
\end{frame}
\begin{frame}{随机游走的两类误差}
\pic{mc_error}
\small 增大路径数降低抽样方差；粗网格总误差受离散偏差限制。每配置独立 30 批。
\end{frame}
\begin{frame}{固定路径数的网格影响}
\pic{mc_grid}
\small 离散偏差持续下降；固定 $M=4096$ 的总 RMSE 还受抽样噪声影响，可遮住网格改善。
\end{frame}
\begin{frame}{正确零边界的 Gaussian 直接卷积}
\[
v=e^{\beta x+\gamma t}u,\quad g=e^{\beta x}f,\quad v_t=Dv_{xx}
\]
\[
u(x,T)=e^{-\beta x-\gamma T}\int_{\mathbb R}G_T(z)\widetilde g(x-z)\,dz
\]
\begin{enumerate}
\item 对变换后的 $g$ 作两端奇延拓，形成周期 2 的辅助输入。
\item Gaussian 宽窗口直接内积，输入只读、输出独立。
\item 乘指数因子变回原解，记录端点置零前残差。
\end{enumerate}
\begin{block}{实现}OpenMP 按输出点并行；RcppArmadillo 的 \texttt{subvec} 与 \texttt{dot} 完成每个窗口。\end{block}
\end{frame}
\begin{frame}{卷积核大小的选择}
\[
s_T=\sqrt{2DT},\quad K=\lceil c s_T/h\rceil,\quad W=2K+1
\]
\begin{itemize}
\item $c$ 决定物理覆盖，$h$ 将范围换算成数组长度。
\item 左右对称加中心点，使奇延拓边界贡献成对抵消。
\item $N=512,c=3$ 时 $K=344,W=689$。
\item 截断后不重新归一化。更宽的核需要更多计算。
\end{itemize}
\begin{exampleblock}{实验控制}核大小实验固定网格；网格实验固定物理覆盖。\end{exampleblock}
\end{frame}
\begin{frame}{核大小与精度、成本}
\pic{kernel}
\small $W=231$ 到 $917$，RMSE 从 $@@K1ERR@@$ 降至 $@@K4ERR@@$；$W=1375$ 为 $@@K6ERR@@$。
\end{frame}
\begin{frame}{宽核下的网格求积误差}
\pic{grid}
\small 多项式初值 $e^{-2x}x(1-x)$，$c=8$；本组观测阶 @@MINORDER@@--@@MAXORDER@@，不推广为任意初值的统一结论。
\end{frame}
\begin{frame}{并行任务与计时范围}
\begin{itemize}
\item R 串行调度；C++ 用 OpenMP 并行循环，BLAS 单线程。
\item MC 私有 RNG、固定块序归约；卷积共享完整奇延拓。
\item 串并行预测与 MC 计数一致，预热后重复计时 7 次。
\item $t_{\rm solve}$ 含 R/C++ 调用、fork/join、计算与归约。
\end{itemize}
\[
t_{\rm new}=t_{\rm input}+t_{\rm solve},\qquad
t_{\rm once}=t_{\rm new}+t_{\rm OpenMP\ init}
\]
\begin{block}{首次请求及核验：单次 R 包装调用观测}两线程 @@INIT2@@ ms，四线程 @@INIT4@@ ms，含检查且可复用既有线程。$t_{\rm once}$ 不是新 R 进程冷启动；随机流分配、编译、预热与真值核验不计时。\end{block}
\end{frame}
\begin{frame}{并行收益取决于任务规模}
\pic{speedup}
\small 本次大任务四线程：MC \textbf{@@MCSPEED@@ 倍}，卷积 \textbf{@@CVSPEED@@ 倍}。@@SUPERIDEAL@@
\end{frame}
\begin{frame}{共同查询任务的精度与成本}
\pic{cost}
\small 共同 8 点、两线程。MC $M=16384$ 与卷积 $c=2$ 的新输入成本比为 \textbf{@@RATIO@@}，加首次请求核验观测后为 @@ONCERATIO@@；结论限于本次问题。
\end{frame}
\end{document}
"""
(report/"slides.tex").write_text(fill(slides,V),encoding="utf-8")
native=[
dict(title="漂移扩散方程数值实验",body=["反向随机游走与 Gaussian 宽核直接卷积","临时实验报告 · C++ OpenMP",date_zh]),
dict(title="模型与比较任务",body=["uₜ = 0.2uₓ + 0.05uₓₓ，0 < x < 1，T = 0.5","u(0,t) = u(1,t) = 0，f(x) = exp(−2x) sin(πx)","解析解：exp[−2x − (0.05π² + 0.2)t] sin(πx)","内部 RMSE 与边界残差分开记录","R 层串行组织；C++ OpenMP 比较 1、2、4 线程"]),
dict(title="反向随机游走",body=["p± = Dτ/h² ± aτ/(2h)，p₀ = 1 − 2Dτ/h²","从查询点出发，逐步吸收，触边贡献为零","分母包含全部 M 条路径，每查询点固定 16 个逻辑块","OpenMP 分配块，私有 RNG，按逻辑块顺序归约",f'{S["tests"]} 项串行 E0 通过；另 {S["openmp_tests"]} 项 OpenMP 集成核验']),
dict(title="随机游走的两类误差",chart="mc",body=["每配置 30 批，误差按单批预算汇总","增大 M 降低抽样方差，总误差还含离散偏差"]),
dict(title="固定路径数的网格影响",chart="mcgrid",body=["M = 4096，查询 x = 1/8","离散偏差持续下降，有限样本总误差仍可能波动"]),
dict(title="正确零边界的 Gaussian 直接卷积",body=["指数变换：v = exp(βx + γt)u，g = exp(βx)f","变换后 vₜ = Dvₓₓ，β = a/(2D)，γ = a²/(4D)","对 g 作两端奇延拓，形成周期 2 的辅助填充","Gaussian 宽窗口内积，再乘 exp(−βx − γT)","OpenMP 按输出点并行；每窗口用 Armadillo dot","先记录端点残差，再将输出端点设零"]),
dict(title="卷积核大小的选择",body=["扩散标准差 s = √(2DT)","K = ceil(c s/h)，核长度 W = 2K + 1","c 决定物理覆盖，h 将距离换算成格点数","左右各 K 点加中心点，保持对称与边界抵消","N = 512，c = 3：K = 344，W = 689","截断后不重新归一化，更宽的核需要更多计算"]),
dict(title="核大小与精度、成本",chart="kernel",body=["固定 N = 512，C++ OpenMP 使用 2 个线程",f'W = 917：RMSE {k4["RMSE"]:.2e}；W = 1375：{k6["RMSE"]:.2e}']),
dict(title="宽核下的网格求积误差",chart="grid",body=["多项式初值 exp(−2x)x(1−x)，固定物理覆盖 c = 8",f'本组观测阶 {V["MINORDER"]}—{V["MAXORDER"]}，限于当前输入和设置']),
dict(title="并行任务与计时范围",body=["R 串行组织；C++ OpenMP；求解含调用及 fork/join","新输入成本 = 输入准备 + 求解；无进程间通信","随机流分配在计时外；调用外准备字段为零","首次请求核验含 R 包装检查，可复用既有线程",f'本次单次观测：2 线程 {V["INIT2"]} ms，4 线程 {V["INIT4"]} ms',"t_once 加该观测，不是新 R 进程端到端冷启动"]),
dict(title="并行收益取决于任务规模",chart="speedup",body=[f'本次四线程中位时间比：MC {pm["speedup"]:.2f}，卷积 {pc["speedup"]:.2f}',"个别加速比略高于线程数可受计时、频率及调度波动影响，不作超线性保证"]),
dict(title="共同查询任务的精度与成本",chart="cost",body=[f'共同 8 点、2 线程：MC M=16384 与卷积 c=2',f'新输入成本比 {V["RATIO"]}；加首次请求观测后 {V["ONCERATIO"]}；限于本次问题'])]
for s in native:s["notes"]=f"模型与算法：numeric-experiment-plan.md。并行依据：idea.md 与课程 computer-skill/openmp-basics.md。实测数据：results/{run_id}/raw/ 与 summary/。Windows11，i7-11800H，R4.6.1，RcppArmadillo15.6.0-1。R 层串行组织实验；C++ OpenMP。原始记录保留随机状态和计时。{S['tests']} 项串行 E0 与另 {S['openmp_tests']} 项 OpenMP 集成核验分开记录。t_prepare_call=0，调用内准备及结果包装包含在 t_solve；seedman$get 随机流分配在计时外。t_openmp_init 是首次请求该线程数并核验的 R 包装调用单次观测，包含检查且可能复用 libgomp 既有线程；t_once 只加该观测，不是新 R 进程端到端冷启动，也不是纯 C++ 核心耗时。略超理想线的实测加速受计时、调度及频率变化影响，不是性能保证。"
(report/"slide_content.json").write_text(json.dumps(native,ensure_ascii=False,indent=2),encoding="utf-8")
print("Wrote report.tex, slides.tex and PowerPoint content.")
