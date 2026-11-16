# 不同数值方法求解PDE

我想做一个漂移扩散方程（Dirft Diffusion Equation）的数值实验：用几种方法求同一个问题，比较算得准不准、速度怎么样。可以用到课程里学习过的大量工具。

$$
    \begin{cases}
        & u_t(\boldsymbol{x},t)=\boldsymbol{a}\cdot \nabla u(\boldsymbol{x},t)+\sigma^2 \cdot \Delta u(\boldsymbol{x},t)\\
        & u_t(\boldsymbol{x},0) = f_0(\boldsymbol{x})
    \end{cases}
$$

这个方程的解析解是：

\[
\begin{aligned}
u(x,t)
=
&\int_{\mathbb R^d}
f_0(y)
\frac{1}{\sqrt{4\pi\sigma^2t}}
\exp\left(
-\frac{(y-x-at)^2}{4\sigma^2t}
\right)\,dy.\\
= & (G_t \ast f_0)(x + at) 
\end{aligned}
\]

其中

$$
    G_t = \frac{1}{\sqrt{4\pi \sigma^2 t}} \exp\left(-\frac{z^2}{4\sigma^2 t}\right)
$$

是Guassian kernel。

**我的想法：**

- 可以把该方程的差分格式看作**反向随机游走**，从而通过模拟随机游走求解该方程。这也是理解方程中出现Guassian kernel的一种方式。求解过程需要使用Monte Carlo模拟，可以使用**R多进程并行**，**OpenMP并行**等

- 可以直接计算**Guassian kernel卷积**，模拟课程中图像卷积的算法的实现方式。可以使用**RcppArmadillo**，**RcppEigen**等等，算法还可以并行实现

- 可以使用显示差分格式的方法，把它转化迭代问题$\boldsymbol{U}^{(n+1)}=B\boldsymbol{U}^{(n)}$的迭代，而且矩阵$B$是稀疏的，可以使用**RcppEigen**中处理稀疏矩阵的方法

- 可以使用物理神经网络（PINN）进行求解，思路很简单，就是把方程本身和边界条件加权设计成loss函数然后训练。可以使用**PyTorch**

**具体方法你可以直接问ChatGPT，也可以看我下面写的。但是idea绝对是我原创的**

暂时先做一维，区间两端固定为零：

$$
\begin{cases}
u_t=a u_x+D u_{xx}, &0<x<1,\\
u(x,0)=f_0(x),\\
u(0,t)=u(1,t)=0.
\end{cases}
$$

可以把它想成一个信号一边移动、一边变平滑，到达边界后逐渐流失。会选一个有简单解析解的例子检查程序，数学推导只讲到能理解算法的程度。边界怎么处理、数组怎么存、循环怎么写，都是这个项目要考虑的内容。

## 1. 随机游走模拟

取空间步长 \(h>0\)，时间步长 \(\tau>0\)，记
\[
u_i^n\approx u(ih,n\tau).
\]

使用时间前向差分、空间中心差分：

\[
\frac{u_i^{n+1}-u_i^n}{\tau}
=
a\frac{u_{i+1}^n-u_{i-1}^n}{2h}
+
\sigma^2\frac{u_{i+1}^n-2u_i^n+u_{i-1}^n}{h^2}.
\]

整理得到
\[
\boxed{
u_i^{n+1}
=p_+u_{i+1}^n+p_-u_{i-1}^n+p_0u_i^n
}
\]

其中
\[
\begin{aligned}
p_+&=\frac{\sigma^2\tau}{h^2}+\frac{a\tau}{2h},\\
p_-&=\frac{\sigma^2\tau}{h^2}-\frac{a\tau}{2h},\\
p_0&=1-\frac{2\sigma^2\tau}{h^2}.
\end{aligned}
\]

注意到
\[
p_++p_-+p_0=1.
\]

只要保证三个系数非负，就可以把它们解释为概率。

**为什么等价于随机游走：**
定义随机游走
\[
X_{n+1}=X_n+h\xi_{n+1},
\]

其中 \(\xi_1,\xi_2,\dots\) 独立同分布，并且
\[
P(\xi_k=1)=p_+,\qquad P(\xi_k=-1)=p_-.
\]

让游走从 \(X_0=ih\) 出发。

因为
\[
u_i^0=f_0(ih),
\]

经过一次差分迭代，
\[
\begin{aligned}
u_i^1
&=p_+f_0(ih+h)+p_-f_0(ih-h)\\
&=\mathbb E[f_0(X_1)\mid X_0=ih].
\end{aligned}
\]

继续迭代，利用独立增量和全期望公式，可以得到
\[
\boxed{
u_i^n=\mathbb E[f_0(X_n)\mid X_0=ih].
}
\]

因此不需要在整个空间网格上反复计算差分。我们也可以使用 Monte Carlo 方法直接估计某一点的解。

**为什么会出现Guassian Kernel：**

每一步随机游走的平均位移为
\[
\begin{aligned}
\mathbb E[h\xi_k]
&=h(p_+-p_-)\\
&=\frac{ah^2}{2\sigma^2}\\
&=a\tau.
\end{aligned}
\]

它的方差为
\[
\begin{aligned}
\operatorname{Var}(h\xi_k)
&=h^2-a^2\tau^2\\
&=2\sigma^2\tau-a^2\tau^2.
\end{aligned}
\]

因为 \(X_N=x+h\sum_{k=1}^N\xi_k\)，且 \(N\tau=t\)，所以
\[
\mathbb E[X_N]=x+at
\]

以及
\[
\operatorname{Var}(X_N)
=N(2\sigma^2\tau-a^2\tau^2)
=2\sigma^2t-a^2t\tau.
\]

当 \(h\to0\) 时，\(\tau\to0\)、\(N\to\infty\)。由中心极限定理对应的三角阵列极限可知
\[
\boxed{
X_N\xrightarrow{d}N(x+at,\,2\sigma^2t).
}
\]

随机游走终点分布逐渐逼近 Gaussian 分布

因此，对于固定的 \(x,t\)，上述差分解的极限为
\[
u(x,t)=\mathbb E[f_0(X_t)],
\]

其中
\[
X_t\sim N(x+at,2\sigma^2t).
\]

将期望写成积分，得到

\[
\boxed{
u(x,t)
=
\int_{\mathbb R}
f_0(y)
\frac{1}{\sqrt{4\pi\sigma^2t}}
\exp\left(
-\frac{(y-x-at)^2}{4\sigma^2t}
\right)\,dy.
}
\]

记 Gaussian 核为
\[
G_t(z)=
\frac{1}{\sqrt{4\pi\sigma^2t}}
\exp\left(-\frac{z^2}{4\sigma^2t}\right),
\]

就有
\[
\boxed{u(x,t)=(G_t*f_0)(x+at).}
\]

这就从差分格式和随机游走出发，自然地得到了 Gaussian 卷积。

从要求解的位置出发，模拟很多条向左、向右或停留的随机路径，再对终点上的初值取平均。路径碰到边界就停止，贡献记为零。

这部分可以先用 **R 或 NumPy** 写一个基础版本，再用 **Rcpp/C++** 写路径循环。不同路径彼此独立，适合用 **R 多进程** 或 **OpenMP** 并行。

我比较想看的是：路径数增加后误差下降多少，串行和并行的差距有多大，以及边界附近的路径提前结束会怎样影响计算量。并行可以先选一种做，时间够再比较另一种。

## 2. 卷积实现与 Gaussian 核

差分中的局部更新可以写成一个三点卷积，正好能用上课里讲的向量卷积和图像卷积。

可以比较 **NumPy 切片、R 向量化、Rcpp 和 RcppArmadillo** 的实现。公式相同，主要看循环、临时数组和内存访问带来的差别。规模大一些时，也可以试试 **OpenMP** 并行更新。

Gaussian 卷积可以留作一个小对照：看忽略边界以后，算出的曲线会在哪里出问题。这里采用固定零边界，一次普通 Gaussian 卷积不能直接当成这个问题的精确解。

## 3. 有限差分法

### 3.1. 空间离散：把 PDE 转化为矩阵方程
先考虑有限区间 \(x\in[0,L]\)，取网格点
\[
x_i=ih,\qquad h=\frac{L}{N+1},
\quad i=0,1,\dots,N+1.
\]

为了简单，假设 Dirichlet 边界条件
\[
u(0,t)=u(L,t)=0.
\]

定义内部网格点上的函数值向量
\[
\boldsymbol U(t)=
\begin{pmatrix}
u_1(t)\\u_2(t)\\\vdots\\u_N(t)
\end{pmatrix},
\qquad u_i(t)\approx u(x_i,t).
\]

利用中心差分：
\[
u_x(x_i,t)\approx\frac{u_{i+1}-u_{i-1}}{2h},
\]

\[
u_{xx}(x_i,t)\approx
\frac{u_{i+1}-2u_i+u_{i-1}}{h^2}.
\]

于是得到
\[
\frac{d\boldsymbol U}{dt}=A\boldsymbol U,
\]

其中
\[
\boxed{
A=\frac{a}{2h}D_1+\frac{\sigma^2}{h^2}D_2
}
\]

且
\[
D_1=
\begin{pmatrix}
0&1& &\\
-1&0&1&\\
&\ddots&\ddots&\ddots\\
&&-1&0
\end{pmatrix},
\qquad
D_2=
\begin{pmatrix}
-2&1&&\\
1&-2&1&\\
&\ddots&\ddots&\ddots\\
&&1&-2
\end{pmatrix}.
\]

注意：\(A\) 是三对角矩阵，其非零元素最多只有 \(3N-2\) 个，而一般的 \(N\times N\) 矩阵有 \(N^2\) 个元素。

这正是有限差分法计算高效的重要原因。
### 3.2 时间离散：显式与隐式差分
设 \(t_n=n\tau\)，记 \(\boldsymbol U^n\approx\boldsymbol U(t_n)\)。
方法一：显式 Euler 法
采用前向差分
\[
\frac{\boldsymbol U^{n+1}-\boldsymbol U^n}{\tau}
=A\boldsymbol U^n.
\]

得到
\[
\boxed{
\boldsymbol U^{n+1}=(I+\tau A)\boldsymbol U^n.
}
\]

这就是我们之前推导的随机游走差分格式。

因为 \(I+\tau A\) 是稀疏矩阵，所以每个时间步只需要做一次稀疏矩阵与向量的乘法。

但缺点是需要满足稳定性条件。例如 \(a=0\) 时，必须有
\[
\tau\le\frac{h^2}{2\sigma^2}.
\]

当空间网格很密时，时间步长就必须很小。
方法二：隐式 Euler 法
这就是你记得的转化为求解线性方程组的方法。

我们将右边的 \(\boldsymbol U^n\) 替换为 \(\boldsymbol U^{n+1}\)：
\[
\frac{\boldsymbol U^{n+1}-\boldsymbol U^n}{\tau}
=A\boldsymbol U^{n+1}.
\]

整理得到
\[
\boxed{
(I-\tau A)\boldsymbol U^{n+1}
=\boldsymbol U^n.
}
\]

记 \(B=I-\tau A\)，就有
\[
\boxed{B\boldsymbol U^{n+1}=\boldsymbol U^n.}
\]

每个时间步求解一个线性方程组即可。

具体来说，记
\[
r=\frac{\sigma^2\tau}{h^2},
\qquad s=\frac{a\tau}{2h}.
\]

那么
\[
B=
\begin{pmatrix}
1+2r&-r-s&&&\\
-r+s&1+2r&-r-s&&\\
&-r+s&1+2r&\ddots&\\
&&\ddots&\ddots&-r-s\\
&&&-r+s&1+2r
\end{pmatrix}.
\]

稀疏矩阵 B 的结构（N = 20）
只需存储三条对角线，而不是完整的 N×N 矩阵。
由于 \(B\) 是三对角矩阵，可以使用 Thomas 算法（追赶法），每个时间步的计算复杂度为 \(O(N)\)，而不是一般高斯消元的 \(O(N^3)\)。

而且所有时间步使用同一个 \(B\)，可以预先做一次 LU 分解，再反复使用分解结果求解。
## 3.3 一个更好的方法：Crank–Nicolson 格式
如果希望同时获得较好的稳定性和精度，可以使用 Crank–Nicolson 方法。

它相当于在相邻两个时间层对空间算子取平均：

\[
\frac{\boldsymbol U^{n+1}-\boldsymbol U^n}{\tau}
=
\frac12 A(\boldsymbol U^{n+1}+\boldsymbol U^n).
\]

整理得到

\[
\boxed{
\left(I-\frac{\tau}{2}A\right)\boldsymbol U^{n+1}
=
\left(I+\frac{\tau}{2}A\right)\boldsymbol U^n.
}
\]

相比之下：
方法	时间精度	每步计算	稳定性
显式 Euler	一阶	稀疏矩阵乘法	条件稳定
隐式 Euler	一阶	稀疏线性求解	无条件稳定
Crank–Nicolson	二阶	稀疏线性求解	无条件稳定

这里的无条件稳定针对标准扩散问题的线性稳定性而言，不代表任意大时间步都能保持足够精度或解的非负性。

空间中心差分通常具有二阶精度，因此 Crank–Nicolson 格式的整体截断误差为
\[
O(\tau^2+h^2).
\]
把空间划成网格，用相邻点的差来近似导数，再一步步推进。准备做显式和隐式两种：显式直接更新数组；隐式每步解一个三对角线性系统。

显式部分可以用 **R/Python** 写参考版本，再用 **Rcpp** 优化。隐式部分能用上 **Eigen/RcppEigen 的稀疏矩阵和线性方程组求解**，Python 端也可以用 **SciPy** 核对结果。

可以比较网格加密后的误差、显式和隐式达到相同精度需要的时间，以及稀疏存储和重复使用矩阵分解能省多少内存与计算。矩阵首末行和边界值也需要自己处理。

## 4. 神经网络求解

这一部分想试试物理信息神经网络（PINN）：输入位置和时间，让网络输出解，用方程、初值和边界条件构造损失函数。

主要使用 **Python 和 PyTorch**，会涉及网络搭建、自动求导、损失函数和训练。**NumPy** 用来整理测试网格和比较结果。有余力可以试 **GPU**，看看批量训练和查询的耗时。

我对这部分还不太熟，想从一个小网络开始。可以比较两种边界处理：在损失里惩罚边界误差，或者通过函数构造直接保证边界为零。之后再看训练点数量、损失权重对精度的影响，并把训练时间和查询时间分开记录。

## 4.1 PINN的构造
仍考虑一维方程
\[
\begin{cases}
u_t(x,t)=a u_x(x,t)+\sigma^2u_{xx}(x,t),\\
u(x,0)=f_0(x)=e^{-x^2}.
\end{cases}
\]

为了便于数值计算，取
\[
x\in[-8,8],\qquad t\in[0,1],
\]

以及参数
\[
a=0.7,\qquad \sigma=1.
\]

我们在有限区间两端增加近似零边界条件：
\[
u(-8,t)=u(8,t)=0.
\]

由于初值在两端非常接近零，而且本例的计算时间较短，因此这种截断带来的误差很小。

前面已经推导出整个实数轴上的精确解：

\[
\boxed{
u_{\mathrm{exact}}(x,t)
=
\frac{1}{\sqrt{1+4\sigma^2t}}
\exp\left(
-\frac{(x+at)^2}{1+4\sigma^2t}
\right).
}
\]

我们可以利用它检验 PINN 的计算精度，并与有限差分法比较。
构造一个前馈神经网络
\[
u_\theta:\mathbb R^2\longrightarrow\mathbb R,
\]

其中输入是空间位置和时间 \((x,t)\)，输出是函数值 \(u_\theta(x,t)\)。

例如采用 4 个隐藏层、每层 64 个神经元的网络。

注意激活函数要使用 Tanh 而不是 ReLU ，原因：方程包含二阶导数 \(u_{xx}\)。

ReLU 网络关于输入的二阶经典导数几乎处处为零，不适合直接用于这种强形式 PDE 残差；Tanh 是光滑函数，可以进行高阶自动微分。

### 4.2 定义PDE残差

定义

\[
\boxed{
r_\theta(x,t)
=
\frac{\partial u_\theta}{\partial t}
-a\frac{\partial u_\theta}{\partial x}
-\sigma^2\frac{\partial^2u_\theta}{\partial x^2}.
}
\]

如果神经网络表示的是真正的 PDE 解，那么整个求解区域中都有
\[
r_\theta(x,t)=0.
\]

PINN 的思路就是从时空区域随机采样一些点，要求网络在这些点上尽量满足该方程。

设内部配点为
\[
\mathcal D_f=\{(x_i^f,t_i^f)\}_{i=1}^{N_f}.
\]

定义 PDE 损失：

\[
\boxed{
\mathcal L_{\mathrm{PDE}}(\theta)
=
\frac1{N_f}\sum_{i=1}^{N_f}
|r_\theta(x_i^f,t_i^f)|^2.
}
\]

这里的所有偏导数都使用 PyTorch 自动微分计算，而不是有限差分近似。

### 4.3 初始条件和边界条件

我们还需要约束网络满足初始条件。设初始配点为 \(\{x_i^0\}_{i=1}^{N_0}\)，定义

\[
\boxed{
\mathcal L_{\mathrm{IC}}(\theta)
=
\frac1{N_0}\sum_{i=1}^{N_0}
|u_\theta(x_i^0,0)-e^{-(x_i^0)^2}|^2.
}
\]

类似地，对边界点 \(\{t_i^b\}_{i=1}^{N_b}\)：

\[
\boxed{
\mathcal L_{\mathrm{BC}}(\theta)
=
\frac1{N_b}\sum_{i=1}^{N_b}
\left(
|u_\theta(-8,t_i^b)|^2+
|u_\theta(8,t_i^b)|^2
\right).
}
\]

### 4.4 最终优化问题

将三个损失函数相加：

\[
\boxed{
\min_\theta\;
\mathcal L(\theta)
=
\lambda_f\mathcal L_{\mathrm{PDE}}
+\lambda_0\mathcal L_{\mathrm{IC}}
+\lambda_b\mathcal L_{\mathrm{BC}}.
}
\]

其中 \(\lambda_f,\lambda_0,\lambda_b>0\) 是损失权重。

训练完成后，神经网络本身就是数值解：
\[
\boxed{\hat u(x,t)=u_{\hat\theta}(x,t).}
\]

PINN 与有限差分法有一个关键区别：它不需要在规则网格上定义全部未知数，也不需要在每个时间步推进解。 网络学习的是整个时空区域上的函数，因此可以直接输入任意 \((x,t)\) 来计算近似解。

不过，这并不意味着 PINN 总是比有限差分法高效。对于当前这样的一维线性抛物方程，稀疏有限差分通常更快、更稳定，也更容易获得高精度结果。

## 5. 其他可以尝试的内容

- 扩展到二维，观察图像或局部信号怎样移动、扩散，能进一步用上二维数组和图像卷积。
- 改变边界值，看看差分、随机游走和网络分别要怎样调整。
- 一次查询多个时刻，比较差分保存中间结果和训练后网络查询的成本。

## 大致怎么做

先把差分和边界处理写对，再加随机游走和 PINN。卷积作为差分的另一种实现来比较，并行先选一个后端。二维和 GPU 看进度再决定。

两个人可以一人侧重差分、Rcpp 和稀疏矩阵，另一人侧重随机模拟、并行和 PyTorch，具体分工可以再商量。

代码用 **Git/GitHub** 管理，保留运行参数、随机种子和原始记录，在 **README** 里说明运行方式和各自做的部分。报告里放解的图、误差、耗时和内存比较；随机方法多跑几次，看看结果波动。这样能对应课程要求里的实现、正确性验证、性能比较和复现。
