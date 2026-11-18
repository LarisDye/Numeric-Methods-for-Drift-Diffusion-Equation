# 主初值具有闭式解析解；poly 初值用来观察宽核下的网格求积误差。
# x 可以是单个位置或位置向量，指数因子与后面的消漂移变换一致。
initial <- function(x,profile="main",a=.2,D=.05) {
  z <- if(profile=="poly") x*(1-x) else sin(pi*x)
  exp(-a*x/(2*D))*z
}
# 连续参照：main 用解析公式，poly 用截断正弦谱级数。
# highest 只控制 poly 的最高奇数波数；正式实验比较 31/63 检查截断稳定性。
exact <- function(x,T=.5,a=.2,D=.05,profile="main",highest=63L) {
  if(T==0) return(initial(x,profile,a,D))
  z <- if(profile=="poly") {
    k<-seq(1L,highest,by=2L)
    # outer 形成所有位置与波数的组合，矩阵乘法对每个位置完成级数求和。
    as.vector(sin(outer(x,k*pi)) %*% (8/(pi^3*k^3)*exp(-D*(k*pi)^2*T)))
  } else sin(pi*x)*exp(-D*pi^2*T)
  exp(-a*x/(2*D)-a*a*T/(4*D))*z
}
# N 是空间段数：共 N+1 节点，数学/C++ 节点 j 在 R 中存于 j+1。
# n 是路径最多步数。此函数建立上下文；概率合法性由 MC 使用前另行检查。
make_problem <- function(N,n=1L,profile="main",a=.2,D=.05,T=.5) {
  stopifnot(length(N)==1,N>=2,N==as.integer(N),D>0,T>=0,n>=1,n==as.integer(n))
  # 显式钳零消除 sin(pi) 的浮点残差，使输入严格符合齐次边界约定。
  x<-seq(0,1,length.out=N+1L);f<-initial(x,profile,a,D);f[c(1L,N+1L)]<-0
  # h=1/N，dt=T/n；probs 依次是左移、停留、右移的概率。
  r<-D*(T/n)*N^2;b<-a*(T/n)*N/2
  list(N=as.integer(N),n=as.integer(n),x=x,f=f,a=a,D=D,T=T,h=1/N,dt=T/n,
       probs=c(r-b,1-2*r,r+b),profile=profile)
}
# 拒绝非法概率，不裁剪或归一化，否则将悄悄改变离散随机过程。
check_probs <- function(p) {
  if(length(p)!=3 || any(!is.finite(p)) || any(p<0) || abs(sum(p)-1)>1e-14)
    stop("Invalid random walk probabilities; refine spatial/time grid")
  invisible(TRUE)
}
# 只允许落在网格上的物理位置，不插值；返回从 0 开始的节点编号。
query_indices <- function(x,N) {
  stopifnot(all(is.finite(x)),all(x>=0&x<=1),max(abs(x*N-round(x*N)))<1e-10)
  as.integer(round(x*N))
}
