# R 奇延拓参考：先映射到周期 2N，后半周期反射并取负号。
# 输入节点 j 可为负数或超过 N；R 读取 g 时需将数学下标再加 1。
odd_extension <- function(g,j) {
  N<-length(g)-1L;r<-((j%%(2*N))+2*N)%%(2*N)
  ifelse(r<=N,g[pmin(r,2*N-r)+1L],-g[pmin(r,2*N-r)+1L])
}
# 串行逐项求和用于核验 Armadillo 滑窗；不负责正式性能测量。
# padded 的第 j+1 项对应窗口左端 j-K，窗口共 length(weights) 项。
gaussian_block_r <- function(padded,weights,scale,query_j) {
  vapply(query_j,function(j) {
    ans<-0;for(t in seq_along(weights))ans<-ans+padded[j+t]*weights[t]
    scale[j+1L]*ans
  },numeric(1))
}
# c 决定物理覆盖：半径为 c*sqrt(2DT)，除以 h 后向上取整得到 K。
# 借用 make_problem 生成网格/初值，不使用其随机游走概率和 n=1。
prepare_gaussian <- function(N,c,profile="main",a=.2,D=.05,T=.5) {
  ctx<-make_problem(N,1L,profile,a,D,T)
  K<-as.integer(ceiling(c*sqrt(2*D*T)*N))
  # 底层热核公式只接受 T>0；零时刻在 R 中直接走初值快捷路径。
  if(T==0) return(list(N=N,f=ctx$f,T=0,K=0,W=1,profile=profile))
  z<-prepare_gaussian_cpp(ctx$f,a,D,T,K)
  # c_eff 是向上取整后的实际标准差覆盖倍数，可能略大于指定 c。
  c(z,list(N=N,h=1/N,c=c,c_eff=K/N/sqrt(2*D*T),a=a,D=D,T=T,profile=profile))
}
# 只计算 query_j 指定的节点；每个 C++ 任务读取完整填充的一个窗口。
solve_gaussian <- function(ctx,query_j,threads=1L) {
  stopifnot(length(threads)==1L,is.finite(threads),threads>=1L,threads==as.integer(threads))
  if(ctx$T==0) return(list(value=ctx$f[query_j+1L],boundary_before=0,threads_used=1L))
  out<-gaussian_openmp_arma(ctx$padded,ctx$weights,ctx$scale,as.integer(query_j),as.integer(threads))
  if(out$threads_used!=threads)stop("OpenMP team size differs from request")
  values<-out$value
  # 必须在钳零之前保存原始边界残差，否则人为赋零会掩盖延拓错误。
  # 若本次未查询端点，则诊断为 NA，不额外计算端点。
  bound<-which(query_j==0L|query_j==ctx$N)
  before<-if(length(bound))max(abs(values[bound])) else NA_real_
  values[bound]<-0
  list(value=values,boundary_before=before,threads_used=out$threads_used)
}
