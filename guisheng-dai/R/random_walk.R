# 串行可读参考：每个查询点从 j 出发，逐步回溯到初值，触边即吸收。
# f 的 R 下标是 pos+1；返回统计量而非保存每条完整路径。
walk_block_r <- function(f,j,n,probs,M) {
  check_probs(probs);N<-length(f)-1L;s<-s2<-alive<-steps<-0
  if(j>0L&&j<N) for(path in seq_len(M)) {
    pos<-j
    if(n>0L) for(t in seq_len(n)) {
      z<-runif(1)
      if(z<probs[1]) pos<-pos-1L else if(z>=probs[1]+probs[2]) pos<-pos+1L
      # 触边那一步也属于实际工作量；之后不继续消耗随机数。
      steps<-steps+1
      if(pos==0L||pos==N) break
    }
    # 被吸收样本的收益是零，仍计入分母 M；alive 只用于存活率诊断。
    if(pos>0L&&pos<N){y<-f[pos+1L];s<-s+y;s2<-s2+y*y;alive<-alive+1}
  }
  c(M=M,sum_y=s,sum_y2=s2,alive=alive,sum_steps=steps)
}
# 小网格独立参照：内部转移矩阵 Q 的边缘行和可小于 1，缺失质量表示吸收。
# 它只核验离散链期望/方差，不是本项目的第三种正式有限差分求解方法。
discrete_reference <- function(ctx) {
  check_probs(ctx$probs);m<-ctx$N-1L;p<-ctx$probs
  Q<-diag(p[2],m)
  if(m>1L){Q[cbind(2:m,1:(m-1))]<-p[1];Q[cbind(1:(m-1),2:m)]<-p[3]}
  # 分别传播 f 与 f^2，再以 E[Y^2]-E[Y]^2 求收益方差。
  u<-ctx$f[2:ctx$N];v<-u^2
  if(ctx$T>0) for(t in seq_len(ctx$n)){u<-as.vector(Q%*%u);v<-as.vector(Q%*%v)}
  list(mean=c(0,u,0),variance=c(0,pmax(v-u*u,0),0),Q=Q)
}
# Independent logical blocks carry RNG state; thread scheduling cannot change it.
# 正式包装：M 是每个查询点的预算，seeds 固定为每点 16 项逻辑块初态。
# R 串行打包和归约，只有 walk_blocks_openmp 内部创建 OpenMP 线程组。
solve_mc <- function(ctx,query_j,M,seeds,threads=1L) {
  check_probs(ctx$probs)
  stopifnot(length(threads)==1L,is.finite(threads),threads>=1L,threads==as.integer(threads))
  stopifnot(M>=2L,M==as.integer(M),all(query_j>=0L&query_j<=ctx$N))
  # 零时刻已有精确初值，不模拟路径，也不启动并行计算。
  if(ctx$T==0) return(list(value=ctx$f[query_j+1L],se=rep(0,length(query_j)),
    alive=ifelse(query_j>0&query_j<ctx$N,M,0),steps=rep(0,length(query_j)),
    blocks=data.frame(),threads_used=1L))
  # 商平均分配，余数从前面的块补齐；逻辑块数不随 threads 改变。
  counts<-rep(M%/%16L,16);if(M%%16L)counts[seq_len(M%%16L)]<-counts[seq_len(M%%16L)]+1L
  # 小预算可能产生零路径块：略过其计算，但保留种子列表的固定布局。
  nq<-length(query_j);keep<-rep(counts>0L,nq)
  query_id<-rep(seq_len(nq),each=16L)[keep]
  block_id<-rep(seq_len(16L),nq)[keep]
  jobs_j<-as.integer(rep(query_j,each=16L)[keep])
  budgets<-as.integer(rep(counts,nq)[keep])
  stopifnot(length(seeds)==nq*16L)
  selected<-seeds[keep]
  # .Random.seed 首项是 RNG 类型标记；C++ 只接收后六项 CMRG 状态。
  states<-do.call(rbind,lapply(selected,function(z)as.integer(z$state[-1L])))
  ans<-walk_blocks_openmp(ctx$f,ctx$n,ctx$probs,jobs_j,budgets,states,as.integer(threads))
  if(ans$threads_used!=threads)stop("OpenMP team size differs from request")
  stats<-ans$stats
  # stats 各列为 M、sum_y、sum_y2、alive、sum_steps；记录种子 ID 便于复查。
  blocks<-data.frame(query_id=query_id,j=jobs_j,block_id=block_id,
    seed_id=vapply(selected,function(z)z$id,character(1)),M_block=stats[,1],
    sum_y=stats[,2],sum_y2=stats[,3],alive=stats[,4],sum_steps=stats[,5])
  vals<-se<-alive<-steps<-numeric(nq)
  # The same ascending block order for every thread count, no floating reduction.
  # 固定加法顺序避免并行浮点 reduction 随线程数改变末位结果。
  for(q in seq_len(nq)){
    a<-numeric(5)
    for(k in which(query_id==q))a<-a+stats[k,]
    # 全部 M 条样本（含吸收的零收益）参与均值与样本方差。
    vals[q]<-a[2]/M
    # se 是单次估计的标准误；不是多批独立估计之间的样本标准差。
    se[q]<-sqrt(max(0,(a[3]-a[2]^2/M)/(M-1))/M)
    alive[q]<-a[4];steps[q]<-a[5]
  }
  list(value=vals,se=se,alive=alive,steps=steps,blocks=blocks,threads_used=ans$threads_used)
}
