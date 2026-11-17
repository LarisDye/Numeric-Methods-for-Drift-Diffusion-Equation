walk_block_r <- function(f,j,n,probs,M) {
  check_probs(probs);N<-length(f)-1L;s<-s2<-alive<-steps<-0
  if(j>0L&&j<N) for(path in seq_len(M)) {
    pos<-j
    if(n>0L) for(t in seq_len(n)) {
      z<-runif(1)
      if(z<probs[1]) pos<-pos-1L else if(z>=probs[1]+probs[2]) pos<-pos+1L
      steps<-steps+1
      if(pos==0L||pos==N) break
    }
    if(pos>0L&&pos<N){y<-f[pos+1L];s<-s+y;s2<-s2+y*y;alive<-alive+1}
  }
  c(M=M,sum_y=s,sum_y2=s2,alive=alive,sum_steps=steps)
}
discrete_reference <- function(ctx) {
  check_probs(ctx$probs);m<-ctx$N-1L;p<-ctx$probs
  Q<-diag(p[2],m)
  if(m>1L){Q[cbind(2:m,1:(m-1))]<-p[1];Q[cbind(1:(m-1),2:m)]<-p[3]}
  u<-ctx$f[2:ctx$N];v<-u^2
  if(ctx$T>0) for(t in seq_len(ctx$n)){u<-as.vector(Q%*%u);v<-as.vector(Q%*%v)}
  list(mean=c(0,u,0),variance=c(0,pmax(v-u*u,0),0),Q=Q)
}
# Independent logical blocks carry RNG state; thread scheduling cannot change it.
solve_mc <- function(ctx,query_j,M,seeds,threads=1L) {
  check_probs(ctx$probs)
  stopifnot(length(threads)==1L,is.finite(threads),threads>=1L,threads==as.integer(threads))
  stopifnot(M>=2L,M==as.integer(M),all(query_j>=0L&query_j<=ctx$N))
  if(ctx$T==0) return(list(value=ctx$f[query_j+1L],se=rep(0,length(query_j)),
    alive=ifelse(query_j>0&query_j<ctx$N,M,0),steps=rep(0,length(query_j)),
    blocks=data.frame(),threads_used=1L))
  counts<-rep(M%/%16L,16);if(M%%16L)counts[seq_len(M%%16L)]<-counts[seq_len(M%%16L)]+1L
  nq<-length(query_j);keep<-rep(counts>0L,nq)
  query_id<-rep(seq_len(nq),each=16L)[keep]
  block_id<-rep(seq_len(16L),nq)[keep]
  jobs_j<-as.integer(rep(query_j,each=16L)[keep])
  budgets<-as.integer(rep(counts,nq)[keep])
  stopifnot(length(seeds)==nq*16L)
  selected<-seeds[keep]
  states<-do.call(rbind,lapply(selected,function(z)as.integer(z$state[-1L])))
  ans<-walk_blocks_openmp(ctx$f,ctx$n,ctx$probs,jobs_j,budgets,states,as.integer(threads))
  if(ans$threads_used!=threads)stop("OpenMP team size differs from request")
  stats<-ans$stats
  blocks<-data.frame(query_id=query_id,j=jobs_j,block_id=block_id,
    seed_id=vapply(selected,function(z)z$id,character(1)),M_block=stats[,1],
    sum_y=stats[,2],sum_y2=stats[,3],alive=stats[,4],sum_steps=stats[,5])
  vals<-se<-alive<-steps<-numeric(nq)
  # The same ascending block order for every thread count, no floating reduction.
  for(q in seq_len(nq)){
    a<-numeric(5)
    for(k in which(query_id==q))a<-a+stats[k,]
    vals[q]<-a[2]/M
    se[q]<-sqrt(max(0,(a[3]-a[2]^2/M)/(M-1))/M)
    alive[q]<-a[4];steps[q]<-a[5]
  }
  list(value=vals,se=se,alive=alive,steps=steps,blocks=blocks,threads_used=ans$threads_used)
}
