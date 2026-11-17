odd_extension <- function(g,j) {
  N<-length(g)-1L;r<-((j%%(2*N))+2*N)%%(2*N)
  ifelse(r<=N,g[pmin(r,2*N-r)+1L],-g[pmin(r,2*N-r)+1L])
}
gaussian_block_r <- function(padded,weights,scale,query_j) {
  vapply(query_j,function(j) {
    ans<-0;for(t in seq_along(weights))ans<-ans+padded[j+t]*weights[t]
    scale[j+1L]*ans
  },numeric(1))
}
prepare_gaussian <- function(N,c,profile="main",a=.2,D=.05,T=.5) {
  ctx<-make_problem(N,1L,profile,a,D,T)
  K<-as.integer(ceiling(c*sqrt(2*D*T)*N))
  if(T==0) return(list(N=N,f=ctx$f,T=0,K=0,W=1,profile=profile))
  z<-prepare_gaussian_cpp(ctx$f,a,D,T,K)
  c(z,list(N=N,h=1/N,c=c,c_eff=K/N/sqrt(2*D*T),a=a,D=D,T=T,profile=profile))
}
solve_gaussian <- function(ctx,query_j,threads=1L) {
  stopifnot(length(threads)==1L,is.finite(threads),threads>=1L,threads==as.integer(threads))
  if(ctx$T==0) return(list(value=ctx$f[query_j+1L],boundary_before=0,threads_used=1L))
  out<-gaussian_openmp_arma(ctx$padded,ctx$weights,ctx$scale,as.integer(query_j),as.integer(threads))
  if(out$threads_used!=threads)stop("OpenMP team size differs from request")
  values<-out$value
  bound<-which(query_j==0L|query_j==ctx$N)
  before<-if(length(bound))max(abs(values[bound])) else NA_real_
  values[bound]<-0
  list(value=values,boundary_before=before,threads_used=out$threads_used)
}
