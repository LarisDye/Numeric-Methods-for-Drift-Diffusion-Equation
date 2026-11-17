initial <- function(x,profile="main",a=.2,D=.05) {
  z <- if(profile=="poly") x*(1-x) else sin(pi*x)
  exp(-a*x/(2*D))*z
}
exact <- function(x,T=.5,a=.2,D=.05,profile="main",highest=63L) {
  if(T==0) return(initial(x,profile,a,D))
  z <- if(profile=="poly") {
    k<-seq(1L,highest,by=2L)
    as.vector(sin(outer(x,k*pi)) %*% (8/(pi^3*k^3)*exp(-D*(k*pi)^2*T)))
  } else sin(pi*x)*exp(-D*pi^2*T)
  exp(-a*x/(2*D)-a*a*T/(4*D))*z
}
make_problem <- function(N,n=1L,profile="main",a=.2,D=.05,T=.5) {
  stopifnot(length(N)==1,N>=2,N==as.integer(N),D>0,T>=0,n>=1,n==as.integer(n))
  x<-seq(0,1,length.out=N+1L);f<-initial(x,profile,a,D);f[c(1L,N+1L)]<-0
  r<-D*(T/n)*N^2;b<-a*(T/n)*N/2
  list(N=as.integer(N),n=as.integer(n),x=x,f=f,a=a,D=D,T=T,h=1/N,dt=T/n,
       probs=c(r-b,1-2*r,r+b),profile=profile)
}
check_probs <- function(p) {
  if(length(p)!=3 || any(!is.finite(p)) || any(p<0) || abs(sum(p)-1)>1e-14)
    stop("Invalid random walk probabilities; refine spatial/time grid")
  invisible(TRUE)
}
query_indices <- function(x,N) {
  stopifnot(all(is.finite(x)),all(x>=0&x<=1),max(abs(x*N-round(x*N)))<1e-10)
  as.integer(round(x*N))
}
