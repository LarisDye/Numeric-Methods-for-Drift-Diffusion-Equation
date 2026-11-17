#include <Rcpp.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]

namespace {
// Thread-local MRG32k3a, with the same integer recurrence and scale as R's
// L'Ecuyer-CMRG generator (R src/main/RNG.c). Stream splitting remains in R,
// via parallel::nextRNGStream; a stream belongs to a logical block, never to
// an OpenMP thread. Signed R seed integers carry uint32_t state bit patterns.
struct CmrgState {
  std::uint32_t s[6];
  double uniform() {
    constexpr std::int64_t m1=4294967087LL, m2=4294944443LL;
    std::int64_t p1=(1403580LL*s[1]-810728LL*s[0])%m1;
    std::int64_t p2=(527612LL*s[5]-1370589LL*s[3])%m2;
    if(p1<0) p1+=m1;
    if(p2<0) p2+=m2;
    s[0]=s[1]; s[1]=s[2]; s[2]=static_cast<std::uint32_t>(p1);
    s[3]=s[4]; s[4]=s[5]; s[5]=static_cast<std::uint32_t>(p2);
    return static_cast<double>(p1>p2?p1-p2:p1-p2+m1)*2.328306549295727688e-10;
  }
};
}

// [[Rcpp::export]]
Rcpp::NumericVector walk_block_cpp(Rcpp::NumericVector f, int j, int n,
                                   Rcpp::NumericVector probs, int M) {
  const int N=f.size()-1;
  if(N<2 || j<0 || j>N || n<0 || M<1 || probs.size()!=3)
    Rcpp::stop("Invalid grid, query or budget");
  double pm=probs[0],p0=probs[1],pp=probs[2];
  if(!std::isfinite(pm+p0+pp) || pm<0 || p0<0 || pp<0 ||
     std::abs(pm+p0+pp-1)>1e-14) Rcpp::stop("Invalid probabilities");
  for(double x:f) if(!std::isfinite(x)) Rcpp::stop("Nonfinite initial value");
  if(std::abs(f[0])>1e-12 || std::abs(f[N])>1e-12) Rcpp::stop("Nonzero boundary");
  double s=0,s2=0,alive=0,steps=0;
  if(j>0 && j<N) for(int path=0;path<M;++path) {
    if((path&4095)==0) Rcpp::checkUserInterrupt();
    int pos=j;
    for(int t=0;t<n;++t) {
      double z=R::runif(0,1);
      if(z<pm) --pos; else if(z>=pm+p0) ++pos;
      ++steps;
      if(pos==0 || pos==N) break;
    }
    if(pos>0 && pos<N) {double y=f[pos];s+=y;s2+=y*y;++alive;}
  }
  return Rcpp::NumericVector::create(Rcpp::Named("M")=M,Rcpp::Named("sum_y")=s,
      Rcpp::Named("sum_y2")=s2,Rcpp::Named("alive")=alive,Rcpp::Named("sum_steps")=steps);
}

// [[Rcpp::export]]
Rcpp::List walk_blocks_openmp(Rcpp::NumericVector f, int n,
  Rcpp::NumericVector probs, Rcpp::IntegerVector jobs_j,
  Rcpp::IntegerVector counts, Rcpp::IntegerMatrix states, int threads) {
  // Validate and materialize every R object before entering the team region.
  if(f.size()<3 || f.size()>std::numeric_limits<int>::max() || n<0 ||
     probs.size()!=3 || threads<1 || jobs_j.size()>std::numeric_limits<int>::max())
    Rcpp::stop("Invalid grid, probabilities, steps or thread count");
  const int N=static_cast<int>(f.size())-1;
  const int B=static_cast<int>(jobs_j.size());
  if(counts.size()!=B || states.nrow()!=B || states.ncol()!=6)
    Rcpp::stop("One count and six CMRG state integers are required per logical block");
  const double pm=probs[0], p0=probs[1], pp=probs[2];
  if(!std::isfinite(pm+p0+pp) || pm<0 || p0<0 || pp<0 ||
     std::abs(pm+p0+pp-1)>1e-14) Rcpp::stop("Invalid probabilities");
  for(double x:f) if(!std::isfinite(x)) Rcpp::stop("Nonfinite initial value");
  if(std::abs(f[0])>1e-12 || std::abs(f[N])>1e-12)
    Rcpp::stop("Nonzero boundary");
#ifndef _OPENMP
  if(threads>1) Rcpp::stop("This binary was compiled without OpenMP");
#endif
  const double* fp=f.begin();
  const int* jp=jobs_j.begin();
  const int* cp=counts.begin();
  const int* sp=states.begin();
  const std::size_t stride=static_cast<std::size_t>(B);
  std::vector<CmrgState> initial(B);
  for(int b=0;b<B;++b) {
    if(jp[b]<0 || jp[b]>N || cp[b]<1) Rcpp::stop("Invalid block query or path count");
    bool first_nonzero=false, second_nonzero=false;
    for(int k=0;k<6;++k) {
      const std::uint32_t value=static_cast<std::uint32_t>(sp[b+stride*k]);
      if(value>=(k<3?4294967087ULL:4294944443ULL)) Rcpp::stop("Invalid CMRG state");
      initial[b].s[k]=value;
      if(k<3) first_nonzero=first_nonzero || value!=0;
      else second_nonzero=second_nonzero || value!=0;
    }
    if(!first_nonzero || !second_nonzero) Rcpp::stop("An all-zero CMRG component is invalid");
  }
  Rcpp::NumericMatrix stats(B,5);
  double* out=stats.begin();
  int actual_threads=1;
  Rcpp::checkUserInterrupt();
  // The caller uses a fixed 16 logical blocks per query. Each block writes
  // only its own row, consumes its own stream and sums paths in fixed order.
  // There are no R/Rcpp calls and no floating-point reductions in this region.
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if(threads>1)
#endif
  {
#ifdef _OPENMP
#pragma omp single
    actual_threads=omp_get_num_threads();
#pragma omp for schedule(static)
#endif
    for(int b=0;b<B;++b) {
      CmrgState rng=initial[b];
      double sum=0, sum2=0, alive=0, steps=0;
      const int j=jp[b], M=cp[b];
      if(j>0 && j<N) for(int path=0;path<M;++path) {
        int pos=j;
        for(int t=0;t<n;++t) {
          const double z=rng.uniform();
          if(z<pm) --pos; else if(z>=pm+p0) ++pos;
          ++steps;
          if(pos==0 || pos==N) break;
        }
        if(pos>0 && pos<N) {
          const double y=fp[pos]; sum+=y; sum2+=y*y; ++alive;
        }
      }
      out[b]=M; out[b+stride]=sum; out[b+2*stride]=sum2;
      out[b+3*stride]=alive; out[b+4*stride]=steps;
    }
  }
  Rcpp::checkUserInterrupt();
  Rcpp::colnames(stats)=Rcpp::CharacterVector::create("M","sum_y","sum_y2","alive","sum_steps");
  return Rcpp::List::create(Rcpp::Named("stats")=stats,
                            Rcpp::Named("threads_used")=actual_threads);
}

// [[Rcpp::export]]
Rcpp::List openmp_info_cpp(int requested=1) {
  if(requested<1) Rcpp::stop("Requested thread count must be positive");
  int actual=1, maximum=1, version=0;
  bool compiled=false;
#ifdef _OPENMP
  compiled=true; version=_OPENMP; maximum=omp_get_max_threads();
#pragma omp parallel num_threads(requested) if(requested>1)
  {
#pragma omp single
    actual=omp_get_num_threads();
  }
#endif
#ifdef __VERSION__
  const char* compiler=__VERSION__;
#else
  const char* compiler="unknown";
#endif
  return Rcpp::List::create(Rcpp::Named("compiled")=compiled,
    Rcpp::Named("_OPENMP")=version, Rcpp::Named("max_threads")=maximum,
    Rcpp::Named("actual_threads")=actual, Rcpp::Named("compiler")=compiler);
}
