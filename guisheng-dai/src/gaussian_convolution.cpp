#define ARMA_DONT_USE_OPENMP
#include <RcppArmadillo.h>
#include <omp.h>
#include <chrono>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
double steady_seconds_cpp() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// [[Rcpp::export]]
Rcpp::List prepare_gaussian_cpp(const arma::vec& f, double a, double D, double T, int K) {
  int N=f.n_elem-1;
  if(N<2 || !f.is_finite() || !std::isfinite(a+D+T) || D<=0 || T<=0 || K<0)
    Rcpp::stop("Invalid Gaussian inputs");
  if(std::abs(f[0])>1e-12 || std::abs(f[N])>1e-12) Rcpp::stop("Nonzero boundary");
  double h=1.0/N, beta=a/(2*D),gamma=a*a/(4*D);
  arma::vec x=arma::linspace<arma::vec>(0,1,N+1);
  arma::vec g=arma::exp(beta*x)%f;g[0]=g[N]=0;
  arma::vec weights(2*K+1),padded(N+2*K+1),scale=arma::exp(-beta*x-gamma*T);
  for(int l=-K;l<=K;++l)
    weights[l+K]=h/std::sqrt(4*arma::datum::pi*D*T)*std::exp(-(l*h)*(l*h)/(4*D*T));
  for(int t=0;t<N+2*K+1;++t) {
    int r=((t-K)%(2*N)+2*N)%(2*N);
    padded[t]=r<=N?g[r]:-g[2*N-r];
  }
  return Rcpp::List::create(Rcpp::Named("padded")=padded,Rcpp::Named("weights")=weights,
     Rcpp::Named("scale")=scale,Rcpp::Named("K")=K,Rcpp::Named("W")=2*K+1,
     Rcpp::Named("kernel_sum")=arma::accu(weights));
}
// [[Rcpp::export]]
Rcpp::NumericVector gaussian_block_arma(const arma::vec& padded,const arma::vec& weights,
  const arma::vec& scale,Rcpp::IntegerVector query_j) {
  int N=scale.n_elem-1, W=weights.n_elem;
  if(W<1 || W%2!=1 || padded.n_elem!=scale.n_elem+W-1) Rcpp::stop("Invalid window");
  Rcpp::NumericVector out(query_j.size());
  for(int q=0;q<query_j.size();++q) {
    int j=query_j[q];if(j<0||j>N)Rcpp::stop("Invalid query");
    out[q]=scale[j]*arma::dot(padded.subvec(j,j+W-1),weights);
  }
  return out; // Preserve pre-clamp endpoint residuals for validation.
}

// Parallelise independent output windows, as in the course image-convolution
// example. Disable Armadillo's internal OpenMP above to avoid nested teams.
// [[Rcpp::export]]
Rcpp::List gaussian_openmp_arma(const arma::vec& padded, const arma::vec& weights,
                              const arma::vec& scale, Rcpp::IntegerVector query_j,
                              int threads = 1) {
  const int N=scale.n_elem-1, W=weights.n_elem, nq=query_j.size();
  if(N<2 || W<1 || W%2!=1 || padded.n_elem!=scale.n_elem+W-1 ||
     !padded.is_finite() || !weights.is_finite() || !scale.is_finite())
    Rcpp::stop("Invalid finite Gaussian window");
  if(threads<1) Rcpp::stop("threads must be positive");
  for(int j:query_j) if(j==NA_INTEGER || j<0 || j>N) Rcpp::stop("Invalid query");
  Rcpp::NumericVector out(nq);
  const int* query=query_j.begin();
  double* values=out.begin();
  int used=1;
  // Worker threads only read input and write their own output entries.
  #pragma omp parallel num_threads(threads) if(threads>1) shared(used)
  {
    #pragma omp single
    used=omp_get_num_threads();
    #pragma omp for schedule(static)
    for(int q=0;q<nq;++q) {
      const int j=query[q];
      values[q]=scale[j]*arma::dot(padded.subvec(j,j+W-1),weights);
    }
  }
  Rcpp::checkUserInterrupt();
  return Rcpp::List::create(Rcpp::Named("value")=out,
                            Rcpp::Named("threads_used")=used);
}
