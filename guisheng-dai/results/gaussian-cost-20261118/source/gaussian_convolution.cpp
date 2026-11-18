// 仅禁止 Armadillo 自行创建内部 OpenMP 团队；下方显式编写的 OpenMP
// 输出窗口并行仍然启用，从而避免外层窗口并行与内层向量运算嵌套。
#define ARMA_DONT_USE_OPENMP
#include <RcppArmadillo.h>
#include <omp.h>
#include <chrono>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
// 使用单调时钟测量耗时，不受系统日期/时间校准影响；返回值单位为秒，
// 起点没有日历含义，仅应对同一进程中的两次读数作差。
// [[Rcpp::export]]
double steady_seconds_cpp() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// 准备一次完整输入：f 含区间 [0,1] 上的 N+1 个节点（下标 0..N）；
// a、D、T 为漂移系数、扩散系数和目标时刻，K 为核的半宽（W=2K+1）。
// 返回奇延拓数据 padded、离散高斯权重 weights、恢复原变量的 scale，
// 以及核尺寸和权重和；这些只读数组可供多个查询窗口复用。
// [[Rcpp::export]]
Rcpp::List prepare_gaussian_cpp(const arma::vec& f, double a, double D, double T, int K) {
  int N=f.n_elem-1;
  if(N<2 || !f.is_finite() || !std::isfinite(a+D+T) || D<=0 || T<=0 || K<0)
    Rcpp::stop("Invalid Gaussian inputs");
  if(std::abs(f[0])>1e-12 || std::abs(f[N])>1e-12) Rcpp::stop("Nonzero boundary");
  double h=1.0/N, beta=a/(2*D),gamma=a*a/(4*D);
  arma::vec x=arma::linspace<arma::vec>(0,1,N+1);
  // 指数变换 v=e^(beta*x+gamma*t)u 将漂移扩散方程化为热方程：
  // 初值 g=e^(beta*x)f，卷积后乘 e^(-beta*x-gamma*T) 恢复 u。
  arma::vec g=arma::exp(beta*x)%f;g[0]=g[N]=0;
  arma::vec weights(2*K+1),padded(N+2*K+1),scale=arma::exp(-beta*x-gamma*T);
  // weights[l+K]=h*G_T(l*h)，h 来自积分求和的网格宽度。
  // 有限核截掉的尾部质量保留为截断误差，不能再归一化为权重和 1；
  // 重新归一化会改变所逼近的热核积分，并掩盖核宽实验中的截断效应。
  for(int l=-K;l<=K;++l)
    weights[l+K]=h/std::sqrt(4*arma::datum::pi*D*T)*std::exp(-(l*h)*(l*h)/(4*D*T));
  for(int t=0;t<N+2*K+1;++t) {
    // padded[t] 对应原网格整数下标 t-K。双重取模把负下标也映射到
    // 0..2N-1；超过 N 时反射并变号，构成周期 2N 的奇延拓。
    // 这同时处理左右零 Dirichlet 边界；K>N 时也可重复跨越边界。
    int r=((t-K)%(2*N)+2*N)%(2*N);
    padded[t]=r<=N?g[r]:-g[2*N-r];
  }
  return Rcpp::List::create(Rcpp::Named("padded")=padded,Rcpp::Named("weights")=weights,
     Rcpp::Named("scale")=scale,Rcpp::Named("K")=K,Rcpp::Named("W")=2*K+1,
     Rcpp::Named("kernel_sum")=arma::accu(weights));
}
// 串行窗口求值：query_j 使用原网格的 0..N 下标，可只查询部分节点。
// padded[j..j+W-1] 对应原网格 j-K..j+K；高斯核对称，
// 因此与该窗口直接点乘等价于卷积，无需另做核的方向翻转。
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
  return out; // 保留端点钳零前的舍入残差，供 R 层核验边界处理是否正确。
}

// 仿照图像卷积逐个计算输出窗口，只在彼此独立的查询节点之间并行；
// 每个窗口仍读取完整奇延拓输入，不把线程任务的分界误当成物理边界。
// 与串行函数使用相同权重和点乘顺序，返回 value 以及实际 threads_used。
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
  // 在主线程中完成校验与 R 对象分配。工作线程只读取输入并写各自输出项，
  // 不调用 R/Rcpp API，也不共享可变累加器；即使查询下标重复仍无写入竞争。
  #pragma omp parallel num_threads(threads) if(threads>1) shared(used)
  {
    #pragma omp single
    // single 由团队中的一个线程记录真实团队规模，供正式实验核验并行生效。
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
