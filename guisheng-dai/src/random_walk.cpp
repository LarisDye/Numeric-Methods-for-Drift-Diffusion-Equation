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
// 每个逻辑块拥有一个 MRG32k3a 状态，递推式与缩放常数对应 R 的
// L'Ecuyer-CMRG（R src/main/RNG.c）。R 层用 parallel::nextRNGStream 分流，
// 随机流绑定逻辑块而非线程，所以改变 OpenMP 线程数不会改变块内的抽样序列。
// R 的有符号整数承载 32 位状态的位模式；转成 uint32_t 恢复对应的无符号值。
struct CmrgState {
  std::uint32_t s[6];
  double uniform() {
    // 两个三阶递推分量分别取模，再组合成 (0, 1) 内的均匀数；
    // 采用 64 位中间整数，避免系数与 32 位状态相乘时溢出。
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

// 串行基准：f 含 N+1 个节点（C++ 下标 0..N），j 为起点下标，n 为步数上限，
// probs 按“左、停、右”排列，M 为路径总数。这里使用 R 的 RNG，供串行一致性核验。
// 返回总路径数、一阶/二阶贡献和、存活数、实际步数；R 层据此计算均值和标准误。
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
  // 边界起点直接贡献 0。内部路径首次触及任一端点后立即吸收，
  // 不再产生随机数或累计步数，也不会从另一侧重新进入区间。
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
  // 被吸收路径的贡献是 0，但仍计入 M；只除以 alive 会错误地估计条件期望。
  return Rcpp::NumericVector::create(Rcpp::Named("M")=M,Rcpp::Named("sum_y")=s,
      Rcpp::Named("sum_y2")=s2,Rcpp::Named("alive")=alive,Rcpp::Named("sum_steps")=steps);
}

// 并行批处理：jobs_j[b] 与 counts[b] 指定第 b 个逻辑块的起点和路径数；
// states 是 B×6 的 CMRG 初态矩阵，threads 是请求的 OpenMP 线程数。
// 返回 B×5 的逐块统计矩阵和实测线程数，由 R 按固定块顺序汇总。
// [[Rcpp::export]]
Rcpp::List walk_blocks_openmp(Rcpp::NumericVector f, int n,
  Rcpp::NumericVector probs, Rcpp::IntegerVector jobs_j,
  Rcpp::IntegerVector counts, Rcpp::IntegerMatrix states, int threads) {
  // 所有输入校验、R 对象分配和指针获取均在主线程完成；
  // 并行区域中不调用 R/Rcpp API，避免非线程安全的解释器或 RNG 访问。
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
      // R 矩阵按列存储：第 b 行第 k 列的线性偏移是 b+B*k，
      // 不能按 C++ 常见的逐行布局读取，否则六个种子状态会错配。
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
  // R 调用方固定每个查询点划分 16 个逻辑块；线程仅负责调度这些块。
  // 每块独占 RNG、局部累加器和结果矩阵的一行，块内路径始终顺序累加。
  // 此处不使用浮点 reduction，避免线程分组改变求和顺序而影响可复现性。
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if(threads>1)
#endif
  {
#ifdef _OPENMP
#pragma omp single
    // 在线程团队内测量实际线程数，不能把请求值当作运行时实际值。
    actual_threads=omp_get_num_threads();
#pragma omp for schedule(static)
#endif
    for(int b=0;b<B;++b) {
      // 复制该逻辑块的状态到线程栈，初态数组保持只读；
      // 同一线程执行多个块时，各块的随机流也互不延续。
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
      // stats 同样按列存储；不同 b 只写自己的行，不发生输出位置竞争。
      out[b]=M; out[b+stride]=sum; out[b+2*stride]=sum2;
      out[b+3*stride]=alive; out[b+4*stride]=steps;
    }
  }
  Rcpp::checkUserInterrupt();
  Rcpp::colnames(stats)=Rcpp::CharacterVector::create("M","sum_y","sum_y2","alive","sum_steps");
  return Rcpp::List::create(Rcpp::Named("stats")=stats,
                            Rcpp::Named("threads_used")=actual_threads);
}

// 核验编译与运行环境：_OPENMP 是编译器支持的规范版本日期宏，
// max_threads 反映运行时的默认线程数设置；actual_threads 来自真实创建的测试团队。
// 未启用 OpenMP 编译时仍可返回串行信息，供 R 层给出明确诊断。
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
