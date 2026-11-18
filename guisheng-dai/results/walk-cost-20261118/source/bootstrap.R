# 仅保留一层显式 OpenMP：禁止动态调整/嵌套，BLAS 库保持单线程。
# PASSIVE 让等待中的线程尽量休眠，避免空转干扰其他测量。
Sys.setenv(OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1", OMP_WAIT_POLICY="PASSIVE",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
options(warn=1)
# 准备阶段检查依赖；求解器和性能计时中不临时安装软件包。
for (p in c("Rcpp", "RcppArmadillo")) if (!requireNamespace(p,quietly=TRUE)) stop("Install required package: ",p)
# sourceCpp 生成可直接从 R 调用的 C++ 包装，缓存可复用已编译的核心。
compile_kernels <- function() {
  dir.create(".cache/rcpp",recursive=TRUE,showWarnings=FALSE)
  for(f in c("src/random_walk.cpp","src/gaussian_convolution.cpp"))
    Rcpp::sourceCpp(f,cacheDir=".cache/rcpp",rebuild=FALSE,showOutput=FALSE,verbose=FALSE)
}
# 相对路径以 guisheng-dai/ 为根；函数定义加载后，再统一加载原生核心。
# local=.GlobalEnv 使实验脚本和交互式调用共享同一组函数名。
load_project <- function(compile=TRUE) {
  for(f in c("problem","random_walk","gaussian_convolution","parallel","rng","metrics")) source(file.path("R",paste0(f,".R")),local=.GlobalEnv)
  if(compile) compile_kernels()
}
