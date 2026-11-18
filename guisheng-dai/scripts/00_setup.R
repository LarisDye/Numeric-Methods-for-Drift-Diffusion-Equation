# 准备阶段编译并打印实际 R 环境；不运行实验、不产生性能结论。
# 请从 guisheng-dai/ 目录调用，相对路径和缓存均以此为根。
source("R/bootstrap.R")
compile_kernels()
cat("Compiled Rcpp and RcppArmadillo kernels successfully.\n")
print(sessionInfo())
