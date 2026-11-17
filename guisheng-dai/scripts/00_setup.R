source("R/bootstrap.R")
compile_kernels()
cat("Compiled Rcpp and RcppArmadillo kernels successfully.\n")
print(sessionInfo())
