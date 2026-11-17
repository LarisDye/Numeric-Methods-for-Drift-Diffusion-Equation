Sys.setenv(OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1", OMP_WAIT_POLICY="PASSIVE",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
options(warn=1)
for (p in c("Rcpp", "RcppArmadillo")) if (!requireNamespace(p,quietly=TRUE)) stop("Install required package: ",p)
compile_kernels <- function() {
  dir.create(".cache/rcpp",recursive=TRUE,showWarnings=FALSE)
  for(f in c("src/random_walk.cpp","src/gaussian_convolution.cpp"))
    Rcpp::sourceCpp(f,cacheDir=".cache/rcpp",rebuild=FALSE,showOutput=FALSE,verbose=FALSE)
}
load_project <- function(compile=TRUE) {
  for(f in c("problem","random_walk","gaussian_convolution","parallel","rng","metrics")) source(file.path("R",paste0(f,".R")),local=.GlobalEnv)
  if(compile) compile_kernels()
}
