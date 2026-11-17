# R orchestrates experiments serially. Only C++ solvers create OpenMP teams.
# A future R reference parallel implementation must use R multiprocessing.
initialize_openmp <- function(threads) {
  stopifnot(length(threads)==1L,is.finite(threads),threads>=1L,threads==as.integer(threads))
  info<-openmp_info_cpp(as.integer(threads))
  if(!isTRUE(info$compiled) || info$actual_threads!=threads)
    stop("OpenMP unavailable or actual team size differs from request")
  info
}
set_context <- function(ctx,threads=1L) {
  # Compatibility helper for serial tests; solvers now receive ctx explicitly.
  # There is no process pool, cluster export or IPC.
  invisible(ctx)
}
