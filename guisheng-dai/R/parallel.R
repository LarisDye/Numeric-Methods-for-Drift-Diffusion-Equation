# R orchestrates experiments serially. Only C++ solvers create OpenMP teams.
# A future R reference parallel implementation must use R multiprocessing.
# 本次 R 只串行组织任务，真正的并行区位于两个 C++ 求解器中。
initialize_openmp <- function(threads) {
  stopifnot(length(threads)==1L,is.finite(threads),threads>=1L,threads==as.integer(threads))
  # 在原生并行区内测实际组大小，不能仅凭请求值声称使用了若干线程。
  info<-openmp_info_cpp(as.integer(threads))
  if(!isTRUE(info$compiled) || info$actual_threads!=threads)
    stop("OpenMP unavailable or actual team size differs from request")
  info
}
set_context <- function(ctx,threads=1L) {
  # Compatibility helper for serial tests; solvers now receive ctx explicitly.
  # There is no process pool, cluster export or IPC.
  # 兼容旧测试入口：不更新全局状态、不导出对象，也不创建 PSOCK 工作进程。
  invisible(ctx)
}
