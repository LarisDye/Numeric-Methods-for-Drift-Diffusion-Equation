# 实验配置：这里只记录参数和重复策略，实际 PDE 参数还需与 runner 的默认调用一致。
# threads 是 C++ OpenMP 线程数；R 层不创建工作进程。
experiment_config <- function(mode = "full", threads = 2L) {
  list(mode=mode, threads=threads, backend="C++ OpenMP", a=.2, D=.05, T=.5, seed=20261116L,
       # 独立 MC 批次用于统计波动；性能计时轮次与之分开。
       mc_reps=if(mode=="smoke") 3L else 30L,
       timing_reps=if(mode=="smoke") 2L else 7L,
       # 短任务连续运行多次再除以次数，减轻单次墙钟分辨率的影响。
       batch_target=if(mode=="smoke") .04 else .2,
       parallel_degrees=if(mode=="smoke") 1L else c(1L,2L,4L))
}
