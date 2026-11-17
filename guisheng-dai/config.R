experiment_config <- function(mode = "full", threads = 2L) {
  list(mode=mode, threads=threads, backend="C++ OpenMP", a=.2, D=.05, T=.5, seed=20261116L,
       mc_reps=if(mode=="smoke") 3L else 30L,
       timing_reps=if(mode=="smoke") 2L else 7L,
       batch_target=if(mode=="smoke") .04 else .2,
       parallel_degrees=if(mode=="smoke") 1L else c(1L,2L,4L))
}
