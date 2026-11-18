list(a = 0.2, D = 0.05, T = 0.5, profile = "main", kernel_N = 512L, 
    kernel_threads = 2L, cutoffs = c(1, 2, 3, 4, 6, 8), parallel_N = c(512L, 
    8192L), parallel_c = 6, parallel_threads = c(1L, 2L, 4L, 
    8L), reps = 9L, batch_target = 0.2, calibration_target = 0.04, 
    max_batch = 16384L, cpu = "Intel Core i7-11800H", physical_cores = 8L, 
    logical_cores = 16L, timing_scope = c(t_input = "Full prepare_gaussian call", 
    t_solve = "Full solve_gaussian call with a prepared input", 
    t_reused_new_input = "Full prepare_gaussian followed by solve_gaussian, independently measured"
    ), excluded = "R startup, compilation, warmup, calibration, gc before batch, verification, file I/O", 
    baseline = "Same solve_gaussian wrapper, one thread; OpenMP parallel region has if(threads>1)", 
    statistics = "9 batch means per configuration; median and quartiles across batches; wall-clock seconds", 
    speedup = "Ratio of median single-thread time to median p-thread time for the same task", 
    query = "All N+1 nodes including endpoints; exact reference and boundary checks outside timing")
