list(a = 0.2, D = 0.05, T = 0.5, query_x = 0.5, q = 1L, grid_M = 16384L, 
    grid_threads = 2L, parallel_N = 64L, parallel_n = 512L, parallel_M = 65536L, 
    parallel_threads = c(1L, 2L, 4L, 8L), reps = 9L, batch_target = 0.2, 
    max_batch = 256L, seed = 20261118L, cpu = "Intel Core i7-11800H", 
    physical_cores = 8L, logical_cores = 16L, timing_scope = "Warmed full R solve_mc call, including packing, validation, C++ and reduction", 
    excluded = "Compilation, context and seed preparation, warmup, gc, checks, file I/O", 
    serial_baseline = "Identical C++ source compiled without the OpenMP plugin; identical R wrapper")
