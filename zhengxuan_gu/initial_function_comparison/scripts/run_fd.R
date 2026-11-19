#!/usr/bin/env Rscript
# 从本实验统一生成的初始网格出发，调用方法三的同一套 Rcpp 求解器。
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
project <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg))))
args <- commandArgs(TRUE)
if (length(args) > 2L) stop("Usage: Rscript run_fd.R [initial_grids.csv] [output_directory]")
input <- if (length(args)) args[1] else file.path(project, "results/reference/initial_grids.csv")
out <- if (length(args) > 1L) args[2] else file.path(project, "results/fd")
if (!file.exists(input)) stop("Run scripts/make_reference.py to create ", input)
fd_dir <- file.path(dirname(project), "method_3_finite_difference")
source(file.path(fd_dir, "R/solver.R"))
load_fd_solver(fd_dir)

a <- 0.2
D <- 0.05
final_time <- 0.5
times <- c(0, 1/32, 1/8, final_time)
cases <- c("baseline", "center", "near", "compact")
methods <- c("explicit_euler", "implicit_euler", "crank_nicolson")
grids <- c(64L, 128L)
rounds <- 9L
target_batch_seconds <- 0.010
order_seed <- 2026L

initial_data <- read.csv(input, stringsAsFactors = FALSE)
stopifnot(all(c("case", "N", "x", "initial") %in% names(initial_data)),
          setequal(unique(initial_data$case), cases), setequal(unique(initial_data$N), grids))
configs <- expand.grid(case = cases, method = methods, N = grids, stringsAsFactors = FALSE)
configs$n <- as.integer(configs$N^2 / 16)  # D*dt/h^2 = 0.4 on both grids.

get_initial <- function(case, N) {
  values <- initial_data[initial_data$case == case & initial_data$N == N, ]
  values <- values[order(values$x), ]
  stopifnot(nrow(values) == N+1L, all(values$x == (0:N)/N),
            all(is.finite(values$initial)), all(abs(values$initial[c(1L, N+1L)]) < 1e-14))
  values$initial[c(1L, N+1L)] <- 0
  values$initial
}
initial <- Map(get_initial, configs$case, configs$N)

# 每个快照直接从同一初值推进；dt 与一次推进到 T 的步长完全相同。
# 128 段的输出仅抽取偶数节点，因此不需要空间或时间插值。
profiles <- vector("list", nrow(configs)*length(times))
k <- 0L
for (i in seq_len(nrow(configs))) {
  cfg <- configs[i, ]
  shared_nodes <- 1L + (0:64L)*(cfg$N/64L)
  for (t in times) {
    steps <- t/final_time*cfg$n
    stopifnot(steps == trunc(steps))
    u <- if (t == 0) initial[[i]] else {
      solve_fd(cfg$method, initial[[i]], a, D, T = t, n = steps)
    }
    stopifnot(all(is.finite(u)), all(u[c(1L, cfg$N+1L)] == 0),
              min(u) >= -1e-13, max(u) <= max(initial[[i]]) + 1e-13)
    k <- k + 1L
    profiles[[k]] <- data.frame(case = cfg$case, method = cfg$method, N = cfg$N,
                                 x = (0:64L)/64L, t, prediction = u[shared_nodes])
  }
}

# 各配置先暖机。每次完整求解重新准备缓冲区和必要的分解，并推进到 T。
measure <- function(i, batch) {
  benchmark_fd(configs$method[i], initial[[i]], a, D, T = final_time,
               n = configs$n[i], batch = batch)
}
batches <- integer(nrow(configs))
for (i in seq_len(nrow(configs))) {
  invisible(measure(i, 3L))
  pilot <- median(replicate(3L, measure(i, 5L)))
  stopifnot(is.finite(pilot), pilot > 0)
  batches[i] <- max(1L, as.integer(ceiling(target_batch_seconds/pilot)))
}
set.seed(order_seed)
measurements <- vector("list", rounds*nrow(configs))
k <- 0L
for (round_id in seq_len(rounds)) {
  for (i in sample.int(nrow(configs))) {
    per_solve_seconds <- measure(i, batches[i])
    k <- k + 1L
    measurements[[k]] <- data.frame(configs[i, ], round_id, measurement_order = k,
                                     batch = batches[i], per_solve_seconds,
                                     batch_seconds = per_solve_seconds*batches[i])
  }
}
raw <- do.call(rbind, measurements)
stopifnot(all(is.finite(raw$per_solve_seconds)), all(raw$per_solve_seconds > 0))
timings <- configs
for (i in seq_len(nrow(configs))) {
  keep <- raw$case == configs$case[i] & raw$method == configs$method[i] & raw$N == configs$N[i]
  samples <- raw$per_solve_seconds[keep]
  stopifnot(length(samples) == rounds)
  bounds <- quantile(samples, c(0.25, 0.75), names = FALSE)
  timings$solve_seconds[i] <- median(samples)
  timings$q25_seconds[i] <- bounds[1]
  timings$q75_seconds[i] <- bounds[2]
}

dir.create(out, recursive = TRUE, showWarnings = FALSE)
save_table <- function(data, name) {
  write.csv(data, file.path(out, name), row.names = FALSE, fileEncoding = "UTF-8")
}
save_table(do.call(rbind, profiles), "profiles.csv")
save_table(timings, "timings.csv")
save_table(raw, "timing_batches.csv")
writeLines(c("D=0.05, a=0.2, T=0.5, Dirichlet endpoints=0, single-thread float64.",
             "N=64 or 128; n=N^2/16; D*dt/h^2=0.4. Common output x=j/64.",
             "Snapshot times=0, 1/32, 1/8, 1/2. Timing covers one complete solve to T.",
             "C++ steady_clock: new buffers, factorization when applicable and time stepping.",
             "Excludes compilation, R dispatch and result wrapping, validation, stencil setup, initial sampling, scoring and I/O.",
             "Warm-up and pilot, then 9 randomly interleaved rounds. Order seed=2026; batch target=10 ms.",
             paste("Recorded UTC:", format(Sys.time(), tz = "UTC", usetz = TRUE)),
             paste("Rcpp", as.character(packageVersion("Rcpp"))), capture.output(sessionInfo())),
           file.path(out, "environment.txt"))
cat("Wrote", length(profiles)*65L, "predictions and", nrow(timings), "timing summaries.\n")
