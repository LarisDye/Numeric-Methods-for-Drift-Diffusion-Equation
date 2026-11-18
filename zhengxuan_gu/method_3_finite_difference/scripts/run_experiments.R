#!/usr/bin/env Rscript
# 在任意目录运行。可选参数是输出目录，例如 Rscript scripts/run_experiments.R /tmp/fd-run
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
module_dir <- dirname(dirname(normalizePath(sub("^--file=", "", file_arg))))
args <- commandArgs(TRUE)
if (length(args) > 1L) stop("Usage: Rscript run_experiments.R [output_directory]")
out_dir <- if (length(args)) args[1] else file.path(module_dir, "results")
source(file.path(module_dir, "R", "solver.R"))
load_fd_solver(module_dir)

a <- 0.2
D <- 0.05
final_time <- 0.5
methods <- c("explicit_euler", "implicit_euler", "crank_nicolson")
grids <- c(16L, 32L, 64L, 128L)
rounds <- 9L
target_seconds <- 0.01
order_seed <- 2026L
rmse <- function(x) sqrt(mean(x^2))
# 解析终态只在误差评价时调用。初始两端显式赋值以避免 sin(pi) 的浮点舍入。
initial <- function(x) {
  y <- exp(-a * x / (2 * D)) * sin(pi * x)
  y[c(1, length(y))] <- 0
  y
}
decay <- D * pi^2 + a^2 / (4 * D)
evaluate <- function(method, N, n) {
  x <- (0:N) / N
  f <- initial(x)
  u <- solve_fd(method, f, a = a, D = D, T = final_time, n = n)
  truth <- exp(-decay * final_time) * f
  stopifnot(length(u) == N + 1L, all(is.finite(u)), u[1] == 0, u[N + 1L] == 0)
  data.frame(method, N, n, x, prediction = u, exact = truth, error = u - truth)
}

configs <- expand.grid(N = grids, method = methods, stringsAsFactors = FALSE)
configs$n <- as.integer(configs$N^2 / 16L) # 对给定 T、D，此时 D*dt/h² = 0.4
profiles <- vector("list", nrow(configs))
grid_rows <- vector("list", nrow(configs))
for (i in seq_len(nrow(configs))) {
  N <- configs$N[i]
  n <- configs$n[i]
  method <- configs$method[i]
  p <- evaluate(method, N, n)
  profiles[[i]] <- p
  err <- p$error[2:N]
  grid_rows[[i]] <- data.frame(method, N, n, h = 1/N, dt = final_time/n,
                               r = D * (final_time/n) * N^2,
                               rmse = rmse(err), max_abs_error = max(abs(err)),
                               common15_rmse = rmse(p$error[1L + (1:15) * (N/16L)]))
}

# 所有批次从同一组初值重新求解。先预热并确定每批次数，再随机交错测量各配置。
initial_grids <- lapply(configs$N, function(N) initial((0:N)/N))
measure <- function(i, batch) {
  benchmark_fd(configs$method[i], initial_grids[[i]], a = a, D = D,
               T = final_time, n = configs$n[i], batch = batch)
}
batch_sizes <- integer(nrow(configs))
for (i in seq_len(nrow(configs))) {
  invisible(measure(i, 3L))
  pilot <- median(replicate(3L, measure(i, 5L)))
  stopifnot(is.finite(pilot), pilot > 0)
  batch_sizes[i] <- max(1L, as.integer(ceiling(target_seconds / pilot)))
}
set.seed(order_seed)
timings <- vector("list", rounds * nrow(configs))
k <- 0L
for (round_id in seq_len(rounds)) {
  for (i in sample.int(nrow(configs))) {
    k <- k + 1L
    seconds <- measure(i, batch_sizes[i])
    timings[[k]] <- data.frame(method = configs$method[i], N = configs$N[i], n = configs$n[i],
                               round_id, measurement_order = k, batch = batch_sizes[i],
                               batch_ms = 1000 * seconds * batch_sizes[i],
                               per_solve_ms = 1000 * seconds)
  }
}
timing_df <- do.call(rbind, timings)
grid_df <- do.call(rbind, grid_rows)
grid_df$order <- NA_real_
for (method in methods) {
  rows <- which(grid_df$method == method)
  grid_df$order[rows[-1L]] <- -diff(log2(grid_df$rmse[rows]))
}
for (i in seq_len(nrow(grid_df))) {
  times <- timing_df$per_solve_ms[timing_df$method == grid_df$method[i] &
                                  timing_df$N == grid_df$N[i]]
  stopifnot(length(times) == rounds, all(times > 0))
  grid_df$median_ms[i] <- median(times)
  bounds <- quantile(times, c(0.25, 0.75), names = FALSE)
  grid_df$q25_ms[i] <- bounds[1]
  grid_df$q75_ms[i] <- bounds[2]
}

# 固定 N：通过线性系统 U'=LU 的矩阵指数，把空间误差与时间误差逐点分开。
# 下对角对应左邻点 (D/h²-a/(2h))，上对角对应右邻点。两端零值无需进入右端向量。
N <- 64L
h <- 1/N
operator <- diag(-2*D/h^2, N - 1L)
j <- seq_len(N - 2L)
operator[cbind(j+1L, j)] <- D/h^2 - a/(2*h)
operator[cbind(j, j+1L)] <- D/h^2 + a/(2*h)
f <- initial((0:N)/N)[2:N]
eig <- eigen(operator)
space_solution <- drop(eig$vectors %*% (exp(final_time * eig$values) * solve(eig$vectors, f)))
space_error <- space_solution - exp(-decay * final_time) * f
time_rows <- list()
for (method in methods) {
  for (n in c(256L, 512L, 1024L)) {
    p <- evaluate(method, N, n)
    time_error <- p$prediction[2:N] - space_solution
    combined <- p$error[2:N]
    time_rows[[length(time_rows) + 1L]] <- data.frame(
      method, N, n, dt = final_time/n, rmse = rmse(combined), max_abs_error = max(abs(combined)),
      space_rmse = rmse(space_error), time_rmse = rmse(time_error),
      opposite_sign_nodes = sum(space_error * time_error < 0))
  }
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_table <- function(x, name) {
  write.csv(x, file.path(out_dir, name), row.names = FALSE, na = "", fileEncoding = "UTF-8")
}
write_table(grid_df, "grid_summary.csv")
write_table(timing_df, "timings.csv")
write_table(do.call(rbind, profiles), "terminal_profiles.csv")
write_table(do.call(rbind, time_rows), "time_refinement.csv")

hardware <- character(0)
if (Sys.info()[["sysname"]] == "Darwin" && nzchar(Sys.which("system_profiler"))) {
  info <- system2("system_profiler", "SPHardwareDataType", stdout = TRUE, stderr = FALSE)
  hardware <- trimws(grep("Chip:|Processor Name:|Total Number of Cores:|Memory:", info, value = TRUE))
}
writeLines(c(paste("Recorded (UTC):", format(Sys.time(), tz = "UTC", usetz = TRUE)), hardware,
             "Domain=[0,1]; a=0.2; D=0.05; T=0.5; f=exp(-2*x)*sin(pi*x); endpoints=0",
             "Precision=float64; solver_threads=1; nodes x_j=j/N for j=0,...,N",
             paste("Timing rounds:", rounds, "  order seed:", order_seed, "  target batch ms:", 1000*target_seconds),
             "One C++ steady_clock batch / batch size; includes per-solve allocation, implicit factorization and stepping.",
             "R/C++ marshalling, validation, basic stencil coefficients, initial function sampling, compilation, reference, error metrics and I/O are outside timed batches.",
             paste("Rcpp:", as.character(packageVersion("Rcpp"))), capture.output(sessionInfo())),
           file.path(out_dir, "environment.txt"))
print(grid_df[, c("method", "N", "n", "rmse", "order", "median_ms")], row.names = FALSE)
cat("Results:", normalizePath(out_dir), "\n")
