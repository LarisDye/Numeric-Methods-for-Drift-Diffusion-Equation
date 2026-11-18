#!/usr/bin/env Rscript
# 固定 64 段空间网格，只改变隐式格式走到 T 的时间步数。
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
method_dir <- dirname(dirname(normalizePath(script)))
source(file.path(method_dir, "R", "solver.R"))
load_fd_solver(method_dir)

N <- 64L
a <- 0.2
D <- 0.05
final_time <- 0.5
x <- (0:N) / N
initial <- exp(-a*x/(2*D)) * sin(pi*x)
initial[c(1L, N+1L)] <- 0
interior <- 2:N
exact <- initial[interior] * exp(-(D*pi^2 + a^2/(4*D))*final_time)
configs <- rbind(
  data.frame(n = 256L, method = "explicit_euler"),
  expand.grid(n = c(4L, 8L, 16L, 32L, 64L, 256L),
              method = c("implicit_euler", "crank_nicolson"), stringsAsFactors = FALSE)
)

rows <- lapply(seq_len(nrow(configs)), function(i) {
  n <- configs$n[i]
  method <- configs$method[i]
  u <- solve_fd(method, initial, a = a, D = D, T = final_time, n = n)
  stopifnot(length(u) == N + 1L, all(is.finite(u)))
  errors <- u[interior] - exact
  data.frame(method, N, n, dt = final_time/n, rmse = sqrt(mean(errors^2)),
             max_abs_error = max(abs(errors)), boundary_max_error = max(abs(u[c(1L, N+1L)])))
})
results <- do.call(rbind, rows)
stopifnot(all(results$boundary_max_error == 0))

output_dir <- file.path(method_dir, "results")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(results, file.path(output_dir, "large_step_check.csv"), row.names = FALSE)
print(results, row.names = FALSE)
