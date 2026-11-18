# Run with Rscript --vanilla tests/test_solver.R (or pass the absolute path).
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
method_dir <- dirname(dirname(normalizePath(script)))
source(file.path(method_dir, "R", "solver.R"))
load_fd_solver(method_dir)

schemes <- c("explicit_euler", "implicit_euler", "crank_nicolson")
close_to <- function(actual, expected, tolerance = 1e-12) {
  stopifnot(length(actual) == length(expected),
            all(is.finite(actual)), max(abs(actual - expected)) < tolerance)
}
rejects <- function(expr) stopifnot(inherits(try(expr, silent = TRUE), "try-error"))

# A left-adjacent impulse catches reversed drift, periodic wrap and in-place updates.
impulse <- c(0, 1, 0, 0, 0)
close_to(solve_fd("explicit_euler", impulse, T = 0.1, n = 1),
         c(0, 0.84, 0.04, 0, 0))

# Independent dense R oracle for both implicit schemes and the explicit stencil.
# A nonsymmetric initial grid, both signs of drift, and varying time horizons
# exercise the first/last interior rows and repeated use of the same factors.
theta <- c(explicit_euler = 0, implicit_euler = 1, crank_nicolson = 0.5)
initial <- c(0, 0.17, -0.4, 0.8, 1.1, 0)
for (a in c(-0.2, 0.2)) {
  N <- length(initial) - 1L
  h <- 1/N
  identity <- diag(N-1L)
  shift <- row(identity) - col(identity)
  derivative <- ((shift == -1L) - (shift == 1L))/(2*h)
  laplacian <- ((abs(shift) == 1L) - 2*identity)/h^2
  dt <- 0.02
  L <- dt*(a*derivative + 0.05*laplacian)
  for (method in schemes) {
    weight <- theta[[method]]
    expected <- initial[2:N]
    for (k in seq_len(5)) {
      expected <- solve(identity-weight*L, (identity+(1-weight)*L) %*% expected)
      close_to(solve_fd(method, initial, a, T = k*dt, n = k), c(0, expected, 0))
    }
  }
}

for (method in schemes) {
  close_to(solve_fd(method, numeric(17), n = 16), numeric(17))
  # N=2 leaves a single interior row and no off-diagonal elements.
  r <- 0.05*0.1/(1/2)^2
  w <- theta[[method]]
  close_to(solve_fd(method, c(0, 1, 0), T = 0.1, n = 1),
           c(0, (1-2*(1-w)*r)/(1+2*w*r), 0))
  stopifnot(benchmark_fd(method, impulse, T = 0.1, n = 1, batch = 5) > 0)

  # Refinement follows dt proportional to h^2, using the analytic eigenmode.
  errors <- sapply(c(16, 32, 64), function(N) {
    x <- seq(0, 1, length.out = N+1)
    f <- exp(-2*x)*sin(pi*x)
    y <- solve_fd(method, f, n = N^2/16)
    stopifnot(y[1] == 0, y[N+1] == 0, min(y) >= 0, max(y) <= max(f))
    truth <- f*exp(-(0.05*pi^2+0.2)*0.5)
    sqrt(mean((y[2:N]-truth[2:N])^2))
  })
  ratios <- errors[1:2]/errors[2:3]
  stopifnot(all(ratios > 3.5 & ratios < 4.5))
}

rejects(solve_fd("explicit_euler", numeric(65), n = 128))
rejects(solve_fd("explicit_euler", c(1, 0, 0), n = 10))
rejects(solve_fd("implicit_euler", c(0, NaN, 0), n = 10))
rejects(solve_fd("implicit_euler", impulse, n = 1.5))
rejects(solve_fd("crank_nicolson", impulse, D = -0.1, n = 1))
rejects(solve_fd("unknown", impulse, n = 1))
rejects(benchmark_fd("explicit_euler", impulse, n = 1, batch = 0))
cat("Passed: one-step stencil, dense reference, boundaries, small grids, timing, convergence and input checks.\n")
