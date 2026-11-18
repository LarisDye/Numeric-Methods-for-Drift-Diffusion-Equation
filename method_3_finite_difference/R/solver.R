# source this file, then call load_fd_solver() with the method directory.
.fd_native <- new.env(parent = environment())

load_fd_solver <- function(method_dir = ".") {
  if (!requireNamespace("Rcpp", quietly = TRUE)) stop("The Rcpp package is required.")
  source_file <- normalizePath(file.path(method_dir, "src", "solver.cpp"), mustWork = TRUE)
  Rcpp::sourceCpp(source_file, env = .fd_native)
  invisible(NULL)
}

positive_integer <- function(value, name) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value < 1 || value > .Machine$integer.max || value != trunc(value)) {
    stop(name, " must be a positive integer.", call. = FALSE)
  }
  as.integer(value)
}

check_fd_inputs <- function(method, f_grid, a, D, T) {
  if (!exists("fd_solve_cpp", envir = .fd_native, inherits = FALSE))
    stop("Call load_fd_solver() first.", call. = FALSE)
  if (!is.character(method) || length(method) != 1L || is.na(method))
    stop("method must be a single method name.", call. = FALSE)
  if (!is.numeric(f_grid) || !is.null(dim(f_grid)))
    stop("f_grid must be a numeric vector including both endpoints.", call. = FALSE)
  for (v in list(a, D, T)) {
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v))
      stop("a, D and T must be finite numeric scalars.", call. = FALSE)
  }
}

solve_fd <- function(method, f_grid, a = 0.2, D = 0.05, T = 0.5, n) {
  check_fd_inputs(method, f_grid, a, D, T)
  .fd_native$fd_solve_cpp(method, as.double(f_grid), a, D, T, positive_integer(n, "n"))
}

benchmark_fd <- function(method, f_grid, a = 0.2, D = 0.05, T = 0.5, n,
                         batch = 1L) {
  check_fd_inputs(method, f_grid, a, D, T)
  .fd_native$fd_benchmark_cpp(method, as.double(f_grid), a, D, T,
                              positive_integer(n, "n"), positive_integer(batch, "batch"))
}
