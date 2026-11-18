# Run from guisheng-dai/: Rscript --vanilla scripts/08_benchmark_gaussian_cost.R
# Reuse the existing solver without modifying the numerical algorithm.
args <- commandArgs(trailingOnly=TRUE)
opt <- function(name, default) {
  k <- match(paste0("--", name), args)
  if (!is.na(k) && k < length(args)) args[k+1L] else default
}
runid <- opt("run-id", "gaussian-cost-20261118")
outdir <- file.path("results", runid)
if (dir.exists(outdir)) stop("Run directory exists; choose a new run-id")
source("R/bootstrap.R"); load_project()
for (sub in c("raw", "summary", "source"))
  dir.create(file.path(outdir, sub), recursive=TRUE, showWarnings=FALSE)

cfg <- list(a=.2, D=.05, T=.5, profile="main", kernel_N=512L,
  kernel_threads=2L, cutoffs=c(1,2,3,4,6,8), parallel_N=c(512L,8192L),
  parallel_c=6, parallel_threads=c(1L,2L,4L,8L), reps=9L,
  batch_target=.2, calibration_target=.04, max_batch=16384L,
  cpu="Intel Core i7-11800H", physical_cores=8L, logical_cores=16L,
  timing_scope=c(t_input="Full prepare_gaussian call",
    t_solve="Full solve_gaussian call with a prepared input",
    t_reused_new_input="Full prepare_gaussian followed by solve_gaussian, independently measured"),
  excluded="R startup, compilation, warmup, calibration, gc before batch, verification, file I/O",
  baseline="Same solve_gaussian wrapper, one thread; OpenMP parallel region has if(threads>1)",
  statistics="9 batch means per configuration; median and quartiles across batches; wall-clock seconds",
  speedup="Ratio of median single-thread time to median p-thread time for the same task",
  query="All N+1 nodes including endpoints; exact reference and boundary checks outside timing")
dput(cfg, file.path(outdir, "config.R")); saveRDS(cfg, file.path(outdir, "config.rds"))
capture.output(sessionInfo(), file=file.path(outdir, "environment.txt"))
cat("\nStarted (Asia/Shanghai):", format(Sys.time(), tz="Asia/Shanghai"),
    "\n", paste(names(Sys.info()), Sys.info(), collapse="\n"),
    "\n", paste(c("OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS", "OMP_WAIT_POLICY",
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"),
                  Sys.getenv(c("OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS", "OMP_WAIT_POLICY",
                               "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")), collapse="\n"),
    "\nPhysical / logical cores:", parallel::detectCores(logical=FALSE), parallel::detectCores(),
    file=file.path(outdir, "environment.txt"), append=TRUE)
srcfiles <- c("src/gaussian_convolution.cpp", "src/random_walk.cpp", "R/gaussian_convolution.R",
  "R/problem.R", "R/bootstrap.R", "R/metrics.R", "R/parallel.R",
  "scripts/08_benchmark_gaussian_cost.R")
for (f in srcfiles) file.copy(f, file.path(outdir, "source", basename(f)), overwrite=FALSE)
write_csv(data.frame(path=srcfiles, md5=unname(tools::md5sum(srcfiles))),
          file.path(outdir, "source_hashes.csv"))
teams <- lapply(cfg$parallel_threads, initialize_openmp)
saveRDS(teams, file.path(outdir, "compiler_and_teams.rds"))
capture.output(teams, file=file.path(outdir, "compiler_and_teams.txt"))

rows <- list(); checks <- list(); accuracy <- list()
logmsg <- function(msg) {
  line <- paste(format(Sys.time(), tz="Asia/Shanghai", format="%H:%M:%S"), msg)
  cat(line, "\n"); flush.console()
  cat(line, "\n", file=file.path(outdir, "progress.log"), append=TRUE)
}
check <- function(name, ok, max_difference=NA_real_) {
  if (!isTRUE(ok)) stop("FAILED: ", name)
  checks[[length(checks)+1L]] <<- data.frame(check=name, passed=TRUE,
    max_difference=max_difference)
}
checkpoint <- function() {
  if(length(rows)) write_csv(do.call(rbind, rows), file.path(outdir, "raw", "timings.csv"))
  write_csv(do.call(rbind, checks), file.path(outdir, "raw", "checks.csv"))
  if(length(accuracy)) write_csv(do.call(rbind, accuracy), file.path(outdir, "raw", "accuracy.csv"))
  saveRDS(list(timings=rows, checks=checks, accuracy=accuracy),
          file.path(outdir, "raw", "records.rds"))
}
make_row <- function(sweep,N,c,p)
  data.frame(sweep=sweep,N=as.integer(N),c=as.numeric(c),p=as.integer(p),
             stringsAsFactors=FALSE)
configs <- rbind(
  make_row("kernel_width",cfg$kernel_N,cfg$cutoffs,cfg$kernel_threads),
  make_row("parallel_small",cfg$parallel_N[1L],cfg$parallel_c,cfg$parallel_threads),
  make_row("parallel_large",cfg$parallel_N[2L],cfg$parallel_c,cfg$parallel_threads))
configs$q <- configs$N+1L
configs$K <- as.integer(ceiling(configs$c*sqrt(2*cfg$D*cfg$T)*configs$N))
configs$W <- 2L*configs$K+1L
configs$config_id <- with(configs,paste(sweep,paste0("N",N),paste0("c",c),paste0("p",p),sep="_"))
write_csv(configs,file.path(outdir,"configurations.csv"))
contexts <- lapply(seq_len(nrow(configs)), function(i)
  prepare_gaussian(configs$N[i],configs$c[i],cfg$profile,cfg$a,cfg$D,cfg$T))
jobs <- lapply(seq_len(nrow(configs)), function(i) {
  z <- configs[i,]; ctx <- contexts[[i]]; j <- as.integer(0:z$N)
  prepare <- function() prepare_gaussian(z$N,z$c,cfg$profile,cfg$a,cfg$D,cfg$T)
  solve <- function() solve_gaussian(ctx,j,z$p)
  total <- function() solve_gaussian(prepare(),j,z$p)
  list(t_input=prepare,t_solve=solve,t_reused_new_input=total)
})

# Independent R summation verifies the window convention before timing.
small <- prepare_gaussian(16L,3,cfg$profile,cfg$a,cfg$D,cfg$T)
small_j <- as.integer(0:16L)
rref <- gaussian_block_r(small$padded,small$weights,small$scale,small_j)
cppref <- gaussian_block_arma(small$padded,small$weights,small$scale,small_j)
check("small_R_Armadillo_window", max(abs(rref-cppref))<1e-12, max(abs(rref-cppref)))
for (p in cfg$parallel_threads) {
  res <- solve_gaussian(small,small_j,p)
  check(paste0("small_wrapper_p",p), max(abs(res$value-rref))<1e-12,
        max(abs(res$value-rref)))
}
# Cross-thread comparisons use identical input and exactly the same public wrapper.
for (i in seq_len(nrow(configs))) {
  z <- configs[i,]; j <- as.integer(0:z$N); ctx <- contexts[[i]]
  baseline <- solve_gaussian(ctx,j,1L)
  ans <- jobs[[i]]$t_solve()
  d <- max(abs(ans$value-baseline$value))
  check(paste0("output_",z$config_id), identical(ans$value,baseline$value),d)
  check(paste0("threads_",z$config_id),ans$threads_used==z$p)
  check(paste0("finite_",z$config_id),all(is.finite(ans$value)))
  check(paste0("boundary_",z$config_id),all(ans$value[c(1L,z$q)]==0) && ans$boundary_before<1e-12,
        ans$boundary_before)
  fresh <- jobs[[i]]$t_reused_new_input()
  check(paste0("new_input_",z$config_id),identical(fresh$value,ans$value),max(abs(fresh$value-ans$value)))
  err <- rmse(ans$value,exact(j/z$N,cfg$T,cfg$a,cfg$D,cfg$profile))
  accuracy[[length(accuracy)+1L]] <- data.frame(z,rmse=err,
    boundary_before=ans$boundary_before,actual_threads=ans$threads_used,max_difference_baseline=d)
}
wide8 <- prepare_gaussian(512L,8); wide10 <- prepare_gaussian(512L,10)
wide_diff <- max(abs(solve_gaussian(wide8,0:512,1L)$value-solve_gaussian(wide10,0:512,1L)$value))
check("wide_reference_c8_c10",wide_diff<1e-12,wide_diff)
checkpoint()

batch_time <- function(fun,A) {
  t <- clock()
  for (a in seq_len(A)) ans <- fun()
  list(seconds=clock()-t, result=ans)
}
calibrate <- function(fun) {
  for (warm in 1:3) fun()
  A <- 1L
  repeat {
    pilot <- batch_time(fun,A)$seconds
    if (pilot>=cfg$calibration_target || A>=cfg$max_batch) break
    A <- min(cfg$max_batch,2L*A)
  }
  max(1L,min(cfg$max_batch,as.integer(ceiling(A*cfg$batch_target/max(pilot,1e-6)))))
}
phases <- c("t_input","t_solve","t_reused_new_input")
batches <- matrix(NA_integer_,nrow(configs),length(phases),dimnames=list(configs$config_id,phases))
logmsg("Compilation and correctness checks complete; calibrating batches before official timing")
for (i in seq_len(nrow(configs))) for (phase in phases)
  batches[i,phase] <- calibrate(jobs[[i]][[phase]])
write_csv(data.frame(config_id=rownames(batches),batches,check.names=FALSE),
          file.path(outdir,"batch_sizes.csv"))
logmsg("Official timing begins: all configuration and phase orders rotate each round")
runstart <- clock()
for (group in unique(configs$sweep)) {
  ids <- which(configs$sweep==group)
  for (r in seq_len(cfg$reps)) {
    order <- ids[((seq_along(ids)+r-2L)%%length(ids))+1L]
    for (i in order) {
      z <- configs[i,]
      phase_order <- phases[((seq_along(phases)+r+i-3L)%%length(phases))+1L]
      for (phase in phase_order) {
        gc(FALSE)
        A <- batches[i,phase]
        measured <- batch_time(jobs[[i]][[phase]],A)
        if (phase!="t_input") {
          ans <- measured$result
          check(paste0("timing_team_",z$config_id,"_",phase,"_r",r),ans$threads_used==z$p)
          check(paste0("timing_boundary_",z$config_id,"_",phase,"_r",r),
                ans$boundary_before<1e-12 && all(ans$value[c(1L,z$q)]==0),ans$boundary_before)
        }
        rows[[length(rows)+1L]] <- data.frame(z,rep=r,phase=phase,batch_A=A,
          batch_elapsed=measured$seconds,seconds=measured$seconds/A)
      }
    }
    checkpoint(); logmsg(paste(group,"round",r,"of",cfg$reps,"complete"))
  }
}
runtime <- clock()-runstart
raw <- do.call(rbind,rows)
summary <- do.call(rbind,accuracy)
for (phase in phases) for (field in c("median","Q1","Q3"))
  summary[[paste0(phase,"_",field)]] <- NA_real_
for (i in seq_len(nrow(summary))) for (phase in phases) {
  v <- raw$seconds[raw$config_id==summary$config_id[i] & raw$phase==phase]
  stopifnot(length(v)==cfg$reps)
  stat <- timing_summary(v)
  for (field in names(stat)) summary[i,paste0(phase,"_",field)] <- stat[[field]]
}
summary$speedup_solve <- summary$speedup_total <- NA_real_
for (group in c("parallel_small","parallel_large")) {
  ids <- which(summary$sweep==group); baseline <- ids[summary$p[ids]==1L]
  summary$speedup_solve[ids] <- summary$t_solve_median[baseline]/summary$t_solve_median[ids]
  summary$speedup_total[ids] <- summary$t_reused_new_input_median[baseline]/summary$t_reused_new_input_median[ids]
}
write_csv(summary,file.path(outdir,"summary","summary.csv"))
saveRDS(summary,file.path(outdir,"summary","summary.rds"))
writeLines(as.character(runtime),file.path(outdir,"runtime_seconds.txt"))
checkpoint()
check("source_unchanged",identical(unname(tools::md5sum(srcfiles)),
  unname(tools::md5sum(file.path(outdir,"source",basename(srcfiles))))))
checkpoint()
writeLines("Completed",file.path(outdir,"COMPLETED"))
logmsg(paste("All timing experiments complete; official-phase seconds",round(runtime,3)))
print(summary[,c("sweep","N","c","p","W","rmse","t_input_median","t_solve_median",
                 "t_reused_new_input_median","speedup_solve","speedup_total")],row.names=FALSE)
