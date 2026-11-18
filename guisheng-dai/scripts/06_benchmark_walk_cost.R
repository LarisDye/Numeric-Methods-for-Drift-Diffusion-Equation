# Run from guisheng-dai/: Rscript --vanilla scripts/06_benchmark_walk_cost.R --run-id NAME
# Fresh timing experiment only; existing E0--E6 records are never overwritten.
args <- commandArgs(trailingOnly=TRUE)
opt <- function(name, default) {
  k <- match(paste0("--", name), args)
  if (!is.na(k) && k < length(args)) args[k+1L] else default
}
runid <- opt("run-id", "walk-cost-20261118")
source("R/bootstrap.R"); load_project()
outdir <- file.path("results", runid)
if (dir.exists(outdir)) stop("Run directory exists; choose a new run-id")
for (sub in c("raw", "summary", "source"))
  dir.create(file.path(outdir, sub), recursive=TRUE, showWarnings=FALSE)

cfg <- list(a=.2, D=.05, T=.5, query_x=.5, q=1L, grid_M=16384L,
  grid_threads=2L, parallel_N=64L, parallel_n=512L, parallel_M=65536L,
  parallel_threads=c(1L,2L,4L,8L), reps=9L, batch_target=.2,
  max_batch=256L, seed=20261118L, cpu="Intel Core i7-11800H",
  physical_cores=8L, logical_cores=16L,
  timing_scope="Warmed full R solve_mc call, including packing, validation, C++ and reduction",
  excluded="Compilation, context and seed preparation, warmup, gc, checks, file I/O",
  serial_baseline="Identical C++ source compiled without the OpenMP plugin; identical R wrapper")
dput(cfg, file.path(outdir, "config.R")); saveRDS(cfg, file.path(outdir, "config.rds"))
capture.output(sessionInfo(), file=file.path(outdir, "environment.txt"))
cat("\nStarted (Asia/Shanghai):", format(Sys.time(), tz="Asia/Shanghai"),
  "\n", paste(names(Sys.info()), Sys.info(), collapse="\n"),
  "\nOMP_WAIT_POLICY:", Sys.getenv("OMP_WAIT_POLICY"),
  file=file.path(outdir, "environment.txt"), append=TRUE)

# Reuse exactly the existing algorithm and CMRG implementation for true serial timing.
# Removing this plugin disables all #ifdef _OPENMP regions, without rewriting the loop.
cpp <- readLines("src/random_walk.cpp", warn=FALSE)
plugin <- grepl("Rcpp::plugins(openmp)", cpp, fixed=TRUE)
stopifnot(sum(plugin)==1L)
serial_path <- file.path(outdir, "source", "random_walk_serial.cpp")
writeLines(cpp[!plugin], serial_path, useBytes=TRUE)
serial_env <- new.env(parent=.GlobalEnv)
Rcpp::sourceCpp(serial_path, env=serial_env, cacheDir=".cache/rcpp",
                rebuild=FALSE, showOutput=FALSE, verbose=FALSE)
stopifnot(!isTRUE(serial_env$openmp_info_cpp(1L)$compiled))
serial_dispatch <- new.env(parent=environment(solve_mc))
serial_dispatch$walk_blocks_openmp <- serial_env$walk_blocks_openmp
solve_serial <- solve_mc
environment(solve_serial) <- serial_dispatch
teams <- lapply(cfg$parallel_threads, initialize_openmp)
saveRDS(list(serial=serial_env$openmp_info_cpp(1L), openmp=teams),
        file.path(outdir, "compiler_and_teams.rds"))
capture.output(list(serial=serial_env$openmp_info_cpp(1L), openmp=teams),
               file=file.path(outdir, "compiler_and_teams.txt"))
for (f in c("src/random_walk.cpp", "R/random_walk.R", "R/problem.R", "R/rng.R",
            "R/bootstrap.R", "R/metrics.R", "scripts/06_benchmark_walk_cost.R"))
  file.copy(f, file.path(outdir, "source", basename(f)), overwrite=FALSE)

seedman <- new_seed_manager(cfg$seed)
rows <- list(); checks <- list()
logmsg <- function(msg) {
  line <- paste(format(Sys.time(), tz="Asia/Shanghai", format="%H:%M:%S"), msg)
  cat(line, "\n"); flush.console()
  cat(line, "\n", file=file.path(outdir, "progress.log"), append=TRUE)
}
check <- function(name, ok) {
  if (!isTRUE(ok)) stop("FAILED: ", name)
  checks[[length(checks)+1L]] <<- data.frame(check=name, passed=TRUE)
}
same_result <- function(a,b) identical(a[1:5], b[1:5])
checkpoint <- function() {
  write_csv(do.call(rbind, rows), file.path(outdir, "raw", "timings.csv"))
  write_csv(do.call(rbind, checks), file.path(outdir, "raw", "checks.csv"))
  saveRDS(rows, file.path(outdir, "raw", "timings.rds"))
  seedman$save(file.path(outdir, "seed_manifest.rds"))
}

# Small independent reference: R's CMRG must produce the same block paths.
small <- make_problem(8L,8L); s <- seedman$get("small_reference",16L)
serial_small <- solve_serial(small,1L,256L,s,1L)
old_seed <- .Random.seed
expected <- lapply(seq_len(16L), function(b) {
  assign(".Random.seed",s[[b]]$state,.GlobalEnv)
  walk_block_r(small$f,1L,small$n,small$probs,16L)
})
assign(".Random.seed",old_seed,.GlobalEnv)
observed <- as.matrix(serial_small$blocks[,c("M_block","sum_y","sum_y2","alive","sum_steps")])
check("serial_matches_R_CMRG_paths", identical(unname(observed),unname(do.call(rbind,expected))))
for (p in cfg$parallel_threads)
  check(paste0("small_serial_openmp_p",p),same_result(serial_small,solve_mc(small,1L,256L,s,p)))

make_row <- function(sweep,N,n,M,p,backend="openmp")
  data.frame(sweep=sweep,N=as.integer(N),n=as.integer(n),M=as.integer(M),
             p=as.integer(p),backend=backend,stringsAsFactors=FALSE)
configs <- rbind(
  make_row("time_grid",64L,c(256L,512L,1024L,2048L),cfg$grid_M,2L),
  make_row("space_grid",c(8L,16L,32L,64L,128L),1024L,cfg$grid_M,2L),
  make_row("coupled_grid",c(16L,32L,64L,128L),c(32L,128L,512L,2048L),cfg$grid_M,2L),
  make_row("parallel",64L,512L,cfg$parallel_M,1L,"serial"),
  make_row("parallel",64L,512L,cfg$parallel_M,cfg$parallel_threads))
configs$config_id <- with(configs,paste(sweep,backend,paste0("N",N),paste0("n",n),paste0("p",p),sep="_"))
write_csv(configs,file.path(outdir,"configurations.csv"))
contexts <- lapply(seq_len(nrow(configs)),function(i) {
  ctx <- make_problem(configs$N[i],configs$n[i])
  check_probs(ctx$probs);ctx
})
solvers <- lapply(seq_len(nrow(configs)),function(i) {
  ctx<-contexts[[i]];j<-as.integer(configs$N[i]/2L)
  M<-configs$M[i];p<-configs$p[i]
  fun<-if(configs$backend[i]=="serial")solve_serial else solve_mc
  function(seeds)fun(ctx,j,M,seeds,p)
})
runstart <- clock()
logmsg("Compilation and independent correctness checks complete; beginning timing")
for (group in unique(configs$sweep)) {
  ids <- which(configs$sweep==group)
  paired <- lapply(0:cfg$reps,function(r)seedman$get(paste(group,"round",r,sep="/"),16L))
  batches <- integer(nrow(configs))
  for (i in ids) {
    z<-configs[i,]
    fun<-solvers[[i]]
    warmed<-fun(paired[[1]])
    reference<-solve_serial(contexts[[i]],as.integer(z$N/2L),z$M,paired[[1]],1L)
    check(paste0("warm_",z$config_id),same_result(warmed,reference))
    pilot<-elapsed(function()fun(paired[[1]]))
    batches[i]<-max(1L,min(cfg$max_batch,as.integer(ceiling(cfg$batch_target/max(pilot$seconds,1e-6)))))
  }
  for (r in seq_len(cfg$reps)) {
    # Rotate configurations within every sweep, avoiding an always-fast-first ordering.
    order<-ids[((seq_along(ids)+r-2L)%%length(ids))+1L]
    round_results<-list()
    for (i in order) {
      z<-configs[i,];A<-batches[i];seeds<-paired[[r+1L]];fun<-solvers[[i]]
      gc(FALSE);start<-clock()
      for (a in seq_len(A)) ans<-fun(seeds)
      total<-clock()-start
      check(paste0("team_",z$config_id,"_r",r),ans$threads_used==z$p)
      rows[[length(rows)+1L]]<-data.frame(z,query_x=cfg$query_x,q=cfg$q,
        rep=r,batch_A=A,batch_elapsed=total,t_solve=total/A,
        actual_threads=ans$threads_used,estimate=ans$value,se=ans$se,
        alive=ans$alive,sum_steps=ans$steps,mean_steps=ans$steps/z$M)
      round_results[[z$config_id]]<-ans
    }
    if (group=="parallel") {
      ref<-round_results[[configs$config_id[ids[1]]]]
      for (i in ids[-1])check(paste0("paired_",configs$config_id[i],"_r",r),
                            same_result(ref,round_results[[configs$config_id[i]]]))
    }
    checkpoint()
    logmsg(paste(group,"round",r,"of",cfg$reps,"complete"))
  }
}
runtime<-clock()-runstart
writeLines(as.character(runtime),file.path(outdir,"runtime_seconds.txt"))
checkpoint();writeLines("Completed",file.path(outdir,"COMPLETED"))
logmsg(paste("All timing experiments complete; seconds",round(runtime,3)))
