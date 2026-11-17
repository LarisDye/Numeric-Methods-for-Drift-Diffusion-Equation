source("R/bootstrap.R");load_project()
dir.create("results/tests",recursive=TRUE,showWarnings=FALSE)
checks<-list()
check<-function(name,ok,detail="") {
  if(!isTRUE(ok))stop("FAILED: ",name)
  checks[[length(checks)+1L]]<<-data.frame(test=name,passed=TRUE,detail=detail)
  cat("PASS",name,detail,"\n")
}
info<-lapply(c(1L,2L,4L),initialize_openmp)
check("compiled_and_actual_teams",all(vapply(info,function(z)isTRUE(z$compiled),logical(1)))&&
  identical(vapply(info,`[[`,integer(1),"actual_threads"),c(1L,2L,4L)),
  paste("_OPENMP",info[[1]][["_OPENMP"]],"compiler",info[[1]]$compiler))
ctx<-make_problem(64L,256L);manager<-new_seed_manager();seeds<-manager$get("thread_test",48L)
seed_before<-.Random.seed;states_before<-seeds
mc<-lapply(c(1L,2L,4L),function(p)solve_mc(ctx,c(1L,32L,63L),4096L,seeds,p))
check("MC_blockwise_bit_identical",identical(mc[[1]]$blocks,mc[[2]]$blocks)&&identical(mc[[1]]$blocks,mc[[3]]$blocks))
check("MC_aggregate_bit_identical",all(vapply(mc[-1],function(z)
  identical(z[1:4],mc[[1]][1:4]),logical(1))))
check("MC_rng_and_inputs_read_only",identical(seeds,states_before)&&identical(.Random.seed,seed_before))
# Independently compare each logical block with R's serial CMRG generator.
saved<-.Random.seed
serial<-lapply(1:16,function(b){assign(".Random.seed",seeds[[b]]$state,.GlobalEnv)
  walk_block_cpp(ctx$f,1L,ctx$n,ctx$probs,256L)})
assign(".Random.seed",saved,.GlobalEnv)
expected<-do.call(rbind,serial)
actual<-as.matrix(mc[[1]]$blocks[1:16,c("M_block","sum_y","sum_y2","alive","sum_steps")])
check("CMRG_matches_R_serial_paths",identical(unname(actual),unname(expected)))
g<-prepare_gaussian(2048L,4);j<-as.integer(0:2048);before<-g
cv<-lapply(c(1L,2L,4L),function(p)solve_gaussian(g,j,p))
check("Gaussian_output_bit_identical",identical(cv[[1]]$value,cv[[2]]$value)&&identical(cv[[1]]$value,cv[[3]]$value))
check("Gaussian_raw_boundary_and_inputs",all(vapply(cv,function(z)z$boundary_before<1e-12,logical(1)))&&identical(g,before))
write_csv(do.call(rbind,checks),"results/tests/openmp_tests.csv")
cat("All OpenMP integration tests passed.\n")
