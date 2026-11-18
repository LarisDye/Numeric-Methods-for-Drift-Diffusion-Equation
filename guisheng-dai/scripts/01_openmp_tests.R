# 专门的并行集成核验：测试 1/2/4 实际线程及逐位一致性，不用于性能排名。
source("R/bootstrap.R");load_project()
dir.create("results/tests",recursive=TRUE,showWarnings=FALSE)
checks<-list()
check<-function(name,ok,detail="") {
  if(!isTRUE(ok))stop("FAILED: ",name)
  checks[[length(checks)+1L]]<<-data.frame(test=name,passed=TRUE,detail=detail)
  cat("PASS",name,detail,"\n")
}
# 实际并行区内测线程组大小，验证编译插件与运行时配置均已生效。
info<-lapply(c(1L,2L,4L),initialize_openmp)
check("compiled_and_actual_teams",all(vapply(info,function(z)isTRUE(z$compiled),logical(1)))&&
  identical(vapply(info,`[[`,integer(1),"actual_threads"),c(1L,2L,4L)),
  paste("_OPENMP",info[[1]][["_OPENMP"]],"compiler",info[[1]]$compiler))
# 三个查询点各 16 个逻辑块，所有线程数重用完全相同的 48 个初态。
ctx<-make_problem(64L,256L);manager<-new_seed_manager();seeds<-manager$get("thread_test",48L)
seed_before<-.Random.seed;states_before<-seeds
mc<-lapply(c(1L,2L,4L),function(p)solve_mc(ctx,c(1L,32L,63L),4096L,seeds,p))
# 既核验最终估计，也核验逐块统计，避免归约恰好抵消了内部差异。
check("MC_blockwise_bit_identical",identical(mc[[1]]$blocks,mc[[2]]$blocks)&&identical(mc[[1]]$blocks,mc[[3]]$blocks))
check("MC_aggregate_bit_identical",all(vapply(mc[-1],function(z)
  identical(z[1:4],mc[[1]][1:4]),logical(1))))
check("MC_rng_and_inputs_read_only",identical(seeds,states_before)&&identical(.Random.seed,seed_before))
# Independently compare each logical block with R's serial CMRG generator.
# 对第一个查询点逐块恢复 R RNG，核对原生私有 CMRG 与 R 产生相同路径。
saved<-.Random.seed
serial<-lapply(1:16,function(b){assign(".Random.seed",seeds[[b]]$state,.GlobalEnv)
  walk_block_cpp(ctx$f,1L,ctx$n,ctx$probs,256L)})
assign(".Random.seed",saved,.GlobalEnv)
expected<-do.call(rbind,serial)
actual<-as.matrix(mc[[1]]$blocks[1:16,c("M_block","sum_y","sum_y2","alive","sum_steps")])
check("CMRG_matches_R_serial_paths",identical(unname(actual),unname(expected)))
# 卷积每个输出窗口互相独立：线程数变化不应改变内积和原始边界残差。
g<-prepare_gaussian(2048L,4);j<-as.integer(0:2048);before<-g
cv<-lapply(c(1L,2L,4L),function(p)solve_gaussian(g,j,p))
check("Gaussian_output_bit_identical",identical(cv[[1]]$value,cv[[2]]$value)&&identical(cv[[1]]$value,cv[[3]]$value))
check("Gaussian_raw_boundary_and_inputs",all(vapply(cv,function(z)z$boundary_before<1e-12,logical(1)))&&identical(g,before))
write_csv(do.call(rbind,checks),"results/tests/openmp_tests.csv")
cat("All OpenMP integration tests passed.\n")
