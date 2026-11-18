# 实验入口（从 guisheng-dai/ 目录运行）：解析命令行后，依次执行所选实验。
# --mode smoke 只做单线程的小规模检查；full 使用正式参数。
# --threads 是 C++ OpenMP 线程数；本脚本不创建 R 多进程工作池。
# --experiment 可选 all 或 e1...e6；--run-id 决定 results/ 下的独立结果目录。
args<-commandArgs(trailingOnly=TRUE)
opt<-function(name,default){k<-match(paste0("--",name),args);if(!is.na(k)&&k<length(args))args[k+1L] else default}
mode<-opt("mode","full")
if("--workers"%in%args)stop("C++ uses OpenMP; specify --threads rather than --workers")
threads<-as.integer(opt("threads",if(mode=="smoke")"1" else "2"))
if(!mode%in%c("full","smoke")||is.na(threads)||threads<1L)stop("Invalid mode or threads")
if(mode=="smoke"&&threads!=1L)stop("Smoke correctness experiments must be serial")
runid<-opt("run-id",paste0("run-",format(Sys.time(),"%Y%m%d-%H%M%S")))
selected<-tolower(opt("experiment","all"))
# 加载模型、求解器和 Rcpp 接口。首次编译与后续报告生成均不计入 runstart。
source("config.R");source("R/bootstrap.R");load_project()
cfg<-experiment_config(mode,threads);outdir<-file.path("results",runid)
# 已写过原始记录的运行禁止覆盖，防止新旧参数和随机种子混在一起。
if(file.exists(file.path(outdir,"raw","records.rds"))) stop("Run already exists; use a new run-id")
dir.create(file.path(outdir,"raw"),recursive=TRUE,showWarnings=FALSE)
saveRDS(cfg,file.path(outdir,"config.rds"));dput(cfg,file.path(outdir,"config.R"))
capture.output(sessionInfo(),file=file.path(outdir,"environment.txt"))
cat(paste0("\nStarted (Asia/Shanghai): ",format(Sys.time(),tz="Asia/Shanghai"),
 "\nLogical cores: ",parallel::detectCores(),"\nPhysical cores: ",parallel::detectCores(logical=FALSE),
 "\nGit HEAD: ",system2("git",c("rev-parse","HEAD"),stdout=TRUE),"\n"),
 file=file.path(outdir,"environment.txt"),append=TRUE)
# 五类记录先保存在列表中，检查点再统一落盘：逐点预测、计时、路径块统计、
# 一致性检查、OpenMP 初始化。add 用 <<- 更新外层 records，避免反复 rbind。
records<-list(predictions=list(),timings=list(),walk_blocks=list(),checks=list(),initialization=list())
add<-function(kind,row){records[[kind]][[length(records[[kind]])+1L]]<<-row}
# seedman 是保存“任务键 -> CMRG 随机流状态”的闭包；相同键重复取回相同状态。
# 新的精度重复使用新键，线程数不写入配对任务键，因此可跨线程数复现同一路径。
seedman<-new_seed_manager(cfg$seed)
teaminfo<-list();inittimes<-list()
# 整段实验耗时包括求解、校验、日志和中途 checkpoint；其精确截止点见文件末尾。
runstart<-clock()
logmsg<-function(msg){s<-paste(format(Sys.time(),"%H:%M:%S"),msg);cat(s,"\n");flush.console();
 cat(s,"\n",file=file.path(outdir,"progress.log"),append=TRUE)}
# 每种 p 首次使用时触发一次短 OpenMP 区域，核验实际团队大小并记录耗时。
# team 只是线程数检查与缓存，不会建立进程池、导出 R 对象或产生 IPC。
# 后续 C++ 求解调用仍会进入各自的 OpenMP 区域；此处的单次测量不能代表
# 每次调用的开销，也不是在全新 R 进程中反复测得的完整冷启动成本。
team<-function(p) {
 k<-as.character(p)
 if(is.null(inittimes[[k]])){
   z<-elapsed(function()initialize_openmp(p));teaminfo[[k]]<<-z$result;inittimes[[k]]<<-z$seconds
   add("initialization",data.frame(p=p,t_openmp_init=z$seconds,
     actual_threads=z$result$actual_threads,openmp_version=z$result[["_OPENMP"]]))
 }
 as.integer(p)
}
# 为统一入口保留空清理钩子；没有需要 stopCluster 的 R 工作进程。
finish<-function() invisible(NULL)
# 同时保留 RDS（原始对象）、CSV（便于查看）和完整随机流清单。
# 这里只保存已完成的记录，不实现断点恢复；重新运行应换用新的 run-id。
checkpoint<-function(){
 saveRDS(records,file.path(outdir,"raw","records.rds"))
 for(k in names(records))write_csv(bind_records(records[[k]]),file.path(outdir,"raw",paste0(k,".csv")))
 seedman$save(file.path(outdir,"seed_manifest.rds"))
}
# 输入准备在求解计时之前完成。t_input 只包含模型/卷积输入构造，
# team(p) 初始化单独测量；prepare_call=0 表示本实现没有进程间传输步骤。
prepared_mc<-function(N,n,p,profile="main"){
 z<-elapsed(function()make_problem(N,n,profile));p<-team(p)
 list(ctx=z$result,threads=p,input=z$seconds,prepare_call=0,p=p)
}
prepared_cv<-function(N,c,p,profile="main"){
 z<-elapsed(function()prepare_gaussian(N,c,profile));p<-team(p)
 list(ctx=z$result,threads=p,input=z$seconds,prepare_call=0,p=p)
}
# 兼容旧调用位置的空钩子：上下文现在显式传给求解器，不设置全局工作者状态。
set_prepared<-function(z) invisible(NULL)
# t_solve 是一次完整 R 求解包装调用的墙钟时间，包含参数封装、C++ 核心和
# 结果汇总；解析解/离散参照、写文件、绘图等不在这个计时区间内。
# t_reused_new_input = 输入准备 + 求解；t_once 再加本次运行中该 p 的首次
# OpenMP 初始化实测值。t_once 是组合口径，未包含 R 启动、源码编译等成本。
# p 是请求线程数，actual_threads 来自本次 C++ 调用的实际团队，不靠推测填写。
time_record<-function(experiment,id,method,z,rep,seconds,A=1,phase="timing",actual_threads) {
 cx<-z$ctx
 add("timings",data.frame(experiment=experiment,config_id=id,method=method,p=z$p,
  N=cx$N,n=if(is.null(cx$n))NA else cx$n,M=if(exists("current_M",inherits=TRUE))current_M else NA,
  c=if(is.null(cx$c))NA else cx$c,W=if(is.null(cx$W))NA else cx$W,
  rep=rep,phase=phase,batch_A=A,batch_elapsed=seconds*A,t_solve=seconds,
  backend="OpenMP",actual_threads=actual_threads,
  t_input=z$input,t_prepare_call=z$prepare_call,t_openmp_init=inittimes[[as.character(z$p)]],
  t_reused_new_input=z$input+z$prepare_call+seconds,
  t_once=z$input+z$prepare_call+seconds+inittimes[[as.character(z$p)]]))
}
# 参照解在求解结束后计算：truth 用于总误差，u_discrete/variance 用于分离
# 随机误差与网格偏差；walk_blocks 保留每个逻辑块的种子标识和充分统计量。
record_mc<-function(experiment,id,z,j,M,rep,res,phase="accuracy") {
 ref<-discrete_reference(z$ctx);truth<-exact(j/z$ctx$N)
 add("predictions",data.frame(experiment=experiment,config_id=id,method="MC",phase=phase,
  p=z$p,actual_threads=res$threads_used,N=z$ctx$N,n=z$ctx$n,M=M,rep=rep,j=j,x=j/z$ctx$N,
  estimate=res$value,truth=truth,u_discrete=ref$mean[j+1L],variance=ref$variance[j+1L],
  se=res$se,alive=res$alive,sum_steps=res$steps))
 if(nrow(res$blocks))add("walk_blocks",cbind(data.frame(experiment=experiment,config_id=id,p=z$p,rep=rep,phase=phase),res$blocks))
}
# 同时记录实际核宽 W、取整后的截断尺度 c_eff 和权重和，便于解释截断误差。
# boundary_before 是强制端点为零之前的残差；refvalues 可提供宽核数值参照。
record_cv<-function(experiment,id,z,j,res,profile="main",refvalues=NULL){
 cx<-z$ctx
 add("predictions",data.frame(experiment=experiment,config_id=id,method="Gaussian",phase="accuracy",
  p=z$p,actual_threads=res$threads_used,N=cx$N,c=cx$c,K=cx$K,W=cx$W,c_eff=cx$c_eff,kernel_sum=cx$kernel_sum,
  profile=profile,j=j,x=j/cx$N,estimate=res$value,truth=exact(j/cx$N,profile=profile),
  boundary_before=res$boundary_before,u_kernel_ref=if(is.null(refvalues))NA else refvalues))
}
# 短任务采用批量计时：先预热一次，再用 pilot 估算 A，使一批接近目标时长。
# 预热、pilot 和计时前的 gc 均排除在正式 t_solve 之外；每批总耗时除以 A。
# A 次重复仅降低计时分辨率的影响，并不产生 A 份独立的统计样本。
# timing_reps 是计时重复次数；MC 的独立精度重复由各实验单独分配新种子。
benchmark<-function(fun,experiment,id,method,z,M=NA) {
 current_M<<-M
 fun() # 显式预热，排除在正式计时之外。
 pilot<-elapsed(fun);A<-max(1L,min(1024L,as.integer(ceiling(cfg$batch_target/max(pilot$seconds,1e-6)))))
 values<-numeric(cfg$timing_reps);last<-pilot$result
 for(r in seq_len(cfg$timing_reps)){
   gc(FALSE);t<-clock()
   for(a in seq_len(A))last<-fun()
   values[r]<-(clock()-t)/A
   time_record(experiment,id,method,z,r,values[r],A,actual_threads=last$threads_used)
 }
 list(result=last,times=values,A=A)
}
want<-function(x)selected=="all"||selected==tolower(x)
tryCatch({
# 先核验默认线程数，初始化耗时不混入求解计时。
team(threads)
if(want("e1")||want("e2")){
 # E1/E2 共享一组单点 x=1/8 数据：改变 M 观察抽样误差，改变 N、n 观察网格偏差。
 # 正式网格取 (8,8)、(16,32)、(32,128)，保持时间步长与空间步长平方同阶。
 # 最细网格仅取 M=4096，作为网格对照；较粗两网格扫描四种路径数。
 grids<-if(mode=="smoke") data.frame(N=8L,n=8L) else data.frame(N=c(8L,16L,32L),n=c(8L,32L,128L))
 for(g in seq_len(nrow(grids))){
  N<-grids$N[g];n<-grids$n[g]
  Ms<-if(mode=="smoke")256L else if(N==32L)4096L else c(256L,1024L,4096L,16384L)
  if(!want("e2")&&N==32L)next
  for(M in Ms){
   id<-sprintf("N%d_n%d_M%d",N,n,M);logmsg(paste("E1/E2",id))
   z<-prepared_mc(N,n,threads);j<-as.integer(N/8);current_M<-M
   # 每个查询点分为固定 16 个逻辑块；随机流属于块，不属于 OpenMP 线程。
   # warm 使用专用键，不消耗后面 accuracy/r 的独立随机流。
   warm<-seedman$get(paste0("e12/",id,"/warm"),16)
   solve_mc(z$ctx,j,M,warm,z$threads)
   for(r in seq_len(cfg$mc_reps)){
    seeds<-seedman$get(paste0("e12/",id,"/accuracy/",r),16)
    tm<-elapsed(function()solve_mc(z$ctx,j,M,seeds,z$threads))
    record_mc("E1_E2",id,z,j,M,r,tm$result)
    time_record("E1_E2",id,"MC",z,r,tm$seconds,phase="accuracy",actual_threads=tm$result$threads_used)
   }
  }
 }
 checkpoint()
}
if(want("e3")){
 # E3 固定空间网格扫描高斯截断尺度 c，从而改变 K=ceil(c*sqrt(2DT)*N)、W=2K+1。
 # c=8 与 c=10 的差值先验证宽核参照已稳定；这部分校验不计入各配置求解时间。
 N<-if(mode=="smoke")16L else 512L;j<-as.integer(0:N)
 refz<-prepared_cv(N,8,threads);ref<-solve_gaussian(refz$ctx,j,refz$threads)$value
 ref10z<-prepared_cv(N,10,threads);ref10<-solve_gaussian(ref10z$ctx,j,ref10z$threads)$value
 checkdiff<-max(abs(ref-ref10));stopifnot(checkdiff<1e-12)
 add("checks",data.frame(check="kernel_ref_c8_c10",max_difference=checkdiff,passed=TRUE))
 for(cutoff in if(mode=="smoke")c(2,4) else c(1,2,3,4,6,8,10)){
  id<-sprintf("N%d_c%d",N,cutoff);logmsg(paste("E3",id))
  z<-prepared_cv(N,cutoff,threads);current_M<-NA
  tm<-benchmark(function()solve_gaussian(z$ctx,j,z$threads),"E3",id,"Gaussian",z)
  record_cv("E3",id,z,j,tm$result,refvalues=ref)
 }
 checkpoint()
}
if(want("e4")){
 # E4 改用多项式型初值，固定充分宽的 c=8，在网格加密时观察空间离散误差。
 # 该初值可由独立正弦谱展开求真值，避免只在单一正弦模态上检验卷积。
 for(N in if(mode=="smoke")c(16L,32L) else c(16L,32L,64L,128L)){
  id<-sprintf("poly_N%d_c8",N);logmsg(paste("E4",id))
  z<-prepared_cv(N,8,threads,"poly");j<-as.integer(0:N)
  tm<-elapsed(function()solve_gaussian(z$ctx,j,z$threads));current_M<-NA
  record_cv("E4",id,z,j,tm$result,"poly")
  time_record("E4",id,"Gaussian",z,1,tm$seconds,phase="accuracy",actual_threads=tm$result$threads_used)
  # 比较谱参照的两种截断阶数，确认参照误差足够小；此检查位于求解计时之外。
  d<-max(abs(exact(j/N,profile="poly",highest=31)-exact(j/N,profile="poly",highest=63)))
  stopifnot(d<1e-12);add("checks",data.frame(check=paste0("spectral31_63_N",N),max_difference=d,passed=TRUE))
 }
 checkpoint()
}
if(want("e5")){
 # E5 比较线程数与问题规模：正式实验 p=1,2,4，所有团队预先核验。
 # p=1 仍走同一 OpenMP 求解接口，作为相同实现的单线程基线。
 for(p in cfg$parallel_degrees)team(p)
 logmsg("E5 timing stage; no concurrent figure generation")
 Ms<-if(mode=="smoke")1024L else c(4096L,65536L)
 for(M in Ms){
  # paired[[1]] 留给预热/一致性检查/pilot，r+1 对应第 r 轮正式计时。
  # 一轮内各线程数使用完全相同的逻辑块种子，避免随机路径工作量干扰加速比。
  paired<-lapply(0:cfg$timing_reps,function(r)seedman$get(sprintf("e5/mc/M%d/r%d",M,r),16))
  # 各轮轮转 p 的执行顺序，减轻温度、缓存及运行时波动的顺序效应。
  orders<-lapply(seq_len(cfg$timing_reps),function(r){
    v<-cfg$parallel_degrees;v[c(seq.int(1+(r-1)%%length(v),length(v)),if((r-1)%%length(v)>0)seq_len((r-1)%%length(v)))]
  })
  zs<-lapply(cfg$parallel_degrees,function(p)prepared_mc(64L,256L,p));names(zs)<-as.character(cfg$parallel_degrees)
  base<-NULL
  # 先检查估计值、标准误、存活数、步数及逐块统计量跨线程数一致，再正式计时。
  for(p in cfg$parallel_degrees){
   z<-zs[[as.character(p)]];set_prepared(z)
   res<-solve_mc(z$ctx,32L,M,paired[[1]],z$threads)
   checkvec<-c(res$value,res$se,res$alive,res$steps,unlist(res$blocks[6:9]))
   if(is.null(base))base<-checkvec else stopifnot(max(abs(checkvec-base))<1e-12)
  }
  add("checks",data.frame(check=paste0("MC_parallel_M",M),max_difference=0,passed=TRUE))
  # 每种线程数单独校准批量大小，避免慢配置的一批过长、快配置的一批过短。
  batch<-list()
  for(p in cfg$parallel_degrees){
   z<-zs[[as.character(p)]];set_prepared(z)
   t<-elapsed(function()solve_mc(z$ctx,32L,M,paired[[1]],z$threads))$seconds
   batch[[as.character(p)]]<-max(1L,min(256L,as.integer(ceiling(cfg$batch_target/max(t,1e-6)))))
  }
  current_M<-M
  for(r in seq_len(cfg$timing_reps))for(p in orders[[r]]){
   z<-zs[[as.character(p)]];set_prepared(z);A<-batch[[as.character(p)]]
   # 同一批 A 次调用复用本轮种子，计时更稳定；只记录最后一次相同结果。
   # 新一轮 r 才使用新随机流，因此不能把 batch_A 当作独立 MC 重复数。
   gc(FALSE);t<-clock();for(a in seq_len(A))res<-solve_mc(z$ctx,32L,M,paired[[r+1L]],z$threads)
   sec<-(clock()-t)/A;id<-sprintf("mc_N64_n256_M%d",M)
   time_record("E5",id,"MC",z,r,sec,A,actual_threads=res$threads_used);record_mc("E5",id,z,32L,M,r,res,"timing")
  }
  logmsg(paste("E5 MC M",M,"complete"))
 }
 # 卷积采用 c=4，扩大 N 使输出点数与核宽同时增长，观察并行何时抵消调度开销。
 for(N in if(mode=="smoke")128L else c(512L,2048L,8192L)){
  j<-as.integer(0:N);id<-sprintf("conv_N%d_c4",N);current_M<-NA
  zs<-lapply(cfg$parallel_degrees,function(p)prepared_cv(N,4,p));names(zs)<-as.character(cfg$parallel_degrees)
  set_prepared(zs[["1"]]);base<-solve_gaussian(zs[["1"]]$ctx,j,1L)$value;batch<-list();maxdiff<-0
  # 输入准备、单线程参照与跨线程数一致性检查都在正式计时之外。
  for(p in cfg$parallel_degrees){
   z<-zs[[as.character(p)]];set_prepared(z);pilot<-elapsed(function()solve_gaussian(z$ctx,j,z$threads))
   maxdiff<-max(maxdiff,max(abs(base-pilot$result$value)))
   batch[[as.character(p)]]<-max(1L,min(1024L,as.integer(ceiling(cfg$batch_target/max(pilot$seconds,1e-6)))))
  }
  stopifnot(maxdiff<1e-12);add("checks",data.frame(check=paste0("Gaussian_parallel_N",N),max_difference=maxdiff,passed=TRUE))
  for(r in seq_len(cfg$timing_reps)){
   # 卷积也轮转线程顺序；确定性输入无需为每个计时重复生成随机种子。
   ps<-cfg$parallel_degrees;ps<-ps[((seq_along(ps)+r-2L)%%length(ps))+1L]
   for(p in ps){
    z<-zs[[as.character(p)]];set_prepared(z);A<-batch[[as.character(p)]]
    gc(FALSE);t<-clock();for(a in seq_len(A))res<-solve_gaussian(z$ctx,j,z$threads)
    sec<-(clock()-t)/A;time_record("E5",id,"Gaussian",z,r,sec,A,actual_threads=res$threads_used)
    if(r==1L)record_cv("E5",id,z,j,res)
   }
  }
  logmsg(paste("E5 Gaussian N",N,"complete"))
 }
 checkpoint()
}
if(want("e6")){
 # E6 对齐任务口径：两种方法只请求同一组 8 个内部点，并使用相同解析真值。
 # MC 扫描 M，卷积扫描 c，比较精度与“准备新输入 + 求解”的时间成本。
 # 这里的卷积不计算全网格输出；不可直接拿该时间与 E3 的全网格时间相比。
 qx<-seq(1,15,by=2)/16
 for(M in if(mode=="smoke")1024L else c(1024L,4096L,16384L)){
  id<-sprintf("mc_N16_n32_M%d_8points",M);logmsg(paste("E6",id))
  z<-prepared_mc(16L,32L,threads);j<-query_indices(qx,16);current_M<-M
  # 8 个查询点各有 16 块，因而每次需要 128 个独立逻辑块随机流。
  solve_mc(z$ctx,j,M,seedman$get(paste0("e6/",id,"/warm"),128),z$threads)
  for(r in seq_len(cfg$mc_reps)){
   seeds<-seedman$get(paste0("e6/",id,"/accuracy/",r),128)
   tm<-elapsed(function()solve_mc(z$ctx,j,M,seeds,z$threads))
   record_mc("E6",id,z,j,M,r,tm$result);time_record("E6",id,"MC",z,r,tm$seconds,phase="accuracy",actual_threads=tm$result$threads_used)
  }
 }
 for(cutoff in c(2,3,4)){
  id<-sprintf("conv_N512_c%d_8points",cutoff);logmsg(paste("E6",id));z<-prepared_cv(512L,cutoff,threads)
  j<-query_indices(qx,512);current_M<-NA
  tm<-benchmark(function()solve_gaussian(z$ctx,j,z$threads),"E6",id,"Gaussian",z)
  record_cv("E6",id,z,j,tm$result)
 }
 checkpoint()
}
# 运行总耗时在最后一次持久化之前截止：它包含上述阶段内 checkpoint、检查和
# 日志，不包含启动/编译、下面最终保存、绘图或文档生成，不能视为端到端耗时。
runtime<-clock()-runstart
writeLines(as.character(runtime),file.path(outdir,"runtime_seconds.txt"))
saveRDS(teaminfo,file.path(outdir,"openmp_info.rds"))
# 最终记录保存成功后才写 COMPLETED；失败时已完成的阶段仍可能有检查点文件。
checkpoint();writeLines("Completed",file.path(outdir,"COMPLETED"))
logmsg(paste("Complete; runtime seconds",round(runtime,3)))
},finally=finish())
