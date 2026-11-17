args<-commandArgs(trailingOnly=TRUE)
opt<-function(name,default){k<-match(paste0("--",name),args);if(!is.na(k)&&k<length(args))args[k+1L] else default}
mode<-opt("mode","full")
if("--workers"%in%args)stop("C++ uses OpenMP; specify --threads rather than --workers")
threads<-as.integer(opt("threads",if(mode=="smoke")"1" else "2"))
if(!mode%in%c("full","smoke")||is.na(threads)||threads<1L)stop("Invalid mode or threads")
if(mode=="smoke"&&threads!=1L)stop("Smoke correctness experiments must be serial")
runid<-opt("run-id",paste0("run-",format(Sys.time(),"%Y%m%d-%H%M%S")))
selected<-tolower(opt("experiment","all"))
source("config.R");source("R/bootstrap.R");load_project()
cfg<-experiment_config(mode,threads);outdir<-file.path("results",runid)
if(file.exists(file.path(outdir,"raw","records.rds"))) stop("Run already exists; use a new run-id")
dir.create(file.path(outdir,"raw"),recursive=TRUE,showWarnings=FALSE)
saveRDS(cfg,file.path(outdir,"config.rds"));dput(cfg,file.path(outdir,"config.R"))
capture.output(sessionInfo(),file=file.path(outdir,"environment.txt"))
cat(paste0("\nStarted (Asia/Shanghai): ",format(Sys.time(),tz="Asia/Shanghai"),
 "\nLogical cores: ",parallel::detectCores(),"\nPhysical cores: ",parallel::detectCores(logical=FALSE),
 "\nGit HEAD: ",system2("git",c("rev-parse","HEAD"),stdout=TRUE),"\n"),
 file=file.path(outdir,"environment.txt"),append=TRUE)
records<-list(predictions=list(),timings=list(),walk_blocks=list(),checks=list(),initialization=list())
add<-function(kind,row){records[[kind]][[length(records[[kind]])+1L]]<<-row}
seedman<-new_seed_manager(cfg$seed)
teaminfo<-list();inittimes<-list()
runstart<-clock()
logmsg<-function(msg){s<-paste(format(Sys.time(),"%H:%M:%S"),msg);cat(s,"\n");flush.console();
 cat(s,"\n",file=file.path(outdir,"progress.log"),append=TRUE)}
team<-function(p) {
 k<-as.character(p)
 if(is.null(inittimes[[k]])){
   z<-elapsed(function()initialize_openmp(p));teaminfo[[k]]<<-z$result;inittimes[[k]]<<-z$seconds
   add("initialization",data.frame(p=p,t_openmp_init=z$seconds,
     actual_threads=z$result$actual_threads,openmp_version=z$result[["_OPENMP"]]))
 }
 as.integer(p)
}
finish<-function() invisible(NULL)
checkpoint<-function(){
 saveRDS(records,file.path(outdir,"raw","records.rds"))
 for(k in names(records))write_csv(bind_records(records[[k]]),file.path(outdir,"raw",paste0(k,".csv")))
 seedman$save(file.path(outdir,"seed_manifest.rds"))
}
prepared_mc<-function(N,n,p,profile="main"){
 z<-elapsed(function()make_problem(N,n,profile));p<-team(p)
 list(ctx=z$result,threads=p,input=z$seconds,prepare_call=0,p=p)
}
prepared_cv<-function(N,c,p,profile="main"){
 z<-elapsed(function()prepare_gaussian(N,c,profile));p<-team(p)
 list(ctx=z$result,threads=p,input=z$seconds,prepare_call=0,p=p)
}
set_prepared<-function(z) invisible(NULL)
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
record_mc<-function(experiment,id,z,j,M,rep,res,phase="accuracy") {
 ref<-discrete_reference(z$ctx);truth<-exact(j/z$ctx$N)
 add("predictions",data.frame(experiment=experiment,config_id=id,method="MC",phase=phase,
  p=z$p,actual_threads=res$threads_used,N=z$ctx$N,n=z$ctx$n,M=M,rep=rep,j=j,x=j/z$ctx$N,
  estimate=res$value,truth=truth,u_discrete=ref$mean[j+1L],variance=ref$variance[j+1L],
  se=res$se,alive=res$alive,sum_steps=res$steps))
 if(nrow(res$blocks))add("walk_blocks",cbind(data.frame(experiment=experiment,config_id=id,p=z$p,rep=rep,phase=phase),res$blocks))
}
record_cv<-function(experiment,id,z,j,res,profile="main",refvalues=NULL){
 cx<-z$ctx
 add("predictions",data.frame(experiment=experiment,config_id=id,method="Gaussian",phase="accuracy",
  p=z$p,actual_threads=res$threads_used,N=cx$N,c=cx$c,K=cx$K,W=cx$W,c_eff=cx$c_eff,kernel_sum=cx$kernel_sum,
  profile=profile,j=j,x=j/cx$N,estimate=res$value,truth=exact(j/cx$N,profile=profile),
  boundary_before=res$boundary_before,u_kernel_ref=if(is.null(refvalues))NA else refvalues))
}
benchmark<-function(fun,experiment,id,method,z,M=NA) {
 current_M<<-M
 fun() # Explicit warmup, excluded.
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
# Initialise and verify the first OpenMP team, outside solver timing.
team(threads)
if(want("e1")||want("e2")){
 grids<-if(mode=="smoke") data.frame(N=8L,n=8L) else data.frame(N=c(8L,16L,32L),n=c(8L,32L,128L))
 for(g in seq_len(nrow(grids))){
  N<-grids$N[g];n<-grids$n[g]
  Ms<-if(mode=="smoke")256L else if(N==32L)4096L else c(256L,1024L,4096L,16384L)
  if(!want("e2")&&N==32L)next
  for(M in Ms){
   id<-sprintf("N%d_n%d_M%d",N,n,M);logmsg(paste("E1/E2",id))
   z<-prepared_mc(N,n,threads);j<-as.integer(N/8);current_M<-M
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
 for(N in if(mode=="smoke")c(16L,32L) else c(16L,32L,64L,128L)){
  id<-sprintf("poly_N%d_c8",N);logmsg(paste("E4",id))
  z<-prepared_cv(N,8,threads,"poly");j<-as.integer(0:N)
  tm<-elapsed(function()solve_gaussian(z$ctx,j,z$threads));current_M<-NA
  record_cv("E4",id,z,j,tm$result,"poly")
  time_record("E4",id,"Gaussian",z,1,tm$seconds,phase="accuracy",actual_threads=tm$result$threads_used)
  d<-max(abs(exact(j/N,profile="poly",highest=31)-exact(j/N,profile="poly",highest=63)))
  stopifnot(d<1e-12);add("checks",data.frame(check=paste0("spectral31_63_N",N),max_difference=d,passed=TRUE))
 }
 checkpoint()
}
if(want("e5")){
 # Initialise and verify requested team sizes before paired timing.
 for(p in cfg$parallel_degrees)team(p)
 logmsg("E5 timing stage; no concurrent figure generation")
 Ms<-if(mode=="smoke")1024L else c(4096L,65536L)
 for(M in Ms){
  paired<-lapply(0:cfg$timing_reps,function(r)seedman$get(sprintf("e5/mc/M%d/r%d",M,r),16))
  orders<-lapply(seq_len(cfg$timing_reps),function(r){
    v<-cfg$parallel_degrees;v[c(seq.int(1+(r-1)%%length(v),length(v)),if((r-1)%%length(v)>0)seq_len((r-1)%%length(v)))]
  })
  zs<-lapply(cfg$parallel_degrees,function(p)prepared_mc(64L,256L,p));names(zs)<-as.character(cfg$parallel_degrees)
  base<-NULL
  for(p in cfg$parallel_degrees){
   z<-zs[[as.character(p)]];set_prepared(z)
   res<-solve_mc(z$ctx,32L,M,paired[[1]],z$threads)
   checkvec<-c(res$value,res$se,res$alive,res$steps,unlist(res$blocks[6:9]))
   if(is.null(base))base<-checkvec else stopifnot(max(abs(checkvec-base))<1e-12)
  }
  add("checks",data.frame(check=paste0("MC_parallel_M",M),max_difference=0,passed=TRUE))
  # Each round rotates the thread-count order. Batch size is calibrated per count.
  batch<-list()
  for(p in cfg$parallel_degrees){
   z<-zs[[as.character(p)]];set_prepared(z)
   t<-elapsed(function()solve_mc(z$ctx,32L,M,paired[[1]],z$threads))$seconds
   batch[[as.character(p)]]<-max(1L,min(256L,as.integer(ceiling(cfg$batch_target/max(t,1e-6)))))
  }
  current_M<-M
  for(r in seq_len(cfg$timing_reps))for(p in orders[[r]]){
   z<-zs[[as.character(p)]];set_prepared(z);A<-batch[[as.character(p)]]
   gc(FALSE);t<-clock();for(a in seq_len(A))res<-solve_mc(z$ctx,32L,M,paired[[r+1L]],z$threads)
   sec<-(clock()-t)/A;id<-sprintf("mc_N64_n256_M%d",M)
   time_record("E5",id,"MC",z,r,sec,A,actual_threads=res$threads_used);record_mc("E5",id,z,32L,M,r,res,"timing")
  }
  logmsg(paste("E5 MC M",M,"complete"))
 }
 for(N in if(mode=="smoke")128L else c(512L,2048L,8192L)){
  j<-as.integer(0:N);id<-sprintf("conv_N%d_c4",N);current_M<-NA
  zs<-lapply(cfg$parallel_degrees,function(p)prepared_cv(N,4,p));names(zs)<-as.character(cfg$parallel_degrees)
  set_prepared(zs[["1"]]);base<-solve_gaussian(zs[["1"]]$ctx,j,1L)$value;batch<-list();maxdiff<-0
  for(p in cfg$parallel_degrees){
   z<-zs[[as.character(p)]];set_prepared(z);pilot<-elapsed(function()solve_gaussian(z$ctx,j,z$threads))
   maxdiff<-max(maxdiff,max(abs(base-pilot$result$value)))
   batch[[as.character(p)]]<-max(1L,min(1024L,as.integer(ceiling(cfg$batch_target/max(pilot$seconds,1e-6)))))
  }
  stopifnot(maxdiff<1e-12);add("checks",data.frame(check=paste0("Gaussian_parallel_N",N),max_difference=maxdiff,passed=TRUE))
  for(r in seq_len(cfg$timing_reps)){
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
 qx<-seq(1,15,by=2)/16
 for(M in if(mode=="smoke")1024L else c(1024L,4096L,16384L)){
  id<-sprintf("mc_N16_n32_M%d_8points",M);logmsg(paste("E6",id))
  z<-prepared_mc(16L,32L,threads);j<-query_indices(qx,16);current_M<-M
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
runtime<-clock()-runstart
writeLines(as.character(runtime),file.path(outdir,"runtime_seconds.txt"))
saveRDS(teaminfo,file.path(outdir,"openmp_info.rds"))
checkpoint();writeLines("Completed",file.path(outdir,"COMPLETED"))
logmsg(paste("Complete; runtime seconds",round(runtime,3)))
},finally=finish())
