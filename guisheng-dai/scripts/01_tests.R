# E0：全部使用默认一线程。与专门的 OpenMP 集成测试分开记录。
# 参照包括手算、R 路径、离散矩阵期望、有限核恒等式和连续解析解。
source("R/bootstrap.R");load_project()
dir.create("results/tests",recursive=TRUE,showWarnings=FALSE)
# 任一断言失败立即停止；成功项目统一写入可读 CSV。
log<-list();check<-function(name,ok,detail="") {
  if(!isTRUE(ok))stop("FAILED: ",name," ",detail)
  log[[length(log)+1L]]<<-data.frame(test=name,passed=TRUE,detail=detail)
  cat("PASS",name,detail,"\n")
}
# 用于确认非法输入会明确报错，而不是被静默修正。
reject<-function(expr) inherits(tryCatch({force(expr);NULL},error=function(e)e),"error")
# 单步手算：N=4、初值只在首个内部节点非零，检查 Q 的上下对角方向。
ctx<-make_problem(4,1,T=.1);ctx$f<-c(0,1,0,0,0);ref<-discrete_reference(ctx)
check("hand_transition",max(abs(ref$mean-c(0,.84,.04,0,0)))<1e-12)
# 检查负下标、跨端点取负及一个完整周期；随后核对手算宽窗口。
g<-c(0,1,2,3,0)
check("odd_padding",identical(as.numeric(odd_extension(g,c(-1,-2,5,8,9))),c(-1,-2,-3,0,1)))
K<-2L;pad<-odd_extension(g,(-K):(4+K));weights<-c(.1,.2,.4,.2,.1)
y<-gaussian_block_arma(pad,weights,rep(1,5),as.integer(c(0,1,4)))
check("hand_wide_window",max(abs(y-c(0,1,0)))<1e-12)
# 人为提供“随机数”让吸收/停留分支可预测，不依赖某次随机抽样碰巧发生。
force_path<-function(j,z,p,N){steps<-0;for(u in z){if(u<p[1])j<-j-1L else if(u>=sum(p[1:2]))j<-j+1L;steps<-steps+1;if(j==0||j==N) return(c(alive=0,steps=steps))};c(alive=1,steps=steps)}
check("forced_left_absorption",identical(force_path(1,c(.01,.5),ctx$probs,4),c(alive=0,steps=1)))
check("forced_right_absorption",identical(force_path(3,c(.99,.5),ctx$probs,4),c(alive=0,steps=1)))
check("forced_stay",identical(force_path(2,c(.5,.5),ctx$probs,4),c(alive=1,steps=2)))
# 恢复同一 RNG 初态后比较 R 与串行 C++，避免用不同样本比较实现。
ctx<-make_problem(8,8);RNGkind("L'Ecuyer-CMRG");set.seed(20261116);s<-.Random.seed
a<-walk_block_r(ctx$f,1L,8L,ctx$probs,256L);assign(".Random.seed",s,.GlobalEnv)
b<-walk_block_cpp(ctx$f,1L,8L,ctx$probs,256L)
check("R_Rcpp_same_stream",max(abs(a-b))<1e-12)
check("absorbing_boundary",all(walk_block_cpp(ctx$f,0L,8L,ctx$probs,256L)[-1L]==0))
# 用理论收益方差构造统计误差尺度；有限 MC 样本不要求恰好等于期望。
ref<-discrete_reference(ctx);sm<-new_seed_manager();set_context(ctx)
values<-vapply(1:10,function(r)solve_mc(ctx,1L,4096L,sm$get(paste0("test/",r),16L))$value,numeric(1))
z<-abs(mean(values)-ref$mean[2])/sqrt(ref$variance[2]/(10*4096))
check("MC_discrete_expectation",z<4,paste("absolute_z",format(z,digits=6)))
# 比较同一个有限核的 R/C++ 结果及正弦恒等式，区分代码错误与截断误差。
prep<-prepare_gaussian(16,4);j<-as.integer(0:16)
before<-prep
a<-gaussian_block_arma(prep$padded,prep$weights,prep$scale,j)
b<-gaussian_block_r(prep$padded,prep$weights,prep$scale,j)
l<-(-prep$K):prep$K
ident<-prep$scale*sin(pi*j/16)*sum(prep$weights*cos(pi*l/16))
check("R_Armadillo_same_window",max(abs(a-b))<1e-12)
check("finite_kernel_sine_identity",max(abs(a-ident))<1e-12)
check("raw_zero_boundary",max(abs(a[c(1,17)]))<1e-12)
check("read_only_input",identical(prep,before))
# 扩大物理核覆盖后再对连续解析解核验；同时覆盖零初值和零时间退化情形。
wide<-prepare_gaussian(64,8);set_context(wide);out<-solve_gaussian(wide,as.integer(0:64))
check("continuous_exact_solution",max(abs(out$value-exact((0:64)/64)))<1e-12)
zero<-make_problem(8,8);zero$f[]<-0;set_context(zero)
check("zero_initial",solve_mc(zero,1L,256L,sm$get("zero",16))$value==0)
at0<-make_problem(8,8,T=0);set_context(at0)
check("zero_time_MC",solve_mc(at0,1L,256L,sm$get("at0",16))$value==at0$f[2])
at0g<-prepare_gaussian(8,3,T=0);set_context(at0g)
check("zero_time_Gaussian",solve_gaussian(at0g,1L)$value==at0g$f[2])
# 这些输入必须拒绝；不能靠裁剪概率、忽略越界或改写边界来“通过”。
check("invalid_probability_rejected",reject(check_probs(c(-.1,.8,.3))))
check("invalid_diffusion_rejected",reject(make_problem(8,D=-1)))
check("invalid_query_rejected",reject(gaussian_block_arma(prep$padded,prep$weights,prep$scale,99L)))
check("invalid_length_rejected",reject(gaussian_block_arma(prep$padded[-1],prep$weights,prep$scale,1L)))
bad<-ctx$f;bad[1]<-.1
check("nonzero_boundary_rejected",reject(prepare_gaussian_cpp(bad,.2,.05,.5,4L)))
# 多项式初值用截断谱参照：加密最高波数后应在本轮参数下稳定。
check("spectral_reference_31_63",max(abs(exact((0:128)/128,profile="poly",highest=31)-exact((0:128)/128,profile="poly",highest=63)))<1e-12)
write_csv(do.call(rbind,log),"results/tests/e0_tests.csv")
cat("All serial E0 tests passed.\n")
