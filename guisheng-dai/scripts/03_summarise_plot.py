"""Summarise actual R results and generate scientific figures. No solver runs here."""
from pathlib import Path
import json, sys, hashlib, platform, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import psutil
root=Path(__file__).resolve().parents[1]
run=root/"results"/(sys.argv[1] if len(sys.argv)>1 else "openmp-final-20261117")
out=run/"summary";out.mkdir(exist_ok=True)
fig=root/"report/figures";fig.mkdir(parents=True,exist_ok=True)
pred=pd.read_csv(run/"raw/predictions.csv");tim=pd.read_csv(run/"raw/timings.csv")
BLUE="#045073";RED="#EC5858";GREEN="#5FB2A8";DARK="#0D2659"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
"axes.labelcolor":"#222222","text.color":"#222222","axes.titleweight":"bold","axes.grid":True,
"grid.alpha":.2,"figure.dpi":160,"savefig.bbox":"tight","pdf.fonttype":42})
def save(name,f):
 f.tight_layout();f.savefig(fig/(name+".pdf"));f.savefig(fig/(name+".png"),dpi=200);plt.close(f)
def ts(df):
 return pd.Series({"median":df.t_solve.median(),"Q1":df.t_solve.quantile(.25),"Q3":df.t_solve.quantile(.75),
 "warm_new_input":df.t_reused_new_input.median(),"once":df.t_once.median(),"reps":len(df),
 "t_input":df.t_input.iloc[0],"t_prepare_call":df.t_prepare_call.iloc[0],"t_openmp_init":df.t_openmp_init.iloc[0]})
mc=pred[pred.experiment=="E1_E2"].copy(); rows=[]
for (N,M),g in mc.groupby(["N","M"]):
 rows.append(dict(N=int(N),n=int(g.n.iloc[0]),M=int(M),R=len(g),mean=g.estimate.mean(),SD=g.estimate.std(),
  RMSE_discrete=np.sqrt(np.mean((g.estimate-g.u_discrete)**2)),
  RMSE_true=np.sqrt(np.mean((g.estimate-g.truth)**2)),
  discrete=g.u_discrete.iloc[0],truth=g.truth.iloc[0],bias=g.u_discrete.iloc[0]-g.truth.iloc[0],
  variance=g.variance.iloc[0],survival=np.mean(g.alive/M),mean_steps=np.mean(g.sum_steps/M)))
ms=pd.DataFrame(rows);ms.to_csv(out/"mc.csv",index=False)
f,axs=plt.subplots(1,2,figsize=(8.5,3.1))
for ax,N in zip(axs,[8,16]):
 g=ms[ms.N==N]
 ax.loglog(g.M,g.RMSE_discrete,"o-",color=BLUE,label="MC vs discrete expectation")
 ax.loglog(g.M,g.RMSE_true,"s-",color=RED,label="MC vs exact PDE")
 ax.loglog(g.M,np.sqrt(g.variance/g.M),"--",color=GREEN,label="Theory: sqrt(v / M)")
 ax.axhline(abs(g.bias.iloc[0]),color="#777777",ls=":",label="Discrete bias")
 ax.set(xlabel="Paths per estimate M",ylabel="RMSE",title=f"N={N}, n={int(g.n.iloc[0])}")
 ax.legend(fontsize=7.1,loc="lower left")
save("mc_error",f)
f,axs=plt.subplots(1,2,figsize=(8.5,3.0))
g=ms[ms.M==4096].sort_values("N")
axs[0].loglog(1/g.N,abs(g.bias),"o-",color=BLUE);axs[0].invert_xaxis()
axs[0].set(xlabel="Grid spacing h",ylabel="Absolute discrete bias",title="Grid bias at x=1/8")
axs[1].loglog(1/g.N,g.RMSE_true,"s-",color=RED);axs[1].invert_xaxis()
axs[1].set(xlabel="Grid spacing h",ylabel="Observed total RMSE",title="Fixed M=4096, 30 batches")
for ax in axs:
 ax.set_xticks(1/g.N,[f"1/{int(n)}" for n in g.N])
 ax.xaxis.set_minor_formatter(NullFormatter())
save("mc_grid",f)
kernel=pred[(pred.experiment=="E3")&(pred.j>0)&(pred.j<pred.N)].copy();rows=[]
for c,g in kernel.groupby("c"):
 t=tim[(tim.experiment=="E3")&(tim.c==c)]
 rows.append(dict(c=int(c),N=512,K=int(g.K.iloc[0]),W=int(g.W.iloc[0]),c_eff=g.c_eff.iloc[0],
  kernel_sum=g.kernel_sum.iloc[0],RMSE=np.sqrt(np.mean((g.estimate-g.truth)**2)),
  Linf=np.max(abs(g.estimate-g.truth)),trunc_rmse=np.sqrt(np.mean((g.estimate-g.u_kernel_ref)**2)),
  boundary=pred[(pred.experiment=="E3")&(pred.c==c)].boundary_before.max(),**ts(t).to_dict()))
ks=pd.DataFrame(rows);ks.to_csv(out/"kernel.csv",index=False)
f,axs=plt.subplots(1,2,figsize=(8.5,3.0));g=ks[ks.c<=6]
axs[0].semilogy(g.W,g.RMSE,"o-",color=BLUE,label="Exact PDE error")
axs[0].semilogy(g.W,np.maximum(g.trunc_rmse,1e-18),"x--",color=RED,label="Same-grid truncation diagnostic")
axs[0].set(xlabel="Kernel length W",ylabel="Interior RMSE",title="Kernel width, fixed N=512");axs[0].legend(fontsize=8)
axs[1].plot(g.W,1000*g["median"],"o-",color=BLUE,label="Solve including OpenMP call")
axs[1].plot(g.W,1000*g.warm_new_input,"s--",color=GREEN,label="New input, runtime warmed")
axs[1].set(xlabel="Kernel length W",ylabel="Time (ms)",title="Two OpenMP threads");axs[1].legend(fontsize=8)
save("kernel",f)
gp=pred[pred.experiment=="E4"];rows=[]
for N,g in gp.groupby("N"):
 inner=g[(g.j>0)&(g.j<N)]
 rows.append(dict(N=int(N),h=1/N,RMSE=np.sqrt(np.mean((inner.estimate-inner.truth)**2)),
  Linf=np.max(abs(inner.estimate-inner.truth)),boundary=g.boundary_before.max()))
gs=pd.DataFrame(rows).sort_values("N");gs["order"]=np.r_[np.nan,np.log2(gs.RMSE.values[:-1]/gs.RMSE.values[1:])]
gs.to_csv(out/"grid.csv",index=False)
f,ax=plt.subplots(figsize=(6.3,3.0))
ax.loglog(gs.h,gs.RMSE,"o-",color=BLUE,label="Polynomial initial condition")
ax.loglog(gs.h,gs.RMSE.iloc[0]*(gs.h/gs.h.iloc[0])**4,"--",color=GREEN,label="Slope 4 reference")
ax.invert_xaxis();ax.set(xlabel="Grid spacing h",ylabel="Interior RMSE",title="Wide kernel c=8, two OpenMP threads");ax.legend(fontsize=8)
save("grid",f)
pts=[];perf=tim[tim.experiment=="E5"]
for (method,N,M,p),g in perf.groupby(["method","N","M","p"],dropna=False):
 pts.append(dict(method=method,N=int(N),M=None if pd.isna(M) else int(M),p=int(p),**ts(g).to_dict()))
pf=pd.DataFrame(pts)
for i,row in pf.iterrows():
 base=pf[(pf.method==row.method)&(pf.N==row.N)&((pf.M.isna()) if pd.isna(row.M) else pf.M==row.M)&(pf.p==1)]["median"].iloc[0]
 pf.loc[i,"speedup"]=base/row["median"];pf.loc[i,"efficiency"]=base/row["median"]/row.p
pf.to_csv(out/"performance.csv",index=False)
f,axs=plt.subplots(1,2,figsize=(8.5,3.1))
for (M,g),col in zip(pf[pf.method=="MC"].groupby("M"),[BLUE,RED]):
 axs[0].errorbar(g.p,g["median"]*1000,yerr=np.array([g["median"]-g.Q1,g.Q3-g["median"]])*1000,fmt="o-",color=col,label=f"M={int(M)}")
for (N,g),col in zip(pf[pf.method=="Gaussian"].groupby("N"),[BLUE,GREEN,RED]):
 axs[1].errorbar(g.p,g["median"]*1000,yerr=np.array([g["median"]-g.Q1,g.Q3-g["median"]])*1000,fmt="o-",color=col,label=f"N={int(N)}")
for ax,title in zip(axs,["Random walks, N=64, n=256","Direct convolution, c=4"]):
 ax.set(xlabel="OpenMP threads p",ylabel="Solve time (ms)",title=title,xticks=[1,2,4],yscale="log");ax.legend(fontsize=8)
save("parallel_time",f)
f,axs=plt.subplots(1,2,figsize=(8.5,3.0))
for (M,g),col in zip(pf[pf.method=="MC"].groupby("M"),[BLUE,RED]):axs[0].plot(g.p,g.speedup,"o-",color=col,label=f"M={int(M)}")
for (N,g),col in zip(pf[pf.method=="Gaussian"].groupby("N"),[BLUE,GREEN,RED]):axs[1].plot(g.p,g.speedup,"o-",color=col,label=f"N={int(N)}")
for ax,title in zip(axs,["Random walk speedup","Direct convolution speedup"]):
 ax.plot([1,4],[1,4],":",color="#999999",label="Ideal");ax.set(xlabel="OpenMP threads p",ylabel="Speedup",xticks=[1,2,4],title=title);ax.legend(fontsize=8)
save("speedup",f)
cost=pred[pred.experiment=="E6"];rows=[]
for (method,M,c),g in cost.groupby(["method","M","c"],dropna=False):
 t=tim[(tim.experiment=="E6")&(tim.method==method)]
 t=t[t.M==M] if method=="MC" else t[t.c==c]
 rows.append(dict(method=method,M=None if pd.isna(M) else int(M),c=None if pd.isna(c) else int(c),
  RMSE=np.sqrt(np.mean((g.estimate-g.truth)**2)),**ts(t).to_dict()))
cs=pd.DataFrame(rows);cs.to_csv(out/"cost.csv",index=False)
f,ax=plt.subplots(figsize=(7.6,3.3))
for method,g,col,marker in [("MC",cs[cs.method=="MC"],RED,"o"),("Gaussian",cs[cs.method=="Gaussian"],BLUE,"s")]:
 ax.loglog(g.warm_new_input*1000,g.RMSE,marker+"-",color=col,label=method)
 for _,r in g.iterrows():ax.annotate(f"M={int(r.M)}" if method=="MC" else f"c={int(r.c)}",(r.warm_new_input*1000,r.RMSE),xytext=(5,5),textcoords="offset points",fontsize=8)
for eps in [.01,.003]:ax.axhline(eps,color="#777777",ls=":",lw=1);ax.text(cs.warm_new_input.min()*1000,eps*1.05,f"epsilon={eps}",fontsize=8)
ax.set(xlabel="New-input cost, runtime warmed (ms)",ylabel="RMSE over the same 8 points",title="Common task, two OpenMP threads");ax.legend(fontsize=9,loc="lower right")
save("cost",f)
# Export native chart data as a public, machine-readable source.
def series(name,x,y,color):return dict(name=name,x=[float(v) for v in x],y=[float(v) for v in y],color=color.lstrip("#"))
charts={}
g=ms[ms.N==16]
charts["mc"]={"title":"随机游走误差（N=16）","xlabel":"路径数 M","ylabel":"RMSE","xlog":True,"ylog":True,
"series":[series("相对离散期望",g.M,g.RMSE_discrete,BLUE),series("相对解析解",g.M,g.RMSE_true,RED),series("理论标准误",g.M,np.sqrt(g.variance/g.M),GREEN)]}
g=ks[ks.c<=6]
charts["kernel"]={"title":"核大小与误差","xlabel":"核长度 W","ylabel":"内部 RMSE","ylog":True,
"series":[series("相对解析解",g.W,g.RMSE,BLUE)]}
charts["grid"]={"title":"多项式初值的网格误差","xlabel":"N","ylabel":"内部 RMSE","xlog":True,"ylog":True,
"series":[series("c=8 宽核",gs.N,gs.RMSE,BLUE)]}
g=ms[ms.M==4096].sort_values("N")
charts["mcgrid"]={"title":"固定路径数时的网格影响","xlabel":"网格段数 N","ylabel":"误差","xlog":True,"ylog":True,
"series":[series("离散偏差",g.N,abs(g.bias),BLUE),series("MC 总 RMSE",g.N,g.RMSE_true,RED)]}
charts["speedup"]={"title":"并行加速比","xlabel":"OpenMP 线程数 p","ylabel":"加速比",
"series":[series("MC M=65536",pf[(pf.method=="MC")&(pf.M==65536)].p,pf[(pf.method=="MC")&(pf.M==65536)].speedup,RED),
series("卷积 N=8192",pf[(pf.method=="Gaussian")&(pf.N==8192)].p,pf[(pf.method=="Gaussian")&(pf.N==8192)].speedup,BLUE),
series("理想加速",[1,2,4],[1,2,4],"999999")]}
charts["cost"]={"title":"共同 8 点查询的精度与成本","xlabel":"预热后新输入成本 / ms","ylabel":"RMSE","xlog":True,"ylog":True,
"series":[series(method,g.warm_new_input*1000,g.RMSE,col) for method,g,col in [("MC",cs[cs.method=="MC"],RED),("Gaussian",cs[cs.method=="Gaussian"],BLUE)]]}
hardware=dict(cpu="11th Gen Intel Core i7-11800H @ 2.30GHz",physical_cores=psutil.cpu_count(logical=False),logical_cores=os.cpu_count(),
ram_gib=psutil.virtual_memory().total/2**30,os=platform.platform(),python=platform.python_version())
(root/"report/chart_data.json").write_text(json.dumps(charts,ensure_ascii=False,indent=2),encoding="utf-8")
summary=dict(run_id=run.name,date="2026-11-17",backend="OpenMP",openmp_tests=int(len(pd.read_csv(root/"results/tests/openmp_tests.csv"))),mc=ms.to_dict("records"),kernel=ks.to_dict("records"),grid=gs.replace({np.nan:None}).to_dict("records"),
performance=pf.replace({np.nan:None}).to_dict("records"),cost=cs.replace({np.nan:None}).to_dict("records"),
hardware=hardware,runtime=float((run/"runtime_seconds.txt").read_text().strip()),
tests=int(len(pd.read_csv(root/"results/tests/e0_tests.csv"))),raw_prediction_rows=len(pred),raw_timing_rows=len(tim))
(root/"report/summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
(run/"hardware.json").write_text(json.dumps(hardware,indent=2),encoding="utf-8")
sources=list((root/"R").glob("*.R"))+list((root/"src").glob("*.cpp"))+list((root/"scripts").glob("*"))+[root/"config.R"]
pd.DataFrame([dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources if p.is_file()]).to_csv(run/"source_hashes.csv",index=False)
print("MC");print(ms.to_string(index=False));print("KERNEL");print(ks.to_string(index=False));print("GRID");print(gs.to_string(index=False))
print("PERFORMANCE");print(pf.to_string(index=False));print("COST");print(cs.to_string(index=False))
