# 基础误差与性能统计；正式 Python 汇总使用相同的统计含义。
rmse <- function(x,y) sqrt(mean((x-y)^2))
timing_summary <- function(x) c(median=median(x),Q1=unname(quantile(x,.25)),Q3=unname(quantile(x,.75)))
# 统一字符编码，不写 R 行名；不同方法不适用的字段在 CSV 中留空。
write_csv <- function(x,path) write.csv(x,path,row.names=FALSE,fileEncoding="UTF-8",na="")
# MC 与卷积的记录列不同：先取列名并集，补 NA 后按统一列序拼接。
bind_records <- function(xs) {
  if(!length(xs)) return(data.frame())
  keys <- unique(unlist(lapply(xs,names)))
  do.call(rbind,lapply(xs,function(x){for(k in setdiff(keys,names(x))) x[[k]]<-NA; x[,keys,drop=FALSE]}))
}
# 单调墙钟描述用户等待的实际秒数，不将多线程 CPU 时间相加。
clock <- function() steady_seconds_cpp()
# 同时保留结果与耗时，参照解计算和写盘由调用方放到这个区间之外。
elapsed <- function(fun) { t<-clock(); result<-fun(); list(result=result,seconds=clock()-t) }
