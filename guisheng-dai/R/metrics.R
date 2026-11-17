rmse <- function(x,y) sqrt(mean((x-y)^2))
timing_summary <- function(x) c(median=median(x),Q1=unname(quantile(x,.25)),Q3=unname(quantile(x,.75)))
write_csv <- function(x,path) write.csv(x,path,row.names=FALSE,fileEncoding="UTF-8",na="")
bind_records <- function(xs) {
  if(!length(xs)) return(data.frame())
  keys <- unique(unlist(lapply(xs,names)))
  do.call(rbind,lapply(xs,function(x){for(k in setdiff(keys,names(x))) x[[k]]<-NA; x[,keys,drop=FALSE]}))
}
clock <- function() steady_seconds_cpp()
elapsed <- function(fun) { t<-clock(); result<-fun(); list(result=result,seconds=clock()-t) }
