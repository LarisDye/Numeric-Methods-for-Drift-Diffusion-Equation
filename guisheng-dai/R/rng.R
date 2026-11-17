new_seed_manager <- function(seed=20261116L) {
  RNGkind("L'Ecuyer-CMRG");set.seed(seed);state<-.Random.seed
  saved<-list()
  list(get=function(key,count) {
    if(is.null(saved[[key]])){
      saved[[key]]<<-lapply(seq_len(count),function(k){
        item<-list(id=paste0(key,"/",k),state=state)
        state<<-parallel::nextRNGStream(state);item
      })
    }
    stopifnot(length(saved[[key]])==count);saved[[key]]
  },save=function(path) saveRDS(list(master_seed=seed,kind="L'Ecuyer-CMRG",tasks=saved),path))
}
