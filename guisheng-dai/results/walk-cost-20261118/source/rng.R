# 为固定逻辑任务分配随机流；parallel::nextRNGStream 不会启动 R 进程。
# 新 key 按首次请求顺序领取状态，不是用 key 散列生成种子。
new_seed_manager <- function(seed=20261116L) {
  RNGkind("L'Ecuyer-CMRG");set.seed(seed);state<-.Random.seed
  saved<-list()
  # get/save 构成闭包；state 与 saved 在多次调用之间保留。
  list(get=function(key,count) {
    if(is.null(saved[[key]])){
      saved[[key]]<<-lapply(seq_len(count),function(k){
        # 保存的是调用前初态；线程数变化不会改变该逻辑块所属的流。
        item<-list(id=paste0(key,"/",k),state=state)
        state<<-parallel::nextRNGStream(state);item
      })
    }
    # 重用 key 返回相同初态，适合跨线程配对或重复计时，不是独立新样本。
    stopifnot(length(saved[[key]])==count);saved[[key]]
  },save=function(path) saveRDS(list(master_seed=seed,kind="L'Ecuyer-CMRG",tasks=saved),path))
}
