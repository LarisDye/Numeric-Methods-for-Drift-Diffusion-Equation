# Recompute accuracy only; no timing experiment is repeated or modified.
# The initial gaussian-cost-20261118 run measured RMSE on all output nodes.
# Project reports use internal nodes, so retain both and refresh derived summaries.
args <- commandArgs(trailingOnly=TRUE)
runid <- if(length(args)) args[1L] else "gaussian-cost-20261118"
outdir <- file.path("results",runid)
stopifnot(file.exists(file.path(outdir,"COMPLETED")))
cfg <- readRDS(file.path(outdir,"config.rds"))
src <- read.csv(file.path(outdir,"source_hashes.csv"),stringsAsFactors=FALSE)
solverfiles <- src$path[!grepl("^scripts/",src$path)]
stopifnot(identical(unname(tools::md5sum(solverfiles)),
                   src$md5[match(solverfiles,src$path)]))
source("R/bootstrap.R"); load_project()
summary_path <- file.path(outdir,"summary","summary.csv")
summary <- read.csv(summary_path,stringsAsFactors=FALSE)
if(!"rmse_all" %in% names(summary)) summary$rmse_all <- summary$rmse
summary$rmse_internal <- NA_real_
records <- list()
for(i in seq_len(nrow(summary))) {
  z <- summary[i,]
  ctx <- prepare_gaussian(z$N,z$c,cfg$profile,cfg$a,cfg$D,cfg$T)
  j <- as.integer(0:z$N); keep <- j>0L & j<z$N
  res <- solve_gaussian(ctx,j,z$p)
  truth <- exact(j/z$N,cfg$T,cfg$a,cfg$D,cfg$profile)
  all <- rmse(res$value,truth)
  stopifnot(abs(all-z$rmse_all)<1e-15, res$threads_used==z$p)
  inside <- rmse(res$value[keep],truth[keep])
  summary$rmse_internal[i] <- inside
  summary$rmse[i] <- inside
  records[[i]] <- data.frame(config_id=z$config_id,N=z$N,c=z$c,p=z$p,
    rmse_all=all,rmse_internal=inside,internal_nodes=sum(keep),
    previous_rmse_all=z$rmse_all,max_difference_previous=abs(all-z$rmse_all))
}
write_csv(do.call(rbind,records),file.path(outdir,"raw","accuracy_internal.csv"))
write_csv(summary,summary_path)
saveRDS(summary,file.path(outdir,"summary","summary.rds"))
script <- "scripts/08a_gaussian_internal_rmse.R"
file.copy(script,file.path(outdir,"source",basename(script)),overwrite=FALSE)
write_csv(data.frame(path=script,md5=unname(tools::md5sum(script))),
          file.path(outdir,"postprocess_source_hashes.csv"))
writeLines(c("Accuracy-only postprocessing; original raw timings and accuracy.csv are unchanged.",
  "The original raw accuracy.csv field rmse includes all N+1 output nodes.",
  "summary.csv and summary.rds now set rmse=rmse_internal (N-1 internal nodes).",
  "Both rmse_all and rmse_internal are retained, with fresh checks in raw/accuracy_internal.csv.",
  "The numerical source matches source_hashes.csv; only the current benchmark script was corrected for future runs.",
  paste("Completed:",format(Sys.time(),tz="Asia/Shanghai"))),
  file.path(outdir,"accuracy_postprocess.txt"))
print(summary[,c("sweep","N","c","p","rmse_all","rmse_internal")],row.names=FALSE)
