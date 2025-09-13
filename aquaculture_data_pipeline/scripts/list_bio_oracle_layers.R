#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(sdmpredictors)
  library(yaml)
  library(dplyr)
  library(stringr)
  library(readr)
})

cfg_path <- "configs/bio_oracle.yaml"
if (!file.exists(cfg_path)) stop("Missing configs/bio_oracle.yaml")
cfg <- yaml::read_yaml(cfg_path)

layers <- sdmpredictors::list_layers(datasets = "Bio-ORACLE")
layers$search_name <- paste(layers$dataset_code, layers$layer_code, layers$name, sep="|")

rows <- list()
for (slice_name in names(cfg$time_slices)) {
  pats <- cfg$time_slices[[slice_name]]
  pattern <- paste(pats, collapse=".*")
  for (v in cfg$variables) {
    ix <- stringr::str_detect(tolower(layers$search_name), tolower(v)) &
          stringr::str_detect(tolower(layers$search_name), tolower(pattern))
    if (!any(ix)) next
    sel <- layers[ix, c("dataset_code","layer_code","name","units","citation")]
    if (nrow(sel) == 0) next
    sel$variable <- v
    sel$time_slice <- slice_name
    rows[[length(rows)+1]] <- sel
  }
}
out <- if (length(rows)>0) dplyr::bind_rows(rows) else layers[0,]
dir.create("reports", showWarnings = FALSE, recursive = TRUE)
readr::write_csv(out, "reports/bio_oracle_layer_index.csv")
message("Wrote reports/bio_oracle_layer_index.csv with ", nrow(out), " rows.")
