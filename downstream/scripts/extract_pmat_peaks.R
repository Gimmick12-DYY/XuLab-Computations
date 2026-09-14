#!/usr/bin/env Rscript
# Write per-TF peak BEDs from TF1000cells.pmat.mtx.rds for de novo discovery.
# Peaks are the original MACS calls (median ~339 bp), not 1 kb bins.
# Rank by occupancy (fraction of that TF's cells); keep peaks with >= --min-cells.
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(
  matrix = "/work/users/d/y/dyy12/XuLab/data/TF1000cells.pmat.mtx.rds",
  meta   = "/work/users/d/y/dyy12/XuLab/data/TF1000cells.meta.csv",
  out    = "/work/users/d/y/dyy12/XuLab/downstream/motif_pmat",
  min_cells = 2L,
  n_fg = 20000L,
  tfs = NULL
)
i <- 1L
while (i <= length(args)) {
  if (args[[i]] == "--matrix") { opt$matrix <- args[[i + 1]]; i <- i + 2 }
  else if (args[[i]] == "--meta") { opt$meta <- args[[i + 1]]; i <- i + 2 }
  else if (args[[i]] == "--out-root") { opt$out <- args[[i + 1]]; i <- i + 2 }
  else if (args[[i]] == "--min-cells") { opt$min_cells <- as.integer(args[[i + 1]]); i <- i + 2 }
  else if (args[[i]] == "--n-fg") { opt$n_fg <- as.integer(args[[i + 1]]); i <- i + 2 }
  else if (args[[i]] == "--tfs") { opt$tfs <- strsplit(args[[i + 1]], "[, ]+")[[1]]; i <- i + 2 }
  else stop("unknown arg: ", args[[i]])
}

message("[extract] loading ", opt$matrix)
m <- readRDS(opt$matrix)
if (!inherits(m, "CsparseMatrix")) m <- as(m, "CsparseMatrix")
meta <- read.csv(opt$meta, stringsAsFactors = FALSE)
cn <- colnames(m)
rn <- rownames(m)
parts <- strsplit(rn, "[:-]")
chrom <- vapply(parts, `[`, "", 1)
start <- as.integer(vapply(parts, `[`, "", 2))
end   <- as.integer(vapply(parts, `[`, "", 3))

tfs <- opt$tfs
if (is.null(tfs) || !length(tfs)) {
  tfs <- sort(unique(meta$TF))
}
tfs <- unique(toupper(tfs))

dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)
for (tf in tfs) {
  cols <- which(cn %in% meta$barcode[toupper(meta$TF) == tf])
  key <- tolower(tf)
  outdir <- file.path(opt$out, key)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  bed <- file.path(outdir, paste0(key, ".pmat_peaks.bed"))
  if (length(cols) == 0L) {
    message(sprintf("[skip] %s: 0 cells", tf))
    next
  }
  rs <- as.numeric(rowSums(m[, cols, drop = FALSE] > 0))
  keep <- which(rs >= opt$min_cells)
  if (!length(keep)) {
    message(sprintf("[skip] %s: 0 peaks with >=%d cells", tf, opt$min_cells))
    next
  }
  occ <- rs[keep] / length(cols)
  o <- order(occ, decreasing = TRUE)
  if (length(o) > opt$n_fg) o <- o[seq_len(opt$n_fg)]
  idx <- keep[o]
  score <- round(1000 * occ[o])
  write.table(
    data.frame(chrom[idx], start[idx], end[idx],
               sprintf("%s_p%d", key, seq_along(idx) - 1L),
               score, ".", stringsAsFactors = FALSE),
    bed, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE
  )
  message(sprintf("[%s] cells=%d peaks=%d -> %s", tf, length(cols), length(idx), bed))
}
