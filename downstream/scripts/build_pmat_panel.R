#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# build_pmat_panel.R
#
# Build a peak x TF occupancy panel from the ORIGINAL peak-level CUT&Tag matrix
# (TF1000cells.pmat.mtx.rds) -- not the 1 kb bins used for imputation.
#
# Those peaks were MACS-called on the combined fragments: median width ~339 bp,
# so they are already at motif-discovery scale. Occupancy = fraction of a TF's
# cells with >=1 fragment in the peak. Rank-normalize each TF's column the same
# way as build_panel_matrix.py so depth/breadth cannot masquerade as specificity.
#
# Writes:
#   <out-prefix>.peaks.bed          chrom start end name
#   <out-prefix>.occupancy.rds      dgCMatrix peaks x TFs (values in [0,1])
#   <out-prefix>.ranknorm.rds       same shape, percentile ranks of nonzeros
#   <out-prefix>.meta.csv           tf, n_cells, n_peaks_ge2
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(a, b) if (is.null(a) || is.na(a) || a == "") b else a

opt <- list(
  matrix = "/work/users/d/y/dyy12/XuLab/data/TF1000cells.pmat.mtx.rds",
  meta   = "/work/users/d/y/dyy12/XuLab/data/TF1000cells.meta.csv",
  out    = "/work/users/d/y/dyy12/XuLab/downstream/motif/pmat_panel"
)
i <- 1L
while (i <= length(args)) {
  if (args[[i]] == "--matrix") { opt$matrix <- args[[i + 1]]; i <- i + 2 }
  else if (args[[i]] == "--meta") { opt$meta <- args[[i + 1]]; i <- i + 2 }
  else if (args[[i]] == "--out-prefix") { opt$out <- args[[i + 1]]; i <- i + 2 }
  else stop("unknown arg: ", args[[i]])
}

dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)

message("[pmat] loading ", opt$matrix)
m <- readRDS(opt$matrix)
if (!inherits(m, "CsparseMatrix")) m <- as(m, "CsparseMatrix")
meta <- read.csv(opt$meta, stringsAsFactors = FALSE)
stopifnot(all(c("barcode", "TF") %in% names(meta)))
cn <- colnames(m)
rn <- rownames(m)
stopifnot(!is.null(rn), !is.null(cn))

# Peak BED from rownames chr:start-end
parts <- strsplit(rn, "[:-]")
chrom <- vapply(parts, `[`, "", 1)
start <- as.integer(vapply(parts, `[`, "", 2))
end   <- as.integer(vapply(parts, `[`, "", 3))
bed <- data.frame(
  chrom = chrom, start = start, end = end,
  name = sprintf("pmat_%d", seq_along(rn) - 1L),
  stringsAsFactors = FALSE
)
bed_path <- paste0(opt$out, ".peaks.bed")
write.table(bed, bed_path, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
message(sprintf("[pmat] %d peaks (median width %d bp) -> %s",
                nrow(bed), as.integer(median(end - start)), bed_path))

tfs <- sort(unique(meta$TF))
n_tf <- length(tfs)
occ <- Matrix(0, nrow = nrow(m), ncol = n_tf, sparse = TRUE)
colnames(occ) <- tfs
rownames(occ) <- rn
n_cells <- integer(n_tf); names(n_cells) <- tfs
n_ge2 <- integer(n_tf); names(n_ge2) <- tfs

for (i in seq_along(tfs)) {
  tf <- tfs[[i]]
  cols <- which(cn %in% meta$barcode[meta$TF == tf])
  n_cells[[tf]] <- length(cols)
  if (length(cols) == 0L) {
    message(sprintf("[%d/%d] %s: 0 cells", i, n_tf, tf))
    next
  }
  rs <- rowSums(m[, cols, drop = FALSE] > 0)
  n_ge2[[tf]] <- sum(rs >= 2)
  occ[, i] <- rs / length(cols)
  message(sprintf("[%d/%d] %s: cells=%d peaks>=2=%d", i, n_tf, tf, length(cols), n_ge2[[tf]]))
}

# Rank-normalize nonzero entries per TF into (0, 1]
ranknorm <- occ
for (j in seq_len(n_tf)) {
  v <- occ[, j]
  nz <- which(v > 0)
  if (length(nz) == 0L) next
  o <- order(v[nz], method = "radix")
  r <- numeric(length(nz))
  r[o] <- seq_along(nz)
  ranknorm[nz, j] <- r / length(nz)
}

saveRDS(occ, paste0(opt$out, ".occupancy.rds"))
saveRDS(ranknorm, paste0(opt$out, ".ranknorm.rds"))
write.csv(
  data.frame(tf = tfs, n_cells = n_cells[tfs], n_peaks_ge2 = n_ge2[tfs],
             stringsAsFactors = FALSE),
  paste0(opt$out, ".meta.csv"), row.names = FALSE
)

# Dense float32, column-major, so Python can pack the same npz layout as
# build_panel_matrix.py (tf_specific_regions.py consumes that).
f32_path <- paste0(opt$out, ".ranknorm.f32")
con <- file(f32_path, "wb")
for (j in seq_len(n_tf)) {
  writeBin(as.numeric(ranknorm[, j]), con, size = 4, endian = "little")
}
close(con)
writeLines(tfs, paste0(opt$out, ".tfs.txt"))
message("[pmat] wrote ", opt$out, ".{peaks.bed,occupancy.rds,ranknorm.rds,ranknorm.f32,meta.csv}")
