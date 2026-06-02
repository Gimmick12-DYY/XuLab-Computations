#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# save_peak_matrix.R
#
# Convert the mm tier produced by build_peak_matrix.py into a dgCMatrix RDS
# that drops into paths.input_rds of the scBasset / scBasset_TF pipelines.
#
# Reads:
#   <mm-dir>/matrix.mtx.gz
#   <mm-dir>/regions.tsv.gz   (chr:start-end, one per line, in row order)
#   <mm-dir>/barcodes.tsv.gz  (cell barcodes, one per line, in column order)
#
# Writes:
#   <out>     a dgCMatrix RDS with rownames=regions, colnames=barcodes.
#
# Usage:
#   Rscript save_peak_matrix.R --mm-dir /work/.../mm --out /work/.../CTCF_peak_matrix.rds
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(Matrix)
  library(optparse)
})

option_list <- list(
  make_option(c("-m", "--mm-dir"), type = "character",
              help = "Directory with matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz"),
  make_option(c("-o", "--out"),    type = "character",
              help = "Output RDS path (dgCMatrix)")
)
opt <- parse_args(OptionParser(option_list = option_list))

mm_dir <- opt[["mm-dir"]]
out_path <- opt$out
if (is.null(mm_dir) || is.null(out_path)) {
  stop("Both --mm-dir and --out are required.")
}

mtx_path     <- file.path(mm_dir, "matrix.mtx.gz")
regions_path <- file.path(mm_dir, "regions.tsv.gz")
barcodes_path <- file.path(mm_dir, "barcodes.tsv.gz")
for (p in c(mtx_path, regions_path, barcodes_path)) {
  if (!file.exists(p)) stop(sprintf("missing input: %s", p))
}

message(sprintf("[save_peak_matrix] reading %s", mtx_path))
mat <- as(readMM(mtx_path), "CsparseMatrix")   # dgCMatrix

regions <- readLines(gzfile(regions_path))
close(gzfile(regions_path))                    # gzfile() leaks otherwise
barcodes <- readLines(gzfile(barcodes_path))

if (length(regions) != nrow(mat)) {
  stop(sprintf("regions count (%d) != matrix rows (%d)",
               length(regions), nrow(mat)))
}
if (length(barcodes) != ncol(mat)) {
  stop(sprintf("barcodes count (%d) != matrix cols (%d)",
               length(barcodes), ncol(mat)))
}

# rowname format check: <chrom>:<start>-<end>. Chrom names may contain
# underscores (contigs like chr1_KI270706v1_random), so the chrom group is
# "anything not a colon". Mirrors the regex used by scBasset_TF's
# 02_prepare_seqs.py.
bad <- !grepl("^[^:]+:\\d+-\\d+$", regions)
if (any(bad)) {
  stop(sprintf("%d rownames do not match '<chrom>:<start>-<end>' (first bad: %s)",
               sum(bad), regions[which(bad)[1]]))
}

rownames(mat) <- regions
colnames(mat) <- barcodes

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
message(sprintf("[save_peak_matrix] saving %s", out_path))
saveRDS(mat, out_path)

message(sprintf("[save_peak_matrix] done: %d peaks x %d cells, nnz=%d",
                nrow(mat), ncol(mat), length(mat@x)))
