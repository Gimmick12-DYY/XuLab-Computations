#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# export_pmat_tf_mm.R
#
# Subset the colleague's FRAGMENT-CALLED peak matrix (peaks x cells, all TFs
# pooled: data/TF1000cells.pmat.mtx.rds) to ONE TF's cells and write
#   mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
# for the Cicero CDS step (02_build_cds.R).
#
# This REPLACES export_tf_cicero_mm.py, which reconstructed coarse peaks from the
# 1 kb-bin matrix (synthetic fragments at bin midpoints) -- that quantized peaks
# to a 1 kb grid and could never match the lab's fragment-resolution sites. The
# pmat peaks are the real variable-width MACS/CUT&Tag peaks, so Cicero now runs on
# the correct universe.
#
# Per-TF cells come from a barcode list (one per line, plain or .gz), e.g.
# unified/work/<tf>/mm/barcodes.tsv.gz. We intersect it with the pmat columns and
# fail loudly if the barcode formats don't line up.
#
# Usage:
#   Rscript export_pmat_tf_mm.R --pmat data/TF1000cells.pmat.mtx.rds \
#     --barcodes unified/work/rbbp4/mm/barcodes.tsv.gz --out-mm <work>/cicero/mm \
#     --min-cells-per-peak 2 --min-peaks-per-cell 0
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(Matrix)
  library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--pmat", type = "character", help = "peaks x cells RDS (dgCMatrix)"),
  make_option("--barcodes", type = "character",
              help = "one TF-cell barcode per line (plain or .gz)"),
  make_option("--out-mm", type = "character", help = "output mm/ dir"),
  make_option("--min-cells-per-peak", type = "integer", default = 2L),
  make_option("--min-peaks-per-cell", type = "integer", default = 0L)
)))
for (r in c("pmat", "barcodes", "out-mm"))
  if (is.null(opt[[r]])) stop("--", r, " is required")

read_lines_maybe_gz <- function(path) {
  con <- if (grepl("\\.gz$", path)) gzfile(path, "rt") else file(path, "rt")
  on.exit(close(con))
  trimws(readLines(con))
}

message(sprintf("[pmat] reading %s", opt$pmat))
pm <- readRDS(opt$pmat)
if (!is(pm, "CsparseMatrix")) pm <- as(pm, "CsparseMatrix")

want <- read_lines_maybe_gz(opt$barcodes)
want <- want[nzchar(want)]
common <- intersect(colnames(pm), want)
message(sprintf("[pmat] %d peaks x %d cells; TF barcodes=%d matched=%d (%.1f%%)",
                nrow(pm), ncol(pm), length(want), length(common),
                100 * length(common) / max(length(want), 1)))
if (length(common) < 50L)
  stop("too few matched barcodes (", length(common), "). Barcode formats differ?\n",
       "  pmat ex: ", paste(head(colnames(pm), 2), collapse = " | "), "\n",
       "  want ex: ", paste(head(want, 2), collapse = " | "))

sub <- pm[, common, drop = FALSE]

# filter peaks (>= min cells), then optionally cells (>= min peaks)
peak_keep <- Matrix::rowSums(sub > 0) >= opt$`min-cells-per-peak`
sub <- sub[peak_keep, , drop = FALSE]
if (opt$`min-peaks-per-cell` > 0L) {
  cell_keep <- Matrix::colSums(sub > 0) >= opt$`min-peaks-per-cell`
  sub <- sub[, cell_keep, drop = FALSE]
}
message(sprintf("[pmat] kept %d peaks x %d cells (nnz=%d) after min-cells/peak=%d, min-peaks/cell=%d",
                nrow(sub), ncol(sub), length(sub@x),
                opt$`min-cells-per-peak`, opt$`min-peaks-per-cell`))
if (nrow(sub) < 100L || ncol(sub) < 50L)
  stop("subset too small after filtering (peaks=", nrow(sub), ", cells=", ncol(sub), ")")

out <- opt$`out-mm`
dir.create(out, showWarnings = FALSE, recursive = TRUE)
mtx <- file.path(out, "matrix.mtx")
Matrix::writeMM(as(sub, "CsparseMatrix"), mtx)
if (system2("gzip", c("-f", shQuote(mtx))) != 0L)
  stop("gzip failed on ", mtx)

write_gz <- function(x, path) {
  con <- gzfile(path, "wt"); on.exit(close(con)); writeLines(x, con)
}
write_gz(rownames(sub), file.path(out, "regions.tsv.gz"))    # colon-form; 02_build_cds converts
write_gz(colnames(sub), file.path(out, "barcodes.tsv.gz"))
message(sprintf("[pmat] wrote %s/{matrix.mtx.gz,regions.tsv.gz,barcodes.tsv.gz}", out))
