#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# tf_abundance_report.R
#
# Rank the TFs in the combined single-cell CUT&Tag matrix by SIGNAL ABUNDANCE,
# so we can pick an example TF with denser signal than CTCF -- on measured
# numbers, not a biological guess.
#
# For each TF it reports:
#   n_cells              cells assigned to the TF (training size)
#   mean_bins_per_cell   mean # nonzero bins per cell  <- SIGNAL DENSITY
#   median_bins_per_cell median of the same
#   n_bins_covered       # distinct bins with signal in >=1 cell (breadth)
#   frac_genome_covered  n_bins_covered / total bins
#   total_nonzero        total nonzero (bin, cell) entries
#
# "More abundant signal than CTCF" = higher mean_bins_per_cell and/or
# frac_genome_covered than the CTCF row (printed for reference).
#
# Inputs:
#   --matrix  TF1000cells.bin1000.mtx.rds   bins x ALL cells (dgCMatrix; colnames=barcodes)
#   --meta    TF1000cells.meta.csv          PER-CELL: DNA_id, cell, TF, barcode, subset
# Output:
#   --out     tf_abundance_report.tsv       one row per TF, sorted by density desc
#
# Usage (cistopic env has Matrix + optparse):
#   Rscript tf_abundance_report.R \
#     --matrix TF1000cells.bin1000.mtx.rds --meta TF1000cells.meta.csv \
#     --out tf_abundance_report.tsv
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(Matrix)
  library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option('--matrix', type = 'character', default = 'TF1000cells.bin1000.mtx.rds'),
  make_option('--meta',   type = 'character', default = 'TF1000cells.meta.csv'),
  make_option('--out',    type = 'character', default = 'tf_abundance_report.tsv')
)))

message('[abund] Reading matrix: ', opt$matrix)
m <- readRDS(opt$matrix)
if (!inherits(m, 'CsparseMatrix')) m <- as(m, 'CsparseMatrix')
n_bins <- nrow(m); n_cells <- ncol(m)
message(sprintf('[abund] Matrix: %d bins x %d cells', n_bins, n_cells))
if (is.null(colnames(m)))
  stop('Matrix has no colnames; cannot map columns to barcodes/TF.')

message('[abund] Reading meta: ', opt$meta)
meta <- read.csv(opt$meta, stringsAsFactors = FALSE)
if (!all(c('barcode', 'TF') %in% names(meta)))
  stop("meta must have 'barcode' and 'TF' columns (the PER-CELL file); got: ",
       paste(names(meta), collapse = ', '))

# Align matrix columns to meta by barcode.
tf_by_col <- meta$TF[match(colnames(m), meta$barcode)]
n_unmapped <- sum(is.na(tf_by_col))
if (n_unmapped > 0)
  message(sprintf('[abund] WARNING: %d / %d columns have no barcode match in meta (dropped).',
                  n_unmapped, n_cells))

# Nonzeros per cell = peaks/cell. For a dgCMatrix this is diff(m@p) (O(1) per col).
nnz_per_cell <- diff(m@p)

tfs <- sort(unique(tf_by_col[!is.na(tf_by_col)]))
rows <- lapply(tfs, function(tf) {
  cols <- which(tf_by_col == tf)
  sub  <- m[, cols, drop = FALSE]
  covered <- sum(rowSums(sub) > 0)            # bins with signal in >=1 cell
  data.frame(
    TF                   = tf,
    n_cells              = length(cols),
    mean_bins_per_cell   = round(mean(nnz_per_cell[cols]), 2),
    median_bins_per_cell = median(nnz_per_cell[cols]),
    n_bins_covered       = covered,
    frac_genome_covered  = round(covered / n_bins, 4),
    total_nonzero        = sum(nnz_per_cell[cols]),
    stringsAsFactors     = FALSE
  )
})
df <- do.call(rbind, rows)
df <- df[order(-df$mean_bins_per_cell), ]

write.table(df, opt$out, sep = '\t', row.names = FALSE, quote = FALSE)
message('[abund] Wrote ', opt$out)

ctcf <- df[df$TF == 'CTCF', , drop = FALSE]
message('\n--- CTCF reference ---')
print(ctcf, row.names = FALSE)
message('\n--- Top 15 by mean_bins_per_cell (signal density) ---')
print(head(df, 15), row.names = FALSE)
if (nrow(ctcf) == 1L) {
  denser <- df[df$mean_bins_per_cell > ctcf$mean_bins_per_cell, ]
  message(sprintf('\n[abund] %d TFs have denser per-cell signal than CTCF (mean_bins_per_cell > %.2f).',
                  nrow(denser), ctcf$mean_bins_per_cell))
}
