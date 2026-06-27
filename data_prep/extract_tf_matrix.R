#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# extract_tf_matrix.R
#
# Subset the combined single-cell CUT&Tag matrix to ONE TF's cells and save it
# as a per-TF .rds in the SAME format as CTCF_bin1000_mtx.rds (a dgCMatrix with
# rownames = bins "chr:start-end" and colnames = barcodes). The result is a
# drop-in `paths.input_rds` for any pipeline (unified / scBasset / cisTopic /
# PUscOpen) -- exactly how CTCF_bin1000_mtx.rds was produced.
#
# Inputs:
#   --matrix  TF1000cells.bin1000.mtx.rds   bins x ALL cells (dgCMatrix)
#   --meta    TF1000cells.meta.csv          PER-CELL: DNA_id, cell, TF, barcode, subset
#   --tf      e.g. NFYA                      TF to extract (must match meta$TF)
#   --out     <TF>_bin1000_mtx.rds           output .rds (default: <tf>_bin1000_mtx.rds)
#   --min-cells-per-bin  drop bins with signal in fewer than this many of the
#                        TF's cells (default 0 = keep all bins, same row order as CTCF).
#
# Usage (cistopic env):
#   Rscript extract_tf_matrix.R --matrix TF1000cells.bin1000.mtx.rds \
#     --meta TF1000cells.meta.csv --tf NFYA --out NFYA_bin1000_mtx.rds
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(Matrix)
  library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option('--matrix', type = 'character', default = 'TF1000cells.bin1000.mtx.rds'),
  make_option('--meta',   type = 'character', default = 'TF1000cells.meta.csv'),
  make_option('--tf',     type = 'character', default = NULL),
  make_option('--out',    type = 'character', default = NULL),
  make_option('--min-cells-per-bin', type = 'integer', default = 0L)
)))
if (is.null(opt$tf)) stop('--tf is required (e.g. --tf NFYA)')
out <- if (is.null(opt$out)) sprintf('%s_bin1000_mtx.rds', opt$tf) else opt$out

message('[extract] Reading matrix: ', opt$matrix)
m <- readRDS(opt$matrix)
if (!inherits(m, 'CsparseMatrix')) m <- as(m, 'CsparseMatrix')
if (is.null(colnames(m))) stop('Matrix has no colnames (barcodes).')
message(sprintf('[extract] Combined matrix: %d bins x %d cells', nrow(m), ncol(m)))

meta <- read.csv(opt$meta, stringsAsFactors = FALSE)
if (!all(c('barcode', 'TF') %in% names(meta)))
  stop("meta must have 'barcode' and 'TF' columns (the PER-CELL file, not a "
       , "TF->count summary); got: ", paste(names(meta), collapse = ', '))
tf_barcodes <- meta$barcode[meta$TF == opt$tf]
if (length(tf_barcodes) == 0L)
  stop(sprintf("No cells with TF=='%s' in meta. Check spelling / available TFs.", opt$tf))

cols <- which(colnames(m) %in% tf_barcodes)
if (length(cols) == 0L)
  stop(sprintf("TF '%s' has %d cells in meta but NONE match matrix colnames -- barcode "
               , opt$tf, length(tf_barcodes)),
       'format mismatch between meta$barcode and colnames(matrix).')
message(sprintf('[extract] TF %s: %d cells in meta, %d matched in matrix.',
                opt$tf, length(tf_barcodes), length(cols)))

sub <- m[, cols, drop = FALSE]

# Optional: drop bins never (or rarely) seen in this TF's cells. Default keeps
# every bin so the row order/universe stays identical to CTCF_bin1000_mtx.rds.
if (opt$`min-cells-per-bin` > 0L) {
  keep <- rowSums(sub > 0) >= opt$`min-cells-per-bin`
  message(sprintf('[extract] min-cells-per-bin=%d: keeping %d / %d bins.',
                  opt$`min-cells-per-bin`, sum(keep), nrow(sub)))
  sub <- sub[keep, , drop = FALSE]
}

message(sprintf('[extract] Output: %d bins x %d cells, nnz=%d (%.2f%% dense)',
                nrow(sub), ncol(sub), length(sub@x),
                100 * length(sub@x) / (as.numeric(nrow(sub)) * ncol(sub))))
saveRDS(sub, out)
message('[extract] Wrote ', out)
message('[extract] Point a pipeline at it via paths.input_rds: ', normalizePath(out))
