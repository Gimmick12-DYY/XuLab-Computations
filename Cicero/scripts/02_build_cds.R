#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# 02_build_cds.R
#
# Read mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz} written by
# 01_export_rds_to_mm.R and build a Monocle3 cell_data_set + LSI + UMAP that
# 03_run_cicero.R will hand to make_cicero_cds() and run_cicero().
#
# Cicero parses peak names to recover genomic coordinates. It accepts either
# colon/dash form ("chr1:100-200") or underscore form ("chr1_100_200") — we
# rewrite the row names to the underscore form because make_cicero_cds() and
# run_cicero() call strsplit("_") internally in several code paths.
#
# Outputs (under <work>/cds/):
#   cds.rds                          the processed Monocle3 object
#   regions_used_canonical.tsv.gz   colon-form names, one per CDS row
#   regions_used_underscore.tsv.gz  underscore-form names, one per CDS row
#   barcodes_used.tsv.gz            barcodes, one per CDS column
#
# Usage:
#   Rscript 02_build_cds.R --config configs/default.yaml [--work-dir <path>]
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Matrix)
  library(monocle3)
  library(optparse)
  library(yaml)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

option_list <- list(
  make_option("--config",   type = "character", help = "Path to YAML config"),
  make_option("--work-dir", type = "character", default = NULL,
              help = "Override paths.work_dir from the config")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$config)) stop("--config is required")

cfg <- yaml::read_yaml(opt$config)
work_dir <- normalizePath(
  if (!is.null(opt$`work-dir`)) opt$`work-dir` else cfg$paths$work_dir,
  mustWork = FALSE
)
mm_dir  <- file.path(work_dir, "mm")
cds_dir <- file.path(work_dir, "cds")
dir.create(cds_dir, showWarnings = FALSE, recursive = TRUE)

mtx_path <- file.path(mm_dir, "matrix.mtx.gz")
reg_path <- file.path(mm_dir, "regions.tsv.gz")
bar_path <- file.path(mm_dir, "barcodes.tsv.gz")
for (p in c(mtx_path, reg_path, bar_path)) {
  if (!file.exists(p)) stop("Missing input: ", p, " — run 01_export_rds_to_mm.R first")
}

message(sprintf("[02] Reading %s", mtx_path))
mat <- as(Matrix::readMM(mtx_path), "CsparseMatrix")

regions  <- readLines(gzfile(reg_path))
barcodes <- readLines(gzfile(bar_path))
stopifnot(nrow(mat) == length(regions), ncol(mat) == length(barcodes))
dimnames(mat) <- list(regions, barcodes)
message(sprintf("[02] Loaded %d regions x %d cells (nnz=%d)",
                nrow(mat), ncol(mat), length(mat@x)))

# ---- Cicero-style row names (chr_start_end) ---------------------------------
to_underscore <- function(x) {
  out <- gsub("^([^:]+):(\\d+)-(\\d+)$", "\\1_\\2_\\3", x)
  bad <- !grepl("^[^_]+_\\d+_\\d+$", out)
  if (any(bad)) stop("Could not parse region names; first offender: ", out[which(bad)[1]])
  out
}
regions_us <- to_underscore(regions)
rownames(mat) <- regions_us

# ---- Drop zero rows (Cicero requires nonzero peaks) -------------------------
nz <- Matrix::rowSums(mat > 0) > 0
n_drop <- sum(!nz)
if (n_drop > 0) {
  message(sprintf("[02] Dropping %d / %d all-zero rows", n_drop, length(nz)))
  mat        <- mat[nz, , drop = FALSE]
  regions    <- regions[nz]
  regions_us <- regions_us[nz]
}

# ---- Per-peak cell-count filter (recommended >= 2) --------------------------
min_cells <- as.integer(cfg$cicero$min_cells_per_peak %||% 0L)
if (min_cells > 0L) {
  cnts <- Matrix::rowSums(mat > 0)
  keep <- cnts >= min_cells
  message(sprintf("[02] Peak filter (cells >= %d): keep %d / %d",
                  min_cells, sum(keep), length(keep)))
  mat        <- mat[keep, , drop = FALSE]
  regions    <- regions[keep]
  regions_us <- regions_us[keep]
}

# ---- Binarise (Cicero treats peaks as binary accessible/inaccessible) -------
mat <- as(mat > 0, "dgCMatrix") * 1

# ---- Optional cell subsample (memory) ---------------------------------------
max_cells <- as.integer(cfg$cicero$max_cells %||% 0L)
if (max_cells > 0L && ncol(mat) > max_cells) {
  set.seed(cfg$cicero$random_seed %||% 555L)
  idx <- sort(sample(seq_len(ncol(mat)), max_cells))
  mat <- mat[, idx, drop = FALSE]
  message(sprintf("[02] Subsampled cells to %d", max_cells))
}

message(sprintf("[02] Post-filter matrix: %d peaks x %d cells",
                nrow(mat), ncol(mat)))

# ---- Build Monocle3 CDS -----------------------------------------------------
# Cicero's downstream parsers want a `site_name` column in fData; gene_short_name
# is the universal Monocle3 identifier column.
peak_meta <- data.frame(
  site_name        = rownames(mat),
  gene_short_name  = rownames(mat),
  row.names        = rownames(mat),
  stringsAsFactors = FALSE
)
cell_meta <- data.frame(
  cell             = colnames(mat),
  row.names        = colnames(mat),
  stringsAsFactors = FALSE
)

cds <- new_cell_data_set(
  expression_data = mat,
  cell_metadata   = cell_meta,
  gene_metadata   = peak_meta
)

num_dim <- as.integer(cfg$cicero$num_dim %||% 50L)
message(sprintf("[02] preprocess_cds(method='LSI', num_dim=%d)", num_dim))
cds <- detect_genes(cds)
cds <- estimate_size_factors(cds)
cds <- preprocess_cds(cds, method = "LSI", num_dim = num_dim)

message("[02] reduce_dimension(reduction_method='UMAP', preprocess_method='LSI')")
cds <- reduce_dimension(cds, reduction_method = "UMAP", preprocess_method = "LSI")

# ---- Persist ----------------------------------------------------------------
saveRDS(cds, file = file.path(cds_dir, "cds.rds"))

gz <- gzfile(file.path(cds_dir, "regions_used_canonical.tsv.gz"), "w")
writeLines(regions, gz); close(gz)

gz <- gzfile(file.path(cds_dir, "regions_used_underscore.tsv.gz"), "w")
writeLines(regions_us, gz); close(gz)

gz <- gzfile(file.path(cds_dir, "barcodes_used.tsv.gz"), "w")
writeLines(colnames(mat), gz); close(gz)

writeLines(
  c(
    sprintf("n_peaks_in\t%d", nrow(mat)),
    sprintf("n_cells_in\t%d", ncol(mat)),
    sprintf("min_cells_per_peak\t%d", min_cells),
    sprintf("num_dim\t%d", num_dim),
    sprintf("cds_path\t%s", file.path(cds_dir, "cds.rds"))
  ),
  con = file.path(cds_dir, "cds_info.tsv")
)

message(sprintf("[02] Done. CDS at %s", file.path(cds_dir, "cds.rds")))
