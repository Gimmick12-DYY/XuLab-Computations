#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# 01_export_rds_to_mm.R
#
# The dataloader script for the modified cisTopic pipeline for ManyTFs.
#
# Load an .rds file that contains a single-cell regions x cells count matrix
# and export it as a Matrix Market file (.mtx.gz) plus accompanying
# regions.tsv.gz and barcodes.tsv.gz, which is the format consumed by
# scripts/02_build_cistopic_obj.py.
#
# The .rds object may be any of:
#   * a sparse Matrix::dgCMatrix / dgTMatrix / dgRMatrix
#   * a base matrix
#   * a Seurat object (counts from the default assay are used)
#   * a SingleCellExperiment / SummarizedExperiment (assay "counts" is used)
#   * a named list containing one of the above under a "matrix" / "counts" slot
#
# Rownames are expected to be region IDs (e.g. "chr1:1-1000" or "chr1_1_1000")
# and colnames are expected to be cell barcodes. If either is missing, synthetic
# names are generated and a warning is printed.
#
# Usage:
#   Rscript 01_export_rds_to_mm.R \
#       --input  /path/to/CTCF_bin1000_mtx.rds \
#       --outdir /path/to/work/cistopic_ctcf/mm
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Matrix)
  library(optparse)
})

option_list <- list(
  make_option(c("-i", "--input"),  type = "character", help = "Input file (.rds/.RData/.rda/.mtx)"),
  make_option(c("-o", "--outdir"), type = "character", help = "Output directory")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$input) || is.null(opt$outdir))
  stop("Both --input and --outdir are required.")

dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

load_input_object <- function(path) {
  # 1) Standard .rds
  obj <- tryCatch(readRDS(path), error = function(e) e)
  if (!inherits(obj, "error")) {
    message("[01] Loaded with readRDS().")
    return(obj)
  }
  message("[01] readRDS() failed: ", conditionMessage(obj))

  # 2) .RData/.rda (or misnamed file) loaded into a temporary environment
  tmp_env <- new.env(parent = emptyenv())
  ld <- tryCatch(load(path, envir = tmp_env), error = function(e) e)
  if (!inherits(ld, "error")) {
    message("[01] Loaded with load() (RData-style).")
    if (length(ld) == 1L) return(get(ld[[1]], envir = tmp_env))
    return(setNames(lapply(ld, function(nm) get(nm, envir = tmp_env)), ld))
  }
  message("[01] load() failed: ", conditionMessage(ld))

  # 3) Matrix Market with wrong extension
  mm <- tryCatch(Matrix::readMM(path), error = function(e) e)
  if (!inherits(mm, "error")) {
    message("[01] Loaded with Matrix::readMM() (Matrix Market format).")
    return(as(mm, "CsparseMatrix"))
  }
  message("[01] readMM() failed: ", conditionMessage(mm))

  stop(
    "Could not parse input as RDS, RData, or Matrix Market.\n",
    "Please verify the file format and extension."
  )
}

message("[01] Reading ", opt$input)
obj <- load_input_object(opt$input)

# ---- Resolve to a sparse dgCMatrix -------------------------------------------
extract_counts <- function(x) {
  if (inherits(x, "dgCMatrix"))               return(x)
  if (inherits(x, c("dgTMatrix", "dgRMatrix"))) return(as(x, "CsparseMatrix"))
  if (is.matrix(x))                           return(as(x, "CsparseMatrix"))

  # Seurat
  if (inherits(x, "Seurat")) {
    if (!requireNamespace("Seurat", quietly = TRUE))
      stop("Seurat package required for this .rds but not installed.")
    m <- Seurat::GetAssayData(x, layer = "counts")
    if (is.null(m) || length(m) == 0)
      m <- Seurat::GetAssayData(x, slot = "counts")
    return(as(m, "CsparseMatrix"))
  }

  # SingleCellExperiment / SummarizedExperiment
  if (inherits(x, c("SingleCellExperiment", "SummarizedExperiment"))) {
    if (!requireNamespace("SummarizedExperiment", quietly = TRUE))
      stop("SummarizedExperiment required for this .rds but not installed.")
    m <- SummarizedExperiment::assay(x, "counts")
    return(as(m, "CsparseMatrix"))
  }

  # Named list: look for an obvious slot
  if (is.list(x)) {
    for (nm in c("matrix", "counts", "mtx", "mat", "X")) {
      if (!is.null(x[[nm]])) return(extract_counts(x[[nm]]))
    }
  }

  stop("Could not find a count matrix inside the .rds (class: ",
       paste(class(x), collapse = "/"), ").")
}

mat <- extract_counts(obj)
message(sprintf("[01] Matrix is %d regions x %d cells (nnz = %d).",
                nrow(mat), ncol(mat), length(mat@x)))

# ---- Ensure row/col names ----------------------------------------------------
if (is.null(rownames(mat))) {
  warning("No rownames found; synthesising 'region_<i>'.")
  rownames(mat) <- sprintf("region_%010d", seq_len(nrow(mat)))
}
if (is.null(colnames(mat))) {
  warning("No colnames found; synthesising 'cell_<j>'.")
  colnames(mat) <- sprintf("cell_%06d", seq_len(ncol(mat)))
}

# Normalise region separators to 'chr:start-end' if they use underscores.
fix_region_id <- function(x) {
  m <- regmatches(x, regexec("^([^:_]+)[_:]?(\\d+)[-_](\\d+)$", x))
  vapply(seq_along(x), function(k) {
    hit <- m[[k]]
    if (length(hit) == 4L) sprintf("%s:%s-%s", hit[2], hit[3], hit[4]) else x[k]
  }, character(1))
}
rn <- rownames(mat)
if (any(grepl("_", rn) & !grepl(":", rn))) rownames(mat) <- fix_region_id(rn)

# ---- Write out ---------------------------------------------------------------
mm_path     <- file.path(opt$outdir, "matrix.mtx")
gz_path     <- paste0(mm_path, ".gz")
region_path <- file.path(opt$outdir, "regions.tsv.gz")
cell_path   <- file.path(opt$outdir, "barcodes.tsv.gz")

message("[01] Writing ", gz_path)
Matrix::writeMM(mat, mm_path)
system2("gzip", c("-f", mm_path))           # gzip in-place

message("[01] Writing ", region_path)
gz1 <- gzfile(region_path, "w")
writeLines(rownames(mat), gz1); close(gz1)

message("[01] Writing ", cell_path)
gz2 <- gzfile(cell_path, "w")
writeLines(colnames(mat), gz2); close(gz2)

# Small metadata file for the Python side
meta <- list(
  n_regions   = nrow(mat),
  n_cells     = ncol(mat),
  nnz         = length(mat@x),
  source_rds  = normalizePath(opt$input)
)
writeLines(
  c(
    sprintf("n_regions\t%d", meta$n_regions),
    sprintf("n_cells\t%d",   meta$n_cells),
    sprintf("nnz\t%d",       meta$nnz),
    sprintf("source_rds\t%s",meta$source_rds)
  ),
  file.path(opt$outdir, "matrix_info.tsv")
)

message("[01] Done.")
