#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# 00_inspect_rds.R
#
# Read-only sanity check for an .rds file before running 01_export_rds_to_mm.R.
# Prints class(), dim(), slotNames(), assay names (if applicable), a preview of
# row/column names, and a rough sparsity estimate. Writes nothing.
#
# Usage:
#   Rscript 00_inspect_rds.R --input /path/to/CTCF_bin1000_mtx.rds
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option(c("-i", "--input"), type = "character", help = "Input .rds file"),
  make_option(c("-n", "--preview"), type = "integer", default = 5,
              help = "How many row/col names to preview [default %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$input))
  stop("--input is required.")

hr <- function(c = "-") cat(strrep(c, 72), "\n", sep = "")

cat("Reading: ", opt$input, "\n", sep = "")
obj <- readRDS(opt$input)

hr("=")
cat("class(): ", paste(class(obj), collapse = " / "), "\n", sep = "")
cat("typeof():", typeof(obj), "\n", sep = "")
if (isS4(obj)) {
  cat("isS4:    TRUE\n")
  cat("slotNames():\n  ", paste(slotNames(obj), collapse = ", "), "\n", sep = "")
}
if (is.list(obj) && !isS4(obj)) {
  cat("names(): ", paste(names(obj), collapse = ", "), "\n", sep = "")
  cat("lengths():\n"); print(lengths(obj))
}
hr()

describe_matrix <- function(m, tag = "") {
  cat(sprintf("[%s] class: %s\n", tag, paste(class(m), collapse = "/")))
  cat(sprintf("[%s] dim:   %d rows x %d cols\n", tag, nrow(m), ncol(m)))
  if (inherits(m, c("dgCMatrix", "dgTMatrix", "dgRMatrix"))) {
    nnz <- length(m@x)
    cat(sprintf("[%s] nnz:   %d (density = %.4e)\n",
                tag, nnz, nnz / (as.numeric(nrow(m)) * ncol(m))))
  } else if (is.matrix(m)) {
    nnz <- sum(m != 0)
    cat(sprintf("[%s] nnz:   %d (density = %.4e)\n",
                tag, nnz, nnz / (as.numeric(nrow(m)) * ncol(m))))
  }
  rn <- rownames(m); cn <- colnames(m)
  n  <- opt$preview
  if (!is.null(rn))
    cat(sprintf("[%s] rownames head (%d/%d): %s\n",
                tag, min(n, length(rn)), length(rn),
                paste(head(rn, n), collapse = ", ")))
  else
    cat(sprintf("[%s] rownames: <none>\n", tag))
  if (!is.null(cn))
    cat(sprintf("[%s] colnames head (%d/%d): %s\n",
                tag, min(n, length(cn)), length(cn),
                paste(head(cn, n), collapse = ", ")))
  else
    cat(sprintf("[%s] colnames: <none>\n", tag))
}

# --- try common shapes --------------------------------------------------------
if (inherits(obj, c("dgCMatrix", "dgTMatrix", "dgRMatrix")) || is.matrix(obj)) {
  describe_matrix(obj, "matrix")

} else if (inherits(obj, "Seurat")) {
  cat("Seurat object detected.\n")
  if (requireNamespace("Seurat", quietly = TRUE)) {
    cat("Assays: ", paste(Seurat::Assays(obj), collapse = ", "), "\n")
    cat("DefaultAssay: ", Seurat::DefaultAssay(obj), "\n")
    m <- tryCatch(Seurat::GetAssayData(obj, layer = "counts"),
                  error = function(e) Seurat::GetAssayData(obj, slot = "counts"))
    describe_matrix(m, "counts")
  } else {
    cat("(install Seurat to introspect further)\n")
  }

} else if (inherits(obj, c("SingleCellExperiment", "SummarizedExperiment"))) {
  cat("SummarizedExperiment-like object detected.\n")
  if (requireNamespace("SummarizedExperiment", quietly = TRUE)) {
    ns <- SummarizedExperiment::assayNames(obj)
    cat("assayNames(): ", paste(ns, collapse = ", "), "\n")
    pick <- if ("counts" %in% ns) "counts" else ns[1]
    m <- SummarizedExperiment::assay(obj, pick)
    describe_matrix(m, pick)
  } else {
    cat("(install SummarizedExperiment to introspect further)\n")
  }

} else if (is.list(obj)) {
  for (nm in names(obj)) {
    x <- obj[[nm]]
    cat(sprintf("\n$%s\n", nm))
    if (inherits(x, c("dgCMatrix", "dgTMatrix", "dgRMatrix")) || is.matrix(x)) {
      describe_matrix(x, nm)
    } else {
      cat(sprintf("  class: %s\n", paste(class(x), collapse = "/")))
      if (length(x) < 20L) print(x) else cat(sprintf("  length: %d\n", length(x)))
    }
  }

} else {
  cat("Unrecognised top-level shape; printing str() (truncated):\n")
  str(obj, max.level = 2)
}

hr("=")
cat("Done. 01_export_rds_to_mm.R should handle the shape above; ",
    "if it doesn't, share this output.\n", sep = "")
