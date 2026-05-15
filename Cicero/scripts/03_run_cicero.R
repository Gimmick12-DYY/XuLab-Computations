#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# 03_run_cicero.R
#
# Aggregate cells with make_cicero_cds() (k-NN over UMAP) and fit Cicero's
# graphical-Lasso-style co-accessibility scores on the resulting metacell
# matrix. Output is a sparse Peak1 / Peak2 / coaccess table that 05_impute.py
# will turn into a bin-bin adjacency for raw-signal propagation.
#
# Genomic coordinates are derived directly from the underscore-form peak names
# (chr_start_end) written by 02_build_cds.R: per-chromosome length is the max
# of `end` over the peaks present, plus a small buffer. This avoids needing a
# separate hg38.chrom.sizes file in the repo.
#
# Outputs (under <work>/cicero/):
#   coaccess.tsv.gz              Peak1, Peak2, coaccess (Cicero's run_cicero output)
#   cicero_info.tsv              tiny meta dump (k_metacell, window_bp, n_links)
#
# Usage:
#   Rscript 03_run_cicero.R --config configs/default.yaml [--work-dir <path>]
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Matrix)
  library(monocle3)
  library(cicero)
  library(optparse)
  library(yaml)
  library(SingleCellExperiment)
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
cds_dir <- file.path(work_dir, "cds")
out_dir <- file.path(work_dir, "cicero")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cds_path <- file.path(cds_dir, "cds.rds")
if (!file.exists(cds_path))
  stop("Missing CDS: ", cds_path, " — run 02_build_cds.R first")

message(sprintf("[03] Loading CDS from %s", cds_path))
cds <- readRDS(cds_path)

umap <- SingleCellExperiment::reducedDims(cds)[["UMAP"]]
if (is.null(umap)) stop("CDS has no UMAP reduction; re-run 02_build_cds.R.")

k_metacell <- as.integer(cfg$cicero$k_metacell %||% 50L)
set.seed(cfg$cicero$random_seed %||% 555L)
message(sprintf("[03] make_cicero_cds(k = %d) on %d cells", k_metacell, ncol(cds)))
cicero_cds <- make_cicero_cds(cds, reduced_coordinates = umap, k = k_metacell)

# ---- Per-chromosome length from peak names ----------------------------------
peak_names <- rownames(cds)
parts <- strsplit(peak_names, "_", fixed = TRUE)
n_parts <- vapply(parts, length, integer(1))
if (any(n_parts != 3L))
  stop("Peak names must be `chr_start_end` (run 02_build_cds.R first).")
chrs   <- vapply(parts, `[[`, character(1), 1)
ends   <- as.numeric(vapply(parts, `[[`, character(1), 3))
chrom_len <- aggregate(ends, by = list(chrs), FUN = max)
colnames(chrom_len) <- c("V1", "V2")
chrom_len$V2 <- chrom_len$V2 + 1000L  # bin buffer

message(sprintf("[03] Derived %d chromosome lengths from peak names", nrow(chrom_len)))

# ---- Run Cicero -------------------------------------------------------------
window_bp  <- as.integer(cfg$cicero$window_bp  %||% 500000L)
sample_num <- as.integer(cfg$cicero$sample_num %||% 100L)
message(sprintf("[03] run_cicero(window=%d, sample_num=%d)", window_bp, sample_num))

conns <- run_cicero(
  cicero_cds,
  genomic_coords = chrom_len,
  window         = window_bp,
  sample_num     = sample_num,
  silent         = FALSE
)

# Cicero returns Peak1, Peak2, coaccess. Drop NAs and self-links.
n_in <- nrow(conns)
conns <- conns[!is.na(conns$coaccess), , drop = FALSE]
conns <- conns[as.character(conns$Peak1) != as.character(conns$Peak2), , drop = FALSE]
message(sprintf("[03] %d / %d links survive NA + self-link filter", nrow(conns), n_in))

# ---- Persist ----------------------------------------------------------------
out_path <- file.path(out_dir, "coaccess.tsv.gz")
gz <- gzfile(out_path, "w")
write.table(conns, file = gz, sep = "\t",
            quote = FALSE, row.names = FALSE, col.names = TRUE)
close(gz)

writeLines(
  c(
    sprintf("n_peaks\t%d",     nrow(cds)),
    sprintf("n_cells\t%d",     ncol(cds)),
    sprintf("k_metacell\t%d",  k_metacell),
    sprintf("window_bp\t%d",   window_bp),
    sprintf("sample_num\t%d",  sample_num),
    sprintf("n_links\t%d",     nrow(conns)),
    sprintf("table\t%s",       out_path)
  ),
  con = file.path(out_dir, "cicero_info.tsv")
)

message(sprintf("[03] Wrote %d co-accessibility links to %s", nrow(conns), out_path))
