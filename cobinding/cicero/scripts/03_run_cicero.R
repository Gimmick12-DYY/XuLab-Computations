#!/usr/bin/env Rscript
# Cobinding runner. The imputation-era Cicero tree is imputation_legacy/Cicero.
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
              help = "Override paths.work_dir from the config"),
  make_option("--shuffle",  type = "character", default = NULL,
              help = "Override cicero.shuffle (none|colshuf|rowshuf|umap|perm); non-'none' writes to <work>/cicero_shuf/ + shuffle.para.txt (Ren-lab null).")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$config)) stop("--config is required")

cfg <- yaml::read_yaml(opt$config)
work_dir <- normalizePath(
  if (!is.null(opt$`work-dir`)) opt$`work-dir` else cfg$paths$work_dir,
  mustWork = FALSE
)
cds_dir <- file.path(work_dir, "cds")
# shuffle (CLI overrides config). Real run -> <work>/cicero; null run -> <work>/cicero_shuf
# so both share the one cds (same UMAP metacell neighborhoods) without clobbering.
shuffle <- tolower(as.character(opt$shuffle %||% cfg$cicero$shuffle %||% "none"))
out_dir <- file.path(work_dir, if (shuffle %in% c("none", "")) "cicero" else "cicero_shuf")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cds_path <- file.path(cds_dir, "cds.rds")
if (!file.exists(cds_path))
  stop("Missing CDS: ", cds_path, " — run 02_build_cds.R first")

message(sprintf("[03] Loading CDS from %s", cds_path))
cds <- readRDS(cds_path)

umap <- SingleCellExperiment::reducedDims(cds)[["UMAP"]]
if (is.null(umap)) stop("CDS has no UMAP reduction; re-run 02_build_cds.R.")

k_metacell <- as.integer(cfg$cicero$k_metacell %||% 50L)
max_iter <- as.integer(cfg$cicero$metacell_max_iter %||% 5000L)
set.seed(cfg$cicero$random_seed %||% 555L)

# Stock cicero::make_cicero_cds caps neighborhood sampling at `it < 5000`,
# which yields ~4.6k metacells on this RBBP4 CDS. Lab shuffle p-values imply ~8.5k.
# Do not mutate the language object in place (body[[i]] <- ...); that corrupts
# later `<-` calls ("incorrect number of arguments to <-").
if (max_iter != 5000L) {
  f <- cicero::make_cicero_cds
  txt <- paste(deparse(body(f), width.cutoff = 500L), collapse = "\n")
  if (!grepl("it < 5000", txt, fixed = TRUE))
    stop("make_cicero_cds body no longer contains 'it < 5000'; cannot raise the cap")
  txt <- gsub("it < 5000", sprintf("it < %d", max_iter), txt, fixed = TRUE)
  body(f) <- parse(text = txt)[[1]]
  make_cicero_cds <- f
  message(sprintf("[03] patched make_cicero_cds iteration cap 5000 -> %d", max_iter))
}

# Ren Lab null (Li 2021 Nature; Zu 2023 Nature): Cicero scores have no p-values.
# Shuffle the peak-by-cell matrix, rerun Cicero, fit a Gaussian to the null
# coaccess, then test the REAL scores against that null (BH FDR).
# Keep the real UMAP so metacell neighborhoods stay the same.
shuffle_ccres_in_cells <- function(mat) {
  mat <- as(mat, "dgCMatrix")
  n <- nrow(mat)
  dp <- diff(mat@p)
  new_i <- integer(length(mat@i))
  idx <- 1L
  for (j in seq_len(ncol(mat))) {
    nz <- dp[j]
    if (nz > 0L) {
      rows <- sort(sample.int(n, nz) - 1L)
      new_i[idx:(idx + nz - 1L)] <- rows
      idx <- idx + nz
    }
  }
  mat@i <- new_i
  mat
}

get_counts <- function(cds) {
  m <- tryCatch(SingleCellExperiment::counts(cds), error = function(e) NULL)
  if (is.null(m)) m <- SummarizedExperiment::assay(cds, 1)
  as(m, "dgCMatrix")
}

set_counts <- function(cds, mat) {
  dimnames(mat) <- list(rownames(cds), colnames(cds))
  an <- SummarizedExperiment::assayNames(cds)
  SummarizedExperiment::assay(cds, an[[1]]) <- mat
  if ("counts" %in% an)
    SingleCellExperiment::counts(cds) <- mat
  cds
}

if (shuffle %in% c("umap")) {
  message("[03] permuting UMAP coordinates across cells (not the Ren Lab null)")
  umap_shuf <- umap[sample.int(nrow(umap)), , drop = FALSE]
  rownames(umap_shuf) <- rownames(umap)
  umap <- umap_shuf
} else if (shuffle %in% c("colshuf", "cols")) {
  # Zu 2023: shuffle cCRE columns of the cell-by-cCRE matrix (peaks within each cell).
  message("[03] colShuf: permuting accessible peaks independently in each cell")
  cds <- set_counts(cds, shuffle_ccres_in_cells(get_counts(cds)))
} else if (shuffle %in% c("rowshuf", "rows")) {
  message("[03] rowShuf: permuting cells independently for each peak")
  cds <- set_counts(cds, Matrix::t(shuffle_ccres_in_cells(Matrix::t(get_counts(cds)))))
  cds <- estimate_size_factors(cds)
} else if (shuffle %in% c("perm")) {
  # Permute peak accessibility *profiles* across genomic loci; coordinates stay put.
  message("[03] perm: permuting peak count-vectors across genomic coordinates")
  mat <- get_counts(cds)
  rn <- rownames(mat)
  mat <- mat[sample.int(nrow(mat)), , drop = FALSE]
  rownames(mat) <- rn
  cds <- set_counts(cds, mat)
}

message(sprintf("[03] make_cicero_cds(k = %d, max_iter = %d, shuffle = %s) on %d cells",
                k_metacell, max_iter, shuffle, ncol(cds)))
cicero_cds <- make_cicero_cds(cds, reduced_coordinates = umap, k = k_metacell)
message(sprintf("[03] n_metacell = %d", ncol(cicero_cds)))

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
    sprintf("n_metacell\t%d",  ncol(cicero_cds)),  # sample size for the coaccess correlation test
    sprintf("k_metacell\t%d",  k_metacell),
    sprintf("metacell_max_iter\t%d", max_iter),
    sprintf("shuffle\t%s",     shuffle),
    sprintf("window_bp\t%d",   window_bp),
    sprintf("sample_num\t%d",  sample_num),
    sprintf("n_links\t%d",     nrow(conns)),
    sprintf("table\t%s",       out_path)
  ),
  con = file.path(out_dir, "cicero_info.tsv")
)

if (shuffle != "none" && nrow(conns) > 1L) {
  x <- conns$coaccess[is.finite(conns$coaccess)]
  if (requireNamespace("fitdistrplus", quietly = TRUE)) {
    ft <- fitdistrplus::fitdist(x, "norm")
    mu <- unname(ft$estimate[["mean"]])
    sig <- unname(ft$estimate[["sd"]])
    message("[03] fitdistrplus Gaussian MLE on shuffled coaccess")
  } else {
    mu <- mean(x)
    sig <- sqrt(mean((x - mu)^2))  # MLE (same as fitdistrplus for norm)
    message("[03] fitdistrplus not installed; using Gaussian MLE mean/sd")
  }
  para_path <- file.path(out_dir, "shuffle.para.txt")
  write.table(
    data.frame(
      group = basename(dirname(work_dir)), metaCol = "TF",
      meanShuf = mu, stdShuf = sig,
      stringsAsFactors = FALSE
    ),
    file = para_path, sep = "\t", quote = FALSE, row.names = FALSE
  )
  message(sprintf("[03] Wrote shuffle para (mu=%.6g sd=%.6g) to %s", mu, sig, para_path))
}

message(sprintf("[03] Wrote %d co-accessibility links to %s", nrow(conns), out_path))
