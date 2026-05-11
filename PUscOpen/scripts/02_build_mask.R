#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# 02_build_mask.R
#
# Build a conservative genome-wide candidate mask + a per-bin feature table for
# the PU classifier (step 03). The mask is the OR of (selected) annotation
# overlap rules, then MINUS the blacklist and (optionally) low-mappability
# regions.
#
# Outputs (under <work>/mask/):
#   mask.bed              kept bins as BED3 (chrom\tstart\tend) -- a "geometric"
#                         candidate mask before the PU refinement step.
#   mask.tsv              kept bins as chr:start-end strings (matches the
#                         mm/regions.tsv.gz naming convention).
#   bin_features.tsv      every bin in the universe with columns:
#                           bin_idx_orig, bin_name,
#                           in_cCRE_293T, in_cCRE_293T_CA,
#                           in_cCRE_CTCF, in_top_N_CTCF,
#                           motif_hit_count, ctcf_zscore_max,
#                           blacklist, mappability_drop,
#                           mask_keep
#   build_meta.json       parameters + per-rule counts
#
# Annotation files are expected under `paths.annotation_dir` from the YAML
# config (defaults to the cache produced by downstream/prepare_pos_neg_bins.R).
# Missing files cause only the corresponding feature to be NA / 0; the script
# does not download.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(Matrix)
  library(optparse)
  library(dplyr)
  library(stringi)
  library(GenomicRanges)
  library(jsonlite)
})

option_list <- list(
  make_option(c('--config'),    type='character', help='YAML config'),
  make_option(c('--work-dir'),  type='character', default=NULL,
              help='Override paths.work_dir'),
  make_option(c('--annotation-dir'), type='character', default=NULL,
              help='Override paths.annotation_dir')
)
opt <- parse_args(OptionParser(option_list=option_list))
stopifnot(!is.null(opt$config))

# Tiny YAML loader -- we don't add a yaml R dep just for this; rely on python.
read_yaml_py <- function(path) {
  py_file <- tempfile(fileext = '.py')
  writeLines(c(
    "import json, sys, yaml",
    "cfg = yaml.safe_load(open(sys.argv[1])) or {}",
    "print(json.dumps(cfg))"
  ), py_file)
  raw <- system2('python3', c(py_file, path), stdout=TRUE, stderr=TRUE)
  status <- attr(raw, 'status')
  if (!is.null(status) && status != 0) {
    stop(sprintf('Failed to parse YAML %s via python3: %s',
                 path, paste(raw, collapse='\n')))
  }
  jsonlite::fromJSON(paste(raw, collapse=''), simplifyVector=TRUE)
}
cfg <- read_yaml_py(opt$config)

work_dir <- if (!is.null(opt$`work-dir`)) opt$`work-dir` else cfg$paths$work_dir
work_dir <- normalizePath(work_dir, mustWork=FALSE)
dir.create(work_dir, showWarnings=FALSE, recursive=TRUE)
mm_dir   <- file.path(work_dir, 'mm')
mask_dir <- file.path(work_dir, 'mask')
dir.create(mask_dir, showWarnings=FALSE, recursive=TRUE)

ann_dir <- if (!is.null(opt$`annotation-dir`)) opt$`annotation-dir` else cfg$paths$annotation_dir
if (is.null(ann_dir) || !dir.exists(ann_dir))
  stop(sprintf('annotation_dir missing or not a directory: %s', ann_dir))

mask_cfg <- cfg$mask %||% list()
`%||%` <- function(a, b) if (is.null(a)) b else a

# Default seed parameters
positive_seed <- mask_cfg$positive_seed %||% list()
n_top <- as.integer(positive_seed$n_top %||% 10000L)
require_293T_CA <- isTRUE(positive_seed$require_293T_CA %||% TRUE)

# -----------------------------------------------------------------------------
# Bin universe
# -----------------------------------------------------------------------------
reg_path <- file.path(mm_dir, 'regions.tsv.gz')
if (!file.exists(reg_path))
  stop(sprintf('Missing %s. Run 01_export_rds_to_mm.R first.', reg_path))

message('[mask] Reading bin coordinates from ', reg_path)
gz <- gzfile(reg_path, 'r')
bin_names <- readLines(gz); close(gz)
message(sprintf('[mask] %d bins in mm universe.', length(bin_names)))

parts <- stringi::stri_split_regex(bin_names, pattern=':|-', simplify=TRUE)
bins_gr <- GRanges(seqnames=parts[, 1],
                   ranges=IRanges(start=as.numeric(parts[, 2]) + 1,
                                  end=as.numeric(parts[, 3])))
N_bins <- length(bins_gr)

# Helper: BED loader without rtracklayer (which may be missing).
read_bed_simple <- function(path, columns=NULL) {
  if (!file.exists(path)) return(NULL)
  con <- if (grepl('\\.gz$', path)) gzfile(path) else file(path)
  df <- tryCatch(
    read.table(con, sep='\t', stringsAsFactors=FALSE, comment.char='#', quote=''),
    error = function(e) { close(con); stop(e) }
  )
  try(close(con), silent=TRUE)
  if (!is.null(columns) && length(columns) <= ncol(df))
    colnames(df)[seq_along(columns)] <- columns
  df
}

bed_to_gr <- function(df) {
  GRanges(seqnames=df[, 1],
          ranges=IRanges(start=as.numeric(df[, 2]) + 1, end=as.numeric(df[, 3])))
}

overlap_mask <- function(gr, bed_path) {
  if (is.null(bed_path) || !file.exists(bed_path)) return(NULL)
  df <- read_bed_simple(bed_path)
  if (is.null(df)) return(NULL)
  gr_b <- bed_to_gr(df)
  hits <- findOverlaps(gr, gr_b)
  m <- logical(length(gr))
  m[unique(queryHits(hits))] <- TRUE
  m
}

# -----------------------------------------------------------------------------
# Annotation loaders
# -----------------------------------------------------------------------------
files <- list(
  cCRE_293T   = file.path(ann_dir, 'SCREEN-293T-cCRE-ENCFF529BOG.bed'),
  cCRE_293    = file.path(ann_dir, 'SCREEN-293-cCRE-ENCFF439DDQ_ENCFF885SUR_ENCFF128UTY.bed'),
  ctcf_zscore = file.path(ann_dir, 'SCREEN-293-CTCFzscore-ENCSR000DTW-ENCFF128UTY.tsv'),
  ctcf_bound  = file.path(ann_dir, 'GRCh38-cCREs.CTCF-bound.bed'),
  blacklist   = file.path(ann_dir, 'hg38-blacklist.v2.bed.gz')
)
for (nm in names(files))
  message(sprintf('[mask] %-13s %s -> %s', nm, files[[nm]],
                  if (file.exists(files[[nm]])) 'OK' else 'MISSING'))

# in_cCRE_293T (all classes)
mask_293T <- overlap_mask(bins_gr, files$cCRE_293T)
if (is.null(mask_293T)) mask_293T <- logical(N_bins)

# in_cCRE_293T_CA (CA-class only -- chromatin accessible)
mask_293T_CA <- logical(N_bins)
if (file.exists(files$cCRE_293T)) {
  df <- read_bed_simple(files$cCRE_293T)
  if (ncol(df) >= 10) {
    df_ca <- df[grepl('CA', df[, 10]), , drop=FALSE]
    if (nrow(df_ca) > 0) {
      hits <- findOverlaps(bins_gr, bed_to_gr(df_ca))
      mask_293T_CA[unique(queryHits(hits))] <- TRUE
    }
  }
}

# in_cCRE_CTCF (genome-wide CTCF-bound cCRE set, any cell type)
mask_CTCF <- overlap_mask(bins_gr, files$ctcf_bound)
if (is.null(mask_CTCF)) mask_CTCF <- logical(N_bins)

# Top-N CTCF cCRE positives (the Bin_posVSneg.ipynb recipe).
top_mask <- logical(N_bins)
top_score <- numeric(N_bins) * NA_real_
if (file.exists(files$cCRE_293) && file.exists(files$ctcf_zscore)) {
  message('[mask] Building top-N CTCF cCRE positives ...')
  z_df <- read_bed_simple(files$ctcf_zscore, c('cCRE', 'score'))
  bed_293 <- read_bed_simple(files$cCRE_293)
  if (ncol(bed_293) >= 10 && all(c('cCRE', 'score') %in% colnames(z_df))) {
    colnames(bed_293)[c(1, 2, 3, 4, 10)] <- c('chrom', 'start', 'end', 'cCRE', 'class')
    j <- bed_293 %>% dplyr::left_join(z_df, by='cCRE') %>%
      dplyr::filter(grepl('CTCF', class))
    if (require_293T_CA && file.exists(files$cCRE_293T)) {
      bed_293T <- read_bed_simple(files$cCRE_293T)
      if (ncol(bed_293T) >= 10) {
        colnames(bed_293T)[c(1, 2, 3, 4, 10)] <- c('chrom', 'start', 'end', 'cCRE', 'class')
        bed_293T_CA <- bed_293T %>% dplyr::filter(grepl('CA', class))
        j <- j %>% dplyr::filter(cCRE %in% bed_293T_CA$cCRE)
      }
    }
    j <- j %>% dplyr::arrange(dplyr::desc(score)) %>% head(n_top)
    if (nrow(j) > 0) {
      top_gr <- GRanges(seqnames=j$chrom,
                        ranges=IRanges(start=as.numeric(j$start),
                                       end=as.numeric(j$end)))
      h <- findOverlaps(bins_gr, top_gr)
      qh <- queryHits(h); sh <- subjectHits(h)
      top_mask[unique(qh)] <- TRUE
      # For each bin, max CTCF z-score across overlapping cCREs.
      sc <- as.numeric(j$score)
      ord <- order(qh)
      qh_s <- qh[ord]; sh_s <- sh[ord]
      sc_h <- sc[sh_s]
      # tapply on qh_s for max -- vectorised enough for our scale.
      agg <- tapply(sc_h, qh_s, max, na.rm=TRUE)
      top_score[as.integer(names(agg))] <- as.numeric(agg)
    }
  }
}

# Motif hits (optional). If a BED is provided, count overlaps per bin.
motif_count <- integer(N_bins)
motif_bed_path <- mask_cfg$motif_bed
if (!is.null(motif_bed_path) && file.exists(motif_bed_path)) {
  message('[mask] Counting motif hits from ', motif_bed_path)
  motif_df <- read_bed_simple(motif_bed_path)
  motif_gr <- bed_to_gr(motif_df)
  hits <- findOverlaps(bins_gr, motif_gr)
  qh <- queryHits(hits)
  tab <- tabulate(qh, nbins=N_bins)
  motif_count <- as.integer(tab)
} else if (isTRUE(mask_cfg$use_motif)) {
  warning('use_motif: true but motif_bed not given or missing; motif_hit_count is all zeros.')
}

# Blacklist
blacklist_mask <- overlap_mask(bins_gr, files$blacklist)
if (is.null(blacklist_mask)) blacklist_mask <- logical(N_bins)

# Optional low-mappability mask
mappability_drop <- logical(N_bins)
if (!is.null(mask_cfg$exclude_mappability_bed)) {
  mp <- overlap_mask(bins_gr, mask_cfg$exclude_mappability_bed)
  if (!is.null(mp)) mappability_drop <- mp
}

# -----------------------------------------------------------------------------
# Aggregate raw signal per bin so the PU classifier has signal features.
# We compute: total_count, n_cells_observed, max_count_per_bin.
# -----------------------------------------------------------------------------
mtx_path <- file.path(mm_dir, 'matrix.mtx.gz')
total_count <- numeric(N_bins)
n_cells_observed <- integer(N_bins)
max_count_per_bin <- numeric(N_bins)
n_cells_total <- 0L
if (file.exists(mtx_path)) {
  message('[mask] Reading raw matrix for per-bin signal aggregates ...')
  M <- Matrix::readMM(mtx_path)
  M <- as(M, 'CsparseMatrix')
  if (nrow(M) != N_bins)
    stop(sprintf('Matrix rows (%d) != bins (%d).', nrow(M), N_bins))
  n_cells_total <- ncol(M)
  total_count <- as.numeric(Matrix::rowSums(M))
  bin_M <- as(M > 0, 'CsparseMatrix')
  n_cells_observed <- as.integer(Matrix::rowSums(bin_M))
  # Per-row max via triplet representation; vectorised in C.
  M_T <- as(M, 'TsparseMatrix')
  if (length(M_T@x) > 0) {
    row_idx <- M_T@i + 1L
    agg_max <- tapply(M_T@x, row_idx, max)
    max_count_per_bin[as.integer(names(agg_max))] <- as.numeric(agg_max)
  }
  rm(M_T)
}
fraction_cells_observed <- if (n_cells_total > 0)
  n_cells_observed / n_cells_total else numeric(N_bins)

# -----------------------------------------------------------------------------
# Richer per-bin features (item 6 in PUscOpen tuning plan):
#   * distance to nearest annotation hit (top-N CTCF, 293T-CA cCRE, cCRE_CTCF)
#   * neighbour count of annotation hits within fixed windows
#   * chromosome flags (X/Y)
#   * optional GC content (if mask.fasta_bsgenome resolvable)
#   * optional mean mappability (if mask.mappability_bigwig provided)
# Distances are reported as log10(min(d, 1e6) + 1); -1 means no annotation
# available (used by the PU classifier as a sentinel that downstream coerces
# to 0 via NaN filling).
# -----------------------------------------------------------------------------
LOG10_DIST_CAP <- 1e6

nearest_log10_dist <- function(bins_gr, ann_gr) {
  if (is.null(ann_gr) || length(ann_gr) == 0L)
    return(rep(NA_real_, length(bins_gr)))
  d <- suppressWarnings(GenomicRanges::distanceToNearest(
    bins_gr, ann_gr, ignore.strand=TRUE))
  out <- rep(NA_real_, length(bins_gr))
  if (length(d) > 0) {
    qh <- S4Vectors::queryHits(d)
    dd <- S4Vectors::mcols(d)$distance
    out[qh] <- log10(pmin(as.numeric(dd), LOG10_DIST_CAP) + 1)
  }
  out
}

count_in_window <- function(bins_gr, ann_gr, halfwidth_bp) {
  if (is.null(ann_gr) || length(ann_gr) == 0L)
    return(integer(length(bins_gr)))
  bins_ext <- resize(bins_gr,
                     width = width(bins_gr) + 2L * as.integer(halfwidth_bp),
                     fix   = 'center')
  as.integer(GenomicRanges::countOverlaps(bins_ext, ann_gr))
}

# Build helper annotation GRanges (some may be empty)
top_gr <- NULL
if (any(top_mask)) {
  top_gr <- bins_gr[top_mask]
}

ca_gr <- NULL
if (any(mask_293T_CA)) {
  ca_gr <- bins_gr[mask_293T_CA]
}
ctcfb_gr <- NULL
if (any(mask_CTCF)) {
  ctcfb_gr <- bins_gr[mask_CTCF]
}

message('[mask] Computing distance / density / chromosome features ...')
dist_log_top_CTCF <- nearest_log10_dist(bins_gr, top_gr)
dist_log_cCRE_293T_CA <- nearest_log10_dist(bins_gr, ca_gr)
dist_log_cCRE_CTCF <- nearest_log10_dist(bins_gr, ctcfb_gr)

n_top_CTCF_within_50kb <- count_in_window(bins_gr, top_gr, 50000L)
n_cCRE_293T_within_5kb <- count_in_window(bins_gr, ca_gr, 5000L)
n_cCRE_CTCF_within_50kb <- count_in_window(bins_gr, ctcfb_gr, 50000L)

chrom_str <- as.character(seqnames(bins_gr))
is_chrX <- as.integer(chrom_str %in% c('chrX', 'X'))
is_chrY <- as.integer(chrom_str %in% c('chrY', 'Y'))

# Optional: GC content. Two source modes:
#   * mask.gc_bsgenome   = "BSgenome.Hsapiens.UCSC.hg38" (or similar)
#   * mask.gc_fasta_path = "/path/to/hg38.fa" (uses Rsamtools::FaFile)
# If neither resolves, fall back to NA.
gc_content <- rep(NA_real_, N_bins)
gc_source <- 'none'
if (!is.null(mask_cfg$gc_bsgenome)) {
  pkg <- as.character(mask_cfg$gc_bsgenome)
  ok <- tryCatch({
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message('[mask] gc_bsgenome=', pkg, ' not installed; skipping GC.')
      FALSE
    } else {
      suppressPackageStartupMessages(library(pkg, character.only = TRUE))
      requireNamespace('Biostrings', quietly = TRUE)
    }
  }, error = function(e) FALSE)
  if (isTRUE(ok)) {
    bsg <- get(pkg)
    message('[mask] Computing GC content from ', pkg, ' ...')
    gc_content <- as.numeric(Biostrings::letterFrequency(
      Biostrings::getSeq(bsg, bins_gr), letters='GC', as.prob=TRUE))
    gc_source <- paste0('bsgenome:', pkg)
  }
} else if (!is.null(mask_cfg$gc_fasta_path) &&
           file.exists(mask_cfg$gc_fasta_path)) {
  ok <- tryCatch(requireNamespace('Rsamtools', quietly = TRUE) &&
                 requireNamespace('Biostrings', quietly = TRUE),
                 error = function(e) FALSE)
  if (isTRUE(ok)) {
    message('[mask] Computing GC content from FASTA ', mask_cfg$gc_fasta_path)
    fa <- Rsamtools::FaFile(mask_cfg$gc_fasta_path)
    seqs <- tryCatch(Biostrings::getSeq(fa, bins_gr),
                     error = function(e) NULL)
    if (!is.null(seqs)) {
      gc_content <- as.numeric(Biostrings::letterFrequency(
        seqs, letters='GC', as.prob=TRUE))
      gc_source <- paste0('fasta:', mask_cfg$gc_fasta_path)
    }
  } else {
    message('[mask] Rsamtools/Biostrings missing; skipping GC.')
  }
}

# Optional: mean mappability per bin from a bigWig file.
mappability_mean <- rep(NA_real_, N_bins)
mapp_source <- 'none'
if (!is.null(mask_cfg$mappability_bigwig) &&
    file.exists(mask_cfg$mappability_bigwig)) {
  ok <- tryCatch(requireNamespace('rtracklayer', quietly = TRUE),
                 error = function(e) FALSE)
  if (isTRUE(ok)) {
    message('[mask] Reading mappability bigWig ', mask_cfg$mappability_bigwig)
    bw <- tryCatch(
      rtracklayer::import.bw(mask_cfg$mappability_bigwig, which = bins_gr),
      error = function(e) { message('[mask] bigWig import failed: ',
                                    conditionMessage(e)); NULL })
    if (!is.null(bw)) {
      # Weighted mean of score across overlapping intervals per bin.
      ov <- GenomicRanges::findOverlaps(bins_gr, bw)
      if (length(ov) > 0) {
        qh <- S4Vectors::queryHits(ov)
        sh <- S4Vectors::subjectHits(ov)
        widths <- pmin(end(bins_gr)[qh], end(bw)[sh]) -
                  pmax(start(bins_gr)[qh], start(bw)[sh]) + 1L
        widths <- pmax(widths, 0L)
        scores <- as.numeric(bw$score[sh])
        # weighted sum / weight sum per bin.
        ws <- widths * scores
        sum_ws <- tapply(ws,    qh, sum, na.rm=TRUE)
        sum_w  <- tapply(widths, qh, sum, na.rm=TRUE)
        idx <- as.integer(names(sum_w))
        denom <- as.numeric(sum_w); denom[denom == 0] <- NA_real_
        mappability_mean[idx] <- as.numeric(sum_ws) / denom
        mapp_source <- paste0('bigwig:', mask_cfg$mappability_bigwig)
      }
    }
  } else {
    message('[mask] rtracklayer missing; skipping mappability.')
  }
}

# -----------------------------------------------------------------------------
# Build the mask
# -----------------------------------------------------------------------------
keep <- logical(N_bins)
rules <- c()

if (isTRUE(mask_cfg$use_cCRE_293T)) {
  if (isTRUE(mask_cfg$use_cCRE_293T_CA)) {
    keep <- keep | mask_293T_CA
    rules <- c(rules, 'cCRE_293T_CA')
  } else {
    keep <- keep | mask_293T
    rules <- c(rules, 'cCRE_293T')
  }
}
if (isTRUE(mask_cfg$use_cCRE_CTCF)) {
  keep <- keep | mask_CTCF
  rules <- c(rules, 'cCRE_CTCF_bound')
}
if (isTRUE(mask_cfg$use_motif)) {
  keep <- keep | (motif_count > 0)
  rules <- c(rules, 'motif')
}

# If none of the use_* flags were set, fall back to "all bins" so the
# downstream PU classifier still sees a sensible candidate set.
if (length(rules) == 0L) {
  message('[mask] No use_* annotation rule enabled; falling back to all bins.')
  keep <- rep(TRUE, N_bins)
  rules <- c('all_bins')
}

if (isTRUE(mask_cfg$exclude_blacklist)) {
  keep <- keep & !blacklist_mask
  rules <- c(rules, 'minus_blacklist')
}
keep <- keep & !mappability_drop

n_keep <- sum(keep)
message(sprintf('[mask] Final mask: %d / %d bins (%.2f%%) after rules: %s',
                n_keep, N_bins, 100 * n_keep / max(N_bins, 1),
                paste(rules, collapse=' | ')))

# -----------------------------------------------------------------------------
# Persist
# -----------------------------------------------------------------------------
mask_bed_path <- file.path(mask_dir, 'mask.bed')
mask_tsv_path <- file.path(mask_dir, 'mask.tsv')
features_path <- file.path(mask_dir, 'bin_features.tsv')
meta_path     <- file.path(mask_dir, 'build_meta.json')

writeLines(bin_names[keep], mask_tsv_path)
mask_bed_df <- data.frame(chrom = as.character(seqnames(bins_gr))[keep],
                          start = start(bins_gr)[keep] - 1L,
                          end   = end(bins_gr)[keep])
write.table(mask_bed_df, mask_bed_path, sep='\t', quote=FALSE,
            row.names=FALSE, col.names=FALSE)

features <- data.frame(
  bin_idx_orig             = seq_len(N_bins) - 1L,
  bin_name                 = bin_names,
  in_cCRE_293T             = as.integer(mask_293T),
  in_cCRE_293T_CA          = as.integer(mask_293T_CA),
  in_cCRE_CTCF             = as.integer(mask_CTCF),
  in_top_N_CTCF            = as.integer(top_mask),
  ctcf_zscore_max          = top_score,
  motif_hit_count          = motif_count,
  log10_total_count        = log10(total_count + 1),
  log10_n_cells_observed   = log10(n_cells_observed + 1),
  log10_max_count          = log10(max_count_per_bin + 1),
  fraction_cells_observed  = fraction_cells_observed,
  dist_log_top_CTCF        = dist_log_top_CTCF,
  dist_log_cCRE_293T_CA    = dist_log_cCRE_293T_CA,
  dist_log_cCRE_CTCF       = dist_log_cCRE_CTCF,
  n_top_CTCF_within_50kb   = n_top_CTCF_within_50kb,
  n_cCRE_293T_within_5kb   = n_cCRE_293T_within_5kb,
  n_cCRE_CTCF_within_50kb  = n_cCRE_CTCF_within_50kb,
  gc_content               = gc_content,
  mappability_mean         = mappability_mean,
  is_chrX                  = is_chrX,
  is_chrY                  = is_chrY,
  blacklist                = as.integer(blacklist_mask),
  mappability_drop         = as.integer(mappability_drop),
  mask_keep                = as.integer(keep)
)
write.table(features, features_path, sep='\t', quote=FALSE, row.names=FALSE)

meta <- list(
  n_bins_universe = N_bins,
  n_mask_keep     = sum(keep),
  rules_applied   = rules,
  counts = list(
    in_cCRE_293T     = sum(mask_293T),
    in_cCRE_293T_CA  = sum(mask_293T_CA),
    in_cCRE_CTCF     = sum(mask_CTCF),
    in_top_N_CTCF    = sum(top_mask),
    motif_hit_any    = sum(motif_count > 0),
    blacklist        = sum(blacklist_mask),
    mappability_drop = sum(mappability_drop)
  ),
  feature_sources = list(
    gc          = gc_source,
    mappability = mapp_source
  ),
  n_cells_total = n_cells_total,
  files = files,
  positive_seed = list(n_top=n_top, require_293T_CA=require_293T_CA),
  outputs = list(
    mask_bed     = mask_bed_path,
    mask_tsv     = mask_tsv_path,
    bin_features = features_path
  )
)
writeLines(toJSON(meta, pretty=TRUE, auto_unbox=TRUE, na='null'), meta_path)
message('[mask] Wrote ', mask_bed_path)
message('[mask] Wrote ', mask_tsv_path)
message('[mask] Wrote ', features_path)
message('[mask] Wrote ', meta_path)
