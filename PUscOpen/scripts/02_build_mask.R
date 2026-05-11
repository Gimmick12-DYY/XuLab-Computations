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
  raw <- system2('python3', c('-c', sprintf(
    "import yaml, json; print(json.dumps(yaml.safe_load(open(%s)) or {}))",
    sprintf("'%s'", path))), stdout=TRUE, stderr=TRUE)
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
# We only read the column nonzero pattern; counts aren't needed beyond log10.
# -----------------------------------------------------------------------------
mtx_path <- file.path(mm_dir, 'matrix.mtx.gz')
total_count <- numeric(N_bins)
n_cells_observed <- integer(N_bins)
if (file.exists(mtx_path)) {
  message('[mask] Reading raw matrix for per-bin signal aggregates ...')
  M <- Matrix::readMM(mtx_path)
  M <- as(M, 'CsparseMatrix')
  if (nrow(M) != N_bins)
    stop(sprintf('Matrix rows (%d) != bins (%d).', nrow(M), N_bins))
  total_count <- as.numeric(Matrix::rowSums(M))
  bin_M <- as(M > 0, 'CsparseMatrix')
  n_cells_observed <- as.integer(Matrix::rowSums(bin_M))
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
  bin_idx_orig         = seq_len(N_bins) - 1L,
  bin_name             = bin_names,
  in_cCRE_293T         = as.integer(mask_293T),
  in_cCRE_293T_CA      = as.integer(mask_293T_CA),
  in_cCRE_CTCF         = as.integer(mask_CTCF),
  in_top_N_CTCF        = as.integer(top_mask),
  ctcf_zscore_max      = top_score,
  motif_hit_count      = motif_count,
  log10_total_count    = log10(total_count + 1),
  log10_n_cells_observed = log10(n_cells_observed + 1),
  blacklist            = as.integer(blacklist_mask),
  mappability_drop     = as.integer(mappability_drop),
  mask_keep            = as.integer(keep)
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
