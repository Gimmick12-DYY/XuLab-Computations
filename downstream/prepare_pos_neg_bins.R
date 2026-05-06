#!/usr/bin/env Rscript
# -----------------------------------------------------------------------------
# prepare_pos_neg_bins.R
#
# Translation of Bin_posVSneg.ipynb cells 0-6 into a reusable script.
#
# Builds positive and negative CTCF bin index sets from public SCREEN cCRE
# annotations + an ENCODE blacklist:
#   * positives: bins overlapping the top 10K CTCF cCREs (by 293 CTCF z-score)
#                that are also chromatin-accessible in 293T (CA class)
#   * negatives: bins NOT overlapping any GRCh38 CTCF-bound cCRE, sampled to
#                match the positive count (seed = 2026)
#   * blacklist bins (hg38 v2) are excluded from both pools
#
# Outputs (under --out-dir):
#   pos_neg_bins.tsv    bin_idx_orig (0-based row in mm/regions.tsv.gz),
#                       bin_idx_post_blacklist, label, bin_name
#   pos_neg_bins.meta.json  source files, counts, blacklist size, seed
#
# bin_idx_orig is the index into the *original* mm matrix (and therefore into
# the imputed HDF5 row axis), so downstream Python can index arrays directly.
#
# Usage:
#   Rscript prepare_pos_neg_bins.R \
#       --mm-dir   /work/.../cistopic_ctcf/mm \
#       --out-dir  /work/.../downstream/bins \
#       --cache-dir /work/.../downstream/cache
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Matrix)
  library(optparse)
  library(dplyr)
  library(stringi)
  library(GenomicRanges)
  library(rtracklayer)
})

option_list <- list(
  make_option(c('--mm-dir'),  type = 'character',
              help = 'Directory with regions.tsv.gz produced by 01_export_rds_to_mm.R'),
  make_option(c('--out-dir'), type = 'character',
              help = 'Output directory for pos_neg_bins.tsv + meta.json'),
  make_option(c('--cache-dir'), type = 'character', default = NULL,
              help = 'Where to download cCRE/blacklist files (default: <out-dir>/cache)'),
  make_option(c('--n-top'), type = 'integer', default = 10000L,
              help = 'Top-N CTCF cCREs by score [default %default]'),
  make_option(c('--seed'), type = 'integer', default = 2026L,
              help = 'RNG seed for the negative sampler [default %default]'),
  make_option(c('--no-download'), action = 'store_true', default = FALSE,
              help = 'Fail instead of downloading missing input files.')
)
opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opt$`mm-dir`), !is.null(opt$`out-dir`))

mm_dir   <- normalizePath(opt$`mm-dir`,  mustWork = TRUE)
out_dir  <- opt$`out-dir`
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_dir  <- normalizePath(out_dir, mustWork = TRUE)
cache    <- if (is.null(opt$`cache-dir`)) file.path(out_dir, 'cache') else opt$`cache-dir`
dir.create(cache, showWarnings = FALSE, recursive = TRUE)
cache    <- normalizePath(cache, mustWork = TRUE)

URLS <- list(
  ctcf_bound_bed   = 'https://downloads.wenglab.org/GRCh38-cCREs.CTCF-bound.bed',
  cCRE_293_bed     = 'https://downloads.wenglab.org/Registry-V4/ENCFF439DDQ_ENCFF885SUR_ENCFF128UTY.bed',
  ctcf_293_zscores = 'https://downloads.wenglab.org/Registry-V4/SCREEN/Signal-Files/ENCSR000DTW-ENCFF128UTY.tsv',
  cCRE_293T_bed    = 'https://downloads.wenglab.org/Registry-V4/ENCFF529BOG.bed',
  blacklist_bed_gz = 'https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/lists/hg38-blacklist.v2.bed.gz'
)

LOCAL <- list(
  ctcf_bound_bed   = file.path(cache, 'GRCh38-cCREs.CTCF-bound.bed'),
  cCRE_293_bed     = file.path(cache, 'SCREEN-293-cCRE-ENCFF439DDQ_ENCFF885SUR_ENCFF128UTY.bed'),
  ctcf_293_zscores = file.path(cache, 'SCREEN-293-CTCFzscore-ENCSR000DTW-ENCFF128UTY.tsv'),
  cCRE_293T_bed    = file.path(cache, 'SCREEN-293T-cCRE-ENCFF529BOG.bed'),
  blacklist_bed_gz = file.path(cache, 'hg38-blacklist.v2.bed.gz')
)

ensure_file <- function(key) {
  path <- LOCAL[[key]]
  if (file.exists(path)) return(path)
  if (isTRUE(opt$`no-download`))
    stop(sprintf('Missing %s and --no-download was passed.', path))
  message(sprintf('[prep] Downloading %s -> %s', URLS[[key]], path))
  download.file(URLS[[key]], destfile = path, quiet = TRUE, mode = 'wb')
  path
}

for (k in names(URLS)) ensure_file(k)

# -- bin universe ------------------------------------------------------------
reg_path <- file.path(mm_dir, 'regions.tsv.gz')
if (!file.exists(reg_path))
  stop(sprintf('Missing %s. Run 01_export_rds_to_mm.R first.', reg_path))

message('[prep] Reading bin coordinates from ', reg_path)
gz <- gzfile(reg_path, 'r')
bin_names <- readLines(gz)
close(gz)
message(sprintf('[prep] %d bins in mm universe.', length(bin_names)))

parts <- stri_split_regex(bin_names, pattern = ':|-', simplify = TRUE)
if (ncol(parts) < 3L)
  stop('Could not parse bin names as chr:start-end (sample: ',
       paste(head(bin_names, 3L), collapse = ', '), ')')
# Match notebook convention: start as numeric+1 (1-based), end as numeric.
CTCF_bins_gr_raw <- GRanges(
  seqnames = parts[, 1],
  ranges   = IRanges(start = as.numeric(parts[, 2]) + 1,
                     end   = as.numeric(parts[, 3]))
)

# -- blacklist ---------------------------------------------------------------
message('[prep] Reading blacklist BED ', LOCAL$blacklist_bed_gz)
blacklist_gr <- import.bed(LOCAL$blacklist_bed_gz)
hits_bl <- findOverlaps(CTCF_bins_gr_raw, blacklist_gr)
blacklist_indices <- unique(queryHits(hits_bl))
message(sprintf('[prep] Blacklist drops %d / %d bins (%.2f%%).',
                length(blacklist_indices), length(CTCF_bins_gr_raw),
                100 * length(blacklist_indices) / length(CTCF_bins_gr_raw)))

# Mask of original-row indices that survive the blacklist filter.
keep_orig <- setdiff(seq_along(CTCF_bins_gr_raw), blacklist_indices)
CTCF_bins_gr <- CTCF_bins_gr_raw[keep_orig]
# Map post-blacklist (1-based local) -> original (1-based) row index.
local_to_orig <- keep_orig

# -- positive set ------------------------------------------------------------
message('[prep] Building positive set (top ', opt$`n-top`, ' CTCF cCREs)...')
ctcf_293_z <- read.table(LOCAL$ctcf_293_zscores, sep = '\t', stringsAsFactors = FALSE) %>%
  dplyr::rename(cCRE = V1, score = V2)
bed_293 <- read.table(LOCAL$cCRE_293_bed, sep = '\t', stringsAsFactors = FALSE) %>%
  dplyr::rename(chrom = V1, start = V2, end = V3, cCRE = V4, class = V10)
bed_293_CTCF <- bed_293 %>%
  left_join(ctcf_293_z, by = 'cCRE') %>%
  filter(grepl('CTCF', class))

bed_293T <- read.table(LOCAL$cCRE_293T_bed, sep = '\t', stringsAsFactors = FALSE) %>%
  dplyr::rename(chrom = V1, start = V2, end = V3, cCRE = V4, class = V10)
bed_293T_CA <- bed_293T %>% filter(grepl('CA', class))

bed_top <- bed_293_CTCF %>%
  filter(cCRE %in% bed_293T_CA$cCRE) %>%
  arrange(desc(score)) %>%
  head(opt$`n-top`)
message(sprintf('[prep] Top-%d CTCF cCREs after 293T-CA filter: %d rows.',
                opt$`n-top`, nrow(bed_top)))

ctcf_top_gr <- GRanges(
  seqnames = bed_top[, 1],
  ranges   = IRanges(start = as.numeric(bed_top[, 2]),
                     end   = as.numeric(bed_top[, 3]))
)
hits_top <- findOverlaps(CTCF_bins_gr, ctcf_top_gr)
pos_idx_local <- unique(queryHits(hits_top))
message(sprintf('[prep] Positive bin count (post-blacklist): %d', length(pos_idx_local)))

# -- negative set ------------------------------------------------------------
message('[prep] Building negative set ...')
CTCF_bound <- read.table(LOCAL$ctcf_bound_bed, sep = '\t', stringsAsFactors = FALSE) %>%
  dplyr::rename(chrom = V1, start = V2, end = V3, cCRE = V5, class = V6)
ctcf_bound_gr <- GRanges(
  seqnames = CTCF_bound[, 1],
  ranges   = IRanges(start = as.numeric(CTCF_bound[, 2]),
                     end   = as.numeric(CTCF_bound[, 3]))
)
hits_bound <- findOverlaps(CTCF_bins_gr, ctcf_bound_gr)
nonneg_idx_local <- unique(queryHits(hits_bound))
all_local <- seq_along(CTCF_bins_gr)
potential_neg <- setdiff(all_local, nonneg_idx_local)
set.seed(opt$seed)
neg_idx_local <- sample(potential_neg, size = length(pos_idx_local))
message(sprintf('[prep] Negative bin count: %d (sampled from %d candidates).',
                length(neg_idx_local), length(potential_neg)))

# -- map back to original (mm) row indices, write TSV ------------------------
to_orig0 <- function(local_1based) local_to_orig[local_1based] - 1L  # 0-based
pos_idx_orig0 <- to_orig0(pos_idx_local)
neg_idx_orig0 <- to_orig0(neg_idx_local)

tsv_path <- file.path(out_dir, 'pos_neg_bins.tsv')
df <- data.frame(
  bin_idx_orig          = c(pos_idx_orig0, neg_idx_orig0),
  bin_idx_post_blacklist = c(pos_idx_local - 1L, neg_idx_local - 1L),
  label                 = c(rep('pos', length(pos_idx_orig0)),
                            rep('neg', length(neg_idx_orig0))),
  bin_name              = c(bin_names[pos_idx_orig0 + 1L],
                            bin_names[neg_idx_orig0 + 1L]),
  stringsAsFactors      = FALSE
)
write.table(df, tsv_path, sep = '\t', row.names = FALSE, quote = FALSE)
message('[prep] Wrote ', tsv_path)

meta <- list(
  mm_regions       = reg_path,
  n_bins_universe  = length(bin_names),
  n_blacklist      = length(blacklist_indices),
  n_post_blacklist = length(CTCF_bins_gr),
  n_pos            = length(pos_idx_local),
  n_neg            = length(neg_idx_local),
  n_top_cCRE       = opt$`n-top`,
  seed             = opt$seed,
  cache_dir        = cache,
  sources          = URLS
)

# Keep the dependency-free path: write JSON manually so this script does not
# need to require jsonlite.
to_json_scalar <- function(x) {
  if (is.character(x))
    sprintf('"%s"', gsub('"', '\\"', x, fixed = TRUE))
  else
    format(x, scientific = FALSE)
}
to_json <- function(lst, indent = '') {
  inner <- vapply(names(lst), function(k) {
    v <- lst[[k]]
    if (is.list(v)) {
      sub <- to_json(v, paste0(indent, '  '))
      sprintf('%s  "%s": %s', indent, k, sub)
    } else if (length(v) == 1L) {
      sprintf('%s  "%s": %s', indent, k, to_json_scalar(v))
    } else {
      arr <- paste(vapply(v, to_json_scalar, character(1)), collapse = ', ')
      sprintf('%s  "%s": [%s]', indent, k, arr)
    }
  }, character(1))
  paste0('{\n', paste(inner, collapse = ',\n'), '\n', indent, '}')
}
writeLines(to_json(meta), file.path(out_dir, 'pos_neg_bins.meta.json'))
message('[prep] Wrote ', file.path(out_dir, 'pos_neg_bins.meta.json'))
message('[prep] Done.')
