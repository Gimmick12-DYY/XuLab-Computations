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
})

`%+%` <- function(a, b) paste0(a, b)   # string concat for readable help text

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
              help = 'RNG seed for the (random-mode) negative sampler [default %default]'),
  make_option(c('--negative-mode'), type = 'character', default = 'desert',
              help = "How to pick negatives from the candidate pool (post-blacklist bins "
                     %+% "overlapping NO cCRE): 'all_candidates' (use the ENTIRE pool -> "
                     %+% "genome-wide false-positive rate, the discriminating metric), "
                     %+% "'desert' (rank by bulk CTCF coverage, take the strictest = "
                     %+% "lowest-signal/farthest; WARNING: the extreme tail is dead/closed "
                     %+% "chromatin where every method calls 0 peaks -> uninformative), or "
                     %+% "'random' (legacy: uniform sample matched to #positives). 'desert' "
                     %+% "requires --bulk-ctcf-bam. [default %default]"),
  make_option(c('--max-neg'), type = 'double', default = 0,
              help = 'Cap on the number of negatives in all_candidates mode (0 = no cap, '
                     %+% 'use every candidate). If the pool exceeds this, a random subset '
                     %+% 'of this size is taken (seed = --seed). [default %default]'),
  make_option(c('--n-pos'), type = 'double', default = 0,
              help = 'Target number of POSITIVE bins. >0 selects the top-N bins by their '
                     %+% 'best overlapping CTCF z-score (high-confidence first), ignoring '
                     %+% '--n-top. 0 = legacy (bins overlapping the top --n-top cCREs). '
                     %+% '[default %default]'),
  make_option(c('--n-neg'), type = 'double', default = 0,
              help = 'Target number of NEGATIVE bins. >0 takes a random sample of this size '
                     %+% 'from the no-cCRE candidate pool (seed = --seed), overriding '
                     %+% '--negative-mode. Use with --n-pos for a balanced set (e.g. '
                     %+% '100k/100k). 0 = use --negative-mode. [default %default]'),
  make_option(c('--bulk-ctcf-bam'), type = 'character', default = NULL,
              help = 'Coordinate-sorted + indexed (.bai) bulk CTCF ChIP-seq BAM. Per-bin '
                     %+% 'coverage is computed with `samtools bedcov`; lowest = most desert.'),
  make_option(c('--bulk-bam'), type = 'character', default = NULL,
              help = 'Alias for --bulk-ctcf-bam; the TF-agnostic name. The bulk ChIP-seq BAM '
                     %+% 'for the TF being benchmarked (desert ranking via samtools bedcov).'),
  make_option(c('--tf-peaks-bed'), type = 'character', default = NULL,
              help = 'GENERIC (non-CTCF) mode. A narrowPeak/BED of the TF\'s bulk ChIP-seq '
                     %+% 'peaks. Positives become bins overlapping these peaks (ranked by '
                     %+% '--peak-signal-col); the SCREEN/cCRE downloads are skipped. Leave '
                     %+% 'unset to use the CTCF cCRE machinery.'),
  make_option(c('--peak-signal-col'), type = 'integer', default = 7L,
              help = 'Column in --tf-peaks-bed used to rank positive bins (narrowPeak col 7 = '
                     %+% 'signalValue). [default %default]'),
  make_option(c('--exclude-flank-bp'), type = 'double', default = 0,
              help = 'In generic mode, also exclude bins within this many bp of any TF peak '
                     %+% 'from the negative candidate pool (avoids near-peak shoulders). '
                     %+% '[default %default]'),
  make_option(c('--no-exclude-ccre'), action = 'store_true', default = FALSE,
              help = 'Generic mode: by DEFAULT the negative candidate pool also excludes ALL '
                     %+% 'cCREs (293+293T registries), matching the CTCF negative definition '
                     %+% '(non-regulatory genome). Pass this flag to exclude only the TF peaks '
                     %+% 'instead (larger, accessibility-inclusive pool).'),
  make_option(c('--samtools'), type = 'character', default = 'samtools',
              help = 'samtools executable (default %default; needs >=1.x bedcov).'),
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

# GENERIC (TF-agnostic) mode: positives come from a TF ChIP-seq narrowPeak
# instead of CTCF SCREEN cCREs. Detected by --tf-peaks-bed.
generic_mode <- !is.null(opt$`tf-peaks-bed`) && nzchar(opt$`tf-peaks-bed`)
# Bulk BAM: accept the TF-agnostic --bulk-bam, falling back to --bulk-ctcf-bam.
opt$`bulk-ctcf-bam` <- if (!is.null(opt$`bulk-bam`) && nzchar(opt$`bulk-bam`))
  opt$`bulk-bam` else opt$`bulk-ctcf-bam`
if (generic_mode) {
  tf_peaks_path <- normalizePath(opt$`tf-peaks-bed`, mustWork = TRUE)
  message('[prep] GENERIC mode: positives from TF peaks ', tf_peaks_path)
}

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
  url <- URLS[[key]]
  message(sprintf('[prep] Downloading %s -> %s', url, path))
  # Longleaf network can be bursty; use a longer timeout and retry loop.
  old_timeout <- getOption('timeout')
  on.exit(options(timeout = old_timeout), add = TRUE)
  options(timeout = max(600, old_timeout))
  ok <- FALSE
  for (attempt in 1:4) {
    if (attempt > 1L) {
      message(sprintf('[prep] Retry %d/4 for %s', attempt, key))
      Sys.sleep(2L * attempt)
    }
    tryCatch({
      download.file(url, destfile = path, quiet = TRUE, mode = 'wb')
      sz <- suppressWarnings(file.info(path)$size)
      if (!is.na(sz) && sz > 0) ok <- TRUE
    }, error = function(e) {
      message(sprintf('[prep] Download failed (%s): %s', key, conditionMessage(e)))
    })
    if (ok) break
    if (file.exists(path)) file.remove(path)
  }
  if (!ok) {
    stop(sprintf('Failed to download %s after retries: %s', key, url))
  }
  path
}

# Generic mode: positives come from the local TF peaks BED, so we skip the
# CTCF-specific SCREEN files (ctcf_bound_bed, ctcf_293_zscores). We still fetch
# the cCRE registries when the negative pool excludes all cCREs (the default,
# to match the CTCF negative definition).
exclude_all_ccre <- !isTRUE(opt$`no-exclude-ccre`)
if (generic_mode) {
  dl_keys <- if (exclude_all_ccre)
    c('blacklist_bed_gz', 'cCRE_293_bed', 'cCRE_293T_bed') else 'blacklist_bed_gz'
} else {
  dl_keys <- names(URLS)
}
for (k in dl_keys) ensure_file(k)

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
# Avoid hard dependency on rtracklayer::import.bed (not always installed).
blacklist_df <- read.table(LOCAL$blacklist_bed_gz, sep = '\t',
                           stringsAsFactors = FALSE, comment.char = '',
                           quote = '', header = FALSE)
if (ncol(blacklist_df) < 3L) {
  stop('Blacklist BED appears malformed: expected at least 3 columns, got ',
       ncol(blacklist_df))
}
blacklist_gr <- GRanges(
  seqnames = blacklist_df[[1]],
  ranges   = IRanges(start = as.numeric(blacklist_df[[2]]) + 1,
                     end   = as.numeric(blacklist_df[[3]]))
)
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
n_pos_target <- as.integer(opt$`n-pos`)
# `feature_gr`: the TF's binding feature ranges (CTCF cCREs or TF peaks). Used
# below both to EXCLUDE positives from the negative pool and (desert mode) as
# the distance-to-nearest-site tiebreak.
if (generic_mode) {
  message('[prep] Building positive set from TF peaks: ', tf_peaks_path)
  pk <- read.table(tf_peaks_path, sep = '\t', stringsAsFactors = FALSE,
                   comment.char = '', quote = '', header = FALSE)
  if (ncol(pk) < 3L) stop('TF peaks BED needs >= 3 columns (chrom,start,end).')
  sigcol <- as.integer(opt$`peak-signal-col`)
  pk_score <- if (ncol(pk) >= sigcol) suppressWarnings(as.numeric(pk[[sigcol]])) else rep(1, nrow(pk))
  pk_score[is.na(pk_score)] <- 0
  feature_gr <- GRanges(
    seqnames = pk[[1]],
    ranges   = IRanges(start = as.numeric(pk[[2]]) + 1, end = as.numeric(pk[[3]]))
  )
  mcols(feature_gr)$score <- pk_score
  message(sprintf('[prep] %d TF peaks loaded (rank by column %d).', length(feature_gr), sigcol))

  hits_pk <- findOverlaps(CTCF_bins_gr, feature_gr)
  if (length(hits_pk) == 0L)
    stop('No bins overlap any TF peak -- check chrom naming (chr1 vs 1) / coordinates.')
  bin_best <- tapply(mcols(feature_gr)$score[subjectHits(hits_pk)], queryHits(hits_pk), max)
  pos_bins_all <- as.integer(names(bin_best))
  if (n_pos_target > 0) {
    ord  <- order(as.numeric(bin_best), decreasing = TRUE)
    take <- min(n_pos_target, length(pos_bins_all))
    if (take < n_pos_target)
      warning(sprintf('[prep] only %d positive bins available; requested %d', take, n_pos_target))
    pos_idx_local <- pos_bins_all[ord[seq_len(take)]]
    message(sprintf('[prep] Positive bins (top %d by peak signal): %d of %d peak-overlapping bins.',
                    n_pos_target, length(pos_idx_local), length(pos_bins_all)))
  } else {
    pos_idx_local <- pos_bins_all
    message(sprintf('[prep] Positive bins (all peak-overlapping): %d', length(pos_idx_local)))
  }
} else {
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

  # CTCF cCREs (293 CTCF class) that are accessible in 293T, with z-scores.
  bed_ctcf_ca <- bed_293_CTCF %>%
    filter(cCRE %in% bed_293T_CA$cCRE, !is.na(score))

  if (n_pos_target > 0) {
    # Rank POSITIVE BINS by their best (max) overlapping CTCF z-score, take top n_pos.
    ctcf_all_gr <- GRanges(
      seqnames = bed_ctcf_ca$chrom,
      ranges   = IRanges(start = as.numeric(bed_ctcf_ca$start),
                         end   = as.numeric(bed_ctcf_ca$end))
    )
    mcols(ctcf_all_gr)$score <- as.numeric(bed_ctcf_ca$score)
    hits_all <- findOverlaps(CTCF_bins_gr, ctcf_all_gr)
    bin_best_z <- tapply(mcols(ctcf_all_gr)$score[subjectHits(hits_all)], queryHits(hits_all), max)
    pos_bins_all <- as.integer(names(bin_best_z))
    ord <- order(as.numeric(bin_best_z), decreasing = TRUE)
    take <- min(n_pos_target, length(pos_bins_all))
    if (take < n_pos_target)
      warning(sprintf('[prep] only %d positive bins available; requested %d', take, n_pos_target))
    pos_idx_local <- pos_bins_all[ord[seq_len(take)]]
    message(sprintf('[prep] Positive bins (top %d by per-bin CTCF z-score): %d of %d candidate bins.',
                    n_pos_target, length(pos_idx_local), length(pos_bins_all)))
  } else {
    bed_top <- bed_ctcf_ca %>% arrange(desc(score)) %>% head(opt$`n-top`)
    message(sprintf('[prep] Top-%d CTCF cCREs after 293T-CA filter: %d rows.',
                    opt$`n-top`, nrow(bed_top)))
    ctcf_top_gr <- GRanges(
      seqnames = bed_top[, 1],
      ranges   = IRanges(start = as.numeric(bed_top[, 2]), end = as.numeric(bed_top[, 3]))
    )
    pos_idx_local <- unique(queryHits(findOverlaps(CTCF_bins_gr, ctcf_top_gr)))
  }
}
message(sprintf('[prep] Positive bin count (post-blacklist): %d', length(pos_idx_local)))

# -- negative set ------------------------------------------------------------
message('[prep] Building negative set ...')
all_local <- seq_along(CTCF_bins_gr)
if (generic_mode) {
  # Candidate negatives = post-blacklist bins NOT overlapping any TF peak
  # (optionally extended by --exclude-flank-bp), AND (default) NOT overlapping
  # any cCRE -- so the negative pool matches the CTCF definition (non-regulatory
  # genome). Pass --no-exclude-ccre to keep only the TF-peak exclusion.
  excl_gr <- feature_gr
  flank <- as.numeric(opt$`exclude-flank-bp`)
  if (flank > 0) excl_gr <- suppressWarnings(
    GenomicRanges::resize(excl_gr, width = width(excl_gr) + 2 * flank, fix = 'center'))
  peak_idx_local <- unique(queryHits(findOverlaps(CTCF_bins_gr, excl_gr)))

  ccre_idx_local <- integer(0)
  if (exclude_all_ccre) {
    bed_293 <- read.table(LOCAL$cCRE_293_bed, sep = '\t', stringsAsFactors = FALSE)
    bed_293T <- read.table(LOCAL$cCRE_293T_bed, sep = '\t', stringsAsFactors = FALSE)
    all_ccre <- rbind(bed_293[, 1:3], bed_293T[, 1:3])
    names(all_ccre) <- c('chrom', 'start', 'end')
    all_ccre_gr <- GRanges(
      seqnames = all_ccre$chrom,
      ranges   = IRanges(start = as.numeric(all_ccre$start) + 1, end = as.numeric(all_ccre$end))
    )
    ccre_idx_local <- unique(queryHits(findOverlaps(CTCF_bins_gr, all_ccre_gr)))
  }

  nonneg_idx_local <- union(peak_idx_local, ccre_idx_local)
  potential_neg <- setdiff(all_local, nonneg_idx_local)
  ctcf_bound_gr <- feature_gr   # desert distance tiebreak = distance to nearest TF peak
  message(sprintf('[prep] Candidate negatives (no TF peak%s%s): %d / %d post-blacklist',
                  if (flank > 0) sprintf(' +/-%gbp', flank) else '',
                  if (exclude_all_ccre) ' AND no cCRE' else '',
                  length(potential_neg), length(all_local)))
} else {
  CTCF_bound <- read.table(LOCAL$ctcf_bound_bed, sep = '\t', stringsAsFactors = FALSE) %>%
    dplyr::rename(chrom = V1, start = V2, end = V3, cCRE = V5, class = V6)
  ctcf_bound_gr <- GRanges(
    seqnames = CTCF_bound[, 1],
    ranges   = IRanges(start = as.numeric(CTCF_bound[, 2]),
                       end   = as.numeric(CTCF_bound[, 3]))
  )
  nonneg_idx_local <- unique(queryHits(findOverlaps(CTCF_bins_gr, ctcf_bound_gr)))

  # STRICTER candidate pool: a desert bin must overlap NO cCRE at all (not just
  # CTCF-bound) -- non-regulatory genome, not merely "no called CTCF". Union the
  # 293 + 293T cCRE registries as the all-cCRE exclusion set.
  all_ccre <- rbind(bed_293[, c('chrom', 'start', 'end')],
                    bed_293T[, c('chrom', 'start', 'end')])
  all_ccre_gr <- GRanges(
    seqnames = all_ccre$chrom,
    ranges   = IRanges(start = as.numeric(all_ccre$start) + 1, end = as.numeric(all_ccre$end))
  )
  ccre_idx_local <- unique(queryHits(findOverlaps(CTCF_bins_gr, all_ccre_gr)))
  potential_neg <- setdiff(all_local, union(nonneg_idx_local, ccre_idx_local))
  message(sprintf('[prep] Candidate negatives (no cCRE AND no CTCF-bound cCRE): %d / %d post-blacklist',
                  length(potential_neg), length(all_local)))
}
if (length(potential_neg) < length(pos_idx_local))
  stop(sprintf('Only %d candidate negatives for %d positives; loosen the cCRE filter.',
               length(potential_neg), length(pos_idx_local)))

neg_mode <- tolower(opt$`negative-mode`)
bam <- opt$`bulk-ctcf-bam`
if (neg_mode == 'desert' && (is.null(bam) || !nzchar(bam))) {
  warning('[prep] negative-mode=desert but --bulk-ctcf-bam not given; falling back to random.')
  neg_mode <- 'random'
}

neg_signal <- rep(NA_real_, length(pos_idx_local))   # bulk CTCF coverage of chosen negs (desert mode)
neg_dist   <- rep(NA_real_, length(pos_idx_local))   # distance to nearest CTCF-bound site

n_neg_target <- as.integer(opt$`n-neg`)
if (n_neg_target > 0) {
  # Explicit negative count: random sample from the no-cCRE candidate pool.
  # Overrides --negative-mode so a balanced set (e.g. --n-pos 100k --n-neg 100k)
  # is one call. Representative (not the dead-desert extreme), so neg coverage
  # stays method-dependent and nonzero.
  set.seed(opt$seed)
  take <- min(n_neg_target, length(potential_neg))
  if (take < n_neg_target)
    warning(sprintf('[prep] only %d candidate negatives available; requested %d',
                    take, n_neg_target))
  neg_idx_local <- sample(potential_neg, size = take)
  neg_mode <- sprintf('random_n%d', n_neg_target)
  message(sprintf('[prep] Negative bins (random sample, --n-neg): %d of %d candidates.',
                  length(neg_idx_local), length(potential_neg)))
} else if (neg_mode == 'desert') {
  if (!file.exists(bam))
    stop(sprintf('Bulk CTCF BAM not found: %s', bam))
  if (!file.exists(paste0(bam, '.bai')) && !file.exists(sub('\\.bam$', '.bai', bam)))
    stop(sprintf('BAM index (.bai) not found for %s -- run `samtools index %s`', bam, bam))

  message('[prep] Desert mode: ranking ', length(potential_neg),
          ' candidates by bulk CTCF coverage via samtools bedcov')
  cand_gr <- CTCF_bins_gr[potential_neg]
  # 0-based half-open BED for samtools; carry the local index in column 4.
  cand_bed <- file.path(out_dir, '_neg_candidates.bed')
  write.table(
    data.frame(chrom = as.character(seqnames(cand_gr)),
               start = start(cand_gr) - 1L,
               end   = end(cand_gr),
               name  = seq_along(potential_neg)),
    cand_bed, sep = '\t', row.names = FALSE, col.names = FALSE, quote = FALSE)

  # `samtools bedcov BAM BED` -> input BED columns + a trailing coverage column
  # (sum of per-base read depth over each region). Order is preserved.
  cov_lines <- system2(opt$samtools, c('bedcov', shQuote(cand_bed), shQuote(bam)),
                       stdout = TRUE, stderr = '')
  if (length(cov_lines) == 0L)
    stop('samtools bedcov produced no output; check samtools + BAM.')
  cov_df <- read.table(text = cov_lines, sep = '\t', stringsAsFactors = FALSE)
  coverage <- as.numeric(cov_df[[ncol(cov_df)]])
  if (length(coverage) != length(potential_neg))
    stop(sprintf('bedcov row count %d != candidate count %d', length(coverage), length(potential_neg)))
  if (sum(coverage, na.rm = TRUE) <= 0)
    stop('All candidate coverage is 0 -- likely chrom-name mismatch between BAM header '
         %+% "and bins (e.g. 'chr1' vs '1'). Check `samtools idxstats`.")

  # Distance to nearest CTCF-bound site (secondary desertness criterion).
  d2n <- distanceToNearest(cand_gr, ctcf_bound_gr, ignore.strand = TRUE)
  dist_vec <- rep(Inf, length(cand_gr))
  dist_vec[queryHits(d2n)] <- mcols(d2n)$distance

  # Desertness ranking: PRIMARY lowest bulk CTCF coverage, TIEBREAK farthest
  # from any CTCF-bound site. Deterministic (no seed). Take the strictest N.
  ord <- order(coverage, -dist_vec)
  sel <- ord[seq_len(length(pos_idx_local))]
  neg_idx_local <- potential_neg[sel]
  neg_signal    <- coverage[sel]
  neg_dist      <- dist_vec[sel]
  message(sprintf('[prep] Desert negatives: %d selected. Coverage of chosen: max=%.3g (cutoff), '
                  %+% 'median=%.3g, min=%.3g; candidate coverage median=%.3g',
                  length(neg_idx_local), max(neg_signal), median(neg_signal),
                  min(neg_signal), median(coverage)))

  # Diagnostic: full candidate ranking for inspection / threshold tuning.
  rank_tsv <- file.path(out_dir, 'neg_desert_ranking.tsv')
  diag <- data.frame(
    bin_name   = bin_names[local_to_orig[potential_neg]],
    bulk_ctcf_coverage = coverage,
    dist_to_ctcf_bound = dist_vec,
    selected   = potential_neg %in% neg_idx_local
  )
  diag <- diag[order(diag$bulk_ctcf_coverage, -diag$dist_to_ctcf_bound), ]
  write.table(diag, rank_tsv, sep = '\t', row.names = FALSE, quote = FALSE)
  message('[prep] Wrote candidate ranking ', rank_tsv)
  file.remove(cand_bed)
} else if (neg_mode %in% c('all_candidates', 'all')) {
  # Use the ENTIRE candidate pool as negatives (no ranking, no extreme-tail
  # selection). The pool spans the whole non-cCRE genome -- dead deserts AND
  # accessible-but-unbound regions -- so neg coverage becomes a method-dependent
  # genome-wide FALSE-POSITIVE RATE again (the 'absolute desert' tail collapsed
  # it to 0 for every method). Positives stay the cCRE-derived set; classes are
  # intentionally imbalanced, which is fine -- coverage is a per-class fraction
  # and AUROC (Mann-Whitney) is imbalance-robust.
  neg_idx_local <- potential_neg
  cap <- as.numeric(opt$`max-neg`)
  if (cap > 0 && length(neg_idx_local) > cap) {
    set.seed(opt$seed)
    neg_idx_local <- sample(neg_idx_local, size = as.integer(cap))
    message(sprintf('[prep] all_candidates: capped to %d of %d candidates (--max-neg).',
                    length(neg_idx_local), length(potential_neg)))
  } else {
    message(sprintf('[prep] all_candidates: using all %d candidates as negatives.',
                    length(neg_idx_local)))
  }
  message(sprintf('[prep] Class sizes: %d positives, %d negatives (ratio 1:%.1f).',
                  length(pos_idx_local), length(neg_idx_local),
                  length(neg_idx_local) / max(length(pos_idx_local), 1)))
} else {
  set.seed(opt$seed)
  neg_idx_local <- sample(potential_neg, size = length(pos_idx_local))
  message(sprintf('[prep] Negative bin count (random): %d (sampled from %d candidates).',
                  length(neg_idx_local), length(potential_neg)))
}

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
  positive_source  = if (generic_mode) 'tf_peaks_bed' else 'CTCF_SCREEN_cCRE',
  generic_mode     = generic_mode,
  tf_peaks_bed     = if (generic_mode) tf_peaks_path else 'NA',
  n_bins_universe  = length(bin_names),
  n_blacklist      = length(blacklist_indices),
  n_post_blacklist = length(CTCF_bins_gr),
  n_pos            = length(pos_idx_local),
  n_neg            = length(neg_idx_local),
  n_top_cCRE       = if (generic_mode) -1 else opt$`n-top`,
  n_pos_target     = n_pos_target,
  n_neg_target     = n_neg_target,
  negative_mode    = neg_mode,
  bulk_bam         = if (is.null(bam)) 'NA' else bam,
  n_candidate_neg  = length(potential_neg),
  neg_coverage_cutoff = if (neg_mode == 'desert') max(neg_signal) else -1,
  neg_coverage_median = if (neg_mode == 'desert') as.numeric(median(neg_signal)) else -1,
  seed             = opt$seed,
  cache_dir        = cache,
  # Only the sources actually used in this mode (generic uses just the blacklist
  # + the TF peaks BED; the CTCF SCREEN/cCRE URLs are not fetched in generic mode).
  sources          = if (generic_mode)
    list(blacklist_bed_gz = URLS$blacklist_bed_gz, tf_peaks_bed = tf_peaks_path)
    else URLS
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
