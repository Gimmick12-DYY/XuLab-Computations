#!/usr/bin/env Rscript
# Compare our run_cicero table to the lab RBBP4.conns.rds (read-only).
# Writes a text report only under --out (never into /vast/som/xujie_lab).
suppressPackageStartupMessages(library(optparse))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--lab-conns", type = "character"),
  make_option("--ours", type = "character",
              help = "Peak1 Peak2 coaccess tsv or tsv.gz"),
  make_option("--out", type = "character")
)))

lab <- readRDS(opt$`lab-conns`)
lab$Peak1 <- as.character(lab$Peak1)
lab$Peak2 <- as.character(lab$Peak2)
lab <- lab[!is.na(lab$coaccess) & lab$Peak1 != lab$Peak2, , drop = FALSE]

con <- if (grepl("\\.gz$", opt$ours)) gzfile(opt$ours, "rt") else file(opt$ours, "rt")
ours <- read.table(con, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
close(con)
# tolerate extra cols / no header
if (!all(c("Peak1", "Peak2", "coaccess") %in% colnames(ours))) {
  colnames(ours)[1:3] <- c("Peak1", "Peak2", "coaccess")
}
ours$Peak1 <- as.character(ours$Peak1)
ours$Peak2 <- as.character(ours$Peak2)
ours <- ours[!is.na(ours$coaccess) & ours$Peak1 != ours$Peak2, , drop = FALSE]

pair_key <- function(a, b) {
  ifelse(a < b, paste(a, b, sep = "|"), paste(b, a, sep = "|"))
}
lab$key <- pair_key(lab$Peak1, lab$Peak2)
ours$key <- pair_key(ours$Peak1, ours$Peak2)
# max coaccess if both orientations present
lab_u <- aggregate(coaccess ~ key, lab, max)
ours_u <- aggregate(coaccess ~ key, ours, max)

shared <- merge(lab_u, ours_u, by = "key", suffixes = c(".lab", ".ours"))
n_lab <- nrow(lab_u)
n_ours <- nrow(ours_u)
n_int <- nrow(shared)
jacc <- if ((n_lab + n_ours - n_int) > 0) n_int / (n_lab + n_ours - n_int) else 0
sp <- if (n_int >= 3) suppressWarnings(cor(shared$coaccess.lab, shared$coaccess.ours, method = "spearman")) else NA
pe <- if (n_int >= 3) suppressWarnings(cor(shared$coaccess.lab, shared$coaccess.ours, method = "pearson")) else NA

lab_peaks <- unique(c(lab$Peak1, lab$Peak2))
ours_peaks <- unique(c(ours$Peak1, ours$Peak2))
p_int <- length(intersect(lab_peaks, ours_peaks))

lines <- c(
  sprintf("lab conns:  %s  pairs=%s  unique_peaks=%s", opt$`lab-conns`, format(n_lab, big.mark = ","), format(length(lab_peaks), big.mark = ",")),
  sprintf("our conns:  %s  pairs=%s  unique_peaks=%s", opt$ours, format(n_ours, big.mark = ","), format(length(ours_peaks), big.mark = ",")),
  sprintf("peak intersection: %s / lab=%s / ours=%s", format(p_int, big.mark = ","), format(length(lab_peaks), big.mark = ","), format(length(ours_peaks), big.mark = ",")),
  sprintf("unordered pair intersection: %s  jaccard=%.4f", format(n_int, big.mark = ","), jacc),
  sprintf("lab recovered of our pairs: %.3f", if (n_ours) n_int / n_ours else NA),
  sprintf("ours recovered of lab pairs: %.3f", if (n_lab) n_int / n_lab else NA),
  sprintf("shared coaccess spearman=%.4f  pearson=%.4f  n=%s", sp, pe, format(n_int, big.mark = ",")),
  sprintf("lab  coaccess median=%.4f mean=%.4f", median(lab_u$coaccess), mean(lab_u$coaccess)),
  sprintf("ours coaccess median=%.4f mean=%.4f", median(ours_u$coaccess), mean(ours_u$coaccess)),
  if (n_int) sprintf("shared lab  coaccess median=%.4f", median(shared$coaccess.lab)) else NULL,
  if (n_int) sprintf("shared ours coaccess median=%.4f", median(shared$coaccess.ours)) else NULL
)
dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)
writeLines(lines, opt$out)
cat(paste(lines, collapse = "\n"), "\n", sep = "")
