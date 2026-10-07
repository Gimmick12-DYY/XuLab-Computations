#!/usr/bin/env Rscript
# Reproduce Wang/Ren fitConns scoring directly from a raw Cicero conns RDS.
suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--conns-rds", type = "character",
              help = "raw Cicero RDS with Peak1, Peak2, coaccess"),
  make_option("--shuffle-para", type = "character",
              help = "Wang/Ren table containing meanShuf and stdShuf"),
  make_option("--output", type = "character",
              help = "output fitConns .txt or .txt.gz")
)))

for (x in c("conns-rds", "shuffle-para", "output"))
  if (is.null(opt[[x]])) stop("--", x, " is required")
if (!file.exists(opt$`conns-rds`)) stop("missing conns RDS: ", opt$`conns-rds`)
if (!file.exists(opt$`shuffle-para`)) stop("missing shuffle parameters: ", opt$`shuffle-para`)

x <- as.data.table(readRDS(opt$`conns-rds`))
need <- c("Peak1", "Peak2", "coaccess")
if (!all(need %in% names(x)))
  stop("conns RDS must contain: ", paste(need, collapse = ", "))
x <- x[!is.na(coaccess) & as.character(Peak1) != as.character(Peak2), ..need]

para <- fread(opt$`shuffle-para`)
if (!all(c("meanShuf", "stdShuf") %in% names(para)) || nrow(para) < 1L)
  stop("shuffle parameters must contain meanShuf and stdShuf")
mu <- as.numeric(para$meanShuf[[1]])
sd <- as.numeric(para$stdShuf[[1]])
if (!is.finite(mu) || !is.finite(sd) || sd <= 0)
  stop("invalid shuffle Gaussian: mean=", mu, " sd=", sd)

# This is the Ren-lab fitConns calculation: one-sided Gaussian upper tail,
# followed by BH across the full (orientation-preserving) Cicero table.
x[, p := pnorm(coaccess, mean = mu, sd = sd, lower.tail = FALSE)]
x[, nlog10p := -log10(p)]
x[, FDR := p.adjust(p, method = "BH")]
setcolorder(x, c("Peak1", "Peak2", "coaccess", "nlog10p", "FDR", "p"))

dir.create(dirname(opt$output), recursive = TRUE, showWarnings = FALSE)
fwrite(x, opt$output, sep = "\t", quote = FALSE)
message(sprintf(
  "[fitConns] rows=%s mu=%.15g sd=%.15g FDR<=0.05=%s p<=0.05=%s -> %s",
  format(nrow(x), big.mark = ","), mu, sd,
  format(sum(x$FDR <= 0.05), big.mark = ","),
  format(sum(x$p <= 0.05), big.mark = ","), opt$output
))
