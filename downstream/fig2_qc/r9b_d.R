#!/usr/bin/env Rscript
# Round 9b step d (SPEC.md 2-4): one DESeq2 fit, design ~ TF (79 levels), poscounts size factors; for each Fig 2 TF a
# numeric contrast (TF minus mean of the other 78), results() + lfcShrink(type = "ashr").
# Usage: Rscript r9b_d.R consensus|chip1kb -> OUT/round9b_deseq2_multigroup/<arm>/deseq2mg_<TF>.tsv.gz, fit_summary.tsv
suppressPackageStartupMessages({library(DESeq2); library(Matrix); library(ashr); library(BiocParallel)})
ncpu <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")); bp <- MulticoreParam(ncpu); register(bp)
arm <- commandArgs(TRUE)[1]
base <- "/work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc"
inp <- if (arm == "consensus") file.path(base, "round9_deseq2") else file.path(base, "round9b_deseq2_multigroup", arm)
o <- file.path(base, "round9b_deseq2_multigroup", arm); dir.create(o, showWarnings = FALSE, recursive = TRUE)
M <- readMM(file.path(inp, "counts.mtx")); S <- read.delim(file.path(inp, "samples.tsv"))
cnt <- as.matrix(M); storage.mode(cnt) <- "integer"; colnames(cnt) <- S$sample; rm(M)
lv <- sort(unique(S$tf)); cd <- data.frame(row.names = S$sample, TF = factor(S$tf, levels = lv))
dds <- DESeqDataSetFromMatrix(cnt, cd, ~ TF); rm(cnt)
dds <- estimateSizeFactors(dds, type = "poscounts")
t0 <- Sys.time()
dds <- DESeq(dds, quiet = TRUE, parallel = ncpu > 1, BPPARAM = bp)
cat("fit minutes", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "\n")
disp <- dispersions(dds); dmed <- median(disp, na.rm = TRUE)
cat(arm, "regions", nrow(dds), "samples", ncol(dds), "dispersion median", dmed, "\n")
rn <- resultsNames(dds); ref <- lv[1]
qs <- c(.01, .05, .25, .5, .75, .95, .99); summ <- list()
for (tf in c("CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777")) {
  cv <- setNames(numeric(length(rn)), rn)
  others <- setdiff(lv, tf)
  for (x in others) if (x != ref) cv[paste0("TF_", make.names(x), "_vs_", make.names(ref))] <- -1 / length(others)
  if (tf != ref) cv[paste0("TF_", make.names(tf), "_vs_", make.names(ref))] <- 1
  stopifnot(cv["Intercept"] == 0, sum(cv != 0) == length(lv) - 1)
  res <- results(dds, contrast = cv, parallel = ncpu > 1, BPPARAM = bp)
  sh <- lfcShrink(dds, contrast = cv, res = res, type = "ashr", quiet = TRUE)
  out <- data.frame(log2FC_shrunk = sh$log2FoldChange, log2FC_mle = res$log2FoldChange, padj = res$padj, baseMean = res$baseMean)
  write.table(out, gzfile(file.path(o, paste0("deseq2mg_", tf, ".tsv.gz"))), sep = "\t", quote = FALSE, row.names = FALSE)
  pa <- !is.na(out$padj) & out$padj < 0.05
  q <- quantile(out$log2FC_shrunk, qs, na.rm = TRUE)
  summ[[tf]] <- data.frame(TF = tf, dispersion_median = dmed, n_diff_abs = sum(pa & abs(out$log2FC_shrunk) > 1),
                           n_diff_up = sum(pa & out$log2FC_shrunk > 1), n_padj05 = sum(pa),
                           t(setNames(as.numeric(q), paste0("lfc_q", qs * 100))))
  print(summ[[tf]])
}
write.table(do.call(rbind, summ), file.path(o, "fit_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
