#!/usr/bin/env Rscript
# Round 9 DIAGNOSTIC (not part of the frozen spec): refit the same model as r9_deseq2_d.R and save the unshrunken MLE
# log2FC, lfcSE, the coefficient names, dispersion and size-factor summaries.  Usage: Rscript r9_deseq2_f_diag.R <TF>
suppressPackageStartupMessages({library(DESeq2); library(Matrix); library(BiocParallel)})
args <- commandArgs(TRUE); tf <- args[1]
ncpu <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")); bp <- MulticoreParam(ncpu)
o9 <- "/work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/round9_deseq2"
M <- readMM(file.path(o9, "counts.mtx")); S <- read.delim(file.path(o9, "samples.tsv"))
cnt <- as.matrix(M); storage.mode(cnt) <- "integer"; colnames(cnt) <- S$sample
cd <- data.frame(row.names = S$sample, group = factor(ifelse(S$tf == tf, "tf", "rest"), levels = c("rest", "tf")))
dds <- DESeqDataSetFromMatrix(cnt, cd, ~ group)
dds <- estimateSizeFactors(dds, type = "poscounts")
dds <- DESeq(dds, quiet = TRUE, parallel = ncpu > 1, BPPARAM = bp)
cat(tf, "levels:", levels(dds$group), "| resultsNames:", paste(resultsNames(dds), collapse = ", "), "| coef used: group_tf_vs_rest\n")
res <- results(dds, name = "group_tf_vs_rest", parallel = ncpu > 1, BPPARAM = bp)
sf <- sizeFactors(dds); d <- dispersions(dds)
cat(tf, "size factors: median", signif(median(sf), 3), "range", signif(min(sf), 3), "-", signif(max(sf), 3), "n < 0.05:", sum(sf < 0.05), "\n")
cat(tf, "dispersion: median", signif(median(d, na.rm = TRUE), 3), "max", signif(max(d, na.rm = TRUE), 3), "n at max:", sum(d >= max(d, na.rm = TRUE) * 0.999, na.rm = TRUE), "\n")
cat(tf, "MLE log2FC quantiles (1,5,25,50,75,95,99%):", paste(signif(quantile(res$log2FoldChange, c(.01,.05,.25,.5,.75,.95,.99), na.rm = TRUE), 3), collapse = " "), "\n")
reps <- S$tf == tf
out <- data.frame(log2FC_mle = res$log2FoldChange, lfcSE = res$lfcSE, pvalue = res$pvalue, tf_total = rowSums(cnt[, reps, drop = FALSE]))
write.table(out, gzfile(file.path(o9, paste0("diag_mle_", tf, ".tsv.gz"))), sep = "\t", quote = FALSE, row.names = FALSE)
