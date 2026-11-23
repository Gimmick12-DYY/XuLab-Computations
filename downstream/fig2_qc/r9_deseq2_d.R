#!/usr/bin/env Rscript
# Round 9 step d (one panel TF per call): DESeq2 ~ group (TF vs rest), size factors from DESeq2, lfcShrink apeglm.
# Usage: Rscript r9_deseq2_d.R <TF>  -> OUT/round9_deseq2/deseq2_<TF>.tsv.gz (peak, log2FC_shrunk, padj, baseMean)
suppressPackageStartupMessages({library(DESeq2); library(Matrix); library(apeglm); library(BiocParallel)})
# Multicore (same model and results as single-core): workers = SLURM_CPUS_PER_TASK.
ncpu <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")); register(MulticoreParam(ncpu)); bp <- MulticoreParam(ncpu)
args <- commandArgs(TRUE); tf <- args[1]
o9 <- "/work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/round9_deseq2"
M <- readMM(file.path(o9, "counts.mtx")); S <- read.delim(file.path(o9, "samples.tsv"))
cnt <- as.matrix(M); storage.mode(cnt) <- "integer"; colnames(cnt) <- S$sample
cd <- data.frame(row.names = S$sample, group = factor(ifelse(S$tf == tf, "tf", "rest"), levels = c("rest", "tf")))
dds <- DESeqDataSetFromMatrix(cnt, cd, ~ group)
dds <- DESeq(dds, quiet = TRUE, parallel = ncpu > 1, BPPARAM = bp)
res <- results(dds, name = "group_tf_vs_rest", parallel = ncpu > 1, BPPARAM = bp)
sh <- lfcShrink(dds, coef = "group_tf_vs_rest", type = "apeglm", quiet = TRUE, parallel = ncpu > 1, BPPARAM = bp)
out <- data.frame(log2FC_shrunk = sh$log2FoldChange, padj = res$padj, baseMean = res$baseMean)
write.table(out, gzfile(file.path(o9, paste0("deseq2_", tf, ".tsv.gz"))), sep = "\t", quote = FALSE, row.names = FALSE)
cat(tf, "differential (padj<0.05, LFC>1):", sum(out$padj < 0.05 & out$log2FC_shrunk > 1, na.rm = TRUE), "\n")
