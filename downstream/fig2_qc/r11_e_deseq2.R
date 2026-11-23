#!/usr/bin/env Rscript
# Round 11 step e (SPEC.md 5): DESeq2 ~ ChIP (6 levels, 12 samples); per ChIP contrast = ChIP minus mean of the other 5;
# ashr shrinkage. -> O11/deseq2_<ChIP>.tsv.gz, deseq2_summary.tsv
suppressPackageStartupMessages({library(DESeq2); library(ashr)})
o <- "/work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/round11_chipdeseq2"
cnt <- as.matrix(read.delim(gzfile(file.path(o, "counts.tsv.gz")), check.names = FALSE)); storage.mode(cnt) <- "integer"
chip <- sub("_rep[12]$", "", colnames(cnt)); lv <- sort(unique(chip))
dds <- DESeqDataSetFromMatrix(cnt, data.frame(row.names = colnames(cnt), ChIP = factor(chip, levels = lv)), ~ ChIP)
sf <- tryCatch({dds <- estimateSizeFactors(dds); "median-of-ratios"},
               error = function(e) {dds <<- estimateSizeFactors(dds, type = "poscounts"); "poscounts"})
cat("size factors:", sf, round(sizeFactors(dds), 3), "\n")
dds <- DESeq(dds, quiet = TRUE)
dmed <- median(dispersions(dds), na.rm = TRUE); cat("dispersion median", dmed, "\n")
rn <- resultsNames(dds); ref <- lv[1]; S <- list()
for (tf in lv) {
  cv <- setNames(numeric(length(rn)), rn)
  for (x in setdiff(lv, tf)) if (x != ref) cv[paste0("ChIP_", x, "_vs_", ref)] <- -1 / (length(lv) - 1)
  if (tf != ref) cv[paste0("ChIP_", tf, "_vs_", ref)] <- 1
  stopifnot(cv["Intercept"] == 0)
  res <- results(dds, contrast = cv)
  sh <- lfcShrink(dds, contrast = cv, res = res, type = "ashr", quiet = TRUE)
  out <- data.frame(lfc = sh$log2FoldChange, lfc_mle = res$log2FoldChange, padj = res$padj, baseMean = res$baseMean)
  write.table(out, gzfile(file.path(o, paste0("deseq2_", tf, ".tsv.gz"))), sep = "\t", quote = FALSE, row.names = FALSE)
  d <- !is.na(out$padj) & out$padj < 0.05 & out$lfc > 1
  S[[tf]] <- data.frame(ChIP = tf, size_factors = sf, dispersion_median = dmed, n_differential = sum(d),
                        n_padj05 = sum(!is.na(out$padj) & out$padj < 0.05))
  print(S[[tf]])
}
write.table(do.call(rbind, S), file.path(o, "deseq2_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
