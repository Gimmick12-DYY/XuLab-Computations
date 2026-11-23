# Round 9b: DESeq2 multi-group differential-peak 6x6 (declared 2026-11-23, before any result; recipe fixed)

Reason: round 9 fitted one model per panel TF with design ~ group (this TF vs "rest" = the other 78 TFs pooled). Pooling
78 different TFs into one group puts their real differences into the dispersion (median ~31), so nothing was testable.
The standard multi-group DESeq2 setup estimates dispersion from replicate variation within each TF only.
Output is an alternative (fig2_corr_matrix_6x6_deseq2mg, _colz), not the main panel, until the user picks.

1. **Inputs (primary arm, "consensus").** The round 9 consensus peaks (718,577) and the round 9 count matrix and
   replicates, unchanged (downstream/fig2_qc/round9_deseq2/SPEC.md 1-2: libraries holding >= 10% of a TF's reads, else 3
   random cell pseudo-replicates, seed 2026; 199 samples).
2. **Model.** DESeq2 (R 4.5), design ~ TF (79 levels), size factors estimateSizeFactors(type = "poscounts") as in round 9.
   One fit (DESeq(), BiocParallel); dispersions therefore come from within-TF replicate variation only.
3. **Contrasts.** For each of the 6 Fig 2 TFs, a numeric contrast "this TF minus the mean of the other 78 TFs"
   (+1 for the TF, -1/78 for each other TF, written on the model coefficients), results(contrast = ...), then
   lfcShrink(type = "ashr"). apeglm (round 9) only shrinks single coefficients and does not accept numeric contrasts,
   so ashr is used here.
4. **Reported per TF:** the fit's dispersion median, n differential peaks (padj < 0.05 and |shrunken LFC| > 1; and
   padj < 0.05 and shrunken LFC > 1), and quantiles (1, 5, 25, 50, 75, 95, 99%) of the shrunken LFC.
5. **ChIP side.** Unchanged from round 9: per region log2(CPM + 1) of each ChIP minus the mean of the other 5 ChIPs
   (bedtools multicov -q 30, duplicates excluded; CPM over all regions of the arm).
6. **6x6.** Pearson r between our TF's shrunken LFC and the ChIP log2FC, (primary) across all regions, (secondary)
   across regions differential in our TF (padj < 0.05, LFC > 1) or in the ChIP (log2FC > 1). Statistics as round 9:
   own-best rows/columns, diagonal minus off-diagonal, exact 720-permutation p, without CTCF (120), and the same on odd
   vs even + X chromosome halves.
7. **Secondary arm ("chip1kb", declared now, same design and contrasts).** Regions = union of the 6 ChIP peak sets'
   summits +-1 kb, merged, chr1-22 + X, hg38 blacklist v2 removed (more reads per region). Our counts per region per
   replicate with the same replicate rule and seed (read 5' ends); bulk counts and ChIP log2FC as in 5. Same 6x6 and
   statistics.
8. **Block check.** The round 10 4-ChIP presence rows (main set: CTCF, ZBTB7A, ZIC2, ZNF282) are re-scored with these
   LFCs: each presence row takes the LFC of the region it overlaps (first overlap), and the mean per single-ChIP block is
   reported for each of the 4 TFs, both arms.
9. **Rendering.** Plain r and column z-score of the primary arm in the style of fig2_corr_matrix_6x6
   (fig2_corr_matrix_6x6_deseq2mg, _colz); the secondary arm as fig2_corr_matrix_6x6_deseq2mg_chip1kb. Optional block
   in paper_figures/fig2/build.sbatch (ROUND9B=1).
Scripts: downstream/fig2_qc/r9b_*. No changes after results.
