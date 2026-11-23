# Round 9: differential-peak 6x6 with DESeq2 (declared 2026-11-23, before any result; recipe fixed)

User request 2026-11-23: replace the summit-group recipe with a proper differential-peak approach. Output is an
alternative (fig2_corr_matrix_6x6_deseq2, _colz), not the main panel, until the user picks.

1. **Consensus peaks.** Union of the 79 TFs' lab MACS2 peaks (q < 0.05; /vast/.../bigwig/macs2/TF.<TF>) and the 6 bulk
   ChIP peak sets of Fig 2 (CTCF = ENCODE HEK293 ENCFF314ZAL; MAZ, ZBTB7A, ZIC2, ZNF777 = ENCODE HEK293 IDR; ZNF282 =
   GSM2466515 HEK293T), each as summit +-250 bp, merged (bedtools merge), chr1-22 + X, hg38 blacklist v2 removed.
2. **Our counts.** Read 5' ends (fragment beds, read-only) per peak, per replicate. Replicates = the library label in the
   cell barcode (prefix before the last "_" field, e.g. 293T_TFF1_DNA_Pool-1, 293T-frame2_DNA); a library counts as a
   replicate for a TF when it holds >= 10% of that TF's reads; reads from smaller libraries are left out. Libraries are
   partly confounded with TFs (different TFs were run in different libraries), so library cannot be a covariate; it is
   used only to define replicates. A TF with fewer than 2 qualifying libraries is split into 3 random cell
   pseudo-replicates (seed 2026) instead; which TFs fall back is reported.
3. **Our differential.** DESeq2 (R 4.5, Rscript) per panel TF: samples = all replicates of all 79 TFs; design ~ group,
   group = this TF vs rest (the other 78 TFs' replicates); DESeq2 size factors; lfcShrink(type = "apeglm"). Output per
   peak: shrunken log2FC and padj. Differential = padj < 0.05 and shrunken log2FC > 1 (n reported).
4. **ChIP differential.** The local bulk files are merged BAMs (no ENCODE replicate BAMs on Longleaf), so per spec the
   fallback is used: per peak log2(CPM + 1) of each ChIP minus the mean of the other 5 ChIPs. Counts = reads per peak
   (bedtools multicov, duplicates excluded, MAPQ >= 30); CPM over all consensus peaks. Differential = log2FC > 1.
5. **6x6.** Pearson r between our TF's shrunken log2FC and the ChIP's log2FC: (primary) across all consensus peaks;
   (secondary) across the union of peaks differential in our TF or in the ChIP. Rows = our 6 TFs, columns = the 6 ChIPs.
   Statistics: own-best rows/columns, diagonal minus off-diagonal, exact 720-permutation p, without CTCF (120), and the
   same on odd vs even + X chromosome halves (peaks split by chromosome).
6. **Rendering.** Plain r and column z-score, in the style of fig2_corr_matrix_6x6 (paper_figures/fig2, RECIPE =
   "deseq2"); added to build.sbatch.
Scripts: downstream/fig2_qc/r9_deseq2_*.
