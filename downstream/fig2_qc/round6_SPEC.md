# Fig 2 round 6: differential peaks vs bulk ChIP (declared 2026-11-20, before any result was computed)

Panel: CTCF, MAZ, ZBTB7A, ZIC2, ZNF282, ZNF777 (NFYA out). Our cells are HEK293T. Differential sets are called from our
single-cell data only; bulk ChIP is never used to select regions.

## Differential sets (per TF), at full depth, raw single-cell reads (5' ends, not deduplicated)
- **D-A** MACS2 re-calls with the pooled reads of the other 78 TFs as control (pair_motif_composite/recall/peaks_tf,
  -p 0.01 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits). Significant = q < 0.05 (column 9 >= 1.301).
  Ranked by p (column 8); regions = summit +-250 bp; top 1,000 and top 5,000 (fewer if not available).
- **D-B** Count test on candidate regions = HEK293T ATAC union3 peaks plus lab MACS2 q < 0.05 summits +-250 bp of the
  6 panel TFs, merged, chr1-22 and X, minus the hg38 blacklist. Per region: n_i = TF reads, n_o = reads of the other 78
  TFs; one-sided binomial test of n_i given n_i + n_o with expected fraction R_i / (R_i + R_o) (genome-wide read totals);
  log2FC = log2(((n_i + 0.5)/R_i) / ((n_o + 0.5)/R_o)); BH over regions with n_i + n_o > 0. Kept: q < 0.05 and
  log2FC > 1, ranked by p; top 1,000 and top 5,000 (fewer if not available; counts reported).

## Comparison with bulk (rows = sc TF differential set, columns = bulk reference)
References: MAZ, ZBTB7A, ZIC2, ZNF777 = ENCODE HEK293; ZNF282 = GSM2466515 HEK293T; CTCF = ENCODE HEK293 (peaks
ENCFF314ZAL, BAM ENCFF139DCW, as in corr_matrix_6x6); variant with CTCF = GSE103651 HEK293T ChIP-nexus (peaks + pooled BAM).
- **M1 overlap enrichment**: log2 odds ratio of a TF's differential regions overlapping reference j's peaks vs a background of
  candidate regions (D-B candidates not in that TF's differential set) matched on ATAC signal decile (OmniATAC fold
  enrichment, mean over the region) x GC decile, 10 background regions per differential region (with replacement when a
  stratum is small); +0.5 on each cell of the 2x2 table.
- **M2a** Pearson r, over a TF's differential regions, between the sc log2FC (formula above) and bulk log2(CPM + 1)
  (reads in the region, CPM over the BAM's mapped reads).
- **M2b** summit-group version: each TF's differential regions ranked by p and split into 30 groups (180 groups for 6 TFs);
  per group, sc value = log2FC with summed counts, bulk = log2(CPM + 1) of summed reads (CPM over the 180 groups);
  Pearson of each sc track's 180 values with each bulk track's 180 values.

## Statistics and reporting
Each 6x6: own-best rows and columns, diag - offdiag, exact permutation p over all 720 shuffles of the sc row labels,
the same without CTCF (5x5, 120 shuffles), per-TF own vs best other. Chromosome halves (odd autosomes; even autosomes + X):
the selected regions are split by chromosome (no reselection) and every matrix is recomputed per half.
Versions run (no others): 2 differential sets (D-A, D-B) x 2 sizes (1,000, 5,000) x 3 metrics (M1, M2a, M2b) x 2 CTCF
references (ENCODE HEK293, HEK293T nexus) = 24 matrices, each also on the two halves.
Rendering: M1 and M2 6x6 like corr_matrix_6x6 (raw values in cells, own cells outlined, palette C, Arial 7 pt, cell lines
in labels, colour range stated). Outputs results_293T_analysis/fig2_qc/round6_diffpeaks/, Mac figures_fig2_qc/round6_diffpeaks/.
