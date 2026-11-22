# Round 8: differential regions called only inside the Mdrop5 accessibility mask (declared 2026-11-22, before any result)

User request 2026-11-22: "after we have the new masked imputed we should do better on the differential peaks". The round 6
raw-called differential regions (D-B) were mostly low-accessibility sites outside Mdrop5, so here the test is restricted
to accessible bins.
- **Universe.** 1 kb frame bins inside MASK = "Mdrop5" (mean ATAC CPM > 0.0781542 over the two pooled HEK293T ATAC
  bigwigs; regulatome/m46_stage1_imputed/work/masks_extra.npz), chr1-22 + X, hg38 blacklist v2 bins removed.
- **Differential test (per TF, per bin).** Binomial: x = the TF's count in the bin, n = the 79-TF total in the bin,
  p0 = the TF's share of all counts inside the universe; one-sided p, BH q over the universe bins; keep q < 0.05 and
  log2 fold enrichment over the other 78 > 1; rank by p; top 1,000 and top 5,000 bins per TF.
  - R = raw 5' read counts (ctcf_promoter_fp/counts/raw.npz).
  - I = Mdrop5-masked imputed, all calls (raw-positive cells + model calls inside Mdrop5; inputs/imp79_mdrop5.npz).
  - Ia = model-added calls inside Mdrop5 only (inputs/imp79_mdrop5_added.npz).
- **Scores vs bulk (6x6, rows = the variant's sc tracks, columns = bulk ChIP).**
  - M2b summit groups: each TF's differential set (in p order) split into 30 groups (180 groups). Bulk value per group =
    log2(sum of bulk ChIP CPM per 1 kb bin + 1), CPM taken over the 180 groups. sc value per group = log2((C + e)/(F + e))
    minus the same for the leave-one-out 78-TF pool; C = the variant's signal in the group's bins, F = mean of the bins
    2-5 kb away on both sides (8 bins); e = 0.5 for raw counts and 1e-4 for imputed fractions of cells.
  - M1 overlap: log2 odds ratio of the TF's differential bins overlapping each reference's peaks vs all universe bins.
  - Bulk ChIP references as in Fig 2 (CTCF ENCODE HEK293 ENCFF314ZAL peaks / ENCFF139DCW bigWig; MAZ, ZBTB7A, ZIC2,
    ZNF777 ENCODE HEK293; ZNF282 GSM2466515 HEK293T).
- **Statistics.** Own-best rows/columns, diagonal minus off-diagonal, exact 720-permutation p, without CTCF (120), and
  odd / even + X chromosome halves (bins split by chromosome before grouping).
- **Also reported.** Number of significant bins per TF and variant; share of each set inside the ATAC-peak union.
- Outputs results_293T_analysis/fig2_qc/round8_mdrop5_diff/; renders like fig2_corr_matrix_6x6 (colour +-0.2 for r,
  +-2 for M1); Mac figures_fig2_qc/round8_mdrop5_diff/.
