# Figure 2 legend notes (QC: single-cell vs bulk ChIP)

## Main: correlation matrix (fig2_corr_matrix_6x6)
- Pearson r between single-cell raw signal (rows) and bulk ChIP (columns), across summit groups.
- Each reference's preferred top 5,000 summits are split into 30 groups (180 groups in total).
- Bulk value: log CPM within ±250 bp.
- Single-cell value: centre (±100 bp) over flank (1–2 kb), relative to the leave-one-out 79-TF pool, at ZNF777 depth
  (420,339 reads; 3 seeds averaged).
- Colour is clipped at ±0.2 (diverging, centred at 0), and every cell is annotated with its value.
- Statistics: diagonal minus off-diagonal, with an exact p from all 720 row-label permutations; the same without CTCF
  (120 permutations).
- Reference cell lines: CTCF, MAZ, ZBTB7A, ZIC2 and ZNF777 are ENCODE HEK293 (the last four are eGFP-tagged with
  anti-GFP); ZNF282 is GSM2466515, HEK293T HA ChIP-exo.

## Display-only standardised versions of the main matrix (same values, no recomputation)
- fig2_corr_matrix_6x6_rowz: each sc row standardised, (r − row mean) / row SD (population SD); diverging, ±2.
  Same definition as the 11-12 S2 version.
- fig2_corr_matrix_6x6_rowminmax: each sc row scaled to 0–1, (r − row min) / (row max − row min); sequential.
- fig2_corr_matrix_6x6_colz: each ChIP column standardised; same definition as the 11-12 S3 version.
- The statistics belong to the unstandardised matrix: diagonal − off-diagonal = +0.151, p = 0.006 (720 permutations);
  without CTCF p = 0.033 (120). Row standardisation cannot create a diagonal; it only rescales each row.

## Main: peak heatmap (fig2_main_heatmap; per TF fig2_main_heatmap_<TF>; chosen 2026-11-23)
- One shared axis for all three panels: ±1 kb at 50 bp. Bulk ChIP CPM; sc raw 5' end CPM (3-bin running mean);
  sc imputed (Mdrop5) = the 1 kb imputed value of the genomic bin covering each 50 bp position.
- Rows: the reference's top 10,000 ChIP summits; rows whose summit 1 kb bin is outside Mdrop5 are dropped from all three
  panels (CTCF 4.6%, MAZ 2.9%, ZBTB7A 4.6%, ZIC2 7.1%, ZNF282 27.4%, ZNF777 6.4%); out-of-mask flank positions are 0.
  Sorted by bulk ChIP within ±200 bp.
- Noise reduction, identical for all three panels: consecutive sorted rows averaged in groups so ~500 rows are drawn
  (group size 20 for CTCF and MAZ, 17 ZBTB7A, 19 ZIC2, 12 ZNF282, 12 ZNF777), then a
  Gaussian smoothing along each row (sigma 100 bp = 2 bins). Colour 0 to the 95th percentile of the row-averaged,
  smoothed values per panel (turbo). The profile on top is the mean of the unsmoothed rows.

- Rows are the reference's top 10,000 ChIP summits by signalValue (fewer when the file has fewer peaks: ZBTB7A 8,400,
  ZNF282 7,989, ZNF777 6,223), minus the out-of-mask rows above. Mdrop5 = 1 kb bins with mean ATAC CPM > 0.0781542
  over the two pooled HEK293T ATAC-seq bigwigs; imputed includes the raw-positive cells kept by the imputation
  (keep_raw). Each panel has its own colour bar, unit and number of sites underneath; the profile row has its own y-axis.

## Supplement
- **fig2_supp_heatmap_premask:** the same processing as the main heatmap, with the genome-wide pre-mask imputation
  (no mask, so no rows are dropped).
- **fig2_supp_corr_10kb_6x6:** genome-wide 10 kb bins (chr1–22 and X, blacklist bins removed), Pearson r of log2 CPM+1,
  single-cell raw vs bulk ChIP (Paired-Tag style, Zhu et al. 2021). Colour ±0.6.
- **fig2_supp_corr_6x6_imputed_mdrop5 (and _added):** the main-matrix recipe (same 180 summit groups, same bulk values)
  with imputed single-cell tracks under Mdrop5 at 1 kb: summit bin vs the bins 2–5 kb away, relative to the
  leave-one-out 78-TF imputed pool; all calls, or model-added calls only. No diagonal (p = 0.70 for both).

## Alternatives (not the main panel; same rows, order, row grouping and smoothing as the main heatmap)
- **fig2_main_heatmap_v3_imp5kb:** ChIP and raw as the main (±1 kb, 50 bp); imputed on its own ±5 kb axis at its native
  1 kb resolution (11 bins), axis labelled "±5 kb, 1 kb bins".
- **fig2_main_heatmap_v3_rownorm:** imputed row-normalised: (x + pc) / (flank + pc), flank = mean of the 1 kb bins 2–5 kb
  from the summit on both sides, pc = 5th percentile of the nonzero per-site flank means; ratios averaged in the row
  groups, shown as log2 on a diverging map (symmetric at the 95th percentile of |log2|, 0.5 steps). The log2 diverging
  display was chosen over a turbo ratio so that enrichment and depletion read symmetrically. _rownorm_all applies the same
  normalisation to ChIP and raw (raw flank per 50 bp = CPM per kb / 20).
- **Why imputed looks noisy (numbers):** the per-site value is the fraction of cells positive in one 1 kb bin. Median
  positive cells in the summit bin per site: CTCF 3 (of 10,410 cells imputed; 20% of sites have 0), ZIC2 5 (4,896),
  ZNF282 5 (2,959), MAZ 38 (3,968), ZBTB7A 40 (7,005), ZNF777 39 (3,400). Median per-site binomial CV,
  sqrt((1 − p)/(n p)): CTCF 0.50, ZIC2 0.45, ZNF282 0.41, MAZ / ZBTB7A / ZNF777 ≈ 0.16. Sites also differ several-fold in
  overall level (between-site SD / mean 0.7–1.7), which shows as horizontal bands; and within ±1 kb each site has only
  2–3 distinct 1 kb values, so there is no positional shape to form a funnel.
