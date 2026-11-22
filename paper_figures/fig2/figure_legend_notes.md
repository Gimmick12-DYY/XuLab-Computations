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

## Main: peak heatmap (fig2_main_heatmap; per TF fig2_main_heatmap_<TF>)
- **Rows:** the reference's top 10,000 ChIP summits by signalValue (fewer when the file has fewer peaks: ZBTB7A 8,400,
  ZNF282 7,989, ZNF777 6,223). Rows are in the same order in every panel, sorted top to bottom by bulk ChIP signal
  within ±200 bp. This follows Kaya-Okur et al. 2019, Nat Commun, Fig 2g.
- **Grid and window:** all panels are on the same 1 kb genomic grid, ±10 kb (21 bins centred on the 1 kb bin that holds
  the summit).
- **Tracks:**
  - Bulk ChIP: CPM per 1 kb tile (bamCoverage; duplicates ignored, reads extended to 200 bp).
  - sc raw: aggregated 5' read ends, CPM per 1 kb tile.
  - sc imputed: fraction of cells positive per 1 kb bin.
- **Imputed mask (MASK = "Mdrop5", the accessibility mask):** 1 kb bins with mean ATAC CPM > 0.0781542 over the two
  pooled HEK293T ATAC-seq bigwigs. Imputed = model calls inside the mask, applied to the genome-wide pre-mask imputation,
  plus the raw-positive cells, which the imputation keeps (keep_raw). Bins outside the mask are light grey (NA).
- **Colour:** one scale per panel, from 0 to the 98th percentile of that panel's nonzero values (turbo). The colour bar,
  unit and number of sites are shown under each panel.
- **Profile:** the row above each heatmap is the mean signal per bin over all rows, with the y-axis starting at 0 and a
  separate axis for each panel.
- **Share of summits inside Mdrop5:** CTCF 95%, MAZ 97%, ZBTB7A 95%, ZIC2 93%, ZNF282 73%, ZNF777 94%.

## Supplement
- **fig2_supp_heatmap_50bp:** the same rows, with ChIP and raw at ±1 kb in 50 bp bins (raw smoothed with a 3-bin
  running mean). Imputed stays on its native ±10 kb, 1 kb grid under Mdrop5. This panel shows the sharp CTCF footprint
  in raw.
- **fig2_supp_heatmap_premask:** the same as the main heatmap, but with the genome-wide pre-mask imputation (no mask,
  so no grey bins).
- **fig2_supp_corr_10kb_6x6:** genome-wide 10 kb bins (chr1–22 and X, blacklist bins removed), Pearson r of log2 CPM+1,
  single-cell raw vs bulk ChIP (Paired-Tag style, Zhu et al. 2021). Colour ±0.6.
