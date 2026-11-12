# m21_tf_tiers: one per-TF evidence table for the 79 panel TFs (spec, 2026-11-12)

Label: diagnostic. HEK293T. Built only from existing result files; no new fits and no new benchmark reads. Every column cites its source file and the commit (or, for untracked files, the sha256). Written before the table is built.

Columns: cells, fragments, depth tertile; q005 peaks; genes with a linked q005 peak; noise-floor d and d >= 0.04 (m6 repro_v1 per_tf_floor_test.tsv; its "clears" column, 38/79, also shown); own motif in HOCOMOCO v11 (m9 motif set); held-out own-motif rank among 472 HOMER known motifs on the ATAC-masked raw profile (complex_screen/pair_motif_composite/per_tf_raw/own_motif_check.tsv, 3bfdaa3, column val_known_rank); own-motif read enrichment over the pooled other 78 TFs with interval (m10 G1_signal_fraction.tsv, 592fef2, motif_<TF> "all" row); rank of the own column in the m20 promoter-read motif check (m20 part1d, 8dc8af0; 6 TFs); bulk ChIP set and cell line, within-open-chromatin AUROC and its rank among 79 (m20 part2, 8dc8af0); strong phenotype (m16, b527f84); split-half reliability (m7 tf_folds.tsv ceiling); reproducible responders (m19 P4, all directions, 159a12a); OR of our list vs responders with interval and the swapped-TF OR for B2 x P1 and B1 x P4 (all directions; m19 159a12a).

Tier rule (fixed now):
- Line 1, noise floor: d >= 0.04.
- Line 2, motif: held-out own-motif rank <= 10, OR the G1 lower interval bound of enrichment over the pooled other 78 TFs > 1.
- Line 3, bulk ChIP: own TF ranks in the top 10 of 79 by within-open-chromatin AUROC (best of its ChIP sets if more than one).
- Tier 1 = at least two lines met. Tier 2 = exactly one. Tier 3 = none met; split into "tested, none met" and "not testable" (only line 1 testable and not met, i.e. no motif and no ChIP).
- Function supported (separate flag): in B2 x P1 or B1 x P4 (all directions), the OR lower bound > 1 AND OR > the swapped-TF OR.
Missing columns are counted per column. The rule is not changed after the counts are seen.
