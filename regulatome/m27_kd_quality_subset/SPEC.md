# m27_kd_quality_subset: does agreement improve on a subset of TFs with good knockdowns? (spec, 2026-11-13)

Label: diagnostic. HEK293T. Tables only. Committed before any agreement number is computed.

1. Knockdown-quality table per panel TF (X-Atlas/Orion HEK293T, existing outputs only): cells per guide pair and in total (m4 out_v2 arm2 guide_qc, n_cells_all per unit), knockdown % of the target vs GEM-matched NTC (phase1 panel_tf_xatlas_status knockdown_pct), NTC expression of the target non-zero (phase1 ntc_frac_cells_detected > 0), split-half reliability (m7 tf_folds ceiling, Spearman-Brown), reproducible responders (m19 P4, all directions), and the authors' strong-phenotype label: NOT released by the authors; our rebuild (m16) is shown and labelled as a rebuild.
2. Filters on knockdown quality only, fixed now, never on agreement with binding:
   F1 = >= 25 cells, >= 30% knockdown, target detected in NTC cells (the authors' criteria without the classifier, which is not released);
   F2 = F1 plus >= 50 cells in every guide pair (our pipeline minimum);
   F3 = F2 plus split-half reliability >= 0.2. A depth-matched null for knockdown split-half reliability does not exist as built; the fixed cutoff 0.2 (used since m7 for "ceiling >= 0.2") is used instead, and this is stated in the report.
   For each filter: the TFs passing, and for each failing TF the first criterion it fails.
3. On each subset (and on all TFs with a knockdown, for reference), the m22 agreement on the top-5,000 edge table (commit 9b8d8c1), exactly as in m22 Amendments 1-2: high-confidence pairs at K = 2,000 (both halves top K, same sign; m15 double-centred halves): fraction bound for the own TF, a swapped TF of the same depth tertile (mean over that tertile's other TFs) and ATAC/expression/CpG-matched non-responder genes; pooled Mantel-Haenszel OR over TFs (own pairs vs matched genes) with pair-bootstrap and TF-clustered intervals (2,000); per-TF Spearman of score vs |z| (all cells) with gene-bootstrap interval, ATAC-only and swapped-TF controls, and the number of TFs whose interval is above 0 and above both controls; whole-matrix Spearman of score vs |z| (raw and within-TF percentile) with the swapped-TF matrix. Each result also without NRF1 and E4F1.
4. Report plainly, nulls included; how many TFs each filter leaves; whether agreement improves on the subsets.
