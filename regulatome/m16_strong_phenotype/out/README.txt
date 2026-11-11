m16_strong_phenotype (SPEC.md, commit 89f5eef). Label: diagnostic, benchmark already read. HEK293T Perturb-seq (X-Atlas/Orion) only.

Part A - authors strong-phenotype filter, reproduced (m16_classifier.py, job 8922771)
  Per knocked-down panel gene: L1 logistic regression (C=0.1) on 2,000 HVG (log1p CP10k), knockdown cells vs GEM-matched NTC cells, 70/30 split,
  test AUC vs 200 label permutations, BH q. Strong phenotype = >=25 cells AND >=30% knockdown AND q<0.05 AND target non-zero in NTC training cells.
  Caveats: the authors per-perturbation classifier results are not released, so this is our reproduction; HVG selection approximated (seurat_v3-like);
  with 200 permutations the smallest reachable q is about 0.032 (all passing TFs sit there).
  classifier_per_gene.tsv      per knocked-down gene: n_cells, AUC, permutation p, q, criteria c1-c4, strong_phenotype
  strong_phenotype_per_tf.tsv  per panel TF (79): status, kd ceiling, m10 testable, depth tertile
  criteria_counts.tsv          TFs passing each criterion
  ntc_negative_control.tsv     20 NTC-vs-NTC sets through the same classifier (1/20 p<0.05, 0/20 q<0.05)

Part B - simple model vs Perturb-seq, strong vs not-strong (m16_compare.py, job 8923056)
  Per TF: top-1000 genes by q005 peak score x ABC (or promoter-only) vs top-5% |z| responders (Arm 2, double-centred z); Haldane OR,
  gene-bootstrap 95% interval; controls matched genes (promoter-ATAC decile x CpG), swapped TF. TFs with an empty list (no q005 peak linked
  to any universe gene, e.g. SETDB1) are NA and excluded from pooled numbers; empty swapped lists are NA.
  simple_per_tf.tsv, simple_pooled.tsv (median OR, median OR-matched with TF-bootstrap interval, Mantel-Haenszel OR), simple_deciles.tsv, simple_summary.png/.svg
