m17_fig5_panels (SPEC.md, commit 1c2ac06). Label: diagnostic, benchmark already read. HEK293T data only. Knockdown response = Arm 2 raw per-TF z (as m16).
P1 (m17_p1.py, job 8923665): TF x TF matrix. Row = target set of TF i (A: top 1,000 by q005 peak score x ABC; B: top 1,000 by raw promoter reads),
   column = knockdown of TF j. D = mean |z_j| in set minus matched genes (ATAC decile x CpG). Standardisations: raw_difference, column_z (drawn), row_z, log2_OR (tabulated).
   p1_entries.tsv (every entry), p1_summary.tsv (diagonal vs off-diagonal, permutation p over row labels, own-best counts; with and without NRF1/E4F1),
   p1_ownrank.tsv (rank of own knockdown within the row), p1_heatmap_<binding>_<strong10|all>_<raw_difference|column_z>.png/.svg. SETDB1 has no q005-linked gene: row NA in A.
P2 (m17_p2.py): post hoc subset of an already scored test. m7_step2 per_tf_test.tsv restricted to the 10 strong-phenotype TFs; no refit, no new scoring.
   p2_per_tf.tsv (real = test Spearman, base = baseline-only, gain = real - base, references), p2_pooled.tsv (TF-bootstrap 95% intervals), p2_side_by_side_raw_bygene_train.tsv.
   Note the references (100 kb sum, v8 membership) are unsigned as in m7, so a TF whose bound genes go down on knockdown has a negative reference Spearman.
P3 (m17_p3.py): gene example. p3_counts.tsv (genes with 2, 3, 4+ half-confirmed TFs), p3_multi_tf_genes_<dc|raw>_K<2000|10000>.tsv, p3_top20.tsv (dc, K=2000),
   p3_declared_genes.tsv (MYC, EZH2, CCNE1, TXNIP, LDLR, CDKN1A: every TF ranked by |z|). A peak-score percentile near 0.48 means no q005 peak (tied zeros). COSMIC CGC: no local copy, not downloaded.
