m15_gene_to_tf (SPEC.md, commit a5f4c07). Label: diagnostic / development, benchmark already read. HEK293T Perturb-seq only.
Gene-to-TF direction, rank based: every (gene, TF) entry of the knockdown matrix ranked by |z|; the top K (100/500/2000/10000) entries in
half 1 that are confirmed in half 2 (and vice versa) are "confident pairs" (union). Also per_gene_same_top_TF: genes whose strongest TF is the same in both halves.
For each confident pair, the binding percentile of that gene among all genes for that TF under (a) q005 peak score, (b) raw promoter reads,
(c) promoter q005 peak score, (d) imputed promoter. 0.5 = no relationship. Nulls: permuted TF labels, matched genes, swapped TF.
Two readings: dc_* (double-centred z) and raw_* (raw per-TF z).
  *_stats.tsv            survival of top-K pairs across halves; mean binding percentile with interval and nulls, leave-NRF1-out, leave-top-3-TFs-out, ceiling>=0.2 subset
  *_composition.tsv      pairs per TF at each K
  *_pairs_K*.tsv         the confident pairs
  *_signed.tsv           fraction of confirmed pairs where the gene goes down on knockdown, per TF
  *_reverse_lookup.tsv   per gene, TFs ranked by knockdown effect with binding percentiles
  *_statistic_vs_K.png   mean percentile vs K per binding version
