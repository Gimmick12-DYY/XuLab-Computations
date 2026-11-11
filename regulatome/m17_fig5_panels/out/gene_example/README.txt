Fig 5d gene example (m17_fig5_panels SPEC Amendment 2, commit eed0023). Illustration / selection on an already-read benchmark (diagnostic). HEK293T only.
Scripts: m17_p4.py (candidate table), make_gene_tf_panel.py (figures). Knockdown z = m15 double-centred reading (Arm 2); confirmed = in the m15 confident pairs (same top-K in both halves of the cells).
p4_candidates.tsv  genes with >= 3 confirmed TFs at K=2,000 or K=10,000 (81) plus PGAM1/HSPB1/TXNIP. bound = q005 peak in a canonical-ABC-linked element AND gene binding percentile >= 0.75.
                   Ranked by n_both_K10000 (bound and confirmed), then n_both_K2000, then summed |z|. housekeeping_flag by the SPEC prefix list. figure_top_candidate = first
                   unflagged gene whose confirmed TFs are not all E4F1/NRF1/DNMT1 (PRAME).
p4_gene_tf.tsv     every candidate gene x 79 TFs: linked peaks, best element, ABC, signalValue, binding percentile, z half1/half2/all, direction, confirmed K2000/K10000, strong phenotype.
p4_elements.tsv    every q005 peak in a linked element (strongest peak per element per TF). p4_links.tsv: canonical ABC links (ABC >= 0.02, self-promoter included).
fig5d_<gene>_impact.svg/png   circle area = |z| of the gene on that TF knockdown; green = gene down on knockdown, purple = gene up; outlined bold = confirmed; faded = bound, not confirmed;
                              hollow grey = TF without knockdown data. Each TF sits above the linked element holding its strongest peak (all TFs with a linked q005 peak are drawn).
fig5d_<gene>_binding.svg/png  circle area = binding percentile; colour strength = |z|.
TXNIP: MLX is drawn with * (not in the confirmed set). COSMIC CGC not available locally.
