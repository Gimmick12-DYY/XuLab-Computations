# m17_fig5_panels: three checks for Figure 5 panels b and d (spec, 2026-11-11)

Label for every table and figure: diagnostic, benchmark already read. HEK293T data only (our HITAG binding, X-Atlas/Orion HEK293T Perturb-seq). Lab q005 peaks and raw binding; imputed binding only where m7 already scored it (P2). Written before any number below is computed.

## Shared inputs
- Knockdown response: Arm 2, raw per-TF z (m9_abc_diagnosis/work/responses_arm2.npz, key r1; the same matrix as m16 part B), genes >= 10% NTC detection, the 11,228-gene universe. Each TF own gene (HP1: CBX1/3/5; ZUFSP: ZUP1) set to NA for that TF.
- Strong-phenotype TFs: the 10 of m16 (CTCF, DNMT1, E4F1, MEF2A, NRF1, PA2G4, RBBP7, RBPJ, SAFB, SETDB1).
- Matched genes: promoter ATAC decile x CpG island, as in m16.

## P1. TF x TF response matrix (panel b, simple model)
- Target set of TF i: (A) top 1,000 genes by q005 peak signalValue x ABC (m11 wide_score_q0.05; genes with score > 0 only, fewer if fewer); (B) top 1,000 genes by raw promoter reads (m10 counts, +-500 bp of TSS, CPM). A TF with an empty set is NA in that row.
- Entry (i, j), knockdown of TF j: D_ij = mean |z_j| over set i minus mean |z_j| over matched genes (mean of 100 matched draws); OR_ij = Haldane OR of set i among the top-5% |z_j| genes.
- Two standardisations, both shown, chosen here: (1) raw difference D_ij; (2) column z-score: D_ij standardised across rows i within each column j (which target set responds most to knockdown j). The row z-score (D standardised within row i across columns) is tabulated, not drawn.
- Matrices: (S) the 10 strong-phenotype TFs; (All) every TF with a knockdown and a binding set.
- Statistics per matrix and standardisation: diagonal mean minus off-diagonal mean, with a permutation p (10,000 shuffles of row labels against columns, square part only); per TF, the rank of its own knockdown among columns for its target set (1 = top); the count of TFs whose own knockdown is the top column. Also the same with NRF1 and E4F1 removed.
- Heatmaps: PNG and SVG, rows = target sets, columns = knockdowns, same TF order on both axes, diverging colour centred at 0, no clustering.

## P2. ML half of panel b (post hoc subset of an already scored test)
- From m7_step2/per_tf_test.tsv only. No refit, no new scoring of test chromosomes.
- Rows: the 10 strong-phenotype TFs; canonical links (raw_hicfixed and imp_hicfixed); Arm 2; by-gene split primary, by-TF split beside; train-centred target primary, raw centring beside; elastic net, LightGBM, network; knockdown-only and both-target modes.
- Per TF: test Spearman (real), baseline-only Spearman, gain = real minus baseline-only, and the unfitted references (100 kb binding sum, v8 list membership).
- Pooled over the 10 (fewer where a TF is absent from the table): mean real, mean gain with a 2,000-draw TF bootstrap 95% interval (seed 20261111), mean of each reference, model minus reference with the same bootstrap.
- Label: post hoc subset of an already scored test.

## P3. Gene example (panel d)
- From m15 confident-pair tables (dc reading primary, raw reading beside) at K = 2,000 and K = 10,000.
- Genes with >= 2 distinct TFs among confirmed pairs. Per gene and TF: z in each half and direction, binding percentile at the gene (raw promoter reads; q005 peak score), promoter peak yes/no, own motif at promoter yes/no, shared_by = number of 79 TFs with a q005 peak linked to the gene, and COSMIC Cancer Gene Census membership only if a local copy exists (no download; otherwise reported as not available).
- Counts: genes with 2, 3, 4+ confirmed TFs; of those, genes where >= 2 confirmed TFs have binding above the 75th percentile (raw promoter reads; separately peak score).
- Top 20 candidates, ranked by the number of confirmed TFs with raw-promoter binding above the 75th percentile, then by summed |z| (all cells).
- Declared genes, looked up whatever they show: MYC, EZH2, CCNE1, TXNIP, LDLR, CDKN1A. For each, every TF ranked by |z| (all cells) with half z, binding percentiles and confirmed-pair status.
- Report the numbers with their comparison; untested causes are written as unexplained.
