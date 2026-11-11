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

## Amendment 1 (2026-11-11): P4, a many-TF gene example and a locus figure; P2 reliability column
Written before any P4 number is computed.
- Pair definitions per (gene, TF), m15 dc matrices (raw reading beside): responsive = |z| >= 2 in half 1 and in half 2 with the same sign; bound = a q005 peak in an ABC-linked element of the gene (peak signalValue x ABC > 0). Beside it, bound_promoter = raw promoter reads above the TF 75th percentile.
- Per gene: n_responsive, n_bound, n_both, and the expected n_both if binding and response were independent across that gene TFs (n_bound x n_responsive / n_TFs). Top 20 genes by n_both, ties by summed |z| over the n_both TFs. Global count of genes with n_both >= 3, 5, 8.
- Locus figure for the top 3 genes (and NRF1/E4F1 gene PGAM1 as a declared fourth): gene body and ABC-linked elements along the chromosome; one ball per responsive TF; ball area proportional to |z| (all cells); colour by direction (gene down on knockdown vs up); filled = bound (placed above the linked element holding its strongest peak), hollow = responsive without a linked peak (placed in a side group). Bound but non-responsive TFs drawn as small grey dots at their element. PNG and SVG.
- P2 addition: per TF, the knockdown split-half reliability (m7 ceiling column) next to each model test Spearman, and sqrt(reliability) as the largest correlation any predictor could reach; read as a comparison only.

## Amendment 2 (2026-11-11): P4 replaced by the following (before any P4 number is computed)
Illustration and selection work on an already-read benchmark; labelled so in the README.
- Candidates: genes with >= 3 half-confirmed TFs among the m15 dc confident pairs at K = 2,000 or at K = 10,000, plus declared PGAM1, HSPB1, TXNIP.
- Per candidate gene and each of the 79 TFs: every q005 peak in a canonical-ABC-linked element (element coordinates, promoter/self vs distal, ABC score, strongest peak signalValue, peak q), the TF binding percentile for the gene (q005 peak score x ABC, over the 11,228-gene universe); for TFs with a knockdown: z in half 1, half 2, all cells (m15 dc), direction, confirmed at K = 2,000 and K = 10,000, strong-phenotype yes/no.
- bound = peak linked and percentile >= 0.75. Counts per gene: n_both (bound and confirmed), n_bound_only, n_confirmed_only, at K = 10,000 and K = 2,000. Rank by n_both at K = 10,000, then n_both at K = 2,000, then summed |z| over the n_both TFs.
- Housekeeping flag: symbol starts with RPL, RPS, MRPL, MRPS, EIF, EEF, ATP5, NDUF, COX, UQCR, SNRP, PSM, HSP, TUB, ACT, or is GAPDH, PGK1, ENO1, LDHA, PPIA, PGAM1. The "top-ranked candidate" for the figure is the first ranked gene that is not flagged and whose confirmed TFs are not all within E4F1/NRF1/DNMT1; if none, the first ranked gene. Top 10 reported.
- Figure (make_gene_tf_panel.py): locus with gene model (GENCODE v46 basic, transcript with the ABC TSS), promoter and linked distal elements, arcs width by ABC score. Encoding 1: circle area = |z| all cells; fill = direction (down on knockdown vs up), solid outlined = confirmed (K = 10,000), faded no outline = bound but not confirmed, small hollow grey = no knockdown data; confirmed-not-bound TFs listed as text. Encoding 2: area = binding percentile, colour intensity = |z| with direction hue. Genes: the top-ranked candidate, PGAM1, HSPB1, TXNIP (MLX marked "not in confirmed set"). Drawn for every TF bound (peak linked) at the gene, regardless of percentile; the percentile >= 0.75 rule applies to the counts only.
- Amendment 1 P4 definitions are withdrawn; its P2 reliability column stands.

## Amendment 3 (2026-11-11): Fig 5d version B, locus view + effect bars (replaces the circle drawing, which stays on disk marked superseded)
Illustration on an already-read benchmark. Genes: PGAM1, HSPB1. Script make_gene_tf_locus.py.
- Window: gene body plus all canonical ABC-linked elements, 10% padding; if linked elements lie > 200 kb from the TSS, the main view is TSS +/- 20 kb (widened to include the gene) and the distant elements are listed as a count.
- Rows (shared genomic x): ABC arcs element -> TSS, shade by ABC score; ATAC (OmniATAC merged tagAlign, the ABC input) and H3K27ac (SRX18002015/16 tagAlign, the ABC input), Tn5/read 5' ends per 50-bp bin, raw counts, grey; one HITAG row per TF with a confirmed (K = 10,000) effect on the gene and a linked q005 peak, plus confirmed TFs without a linked peak (HSPB1: DNMT1), at most 6; HITAG coverage = read 5' ends from the raw per-TF fragment files (/vast/.../bigwig/fragment/TF.<TF>.bed.gz) per 50-bp bin, CPM by that TF's total reads, 3-bin running mean, same y-scale across TF rows; q005 peak summits and own-motif sites (corrected m10 scans, p < 1e-4) as ticks; pooled CPM of all other panel TFs; gene model (GENCODE v46 basic, transcript with the ABC TSS). Promoter element and enhancers as faint full-height bands.
- Right of each TF row: the gene's z under that TF's knockdown (m15 double-centred, all cells) as a bar, half 1 and half 2 as dots.
- Bottom strip: every TF with a linked q005 peak, ordered by z; dot area = peak-score binding percentile; colour = z (diverging); confirmed outlined; names on confirmed TFs and the 3 largest |z| unconfirmed.
- A track that is essentially empty is reported as such, not rescaled.
