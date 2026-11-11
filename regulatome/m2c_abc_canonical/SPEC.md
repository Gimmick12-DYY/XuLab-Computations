# m2c canonical ABC: spec (copy)
Source: design-thread spec, sha256 a468869d... (2026-11-10 ~18:29Z). This file holds the spec text as relayed to this machine, so its own sha256 differs.

Ask (Yuyang, 2026-11-10 18:29Z): "For the ABC part, I want you to be absolutely sure: we are applying the basic basic ABC modeling in literature,
and see if the ABC called target genes for each TF could overlap with the Perturb-seq TF results".

1. AUDIT.md first: canonical recipe vs m2_abc vs m2b_abc_k27ac per step (code version, candidates, activity, contact, normalisation, threshold,
   promoter handling, genes, TF-target calling).
2. Canonical build with the official EngreitzLab ABC-Enhancer-Gene-Prediction code (latest release in a separate env if it installs, else v1.1.2 in
   documented default mode; record version/commit). Candidates: MACS2 on merged OmniATAC (repo defaults), top 150,000 by reads, summit +/-250 bp,
   merged, hg38 blocklist removed, TSS +/-250 always included; do NOT reuse elements_v2. Activity sqrt(ATAC x H3K27ac) (ATAC omniatac_merged tagAlign;
   H3K27ac = m2b_abc_k27ac merged SRX18002015/16), repo default normalisation. Contact GSE216399 WT Hi-C 5 kb KR with repo power-law scaling/fallback
   (as used for Predictions_hic5kb); power-law-only as a sensitivity arm. ABC = A x C normalised over all candidates within 5 Mb; threshold 0.02;
   promoters and expressed-gene filtering as repo defaults (state them); same protein-coding TSS file as m2_abc.
3. Per-TF targets (fixed, no tuning): bound elements = lab MACS2 peaks /vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.<NAME>/ (read only) at
   q <= 0.05 whose summit overlaps a candidate; targets = genes with ABC >= 0.02 to >= 1 bound element. Report n peaks / bound elements / targets;
   < 20 targets = untestable.
4. Overlap with Perturb-seq (kd_response out_v2, z, >= 10% NTC detection, units averaged per TF, double-centred as in B6/m6; Arm 2 primary, Arm 1
   secondary). Universe = genes in ABC gene set and KD matrix. Responders = top 5% |z| for that TF. Per TF: Fisher OR + p, BH across TFs.
   Nulls (1,000 draws, seed 20261110): (a) same number of elements drawn from all candidates matched on ATAC decile and H3K27ac decile, same
   ABC >= 0.02 rule -> null OR median and empirical p; (b) ATAC-only control = top-N most accessible candidates (N = bound-element count) -> OR_ATAC;
   report OR_TF - OR_ATAC. Pooled: median OR, frac OR > 1, sign test, TF-minus-null sign test, with/without NRF1; plus B6 effect and Wspec for
   continuity. Score all testable TFs (fixed literature procedure, no tuning) but split the table dev vs held-out (split_v1) and note held-out TFs
   were used once before; no variant selection on these results.
5. Outputs per_tf.tsv, pooled.tsv, AUDIT.md, README.md; commit scripts + README + journal entry; push. Chain jobs with Slurm dependencies, modest resources.
