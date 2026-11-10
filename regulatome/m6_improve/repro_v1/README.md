# repro_v1: reproducibility of our own target lists (SPEC_REPRO.md; DESCRIPTIVE; no Perturb-seq read anywhere)
Declared 2026-11-10 ~01:00 local, committed BEFORE anything is run. Spec: ../SPEC_REPRO.md (sha256 d6508f1b48f8143d6273138ba16978d09380784f9ff518488f520631b81e43b2).

Infrastructure reused: the Gate 1 per-cell matrices (regulatome/gate1/cellmats/<TF>.raw.npz, cells x 159,965 active elements, raw read 5' ends; <TF>.cells.npz per-cell reads), the same cells as the TF1000cells qc20 table, and the split-half method of r5_core.py (random permutation of cells, first half = n // 2). The existing median split-half Jaccard 0.107 came from r5_core (core S lists); here the baseline list rule is v8.
Baseline list rule (v8 / B6 list e): per half, CPM = reads at element / total reads of that half's cells x 1e6; gene score = sum of CPM over the gene's ABC power-law links (m2_abc EnhancerPredictionsFull_thr0.02); list = top 1,000 genes with score > 0 (stable sort).

Objective: J_within - J_between, with BOTH components on every row.
- J_within = Jaccard of the TF's half-1 and half-2 lists (cells split randomly into halves, list rebuilt per half with the same rule).
- J_between = mean Jaccard of the TF's half-k list against the other TFs' half-k lists built on the same half index, averaged over k = 1, 2.
- Any method that raises both J_within and J_between is rejected, however large the J_within gain.
- Every method is stratified by current depth (cells-per-TF tertiles of the 79 TFs).
- Gains are reported in equivalent cells using the Run 1 curve (only if the Run 1 gate below accepts it).

Run 1 (depth curve): the 5 deepest TFs (RBBP4 181,396; NFYA 70,292; SOX4 64,652; XPA 50,752; RBBP7 39,619 cells), subsampled without replacement to 1k, 2k, 3.4k, 5k, 10k, 20k, 50k, 100k cells where available, 5 draws each (seed 20261110 + 1000 x TF index + draw, depth-specific), each subsample split into halves.

## Amendment 1 (declared before any Run 1 output exists)
1. J_between at matched depth is PRIMARY: at each depth and draw, pairwise Jaccard among the five subsampled deep TFs' half-lists built at that same depth (4 partners per TF, same half index, averaged over the two halves). J_between against the other 74 TFs at full depth (r5_core-style split, seed 20261029 + TF index) is SECONDARY and remains the reference for Run 2.
2. Slopes, not fitted values: per TF, OLS of J_within on log10(cells); slope with 95% CI from 1,000 bootstraps over draws (draws resampled within each depth), seed 20261110. Intercept spread across the five TFs reported separately (antibody-quality effect). Each slope compared with 0.206.
   Pre-declared gate: if the median within-TF slope's CI includes 0.206, or the slopes are >= 0.10, the equivalent-cells yardstick is accepted and Run 2 may use it. If the slopes are flat (CI includes 0 and below 0.10), stop before reading Run 2 and report.
3. Every per-depth, per-draw, per-half top list is written out (run1/lists/<TF>_N<cells>_d<draw>_h<half>.txt, gitignored), plus the full-depth half lists of the other TFs (run1/lists_full/).
Outputs: depth_curve.tsv, depth_slopes.tsv, GATE.txt (Run 1); methods_per_tf.tsv, methods_summary.tsv (Run 2).
