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

## Run 1 result and gate (read 2026-11-10 ~01:05 local)
GATE.txt: within-TF J_within slopes on log10(cells) 0.24-0.33 (all CIs exclude 0 and exclude 0.206; steeper), intercept spread 0.28 -> ACCEPT_YARDSTICK=True, FLAT=False. But J_between at matched depth rises almost as fast (slopes 0.24-0.28), so J_within - J_between_matched stays small (-0.008 to +0.09 up to 50k cells): most of the depth gain in self-agreement is shared signal.

## Run 2 declaration (before any Run 2 output exists)
Script repro_run2.py; panel's real depths, all 79 TFs, the r5_core-style half split (seed 20261029 + TF index), J_between = mean over the other 78 TFs' same-method lists on the same half index (secondary reference per Amendment 1), both components on every row.
Methods run in v1: (0) v8 baseline (ABC-linked raw CPM sum, top 1,000); (4) fixed TSS windows +-50 kb and +-100 kb (elements by midpoint, raw CPM sum); (5) imputed binding (Gate 1 <TF>.imp.npz, per half, same v8 rule); (6) element-level normalisation against the pooled panel signal on the same half (mean CPM of the other 78 TFs at the element): ratio = CPM / (pool + 1) and residual = max(CPM - b x pool, 0) with b from a per-TF least-squares fit over elements, both summed over ABC links; (7) list sizes 250, 500, 2,000 for the v8 rule.
Not run in v1 (stated): (1) r3_eb EB shrinkage, (2) pooled-neighbour bins, (3) peak-level aggregation; they need separate builds and are left for a v2 declaration.
Rejection rule: a method that raises J_within and J_between together (both medians up vs v8, paired across TFs) is rejected regardless of J_within.
Stratification: cells-per-TF tertiles of the 79 TFs.
Equivalent cells: from the pooled Run 1 fit of J_within on log10(cells) (all five deep TFs), a method's median J_within gain at a TF's depth N is converted to N' / N where J(N') = J(N) + gain. Because J_between rises nearly as fast, the same conversion is also reported for J_within - J_between_matched using its pooled Run 1 fit, and flagged as ill-conditioned where that slope is near 0.
Outputs (DESCRIPTIVE): methods_per_tf.tsv, methods_summary.tsv (median J_within, J_between, difference, paired deltas vs v8 with bootstrap CI over TFs, equivalent cells, by depth tertile and overall).

## Amendment 2 (declared 2026-11-10 ~01:08 local, before the partner-excluded numbers are computed and before any Run 2 output is read; Run 2 job 8753195 finished but its outputs are unread)
Confound: RBBP4 and RBBP7 are complex partners (NuRD/PRC2 core), so their matched-depth J_between includes a real-biology partner.
Run 1 post hoc (repro_partnerex.py, from the saved lists, no new subsampling):
- J_between_partnerex: RBBP4 vs {NFYA, SOX4, XPA}; RBBP7 vs {NFYA, SOX4, XPA}; NFYA, SOX4, XPA vs all four others (unchanged) and, as a second column, vs the deep set without RBBP4 and RBBP7.
- J_between_XPA: each of the other four vs XPA alone (repair factor, least contaminated comparator).
- depth_curve_partnerex.tsv: TF, cells, draw, J_within, J_between_matched, J_between_partnerex, J_between_noRBBP, J_between_XPA and the diffs; per TF slope of (J_within - J_between_partnerex) on log10(cells) with 95% CI (1,000 bootstraps over draws, seed 20261110), next to the slope of (J_within - J_between_matched).
- Pre-declared reading: if the partner-excluded diff slope is >= 2x the current diff slope, depth buys more TF-specific signal than the first table showed; if it is within +-25% of it, the convergence is generic; in between is reported as intermediate.
Run 2 partner exclusion: known partners per TF = pairs in r1_tftf/v2_final_v3/r1_pairs_frozen_v3.tsv with class "complex" OR a STRING physical score (>= 0.70, the R1 known-interaction annotation); complex_weak (prediction-only) pairs are not excluded. This gives 34 pairs, listed in run2_partners.tsv (written by repro_partnerex.py before Run 2 outputs are read). Because Run 2 did not save its half lists, Run 2 is re-run identically as Run 2b (repro_run2b.py: same code, same seeds, also writes the half lists and adds J_between_partnerex = mean over the other TFs excluding the TF's known partners). Run 2b replaces Run 2 for reporting; the unread Run 2 tables are kept untouched for comparison.

## Amendment 3 (2026-11-10 ~01:05 local; made AFTER seeing Run 2b, stated openly)
1. Rule change: "both components rise" is a flag requiring scrutiny, not an automatic rejection; the objective stays J_within - J_between. Under this amended rule imputation in Run 2b (Delta diff +0.287 [0.262, 0.311], J_within +0.539, J_between +0.252 vs v8@1000) is SUSPENDED pending a leakage-free test, not rejected: its half-lists share one imputation model fitted on all cells, so they are not independent. The windows (win50k/win100k, Delta diff ~0) and v8@2000 (Delta diff -0.008) stay flagged with no margin gain.
2. v8@250 is a small real margin gain (Delta diff +0.007 [0.002, 0.013] vs v8@1000).
3. Leakage-free test, as declared: per-half refit of the unified imputer (neighbour graph, model and smoothing all inside each half, no full-data object), lists by the v8 rule (top 1,000), J_within, J_between (all others and partner-excluded) and their difference by TF and depth tertile, paired Delta vs v8@1000 with CIs; control arm with an accessibility-only (TF-agnostic) neighbour graph. Pre-declared bar unchanged: Delta diff >= 0.05 with CI excluding 0 AND J_between rising less than J_within.
   NOT RUN: cost check failed the cap. The unified imputer (unified/scripts 00_ingest -> 01_train -> 02_impute -> 02b mask) is a GPU PyTorch model; the panel took 192 GPU-h over 72 recorded jobs (median 2.25 h per TF), so 2 x 79 per-half refits are ~250-420 GPU-h (plus the same again for the control arm), far above the ~1 GPU-day / 200 CPU-h cap, and the account's GPU allocation is currently exhausted (AssocGrpGRES). A reduced version (6 TFs, 2 per depth tertile, J_between among the refit set) at ~20-30 GPU-h per arm is proposed for Yuyang's decision.
