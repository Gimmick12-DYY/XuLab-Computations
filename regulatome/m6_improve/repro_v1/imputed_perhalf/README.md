# imputed_perhalf: leakage-free test of imputation for TF-specific reproducibility (repro_v1; DESCRIPTIVE; no Perturb-seq)
Declared 2026-11-10 ~01:10 local, committed BEFORE any job is submitted or any output exists.

(a) Rule amendment, stated openly as made AFTER seeing Run 2b (also ../README.md Amendment 3): "both components rise" is a flag for scrutiny, not an automatic rejection; the objective stays J_within - J_between. Imputation in Run 2b (Delta diff +0.287 vs v8@1000, all 79 TFs) is SUSPENDED pending this test, because its two half-lists came from one imputation model fitted on all cells.

(b) Design. Six TFs, two per depth tertile: low ZNF79, PCGF6; mid ZNF143, E4F1; high SOX4, XPA (all have unified/work/<tf>/mm inputs; no swap). Halves = the Run 2b splits (Gate 1 cellmats row order, permutation seed 20261029 + index of the TF in m1_binding/v2/tfs.txt, first half = n // 2), mapped to the unified bin-matrix barcodes. For each TF x half, everything is refit inside that half only: the bin x cell matrix is subset to that half's cells, then 00_ingest, 01_train (model incl. cell bank), 02_impute (sparsity_match threshold) and 02b open-chromatin mask (same ATAC union bed as the original run, data/HEK293T_ATAC_union3.bed.gz, which is TF- and cell-agnostic) run from scratch with the TF's own config and a new work dir (imputed_perhalf/work/<tf>_h<k>). No full-data object (model, ckpt, impute matrix) is read. Imputed bins -> elements via m1_binding/v2/counts/element_bin_map.npz exactly as gate1_prep.py; per half, the summed imputed signal is CPM-normalised and the list is the v8 rule (ABC power-law linked sum), top 1,000.

(c) J_between is matched among the 6 refit TFs on the same half index (5 partners, averaged over halves); partner-excluded version drops known partners per ../run2_partners.tsv (none of the 34 pairs is within these 6, so it equals the matched value; reported anyway). Paired baseline: v8 raw-binding lists for the same 6 TFs on the same halves, with J_within / J_between among the same 6.

(d) Pre-declared reading: imputation adds TF-specific reproducibility only if Delta diff >= 0.05 vs v8 with the CI excluding 0 AND J_between rises less than J_within. Delta diff = (J_within - J_between)_imputed - (J_within - J_between)_v8 on the same 6 TFs; CI = 10,000 bootstrap resamples over the 6 TFs (seed 20261110).

Pre-declared outcomes:
(A) Collapse: Delta diff < 0.05 and its CI includes 0, with a point estimate near zero (<= 0.05) -> leakage explains Run 2b's 0.287; imputation REJECTED for TF-specific reproducibility.
(B) Pass: Delta diff >= 0.05, CI excluding 0, and J_between rising less than J_within -> PASSES as declared; the accessibility-only smoother control is the next test.
(C) Indeterminate: point estimate >= 0.05 but CI touching 0, or any other pattern not in A or B -> six TFs cannot settle it; the only decisive test is the full 2 x 79 refit (~250-420 GPU-h), which is Yuyang's decision and is NOT queued.

(e) The accessibility-only (TF-agnostic) smoother control is deferred (needs a model change); next test only if (B).

Jobs: 12 GPU tasks (perhalf.sbatch, l40-gpu/a100-gpu, gpu_access, 1 GPU, 4 CPU, 16 GB, 8 h, --nice=500 so the AF3 5-seed arrays get the account's GPU slots first), then a CPU summary afterok (perhalf_summary.py). Outputs (DESCRIPTIVE): perhalf_per_tf.tsv, perhalf_summary.tsv, OUTCOME.txt.

## Amendment (2026-11-10, before any output)
If outcome B (pass) lands, the accessibility-only neighbour-graph control is REQUIRED, not optional, and must be declared before its result is read. Reason: the motif thread found motif z scales with cell count more strongly on imputed signal (Spearman 0.60) than raw (0.43/0.37), which is consistent with imputation amplifying depth convergence.
