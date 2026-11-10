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

Run 2b note (2026-11-10): Residual vs pool is −.031 overall but +.016 in the LOW-depth tertile — the one method so far with the correct stratification (helps shallow TFs), though inside Amendment 3's null band. Watch whether Run 2c (peak-level, pooled-neighbour) shows the same low-tertile pattern; two independent methods helping only shallow TFs would be a direction.

## Amendment 4 (2026-11-10, declared before Run 2c is run): Run 2c, methods 3 and 2
Methods 6 (pool ratio/residual) and 7 (list sizes) already ran in Run 2b and are not rerun. No equivalent-cells conversion (retracted,
SPEC_REPRO Amendment 3). Same half splits and seeds as Run 2b (cells permuted with seed 20261029 + TF index in tfs.txt order, first half
= n // 2, cell order = gate1 cellmats). List rule otherwise v8: ABC power-law links (m2_abc thr0.02), gene score = sum over links, top 1,000
genes with score > 0. Baseline = v8@1000, recomputed in the same script (must reproduce Run 2b exactly; asserted).

Method 3, peak-level aggregation. Peaks are called **per half**, from that half's reads only, so nothing is shared between halves
(the full-data peak sets in tf_complex/work/peaks_tmp are not used: they are called on all cells and would leak across halves).
- Reads: /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.<TF>.bed.gz (read-only), filtered to the half's barcodes
  (gate1 cellmats <TF>.cells.npz barcode order).
- Caller: MACS3 (snapatac2 env) callpeak -f BED -g hs --nomodel --shift -75 --extsize 150 --keep-dup all -p 0.01; written to
  run2c/peaks/<TF>_h<k>_peaks.narrowPeak (outputs, not committed).
- peak_fe@1000: element score = max fold enrichment (narrowPeak col 7) of the half's peaks overlapping the element (any bp), 0 if none.
- peak_mask@1000: element score = the half's raw CPM (as v8) where at least one of the half's peaks overlaps the element, else 0.
Method 2, rank on pooled-neighbour bins: element score = its CPM plus the CPM of every active element on the same chromosome whose
midpoint lies within ±k kb of its midpoint (k = 1, 2; i.e. its ±k neighbouring 1-kb bins): nbr_k1@1000, nbr_k2@1000.

Scoring (as Run 2b): per TF x method, J_within, J_between (mean over the other 78 TFs' same-method same-half lists, both halves),
J_between_partnerex (excluding run2_partners.tsv partners), diff and diff_partnerex; by depth tertile (cells-per-TF tertiles, as Run 2b)
and overall; paired Δ vs v8@1000 (mean over TFs, 95% bootstrap CI over TFs, 10,000 resamples, seed 20261110). The low-tertile Δdiff is
reported explicitly for every method.

Null band (depth yardstick, no conversion): Run 1 within-TF diff slope .004–.058 per decade and J_within slope .24–.33 per decade, so a
depth increase that raises J_within by ΔJ_within buys at most ΔJ_within × (0.058 / 0.24) = 0.2417·ΔJ_within in diff (generous end).
A method counts as **beating depth** only if its mean Δdiff > 0.2417·max(ΔJ_within, 0) AND the Δdiff 95% CI excludes 0. Otherwise it
"reproduces the depth curve" = null. Reported per tertile and overall, for diff and diff_partnerex.
Outputs (DESCRIPTIVE): run2c_per_tf.tsv, run2c_summary.tsv, run2c/lists/. Run as one CPU dependency chain (peak array -> lists/scoring).

## Amendment 5 (2026-11-10, declared before the noise-floor test is run; spec SPEC_REPRO.md Amendment 5)
The depth band applies only to methods that raise J_within; for methods that lower it, the comparator is the noise floor. Peak-level
aggregation is not adopted; pooled-neighbour bins are null.

Noise-floor test (script noisefloor.py). Uninformative lists at matched depth and matched list length (1,000), same half splits and seeds
as Run 2b/2c (cells permuted with seed 20261029 + TF index, first half = n // 2), 20 draws of each kind. Draw RNG seed =
20261110 + 100000 x kind + 1000 x draw + 2 x TF index + half (kind a = 1, b = 2).
- (a) perm: within each TF-half, shuffle the per-element CPM vector across the 159,965 active elements, then the v8 rule (sum over ABC
  links, top 1,000 genes with score > 0).
- (b) count-matched random genes: for each TF-half, take that half's v8 list; for each gene record its number of ABC-linked elements; draw
  the same number of genes at random without replacement from all ABC-linked genes so that the element-count histogram is matched exactly
  per stratum (strata: counts 1..15 exact, then 16-20, 21-30, 31-50, 51+; if a stratum runs out, the deficit is drawn from the nearest
  stratum and logged).
Per draw: J_within, J_between (mean over the other 78 TFs' same-kind same-draw same-half lists, both halves) and difference per TF; per
depth tertile (Run 2b tertiles) the mean and median over TFs. Reported over the draws: mean and 95% interval (2.5th-97.5th percentile
of the per-draw values).

### Amendment 5 revision (2026-11-10, before the noise floor was read; a 20-draw job 8796629 was cancelled before writing any output)
- The noise floor is DESCRIPTIVE calibration only, not a gate. 100 draws per floor kind (draw RNG seeds as above, draw = 0..99).
  Reason: with panel-shared, TF-agnostic ABC links both floors sit at about 0 by construction (after permutation, scores track the
  linked-element count, which is the same for every TF and both halves), so a floor below +.003 cannot validate anything.
- Everything is reported on the ABSOLUTE difference scale (J_within - J_between), not as deltas.
- The gate is magnitude, declared after Run 2c and before the floor is read: the low-depth tertile must reach an absolute difference of
  +.04 (a third of the high tertile's +.127) for the shallow-TF avenue to stay open. The best observed is resid_pool at about +.003
  absolute (+.016 as a delta vs v8@1000, low-tertile baseline -.013).

**Closing statement (shallow-TF avenue):** CLOSED on magnitude, whatever the floor returns. No method in Run 2b or 2c brings the low-depth
tertile within an order of magnitude of +.04 absolute (best: resid_pool ~+.003); the four low-tertile "gains" (resid_pool, ratio_pool,
peak_fe, peak_mask) all raise the difference by lowering both components toward the uninformative level. Outputs (DESCRIPTIVE):
noisefloor_per_draw.tsv, noisefloor_summary.tsv.

## Amendment 7, declared before the per-TF read (2026-11-10)
Per-TF test of the observed v8@1000 list against that TF's own uninformative-list draws (noise floor, job 8796692). Route: the existing
noisefloor_per_draw.tsv already holds per-TF draws (79 TFs x 100 draws x 2 kinds = 15,800 rows), so nothing is rerun.
- d = J_within - J_between of the TF's observed v8@1000 lists (run2c_per_tf.tsv, method v8@1000, identical to Run 2b), J_between = mean
  over the other 78 TFs' same-half lists (as in the summary).
- p_perm = (1 + #perm draws with diff >= d) / 101; p_cm likewise against the 100 count-matched draws.
- A TF clears if p_perm <= .05 AND p_cm <= .05.
- BH-adjusted q over the 79 TFs for each null (q_perm, q_cm). With 100 draws the minimum p is 1/101 = .0099.
- in_reproducible10 = TF in {RBBP4, NFYA, SOX4, XPA, ZNF507, RBBP7, NME2, E2F3, MBD3, DDIT3}; overlap reported as a 2x2 with a two-sided
  Fisher exact p; counts by depth tertile (Run 2b tertiles).
- Without NRF1: the NRF1 row is dropped (its draws and d are unchanged), BH and the Fisher test are recomputed over 78 TFs.
- Output per_tf_floor_test.tsv: tf, cells, tertile, Jw, Jb, d, perm_mean, perm_q95, p_perm, cm_mean, cm_q95, p_cm, q_perm, q_cm, clears,
  in_reproducible10, plus the Amendment 7b columns.

## Amendment 7b, declared before the per-TF read (2026-11-10)
- The exact-draw p stays the PRIMARY declared test (clears = p_perm <= .05 AND p_cm <= .05).
- Its BH correction is reported, but it is known in advance to be draw-limited. With 100 draws the minimum p is .0099, so BH can only
  pass if about 25 or more TFs sit at the floor; 10 real TFs would give q = .078. A failure of the exact-p BH is therefore NOT evidence of
  no signal.
- SECONDARY test: for each TF and each null, z = (d - mean of its own 100 draws) / sd of those draws (ddof = 1), with a one-sided normal
  p. BH over the 79 TFs on the z-based p. A TF is called at FDR .05 when z-BH q <= .05 for BOTH nulls (called_fdr05).
- Normality check: Shapiro-Wilk p of each TF's draws per null, and the share of TFs with p < .01.
- Any TF whose draw SD < 1e-4 (either null) is flagged (sd_flag), listed, and not called on the z test.
- Added columns: z_perm, pz_perm, qz_perm, z_cm, pz_cm, qz_cm, shapiro_perm, shapiro_cm, sd_flag, called_fdr05.

## Results (2026-11-10, noise floor job 8796692 and per-TF floor test; DESCRIPTIVE)
Absolute difference d = J_within - J_between, mean over the TFs in each tertile (floors: mean over 100 draws [95% interval over draws]).

| tertile | floor perm | floor count-matched | v8@1000 | resid_pool | ratio_pool | peak_fe | peak_mask | nbr_k1 | nbr_k2 |
|---|---|---|---|---|---|---|---|---|---|
| low | +.0023 [.0005, .0042] | -.0037 [-.0053, -.0023] | -.0133 | +.0029 | -.0065 | -.0040 | -.0042 | -.0192 | -.0214 |
| mid | +.0003 [-.0022, .0027] | +.0009 [-.0009, .0026] | +.0054 | +.0106 | +.0038 | +.0171 | +.0068 | +.0057 | +.0072 |
| high | +.0117 [.0094, .0142] | +.0071 [.0054, .0090] | +.1574 | +.0462 | +.1017 | +.0631 | +.1093 | +.1476 | +.1599 |
| all | +.0048 [.0035, .0061] | +.0015 [.0004, .0025] | | | | | | | |

(1) Low-tertile v8 (-.0133) is BELOW both floors (+.0023 perm, -.0037 count-matched). For the shallowest third, our lists agree with
other TFs' lists more than an uninformative list does.
(2) High-tertile v8 d = +.157, with an interval clear of the uninformative-list null (floor about +.01). The floors sit near 0 partly by
construction (Amendment 6), so the evidence is d > 0 with its interval, not a ratio.

### Per-TF floor test (Amendments 7, 7b; floor_test.py; per_tf_floor_test.tsv)
Route: the existing noisefloor_per_draw.tsv holds per-TF draws (79 x 100 x 2); nothing was rerun.
- PRIMARY (exact-draw p, p_perm <= .05 AND p_cm <= .05): 38 of 79 TFs clear. By tertile: high 26/27, mid 11/26, low 1/26 (ZNF143).
  All 10 reproducible TFs clear.
  PA2G4 is the one high-tertile TF that does not clear (Jw .166, Jb .287).
  Exact-p BH (minimum p 1/101 = .0099; draw-limited, Amendment 7b): 36 TFs have q <= .05 under both nulls (minimum q .023).
- SECONDARY (z vs own draws, BH over 79, both nulls): 36 called at FDR .05 (high 26, mid 9, low 1); POGK and TBX3 clear the primary test
  but are not called. Shapiro-Wilk p < .01 for 2.5% of TFs under each null; no TF has a draw SD < 1e-4.
- Cleared, sorted by d: RBBP4 .489, SOX4 .383, NFYA .366, XPA .329, RBBP7 .270, ZNF507 .259, NME2 .236, E2F3 .226, MBD3 .209, MLX .206,
  E4F1 .188, DDIT3 .169, ZNF703 .163, GLYR1 .127, CTCF .123, SAFB .095, MTF2 .086, DR1 .057, ZNF512B .045, HOXB9 .042, ZBTB7A .035,
  HOXA10 .035, RBPJ .031, HOXB6 .031, ZNF143 .024, MAZ .023, FLYWCH1 .023, ZNF768 .022, HES1 .022, ZNF644 .022, HOXC9 .022, WIZ .021,
  FOXC1 .018, ZIC2 .017, ZNF746 .014, SMARCA1 .012, POGK .011, TBX3 .008.
- Without NRF1 (it does not clear): unchanged, 38 clear / 36 called.
- The 10 were selected on split-half Jaccard, which is a term in d, so their clearing is a sanity check and not independent evidence.
- Per-TF claims tier: d >= .04, the magnitude declared in Amendment 6 before any per-TF number existed (declared there as a low-tertile
  gate, applied here per TF). TFs above the null with d < .04 are reported as above-null with a small effect and are not used for per-TF
  statements. 20 of the 38 have d >= .04 (RBBP4 through HOXB9 .042), all 20 in the high tertile (high 20, mid 0, low 0); no TF outside
  the 38 has d >= .04.
  Nearest exclusion: ZBTB7A, d = .035 (.005 below the line). The line is not moved to include it, the same treatment as the
  EPAS1-NEUROD4 ipTM .390 borderline in the structure screen. (HOXA10 is next, d = .0352 against ZBTB7A's .0354.)
  d for all 79 TFs is published in per_tf_floor_test.tsv, so readers can apply their own cut.
  Observed range over which specificity appears: the 20 TFs at d >= .04 have >= 5,185 cells (ZNF512B); the high/mid tertile
  boundary lies between 4,896 and 5,138 cells. These are OBSERVED in this panel, not a validated cutoff, and unlike the retracted ~22k-cells
  figure they are not extrapolated from a fit.
- Clearing is almost a pure depth ordering (high 26/27, mid 11/26, low 1/26), so this is a depth boundary with a per-TF test behind it,
  not a biological property of the named proteins.
- The Perturb-seq diagonal had a single contributor, NRF1, whose knockdown was the cleanest in the panel. NRF1 lies in the shallowest
  third of the panel by cell count (1,416 cells) and its list does not clear the specificity null (Jw .144, Jb .143, d .0006). The
  benchmark's only positive therefore came from a TF outside the set our own data can support, which is the simplest account of why it
  did not generalise. Whether NRF1's slightly above-median Jw (low-tertile median .096) reflects its targets or its depth cannot be
  separated here.
- Column legend, per_tf_floor_test.tsv: p_perm, p_cm = exact draw p, (1 + #draws >= d) / 101; a value of .0099 means "at the 100-draw
  resolution floor", not a smaller p. z_*, pz_*, qz_* = z against the TF's own 100 draws and its one-sided normal p (BH-adjusted in qz_*).
  For rows with |z| > 8, pz_* is a normal-tail extrapolation, not a frequency; the exact p for those rows is at the 100-draw resolution
  floor (.0099). shapiro_* = Shapiro-Wilk p of the TF's draws; sd_flag = draw SD < 1e-4 under either null (none).
