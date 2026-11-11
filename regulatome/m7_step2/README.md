# m7_step2: Figure 5 Step 2, Regulome model with Perturb-seq train/test separation

The spec and all amendments are in SPEC.md (Amendments 1-5, each committed before the step it governs). The freeze is commit 203c542 (FROZEN.txt holds the sha256 of every script, split file and frozen parameter file). Test chromosomes and held-out TF folds were scored once, after the freeze: jobs 8864797-8864800, with the summary job 8864801.

## Result (section 7 rule, read on the primary arm: raw binding, canonical Hi-C links, Arm 2, train-centred, all 76 testable TFs)

- **Knockdown response: null.** No model in either mode shows binding signal on the by-gene split. All six primary intervals include 0 or lie below it: elastic net knockdown-only +0.00083 [-0.00292, 0.00456], LightGBM +0.00041 [-0.00252, 0.00347], network -0.00556 [-0.00914, -0.00225]. The by-TF split is null as well. The adopted Regulome model is therefore the linear model (elastic net, knockdown-only), and adopted_weights.tsv holds its per-TF weights.
- **Expression: null.** For no model does the rung-3 (+ per-TF binding) gain over rung 2 have an interval above 0 (elastic net -0.00436 [-0.01273, 0.00438]).
- The power-law sensitivity arm is also null for both targets. The imputed-binding arms are reported beside these (decision.tsv, decision_expr.tsv) and cannot change the verdict (Amendment 5.8).

## Files

| file | what |
|---|---|
| splits/gene_split.tsv, splits/tf_folds.tsv, pair_list.tsv, splits/splits.sha256 | splits and the pair list, committed with sha256 before any fit |
| per_tf_test.tsv | one row per binding kind x link set x split x arm x centring x model x mode x TF: real, baseline-only, pseudo (mean of 5), mis-paired (mean of 5), both references, primary, ceiling, cells, claim tier, fold, n test genes |
| pooled.tsv | means over TFs with 2,000-draw TF bootstrap intervals, for all TFs, the 19 claim-tier TFs, and without NRF1; differences against pseudo, mis-paired, both references and the linear model |
| ladder.tsv | expression R2 on test chromosomes per rung, gene-bootstrap intervals, gain over the previous rung, real minus pseudo |
| decision.tsv, decision_expr.tsv | section 7 rule read per model x mode (knockdown) and per model (expression), as Amendment 5 declares |
| adopted_weights.tsv | per-TF elastic-net weights of the adopted linear model (own promoter and distal binding, pair columns) |
| frozen/ | frozen hyperparameters per link set; validation-only tuning tables |

`kind` = binding kind and link set: raw_hicfixed (primary), raw_powerlaw (sensitivity), imp_hicfixed and imp_powerlaw (imputed binding, reported beside the others).

## Centring per table

The knockdown target is the raw per-TF z (out_v2: Arm 2 primary, Arm 1 secondary). `centring = train`: per-TF offset from train genes, and a per-gene offset from training TFs at train genes only, which is dropped for validation and test genes (Amendment 1.1). `centring = raw`: no per-gene offset. Every Spearman is computed against the raw per-TF z on test genes. The section 7 verdict uses `train`, and the `raw` statistic is reported beside it. The two centrings agree on the primary arm for 5 of the 6 by-gene model x mode rows; they disagree for LightGBM both-targets, and that disagreement is reported as the gene centring doing the work (decision.tsv, centrings_agree). The by-TF folds use no offset estimated from a held-out TF.

## Data and counts

- Gene universe: 11,312 genes with an ABC link (power-law set, 19,807 genes) and >= 10% NTC detection in Arm 2 (12,349) or Arm 1 (12,427). Dropped for no chromosome: 0. Dropped as chrY, chrM or non-primary: 0. chrX is in train. Arm 2: train 7,883, validation 1,128, test 2,211. Arm 1: 7,925, 1,131, 2,229. One gene symbol is duplicated among the Arm 2 genes, and its last occurrence is kept.
- TFs: 76 testable (Arm 1: 30); 3 panel TFs have no knockdown (GLYR1, SREBF2, ZNF367). Claim tier = d >= .04 in repro_v1: 20 TFs, 19 testable (GLYR1 has no knockdown). By-TF folds: 16, 15, 15, 15, 15, seeded permutation within depth x ceiling strata (seed 20261111).
- Expression target: NTC mean log1p(CP10k), Arm 2 cells. Expression features: 7 baseline/background columns, then 158 per-TF binding columns (79 TFs x promoter, distal), then 38 pair columns.
- Pairs: 38 = the 35 af3_strong pairs in complex_screen/interface_audit_20261110/triple_support_flagged_v2.tsv, plus the known controls not already among them (CTCF-STAG2, EZH2-MTF2, MAZ-ZNF143). No pair was dropped for a member missing from the panel.
- Links. Canonical Hi-C (m2c_abc_canonical HEK293T_hicfixed, threshold 0.02): 142,706 elements, 54,900 links to universe genes, 0 genes without a self-promoter link, 418 genes with no distal link. Power-law (m2_abc thr0.02): 159,965 elements, 60,461 links, 777 genes without a self-promoter link, 371 with no distal link. Cross-chromosome links: 0 in both sets. Links whose element is on a chromosome in another split: 0 in both sets.
- Raw binding: power-law = Gate 1 panel-element counts. Canonical = per-TF fragment 5' ends counted over canonical elements (Amendment 4). Imputed binding: panel-element imputed tracks, transferred to canonical elements by overlap-fraction weighting. Imputed scope: Arm 2 train-centred with full controls, and Arm 1 train-centred real and baseline-only, using the raw-arm hyperparameters.
- References (no fit): the TF's raw binding summed over elements within 100 kb of the TSS, and membership of the frozen v8 top-1,000 list. Both are computed on the power-law panel build and used for both link sets.

## Scripts

make_splits.py, build_features.py, count_canonical.py, m7.py (all fits), make_jobs.py, select_params.py, chain.sh (stage chain), score.py (test scoring, once), summarise.py, weights.py, run_*.sbatch. dryrun_val*.sbatch exercised the scoring code on validation chromosomes before the freeze (Amendment 5.9). Outputs (features/, fits/, results/, logs/) are not tracked.

## EXPLORATORY post-hoc diagnostics (Amendment 6; declared after the test chromosomes were read; cannot change any verdict)

The script is diag.py, run as jobs 8865840 (D1), 8865841 (D2) and 8865843 (D3). Outputs are in diagnostics_exploratory/, and every table carries the label column "EXPLORATORY (Amendment 6, post-hoc)".

- **D1, expression baseline audit (train and validation only; d1_expression_audit.json).**
  - Target: universe genes with a target number 11,222. min 0.037, median 0.259, max 4.02; none at zero.
  - Spearman with the target on train genes (canonical): promoter ATAC 0.22, promoter H3K27ac 0.31, pooled promoter binding 0.23.
  - Rung-1 elastic net validation R2: 0.040 as fitted, and 0.040 when refitted here on the same genes. Widening to all ABC genes on train and validation chromosomes gives 0.194; undetected genes stay at their NTC mean, and the 30% absent from the matrix are set to 0. ABC genes at >= 10% detection give 0.040. Power-law: 0.032, 0.176 and 0.032.
  - Canonical: every gene's self-promoter element contains its TSS. The element found by coordinates at the TSS is a self-promoter element for 11,312 of 11,312 genes. Spearman of that element's ATAC with the promoter feature is 0.96; the feature sums all of a gene's self-promoter elements (12,821 links).
  - Power-law: 777 genes have no self-promoter link. The remaining 10,535 have one within 499 bp of the TSS, and for 10,372 it contains the TSS. Spearman with the TSS-element ATAC is 0.86.
  - No feature or join error was found.
- **D2, controls on the claim-tier reading (from per_tf_test.tsv; d2_claim_tier_controls.tsv).**
- **D3, signed unfitted references (d3_signed_refs_pooled.tsv, d3_signed_refs_per_tf.tsv).** One sign per TF is estimated on train chromosomes, then applied to the test Spearman. References are on the power-law panel build, Arm 2, both centrings. The baseline-only elastic net beside them is unsigned.

## EXPLORATORY widened-universe expression ladder (Amendment 7; post-hoc; does not change the section 7 expression verdict)

- Universe: all 19,807 ABC genes. Train 13,820, validation 1,916, test 4,071 genes. Genes absent from the matrix (target set to 0): 29.6%, 29.5% and 33.2% of each split. Detected at >= 10%: 57.0%, 58.9% and 54.3%.
- Elastic net and LightGBM, re-tuned on validation (frozen/params_wide_*.json). Freeze aaa5b02; test scored once, in jobs 8869458 and 8869511-8869513.
- Tables in wide_exploratory/: ladder_wide.tsv covers the level target (R2) and the detection target (AUROC, AP). dropone_wide.tsv covers rung 3, level target, all 79 TFs.
- The mis-paired control does not apply, because the per-TF rung holds all 79 TFs at once.
