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
