# Figure 5 Step 2: Regulome model with Perturb-seq train and test sets (spec, 2026-11-11)

Requested by Yuyang 2026-11-11 04:24Z ("keep test and train sets that separate the perturb-seq data"; doubts that a simple regularised regression suffices) and 04:34Z ("we can give it a try"). Written before any result of this version exists. Folder: XuLab/regulatome/m7_step2/. Corrections to this file are made by appending amendments, never by editing declared text.

Nothing here reads a test-set number before section 7 is complete. Quantities that must not be read or relayed first: any metric computed on the test chromosomes or on a held-out TF fold. None has been read as of this writing.

## 1. Prior evidence that must travel with any result
- 2026-11-09 tuning round (m6_improve): ridge and gradient-boosted models, split by TF, scored worse on the 52 tuning TFs than the plain 100 kb binding sum. The by-TF question therefore starts with evidence against it.
- The four earlier Perturb-seq tests were null. Three TFs (E4F1, NME2, XPA) have both a reproducible list and a strong knockdown.
- The 24 held-out TFs of split_v1 were scored once. Results here are "benchmark-trained", never "independent validation".

## 2. Data
- Binding: raw HITAG fragments per TF, 79 TFs (primary). Imputed tracks are a second arm, reported side by side, never mixed.
- Links: ABC links from the current power-law build to start; rerun unchanged on the canonical ABC links (abc_canonical/SPEC.md) once the Hi-C re-export is done. State which link set every table used.
- Knockdown target: existing kd_response out_v2 z, genes with >= 10% NTC detection, units averaged per TF, double-centred (as in B6/m6). Arm 2 primary, Arm 1 secondary.
- Expression target: per-gene mean of log1p(CP10k) over NTC cells from the same dataset, same gene universe. A bulk HEK293T RNA-seq set is added as replication if Yuyang names one.
- Gene universe: genes in the ABC gene set and in the KD matrix.

## 3. Features (identical for all three models)
Per gene: promoter ATAC, promoter H3K27ac, summed ABC score of linked distal elements, number of linked elements, CpG-island promoter flag (baseline); binding pooled over all 79 TFs (panel background).
Per TF and gene: the TF's binding summed over the promoter and over linked distal elements (two values), log1p CPM, standardised within TF.
Pairs: for the pre-declared pair list (correlation + AF3 candidates plus the known controls; file pair_list.tsv committed before fitting), min of the two members' binding at the same element, summed per gene.
For the knockdown target one example is a (TF, gene) pair: the gene's baseline and background features, the knocked-down TF's own two binding values, the pair values for pairs containing that TF, and a TF identity term.

## 4. Splits (files committed with sha256 before any fit)
- By gene: test chromosomes chr3, chr11, chr17, chr20; validation chromosomes chr5, chr12; all others train. The same assignment applies to both targets and every TF.
- By TF: 5 folds over the testable TFs, stratified by depth tertile x knockdown-ceiling tertile, seed 20261111. The split_v1 status of each TF (dev or held-out) is carried as a column and reported, not used.
- All tuning (hyperparameters, feature scaling, early stopping, model choice within a family) uses train and validation only. The test chromosomes are scored once per model, after section 7 is frozen.

## 5. Models
1. Regularised linear regression (elastic net), with a per-TF coefficient on that TF's binding values.
2. Gradient-boosted trees (LightGBM).
3. Small multilayer network (two hidden layers, TF embedding), shared across both targets.
Each is fitted for: expression only; knockdown only; both targets (model 3 jointly, models 1-2 with the expression-model prediction as an added feature).

## 6. Controls (every model, same splits)
- Baseline-only: no per-TF binding, but the per-TF term kept. This is the control that matters most for the by-gene split: a model with a per-TF term can predict knockdown response from accessibility or expression level alone. Binding must add to this.
- Pseudo-TF: each TF's binding replaced by the same number of fragments placed on elements of matched ATAC decile (5 draws).
- Mis-paired: each TF scored with another TF's binding from the same depth tertile.

## 7. Metrics and decision rule (declared now)
- Knockdown, primary: per-TF Spearman between predicted and observed z on test-chromosome genes, minus the same quantity for the baseline-only control; summarised as the mean over TFs, with a paired bootstrap over TFs (2,000 draws). Reported for all testable TFs, for the 20 claim-tier TFs, and without NRF1. Per-TF values are tabulated next to knockdown ceiling and cell count.
- Knockdown, by-TF split: the same statistic on held-out TF folds.
- Expression: held-out R2 on test chromosomes for the four-rung ladder (baseline, + background, + per-TF binding, + pairs), real against pseudo-TF.
- A model "shows binding signal" only if its primary statistic has a bootstrap interval above 0 AND exceeds the pseudo-TF and mis-paired controls, in Arm 2, with the same sign in Arm 1.
- A more complex model replaces the linear one only if its primary statistic exceeds the linear model's with a paired bootstrap interval above 0, on the by-gene split, and it also clears its own controls. Otherwise the linear model is the Regulome model.
- If no model shows binding signal, that is the result. No further variants are fitted against the test chromosomes.

## 8. Outputs
splits/ (with sha256), pair_list.tsv, per_tf_test.tsv, pooled.tsv, ladder.tsv, README.md; fitted weights or attributions per TF for the adopted model. Scripts committed in the Longleaf repo under the user's git identity, files staged by name, no attribution lines.

## Amendment 1, declared 2026-11-11 before any fit, before any split file exists

Four corrections, all made before any feature, fit or metric exists. No test-chromosome or held-out-TF quantity has been read.

**1. The target itself leaks, and must be re-centred on train chromosomes only.** Section 2 takes the knockdown target as the existing double-centred z. Double-centring subtracts per-TF and per-gene means computed over the whole matrix, so the value of a test-chromosome gene is a function of the test genes' own mean. The centring must be recomputed inside the training set: per-TF and per-gene offsets estimated on train chromosomes only, then applied unchanged to validation and test. The same holds for the within-TF standardisation of binding features and for any gene-level normalisation of the expression target. Nothing fitted, scaled or centred may see a test chromosome. It is easy to reintroduce by accident because the double-centred matrix already exists on disk and is convenient to load.

A consequence worth stating: per-gene offsets cannot be estimated for a test gene from the training genes, because they are gene-specific. The per-gene term is therefore dropped for scoring rather than estimated, and the primary statistic is a per-TF Spearman over genes. State in the README which of the two centrings each table used, and report the primary statistic both ways: on the train-centred target and on the raw per-TF z with no gene centring at all. If the two disagree, the gene centring is doing the work and must be reported as such. For the by-TF folds, the same rule applies along the TF axis: no offset may be estimated using a held-out TF's knockdown.

**2. Reference predictors are missing, and they are the ones to beat.** Section 1 records that in the 2026-11-09 round ridge and gradient-boosted models scored worse than the plain 100 kb binding sum. Add two reference predictors, scored identically to the three models, on the same splits, through section 7's rule:
- the plain sum of that TF's raw binding over elements within 100 kb of the gene, no fitting;
- membership of that TF's frozen v8 top-1,000 list, as a binary predictor.
Neither is fitted, and both are reported in the same table as the models. A model that does not beat both is not an improvement however well it clears its controls.

**3. The two targets get separate verdicts.** Expression and knockdown response are different questions. A result on the expression ladder is reported as a statement about expression and never as evidence that the Regulome predicts knockdown response. "No binding signal" is recorded per target. An expression-only success is written as "binding adds to an accessibility model of steady-state expression", which is a weaker and different claim.

**4. The expression model's feature layout is fixed.** For the knockdown target an example is a (TF, gene) pair and the per-TF binding features are that TF's two values. For the expression target an example is a gene, so the per-TF rung of the ladder carries all 79 TFs' two values at once, 158 features, plus the baseline and background rungs. The pair rung carries one summed value per pre-declared pair. State the counts in the README, and do not collapse the 79 into a sum.

**Also recorded, not a correction.** Check and state in the README whether any gene's features draw on an element on a chromosome in another split (ABC links are expected to be within-chromosome).

## Amendment 2, declared 2026-11-11 before any fit

Fixes every choice this spec left open. Written after the split files, before any feature was fitted; no test-chromosome or held-out-TF quantity has been read.

**1. Testable TFs and claim tier.** Testable = the 76 TFs of split_v1 with an Arm 2 knockdown and panel binding (Arm 1: the 30 of these with an Arm 1 knockdown). 3 panel TFs have no knockdown and are not testable (GLYR1, SREBF2, ZNF367). The claim tier is d >= .04 in repro_v1/per_tf_floor_test.tsv: 20 TFs, of which 19 have a knockdown (GLYR1 does not). "The 20 claim-tier TFs" of section 7 therefore reports 19. STAG2 has no knockdown-ceiling value (its replicate correlations are undefined) and forms its own stratum "na". Fold assignment continues the round-robin counter across strata.

**2. What the two centrings are, operationally.** Offsets are estimated on train chromosomes. Per-TF offset: the TF's mean z over train genes (irrelevant to the per-TF Spearman but applied). Per-gene offset: the mean over the training TFs of the TF-offset z at that gene, estimated for train genes only and subtracted from train genes only; validation and test genes get no per-gene offset (Amendment 1). "train-centred" models are fitted to the target with both offsets; "raw" models are fitted to the target with no per-gene offset. Because no per-gene offset can be applied to test genes, every test Spearman is computed against the raw per-TF z of the test gene (a per-TF constant does not change it), so the difference between the two reported statistics is in what the model was fitted to. The primary statistic is the train-centred fit.

**3. Fit protocol.** "train + validation" means: fitted on train chromosomes, with validation chromosomes for hyperparameter selection and early stopping only. No refit on train + validation. Each TF's own gene (HP1: CBX1/3/5; ZUFSP: ZUP1) is excluded from training and scoring for that TF. An undefined Spearman (constant predictor) counts as 0. Hyperparameters are selected once per model, target, mode (knockdown-only or both) and feature condition (real or baseline-only), on Arm 2, train-centred, raw binding, by-gene, by mean per-TF Spearman on validation genes, and frozen for all other runs (pseudo-TF and mis-paired use the real setting; by-TF folds, Arm 1 and the raw-centring fits use the setting of the matching run; expression uses validation R2, per rung). Early stopping (LightGBM, network) is rerun in each fit on the validation chromosomes.

**4. Model definitions.** Elastic net: columns are 7 baseline/background columns, the TF's two binding values, one column per pair (0 unless the TF is a member), plus per-TF copies of the baseline/background and binding columns (TF identity term; in baseline-only the binding copies are absent). Grid: alpha {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} x l1_ratio {.1, .5, .9}. LightGBM: objective l2, learning rate .05, up to 2,000 trees, early stopping 50, bagging .8, lambda_l2 1, TF identity as a categorical column; grid num_leaves {15, 31, 63} x min_data_in_leaf {50, 200} x feature_fraction {.7, 1} (knockdown), {7, 15, 31} x {10, 30} x {.7, 1} (expression). Network: two hidden layers (a shared trunk over baseline/background, then a head per target), TF embedding fed to the knockdown head; expression rungs 3-4 add their binding and pair columns to the expression head; Adam lr 1e-3, weight decay 1e-5, batch 8,192 (knockdown) or 256 (expression), early stopping on validation MSE; grid hidden size {64, 128, 256} x dropout {0, .2} x embedding {8, 16}. By-TF folds: no TF identity term for the elastic net or LightGBM; the network uses the mean of the training-TF embeddings for held-out TFs. Both-target mode for models 1-2: an expression-model prediction column is added, cross-fitted by chromosome on train genes (5 chromosome folds) and from the full-train model on validation and test genes; the expression model used is rung 4 real (real and mis-paired), rung 2 (baseline-only), or rung 4 of the matching pseudo draw. Model 3 joint: same trunk, knockdown and expression heads trained together, expression target in train-gene SD units, equal loss weights.

**5. Features.** Promoter = the gene's self-promoter ABC links; distal = all other links; number of linked elements = all links. Background = pooled CPM of all 79 TFs, two columns (promoter, distal). CpG-island promoter flag = any self-promoter element overlapping a UCSC hg38 cpgIslandExt interval. H3K27ac = RPM from the m2b_abc_k27ac EnhancerList at the same elements. All log1p CPM columns standardised on train-chromosome genes. Pseudo-TF: each TF's element values permuted within ATAC-decile strata (5 draws, seed 20261111 + 1000 x (draw + 1) + TF index); fragment counts per element are permuted, not resampled. Mis-paired: 5 derangements of the TFs within depth tertile (seed 20261111 + 900 + draw). Background, ATAC and H3K27ac are never replaced. Mis-paired is defined only for the knockdown target.

**6. Reference predictors and rule.** Both references (100 kb raw CPM sum around the TSS; v8 top-1,000 membership, with ties) are scored per TF on the same test genes as the models, and reported as Spearman; their "primary statistic" is computed against the elastic-net baseline-only control. "Beats" a reference means the model minus reference mean per-TF Spearman has a paired bootstrap interval above 0.

**7. Scope of controls and arms.** Full controls (baseline, 5 pseudo, 5 mis-paired) are run for Arm 2, train-centred. For the raw-centring statistic and Arm 1, real and baseline-only only, as the rule needs only a sign there. Imputed binding: after the raw arm, Arm 2 train-centred with full controls and Arm 1 train-centred (real, baseline-only), by-gene and by-TF, with the raw-arm hyperparameters. The joint expression head is fed the pseudo or mis-paired columns as the matching run; Arm-1-only genes get the expression prediction from the full-train model. Link set for every table in this round: power-law ABC, thr0.02 (m2_abc).

## Amendment 3, declared 2026-11-11 before any test-chromosome number exists for either link set

Link sets. Section 2 said the build starts on the m2_abc power-law links and is rerun unchanged on the canonical links. Both link sets are now available before any test number has been read, so the roles are fixed here: the PRIMARY link set is canonical ABC with real Hi-C (m2c_abc_canonical HEK293T_hicfixed). The m2_abc power-law link set (104,071 links, the set behind the frozen v8 lists) is the sensitivity arm. Features, splits, models, controls, tuning and the decision rule are identical for both. The gene universe and the split files are unchanged; a gene with no link in a link set keeps its promoter features and gets zero distal features, and the count is reported. The verdicts of section 7 are read on the primary link set. The sensitivity arm is reported beside it and cannot rescue or overturn a verdict. The v8-membership reference predictor stays as frozen (it was built on the power-law links); the 100 kb plain-sum reference does not depend on links.

## Amendment 4, declared 2026-11-11 before any feature on the canonical links

Operational choices for the canonical link set, none changing a declared comparison. (1) The canonical candidates are a different element set from the panel elements behind the Gate 1 matrices (0 of the 109,421 links are at panel element coordinates; 142,706 elements, 500 bp). Raw binding on them is the count of each TF's fragment 5' ends (per-TF fragment BED files; + strand start, - strand end - 1; not Tn5-shifted, not deduplicated) over the canonical elements, as CPM of the reads in elements. ATAC and H3K27ac are the RPM columns of the canonical EnhancerList. Pseudo-TF ATAC deciles use those. (2) The imputed tracks exist only on panel elements; on canonical elements the imputed value is the overlap-fraction-weighted transfer of panel-element CPM, renormalised to CPM. (3) Hyperparameter tuning is run separately on each link set with the same grids and rules, and frozen separately (frozen/params_<link>.json). (4) The two references are computed once on the power-law panel build (raw binding, TSS +/- 100 kb) and used identically for both link sets, as Amendment 3 states. (5) All 11,312 universe genes have a self-promoter link in the canonical file, so the "no link" case is count 0 for promoters; the count of genes with no distal link is reported.

## Amendment 5, declared 2026-11-11 at the freeze, before any test-chromosome or held-out-TF number exists

This is the freeze of section 7. Every fit for both link sets (raw and imputed binding) is complete on train and validation chromosomes. The hyperparameters are frozen in frozen/params_hicfixed.json and frozen/params_powerlaw.json, with the validation tables in frozen/tuning_*.tsv. Nothing below changes a declared comparison. It fixes how the rule is read off the tables, so that no reading choice is left for after the test numbers exist.

**1. Data handling recorded.** The knockdown and expression matrices contain duplicate gene symbols. One symbol has two rows among Arm 2 genes at >= 10% detection. The last occurrence is kept, as in m6_improve load().

**2. Primary statistic and intervals.** Per TF, primary = Spearman(real) minus Spearman(baseline-only) for the same model, mode, arm, centring and split, on the same test genes. Pooled = mean over TFs, with a 2,000-draw bootstrap over TFs (seed 20261111), 95% percentile interval. The TF sets are all testable TFs, the 19 claim-tier TFs with a knockdown, and all testable TFs without NRF1. For the by-TF split, each TF is scored once, from the fold that holds it out, and the five folds are pooled into one set.

**3. Controls.** "Exceeds the pseudo-TF control" means the pooled primary of the real model is greater than the pooled primary of the pseudo-TF model, where each TF's pseudo Spearman is averaged over the 5 draws. The paired bootstrap interval of real minus pseudo is reported beside it. The mis-paired control is read the same way. "Same sign in Arm 1" means the pooled Arm 1 primary (train-centred, real minus baseline-only) has the same sign as the pooled Arm 2 primary.

**4. Knockdown verdict.** Verdicts are read per model (elastic net, LightGBM, network) and per mode (knockdown-only, both targets). They use Arm 2, the train-centred target, all testable TFs, raw binding and canonical links (raw_hicfixed). "Shows binding signal" = interval above 0 AND exceeds pseudo AND exceeds mis-paired AND same sign in Arm 1. The by-gene split is the split section 7 names for replacement. The by-TF split gets the same columns and is reported beside it.

**5. Centrings.** The verdict is read on the train-centred target. The raw-z centring primary and its interval are reported beside it. The two centrings "disagree" if the pooled primaries differ in sign, or if one interval is above 0 and the other is not. A disagreement is reported as the gene centring doing the work.

**6. References and adoption.** "Beats" a reference means the paired interval of model minus reference (mean per-TF Spearman) is above 0, as in Amendment 2.6. The linear model is the elastic net in knockdown-only mode. Any other model and mode replaces it only if, on the by-gene split, it shows binding signal, its primary exceeds the linear primary with a paired interval above 0, and it beats both references. If several qualify, the one with the largest primary is adopted. Otherwise the adopted Regulome model is the linear model, whether or not it shows binding signal. Its per-TF weights are written to adopted_weights.tsv: the effective coefficients on the TF's own promoter and distal binding columns (shared plus the TF's own copy) and on its pair columns, from the frozen by-gene Arm 2 train-centred real fit. If a non-linear model were adopted, its attribution method would be declared by a further amendment before it is run.

**7. Expression verdict.** Per model, at rung 3 (+ per-TF binding), the statement "binding adds to an accessibility model of steady-state expression" is made only if both of the following hold. First, the R2 gain over rung 2 has a paired bootstrap interval above 0, from 2,000 resamples of test genes shared across rungs. Second, real minus pseudo-TF R2 (mean over 5 draws) has a paired interval above 0. Rung 4 (+ pairs) is reported with the same columns and gets no separate verdict. The network's joint expression head is reported at rung 4 only. This verdict is never read as a knockdown result (Amendment 1.3).

**8. Link sets and binding kinds.** Verdicts come from raw_hicfixed only. raw_powerlaw (sensitivity), imp_hicfixed and imp_powerlaw get the same columns, are reported beside it, and cannot rescue or overturn a verdict. Imputed fits use the raw-arm hyperparameters of the same link set.

**9. Scoring code checked before the freeze.** score.py, summarise.py and weights.py were run before this freeze, on validation chromosomes only, with by-gene fits and with by-TF fits scored on each fold's training TFs (dryrun_val.sbatch, dryrun_val_fold.sbatch; outputs under logs/, not committed). Rescored validation Spearman matched the fit-time value to within 1e-6, and to within 1e-6 for the network on the by-TF branch when the training TFs' own embeddings were used. The held-out path uses the mean embedding, as Amendment 2.4 declares. No test-chromosome gene and no held-out TF was read.

**10. Run once.** score.py is run once per binding kind per link set: raw and imputed, canonical and power-law. summarise.py and weights.py then build per_tf_test.tsv, pooled.tsv, ladder.tsv, decision.tsv, decision_expr.tsv and adopted_weights.tsv. No model, feature or hyperparameter is changed after this point. If a scoring script fails, only the failing code path is fixed, by a dated note appended here, with no change to any fitted model or rule.
