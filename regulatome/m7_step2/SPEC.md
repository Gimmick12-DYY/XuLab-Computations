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
