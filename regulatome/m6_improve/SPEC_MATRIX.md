# Comparing our TF x gene matrix directly against the Perturb-seq matrix

Yuyang, 2026-11-09 21:36: "Then why can't we calculate the same number for our ABC results?"

Short answer: we can, and we should have done this before reducing everything to top-1,000 lists.
Nothing stops us from putting our score and their z side by side over the whole gene universe. What we
cannot do is call our number a predicted z, and the reason matters for how the comparison is read.

**Status: descriptive only.** The 24 held-out TFs were scored once and are spent. Nothing here can
establish a new claim; it can only describe what the two matrices have in common.

## 1. What the two numbers are

| | Perturb-seq z | Our score S(t, g) |
|---|---|---|
| what it measures | how much gene g moved when TF t was removed | how much of TF t sits at the elements linked to g |
| sign | signed: up or down | unsigned magnitude, always >= 0 |
| units | SD against GEM-matched non-targeting controls | summed CPM |
| causal status | a consequence of removing the TF | a property of the unperturbed cell |

So S is not a prediction of z and cannot become one without a direction and a dose-response we do not
have. S says a TF is present; z says a gene moved. The honest comparison is a correlation, per TF, not
an agreement statistic.

## 2. Why the sign is informative anyway

For an activator, knocking the TF down should lower its targets, so S correlates **negatively** with
signed z. For a repressor the sign flips. The per-TF sign of corr(S, signed z) is therefore an empirical
activator-or-repressor readout, which the top-1,000 lists threw away by using |z|. Report it per TF with
its interval. Expect it to be small, and treat a TF whose sign is unstable across the two arms as
uncalled.

## 3. The confound that will otherwise dominate

Both matrices have strong gene-level structure. Genes with many linked elements score high for every TF;
genes that respond to everything have large |z| for every knockdown. A raw correlation across genes will
be positive and mean nothing, which is the same trap that produced the element-count result. So:

- Primary: both matrices **double-centred** (subtract gene means and TF means), as the X-Atlas matrix
  already is.
- Secondary: raw, reported beside it so the size of the confound is visible.
- Control in every table: the **element count per gene**, a single column with no TF index, correlated
  against each TF's z. Our per-TF correlations must exceed it or the conclusion is unchanged.
- Partial correlation of S with z controlling for element count and the gene's mean |z|.

## 4. The run

- Universe: genes present in both the expression matrix and our linked-gene set, per arm.
- All overlapping TFs (76 in Arm 2, 30 in Arm 1), both arms, Spearman per TF.
- Four statistics per TF: S vs |z|, S vs signed z, each raw and double-centred.
- Report against the per-TF reproducibility ceiling, since replicate correlation is 0.04 panel-wide and
  caps what any method can reach. A correlation of 0.05 against a ceiling of 0.1 is not the same result
  as 0.05 against a ceiling of 0.6.
- Also score v8 (binding-only) and the ABC-weighted score, so the ABC comparison is made on the full
  matrix rather than on list overlap.

## 5. What would be worth seeing

Not a high correlation. A per-TF correlation that beats the element-count control after double-centring,
in both arms, for more than a handful of TFs. That is the same bar as before, asked of the full matrix
instead of a thresholded list, and the full matrix has more power because it uses every gene rather
than a cut at 1,000.
