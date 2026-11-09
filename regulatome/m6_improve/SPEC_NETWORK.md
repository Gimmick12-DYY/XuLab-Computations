# Borrowing strength across TFs over the Regulome network

Yuyang, 2026-11-09 20:51: "by binding the regulome network ... if we can use the modeling to give
weights and create edges that link individual TFs, we could somehow borrow the poor TFs with some
information from the strong ones."

This is a declared question with its own objective and its own pre-declaration. It is tested on the 52
development TFs only. It reverses the earlier "no target transfer" decision, so it has to be tested, not
assumed.

## 1. Why this needs a sharp null

The tuning already showed that the other TFs' pooled binding scores **higher** than a TF's own
(0.76 vs 0.67 on the B6 effect). Smoothing a TF's score toward other TFs will therefore raise the B6
effect almost automatically, whatever the network says. The B6 effect cannot judge this question.

The only thing worth asking is whether borrowing along **real** edges beats borrowing along **any**
edges. Three comparators, all at the same smoothing strength:

1. **Uniform pooling** — the same amount of smoothing spread over every other TF. This is the panel
   average that already explains most of the signal.
2. **Degree-preserving edge shuffle** — the per-edge shuffle null already used for the Figure 3 network.
3. **No smoothing** (alpha = 0).

A result counts only if graph smoothing beats all three on specificity.

## 2. The score

For TF t with element-level binding score S(t, g):

    S_sm(t) = (1 - alpha_t) * S(t) + alpha_t * sum_u W(t, u) S(u) / sum_u W(t, u)

W from the binding data alone, never from the benchmark: depth-matched co-binding percentile (Pearson,
as agreed for the complex slides), with the structure and motif support used as edge weights where it
exists. Two settings for alpha:

- **Fixed sweep**: alpha in 0, 0.15, 0.3, 0.5, 0.7.
- **Reliability-weighted** (the hierarchical version, and the one that matches Yuyang's point):
  alpha_t = sigma2_own(t) / (sigma2_own(t) + tau2), with sigma2_own from the split-half reproducibility
  of that TF's own binding, estimated from the binding data only. Noisy TFs borrow more, deep TFs borrow
  little. Nothing here is fitted to Perturb-seq.

## 3. Objective

Ceiling-weighted own-versus-other specificity, as in SPEC.md, with the own-knockdown rank. Reported with
and without NRF1, both arms, 1,000 random sets.

## 4. The partner contrast, which is the real prediction

Smoothing makes partners' lists more alike, so lumping partners in with "other knockdowns" would hide
the effect it is meant to show. Split the contrast three ways for every TF:

| | prediction if the edges are real |
|---|---|
| own knockdown | high |
| partner knockdowns (its graph neighbours) | also high |
| non-partner knockdowns | low |

Report own, partner and non-partner response separately, and take **own and partner above non-partner**
as the signature of a real edge. This is a genuine test of the network, not a repair of the gene score:
if partner knockdowns move a TF's targets and unrelated knockdowns do not, the edges carry regulatory
information even where a single TF's own knockdown is too weak to show it.

## 5. Who should gain

Stratify by knockdown ceiling tertile. The claim is specifically that weak TFs gain. If only the strong
TFs gain, the method is relabelling the strong ones and should be rejected.

## 6. Effect on the freeze

The held-out 24 TFs can honestly be scored **once**. Splitting them between two questions leaves 12 each,
which is too few. So hold the single held-out scoring until this question resolves on the development
TFs, then score once with every candidate in the same table: the gene-score candidates from grid6 and
the borrowing candidates from here. That costs an hour and keeps one clean use of the held-out set.
