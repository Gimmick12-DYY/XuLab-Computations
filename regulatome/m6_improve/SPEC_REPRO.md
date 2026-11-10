# Raising the reproducibility of our own target lists

DECLARED 2026-11-10, before any run. Judged entirely on our own data. No Perturb-seq, so nothing here
can spend the held-out TFs or need a new split.

## Why this is the right target

The benchmark programme returned four nulls, and the diagnosis is that for 65 of 79 TFs our own target
list does not agree with itself across a split of our own data (median split-half Jaccard 0.107).
Nothing can validate a list that does not replicate against itself.

Reproducibility is almost entirely a depth effect:

    Jaccard = -0.594 + 0.206 * log10(cells per TF)      r = 0.94, R-squared = 0.87, residual SD 0.040

| target Jaccard | cells per TF implied |
|---|---|
| 0.2 | 7,300 |
| 0.3 | 22,200 |
| 0.4 | 68,000 |

The median TF has 3,400 cells and 69 of 79 sit below the 0.3 requirement. The panel's 830,331 cells are
spread thin rather than insufficient in total.

## The trap that must be guarded

Any method that shrinks a TF's score toward the panel will raise within-TF agreement by making every
TF's list more alike, up to the limit where all lists are identical and the Jaccard is 1. Reproducibility
on its own is therefore not a valid objective.

**Objective: J_within - J_between.** For each TF, J_within is the split-half Jaccard of its own list,
and J_between is the mean Jaccard of that TF's list against the other TFs' lists built on the same
halves. Report both components on every row, never the difference alone. A method that raises both is
rejected however good the first column looks.

## Run 1, the depth curve, which is also the yardstick

Subsample the deep TFs, RBBP4 at 181k cells down through NFYA, SOX4, XPA and RBBP7, to 1k, 2k, 3.4k, 5k,
10k, 20k, 50k and 100k cells, five draws each, and measure J_within and J_between at every depth.

This does two things. It tests whether the cross-TF relationship above is really depth rather than
antibody quality, by tracing the curve inside a single TF. And it converts any later result into
**equivalent cells**: a method that raises J_within by 0.04 can be reported as worth so many thousand
cells per TF, which is the number the lab actually needs for planning.

## Run 2, the methods

All scored on J_within - J_between, both components shown, at the panel's real depths:

1. Empirical-Bayes shrunk lists from the r3_eb machinery.
2. Ranking on pooled-neighbour bins rather than raw summed CPM.
3. Peak-level rather than bin-level aggregation.
4. Fixed windows at 50 and 100 kb, which already beat the ABC-linked sum on the benchmark.
5. Imputed binding.
6. Element-level normalisation against the pooled panel signal, as a ratio and as a residual.
7. List size 250, 500, 1,000, 2,000, since a shorter list is a stricter test of agreement.

Report each method's gain in equivalent cells, and stratify by current depth: a method that only helps
the already-deep TFs does not solve the problem.

## What a win means, and what it does not

A method that raises J_within - J_between materially is a real improvement to the Regulome and is
publishable on its own terms, because reproducibility is a property of our data and needs no external
benchmark to justify it.

It is **not** a licence to re-run the benchmark. The held-out TFs are spent and no fresh ones exist. If
an improved list is ever to be tested against knockdowns, that needs new TFs or a new dataset, declared
separately.
