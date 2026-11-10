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

> **RETRACTED 2026-11-10 by Amendment 3 at the foot of this document.** The table above converts a
> target J_within into cells per TF and is wrong as a planning number, because J_between rises at
> almost the same rate. The numbers 7,300 / 22,200 / 68,000 must not be quoted. The table is left in
> place so that the retraction has something to point at.

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

## Amendment 1, declared 2026-11-10 before Run 1 was read

**J_between is measured at matched depth.** Taking it against the other TFs at full depth changes its
meaning along the curve, because a list built from 1,000 cells is noisy and overlaps less with
everything. Since five deep TFs are subsampled at each depth, the pairwise Jaccard among those five at
the same depth costs nothing and is the primary J_between for Run 1. The full-depth version stays as a
secondary column and remains the right reference for Run 2.

**The test is the slope, not the fitted value.** The cross-TF fit mixes depth with everything that
differs between TFs, antibody quality most of all, so its intercept averages over 79 antibodies and each
within-TF curve will sit at its own level. Depth is confirmed as causal if the within-TF slope is near
0.206 per factor of ten in cells. Report a slope per TF with its interval, and the spread of intercepts
separately, since that spread measures the antibody-quality effect and is worth knowing on its own.

If the slopes come out flat, depth is not the lever, the equivalent-cells yardstick is void and Run 2
must not be read against it.

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

## Amendment 2, declared 2026-11-10 AFTER Run 2b was read, including imputation at +.287

*Header corrected 2026-11-10. An earlier draft of this document said "before Run 2b was read" for both
amendments, which was wrong: Run 2b, imputation included, had already been read. The rule change below
was made after seeing +.287, and is recorded as such here and in the imputed_perhalf README. What was
set before anything was run is the .05 margin bar on the leakage-free refit, and only that.*

**The rejection rule was too strict and is corrected.** "Reject any method where both components rise"
was written to catch shrinkage toward the panel, where within-TF agreement and between-TF agreement
both go to one and the difference goes to nothing. The objective is and remains the difference. A method
that raises both components while raising the difference substantially is not the pathology the rule was
aimed at. Both components rising is a flag requiring scrutiny, not an automatic rejection.

Under the corrected rule, imputation is **suspended, not rejected**: it raises the difference from .009
to .296 in every depth tertile. The objection to it is leakage, not the rule. Its model is shared across
the halves and its neighbours are drawn from the full data, so the halves are not independent and
within-TF agreement is inflated mechanically.

**The leakage-free test, declared before it is run:** everything that touches the data is refit inside
each half, the neighbour graph included. The margin survives only if it stays at least .05 above v8 with
the interval excluding zero. That bar is five times the current baseline difference and was set before
the .287 was seen.

**Control:** the same per-half imputation with a neighbour graph built from accessibility alone, with no
TF information. If a TF-agnostic smoother gives the same gain, the result is convergence to common
structure rather than TF signal.

## What a win means, and what it does not

A method that raises J_within - J_between materially is a real improvement to the Regulome and is
publishable on its own terms, because reproducibility is a property of our data and needs no external
benchmark to justify it.

It is **not** a licence to re-run the benchmark. The held-out TFs are spent and no fresh ones exist. If
an improved list is ever to be tested against knockdowns, that needs new TFs or a new dataset, declared
separately.

## Amendment 3, declared 2026-11-10 after Run 1 and Run 2b were read, before the leakage-free refit

**The equivalent-cells yardstick is void, and the planning number is retracted.**

Run 1 confirmed the causal half of the hypothesis and killed its usefulness in the same table. Within a
single TF, J_within rises with depth exactly as the cross-TF fit said it would: slopes .239 to .329,
median .308, every interval excluding both zero and .206. Depth is causal, the antibody effect is the
.28 spread of intercepts, and that part of Run 1 passed.

But matched-depth J_between rises almost as fast, slopes .24 to .28. The objective, the difference,
therefore grows by only .004 to .058 per factor of ten in cells: NFYA +.019 at 5k and +.061 at 50k,
SOX4 +.091 at 50k, RBBP4 +.041, RBBP7 +.001 at 20k. Eighty to a hundred per cent of what depth buys is
signal shared with every other TF in the panel, and that shared signal is promoter-concentrated.

The partner-contamination explanation was mine and it was wrong. Excluding each TF's complex partners
from J_between changes it by a factor of 1.00 to 1.06; RBBP4 and RBBP7 agree with everyone, not just
with NuRD. The convergence is generic.

**Consequences, all binding on Run 2:**

1. No result may be reported in equivalent cells. Report J_within, J_between and the difference, at the
   panel's real depths, and nothing converted through the depth curve.
2. "Sequence more cells" is not a route to TF-specific lists and must not be offered as one. At the
   observed difference slope, a factor of ten in cells buys about .04, so reaching a difference of 0.2
   from today's 0.01 would take roughly five orders of magnitude. The lab cannot buy its way out.
3. A method now has to beat depth at its own game, by raising the difference rather than J_within. Run
   2's methods are judged on that alone, and a method that merely reproduces the depth curve's shape is
   a null result whatever it does to the first column.
4. The depth curve keeps one legitimate use: as the null. Any method's difference must be compared
   against the difference a depth increase of the same J_within gain would have produced, which is the
   .004-.058 band above.

This amendment is a retraction of a number already given to Yuyang on 2026-11-10 at 05:49 and withdrawn
to him the same day. It is recorded here so that the spec cannot be read later as if the table at the
top still stood.

## Amendment 5, declared 2026-11-10 after Run 2c was read

**Amendment 3's band was underspecified and is corrected here.** It said a method's difference gain must
exceed what a depth increase of the same J_within gain would have bought. For a method that LOWERS
J_within that comparison is degenerate: the band collapses to zero and anything positive passes. Four
methods in Run 2b and 2c exploited that without anyone intending it, since they all raise the difference
by lowering both components.

The band therefore applies only when the method raises J_within. For a method that lowers J_within the
comparator is not the depth curve but the noise floor, defined below.

**The noise floor, which must be measured before any further method is scored.** A list carrying no
information has J_within near zero and J_between near zero, so its difference is also near zero. Any
method that pushes a TF's list toward noise will therefore show a difference near zero and can look like
an improvement wherever the baseline difference is negative. The low-depth tertile is exactly that case:
its baseline difference is -.013, and the best of the four methods reaches about +.003.

Declared test, to be run before the next method: build deliberately uninformative lists at matched depth
and matched list length, by permuting element labels within each half and by drawing elements at random
with the element-count distribution preserved, and report J_within, J_between and the difference by
depth tertile. If the noise floor's difference is not clearly below +.003 in the low tertile, then every
low-tertile gain seen so far is indistinguishable from removing the signal altogether, and this whole
line of attack is void and closes.

**What is not adopted.** Peak-level aggregation costs .19 J_within in the high tertile and -.094 on the
difference there. It is not a candidate for the frozen layer under any reading. Pooled-neighbour bins are
null, both components rising about .04, which is the depth curve again.
