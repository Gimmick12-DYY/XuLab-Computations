# Giving every TF-to-gene edge a direction

Yuyang, 2026-11-09 21:37: "we want to see the activator/repressor difference, otherwise how can you
call that a regulome?"

Correct, and the unsigned rank correlation is demoted to a baseline. This document is about where a
sign can come from, which of those sources we actually have, and how to test the result honestly.

## 1. Why sign is more tractable than target identity

The held-out test failed to show that we can say *which* genes a TF regulates. Sign is a different
problem and a much easier one. Picking the right 1,000 genes out of 15,000 is a ranking problem at
single-gene resolution. Calling a TF an activator is one bit, estimated by pooling the signed response
over all of its candidate targets at once. A TF whose targets fall by a small amount on average, over a
thousand genes, is detectable even when no individual gene is callable. So the previous null does not
predict failure here.

## 1b. Can ABC supply the sign? No, and here is exactly why

ABC is activity x contact, divided by the sum over the gene's elements. Activity is a measure of how
active an element is, contact is how often it touches the promoter, and both are positive, so the
product is positive. There is no term in it that can come out negative. It is also TF-agnostic: the ABC
value of an element-to-gene link is the same number whichever of the 79 TFs is sitting there. A quantity
that does not know which TF it is describing cannot say whether that TF activates or represses.

This is not a limitation of our ATAC-only version. The published ABC with full H3K27ac has the same
property. ABC answers **which element controls which gene, and how strongly**. That is wiring, and it is
genuinely useful, but it is a different question from direction.

It is also the reason ABC weighting hurt the gene score: a TF-agnostic, gene-level factor is exactly the
neighbourhood-density confound we spent the day chasing. In v8 the ABC links are kept as the wiring and
the weight is dropped.

**But the instinct behind the question is right, and here is the version of it that works.** The
direction lives in the *character of the element*, which is what ABC's activity term would be made from
if we fed it histone marks: an element carrying H3K27ac is active, one carrying H3K27me3 is Polycomb
repressed. ABC collapses that into one positive number and throws the distinction away. We do not need
to rebuild ABC to get it back, and we do not need H3K27me3 data either, because **our own panel already
measures it**: EZH2, MTF2 and SUZ12 mark Polycomb elements, MBD3 and RBBP4/7 mark NuRD elements, CBX1
and CBX5 mark HP1 heterochromatin. An element co-bound by those is a repressive context in our own
cells, measured by our own assay. That is source C below, and it is the part of this that is ours.

So: keep ABC for wiring, take the sign from the element's character, and weight the edge by binding.

## 2. Where a sign can come from

Our binding score is a magnitude and carries no direction. Direction has to come from somewhere else.
Four sources, in the order of how much we already have.

**A. The TF's own identity, one sign per TF.** PRC2 (EZH2, MTF2, SUZ12), NuRD (MBD3, RBBP4, RBBP7,
CHD4), HP1 (CBX1, CBX5) and the KRAB zinc fingers are repressive; NRF1, MAZ, NFYA, E2F3 and the rest of
the promoter set are activating. This is an annotation we can write down today. It is the **baseline
every per-edge model has to beat**, and on its own it is already a signed Regulome, just a coarse one.

**B. Element context.** The same TF at an active promoter and at a Polycomb element is not doing the
same thing. We have accessibility at every element and ChromHMM states were used for the NuRD result,
so a per-element state is available. This turns one sign per TF into a sign per edge.

**C. Co-binding at the element, which is our own asset.** A TF sharing an element with PRC2 or NuRD
members is a candidate repressive edge; the same TF at an element with the activating promoter set is a
candidate activating edge. This is the one source that is ours rather than borrowed, and it is what
would make the Regulome more than a list of TFs with labels attached.

**D. Learned from the signed Perturb-seq z**, now that tuning on the benchmark is allowed, with
leave-TF-out cross-validation so no TF's sign is learned from its own knockdown.

**Open question for Yuyang, which changes the plan if the answer is yes:** does the HITAG/Paired-Tag
data include paired RNA from the same cells? Paired-Tag assays often do. If per-cell RNA exists
alongside the binding, we can measure binding-expression covariation directly, in our own cells, with no
borrowed annotation and no benchmark involved. That would be the strongest route by a wide margin and it
would change the order of everything below.

## 3. The honest evaluation problem, and the way out

If the sign is learned from HEK293T Perturb-seq (source D), then HEK293T Perturb-seq cannot also judge
it, and the 24 held-out TFs are already spent. There is no clean test set left in that dataset.

**HCT116 is untouched.** It was dropped before anything was downloaded, and no number from it has ever
been read. It is now the natural test set for a signed model: learn the sign in HEK293T, predict the
direction of each edge in HCT116, score it once. Reviving it for this is a better reason than the one it
was dropped for, since we would now have a real prediction to test rather than a repeat of a measurement
we already had.

## 4. Test design

Prediction: for TF t and gene g, a sign in {activating, repressing} and a confidence.

Scored at two levels:
- **Per TF**: does the predicted sign match the sign of the mean signed z over that TF's targets?
  Reported as the fraction of TFs called correctly, with the TFs whose own knockdown is too weak to
  give a direction marked uncalled rather than wrong.
- **Per edge**: among the edges where the model predicts a sign confidently, does the signed z split
  the way it should? Compared against the source A baseline, which assigns every edge of a TF the same
  sign. **A per-edge model that does not beat the constant-per-TF sign has not earned the word
  per-edge**, and we say so.

Controls alongside, as before: element count, and sign assigned at random with the same marginal
proportions.

## 5. What to do first

1. Answer the paired-RNA question. It reorders everything.
2. Build source A now, in a table: every one of the 79 TFs, its predicted sign, and the evidence. It is
   a day's work, it is publishable as part of the Regulome, and it is the baseline.
3. Measure the per-TF signed response in the HEK293T data we have already read, as a descriptive
   readout, to see how many TFs give a usable direction at all.
4. Only then decide between sources B, C and D, and whether HCT116 is worth reviving.
