# The Perturb-seq programme: what it established, and why every test came out null

Closing summary, 2026-11-10. Four tests were run against the X-Atlas/Orion HEK293T benchmark after the
tuning rule was lifted. All four are null, and they are null for one measurable reason.

## The four results

| Test | Outcome |
|---|---|
| Held-out 24 TFs, gene lists | Candidate -0.14, controls 0.20 to 0.35, all intervals cover zero |
| Direction per TF | Agreement with annotation 25 of 45, independently confirmed at 24 of 45 |
| Full TF by gene matrix | Per-TF correlation within 0.005 of a mis-paired control, every score and variant |
| Per-edge sign layer | Model minus random +0.002 [-0.008, 0.012]; model minus constant sign +0.000 |

The matrix test is the one Yuyang asked for, and it is the strongest of the four because it uses every
gene rather than a list cut at 1,000. Pairing a TF's knockdown with its own score does no better than
pairing it with a different TF's score.

## Why: the test was never powered, and half the reason is ours

Two things have to be true for a TF before the benchmark can say anything about it. Our target list has
to reproduce between halves of our own data, and their knockdown has to be strong enough to move genes.

| | median | TFs clearing 0.2 |
|---|---|---|
| our split-half list agreement (Jaccard) | **0.107** | 14 of 79 |
| their knockdown ceiling | 0.10 | about half |

**Only three TFs clear both bars: E4F1, NME2 and XPA.** The two axes are uncorrelated (r = +0.04), so
being good on one side predicts nothing about the other. A test that could only ever have worked for
three TFs out of seventy-six was not going to produce a panel-wide result, whatever we did to the score.

This is not only a complaint about the benchmark. For sixty-five of seventy-nine TFs, our own target
list does not agree with itself across a split of our own data. Nothing can validate a list that does
not replicate against itself. The limit is per-TF sequencing depth on our side as much as knockdown
strength on theirs.

Only ten TFs have lists that reproduce at 0.3 or better: RBBP4, NFYA, SOX4, XPA, ZNF507, RBBP7, NME2,
E2F3, MBD3, DDIT3.

## The edge-sign layer failed in an informative way

The repressive co-binding flag fires on 81% of edges and the activating flag on 79%, because v8 target
genes sit at busy elements that carry both. Only 1.8% of genes ever departed from the per-TF baseline,
so the layer was barely per-edge at all. This is the neighbourhood-density problem again, in a new
place. A stricter flag, marker enrichment measured against the element's own pooled panel signal, is the
obvious fix and would have to be declared as a separate v2. It has not been run.

## What stands

- The gene layer beats an accessibility-matched control, and the accessibility list is worse than random.
- ABC weighting is harmful to the gene score. The links assign elements to genes and nothing else.
- A plain distance window matches the linked-element machinery, so the model is simpler than it was.
- The TF-TF co-binding edges carry no shared knockdown response, tested against uniform pooling and a
  degree-preserving shuffle.
- Direction ships as annotation, from identity and complex membership, with source and confidence on
  every row. 79 panel TFs annotated; 40 carry a scored direction.

## What to do with it

1. **Make the claims at the resolution the data support.** Panel-wide: binding predicts response better
   than accessibility. Per TF: only where the list reproduces. Name the ten TFs rather than implying
   seventy-nine.
2. **Report the power bound, do not bury it.** Three TFs clearing both bars is a stronger and more
   defensible sentence than any marginal p-value, and it pre-empts the reviewer who asks why the
   validation is thin.
3. **Do not run another test against this benchmark.** The held-out TFs are spent and the power is not
   there. HCT116 only becomes interesting if its knockdown ceilings are measurably better, which should
   be checked before anything is downloaded.
4. **The decision still open with Yuyang** is how prominent the gene layer should be in the paper. This
   result argues for keeping it with the limit stated plainly.
5. **Depth is the actionable lever.** If per-TF target lists matter for the paper, deeper data on a
   chosen subset would move the split-half agreement, and that is the one intervention that would change
   any of these answers.
