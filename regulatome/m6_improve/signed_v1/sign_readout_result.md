# Can the benchmark tell us the direction of each TF? Not at per-TF resolution

Descriptive readout, 2026-11-10, from data already read. The 24 held-out TFs are spent, so nothing here
is a test. Paired RNA does not exist for HITAG, confirmed by Yuyang, so that route is closed.

Files: `tf_sign_annotation.tsv` (source A, all 76 panel TFs with evidence and confidence) and
`tf_sign_readout.tsv` (the per-TF numbers behind everything below).

## What was measured

The B6 tables already carry a signed statistic per TF: `-mean(R over the target list) / SD(R over the
TF's universe)`, where R is the double-centred knockdown response. Positive means the TF's targets fall
when it is removed, so the TF was activating them. Negative means they rise, so it was repressing them.
Scored on the v8 list, with the accessibility-only list as the control.

## The direction does not match what the TFs are known to do

| Subset | agree with annotation |
|---|---|
| all TFs with an annotated direction | 25 of 45 |
| high-confidence annotation only | 11 of 22 |
| high confidence and a usable knockdown | 7 of 14 |

Chance is half. There is no agreement. NRF1 (+0.29) and ZNF143 (+0.14) come out as activators, which is
right, but DNMT1 (+0.14), SETDB1 (+0.09) and PCGF6 (+0.13) come out as activators too, and all three are
repressive machinery.

## But the signal is not nothing, and it is specific to the binding lists

| List scored | median absolute signed value | TFs above 0.1 |
|---|---|---|
| v8 binding-only | 0.13 | 47 of 76 |
| core, binding x ABC | 0.06 | 22 |
| accessibility only (control) | **0.03** | **0 of 76** |

So the target lists show per-TF directional shifts that an accessibility-matched list does not produce
at all, and the shift is reproducible across the two cell sets for 21 of the 30 TFs measured in both.
Something real is moving. It just does not line up with activator and repressor identity.

The pooled gap points the right way and does not reach significance:

| | activator minus repressor | permutation p |
|---|---|---|
| all cells, v8 | +0.063 [-0.050, +0.182] | 0.23 |
| strong-knockdown cells, v8 | +0.113 [-0.004, +0.236] | 0.08 |
| all cells, control list | +0.001 [-0.030, +0.032] | 0.94 |

Both arms lean the same way and the control sits on zero, which is a hint rather than a finding.

## Reading

This is the same verdict as for target identity, from the same cause. The benchmark's knockdowns are
weak, its replicates correlate at 0.04, and it cannot adjudicate direction per TF any more than it could
adjudicate which genes are targets. The annotation is not shown to be wrong; the benchmark is shown to
be unable to judge it.

## What follows

1. **Ship the direction as annotation, not as a measurement.** `tf_sign_annotation.tsv` is a signed
   Regulome today. Label it by source and confidence, and do not claim benchmark support for it.
2. **Add the element-context and co-binding refinement anyway** (sources B and C), because they make the
   sign per edge rather than per TF and they stand on our own data. They should be presented the same
   way, as annotation.
3. **Do not run another direction test against HEK293T.** If a direction test is wanted, HCT116 is
   untouched, but check its knockdown ceilings first: if they are no better than these, it will not
   settle anything either and should not be spent.
4. Treat the pooled gap as a question for better data, not as support.
