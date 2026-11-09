# Held-out result: the Regulome gene layer is not confirmed per-TF

Single scoring of the 24 sealed TFs, job 8702375, committed before reading (20fde83, 46bae6f,
carrying the dev table sha aa0b86ef…). Held-out lists scored against held-out knockdowns only,
1,000 random sets. NRF1 is in the development half, so these numbers are NRF1-free.

## Arm 2, ceiling-weighted specificity

| Row | Wspec [CI] | |
|---|---|---|
| win100k raw (the declared candidate) | -0.14 [-.75, .51] | rank 13 of 24 |
| ABC-sum | -0.01 [-.52, .59] | |
| element count, no binding | 0.20 | control |
| pooled other-TF binding | 0.29 | control |
| accessibility | 0.35 | control |
| win100k + network borrowing | -0.16 | descriptive |

Margin of the candidate over the element-count control: -0.34 [-.88, .20]. Arm 1 is the same picture.

**The pass condition was: Wspec interval excluding zero and a margin over the element-count control
excluding zero. Neither holds. This is the outcome predicted in writing before the table existed.**

## The companion number did not replicate

Own knockdown minus unrelated knockdowns, Arm 2, 100 kb window: 1.09 [.08, 1.97] on the tuning TFs,
0.84 [-.13, 1.75] with NRF1 removed, **-0.14 [-.75, .50] on the held-out TFs**. The dev gap was NRF1
plus tuning noise. This is why it was declared descriptive and barred from converting the test.

## What is frozen

Per the declared freeze rule: keep **binding-only over ABC links** (B6 list e) as the gene layer. The
100 kb window variant is not confirmed and is not adopted, despite leading on the tuning TFs.

Methods wording: *benchmark-tuned on 52 TFs, not confirmed on 24 held out.*

## What stands, and what does not

Stands:
- The gene lists respond more than an accessibility-matched control (B6, three declared reads).
- ABC weighting is actively harmful to the gene score; the links serve only to assign elements to genes.
- A plain distance window is as good as the linked-element machinery, and simpler.

Does not stand:
- Any claim that the lists are TF-specific. A list using no binding at all does as well.
- Any per-TF target call outside NRF1.
- Borrowing strength across the TF-TF network: the edges carry no shared knockdown response.

## Honest reading of the limits

Held-out is 24 TFs, 16 with weight above zero, so this test could not have detected a small effect. It
rules out a large one. The benchmark is also weak on its own terms: median knockdown 51%, replicate
correlation 0.04. The right conclusion is that the gene layer is not demonstrated at per-TF resolution
with the data we have, not that it is disproved.
