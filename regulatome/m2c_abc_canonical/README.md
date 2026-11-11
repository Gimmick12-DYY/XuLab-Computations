# m2c_abc_canonical: the literature ABC recipe, and overlap of ABC-called TF targets with HEK293T Perturb-seq

Spec: `SPEC.md`. Comparison with our earlier ABC builds: `AUDIT.md`. Code: EngreitzLab ABC-Enhancer-Gene-Prediction v1.1.2 (commit e2a88d2, the latest
tagged release), run through its own Snakefile with `config/config.yaml` (repo defaults; declared changes only: chrom sizes, GENCODE v46 protein-coding genes
via alt_genes / alt_TSS, threshold 0.02).

## Build
`dump_hic_fixed.sbatch` (WT Hi-C 5 kb, ICE-balanced, to `inputs/hic5kb_fixed`) -> `abc_canonical.sbatch` (biosamples HEK293T_hicfixed, HEK293T_powerlaw) ->
`overlap.sbatch <biosample>` (`tf_targets_overlap.py`, `pooled.py`; outputs `overlap/<biosample>/per_tf.tsv`, `pooled.tsv`).
Activity sqrt(qnorm ATAC x qnorm H3K27ac): ATAC = m2_abc omniatac_merged tagAlign; H3K27ac = m2b_abc_k27ac merged SRX18002015/16.

**Hi-C bug found and fixed (2026-11-11).** `m2_abc/dump_hic.sh` printed awk `$9`; `cooler dump --join --balanced` has 8 columns, so every file in
`m2_abc/inputs/hic5kb` had an empty value column. ABC did not stop: it set contact to 0 / fell back to the power-law. All earlier "Hi-C" arms never used Hi-C
(m2_abc Predictions_hic5kb, m2b followups k27_hic5kb / oldhic / newhic, the first m2c arm, kept as `results/HEK293T_hic_INVALID_emptyHiC`). Power-law runs are unaffected.
In the fixed arm hic_contact differs from powerlaw_contact on 98% of non-self links (Spearman 0.87). Values are ICE (the mcool has no KR vector) and pixels are limited to 5 Mb,
so ABC's row-mean rescaling sees a mean row sum of about 0.82 (constant within a chromosome).

## Canonical link files (reuse these)
- Hi-C (primary): `regulatome/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv`
- Power-law (sensitivity): `regulatome/m2c_abc_canonical/results/HEK293T_powerlaw/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv`
- Candidates / activity: `results/<biosample>/Neighborhoods/EnhancerList.txt`; all candidate-gene pairs: `Predictions/EnhancerPredictionsAllPutative.tsv.gz` (+ NonExpressedGenes).

## Build stats
MACS2 peaks 634,488; candidates 142,706 (median 500 bp; 20,100 promoter elements).
| | Hi-C 5 kb (primary) | power-law only (sensitivity) |
|---|---|---|
| links at ABC >= 0.02 | 109,421 (22,066 self-promoter) | 83,549 (22,066) |
| non-self links | 87,355 | 61,483 |
| genes with a non-self link | 19,457 | 18,206 |
| non-self links per gene, median / mean | 4 / 4.49 | 3 / 3.38 |
| median distance of non-self links | 92.7 kb | 21.4 kb |
| links > 100 kb | 48.7% | 18.6% |

The two arms share 68,718 links (Jaccard 0.55); ABC Spearman on shared links 0.91. m2_abc power-law had 104,071 links.

## Overlap with Perturb-seq (fixed procedure, no tuning; SPEC step 4)
Universe 11,230 genes. Arm 2: 66 TFs tested, 10 with < 20 targets (untestable), 3 without a knockdown. Arm 1: 27 tested, 3 untestable. Held-out TFs were used once before (m6 heldout_v1); no selection was made here.
Arm 2, all TFs, Hi-C: median OR 1.012, OR > 1 in 59% (sign p 0.18), matched-element null median OR 1.004, above null 50% (p 1.0), minus ATAC top-N median -0.012, above ATAC 47% (p 0.71), BH <= 0.10: 0 TFs. Without NRF1: 1.009, 59%, p 0.21.
Arm 2, power-law: median OR 1.042, OR > 1 in 62% (p 0.064), above null 59% (p 0.18), above ATAC 41% (p 0.21), BH <= 0.10: NRF1 (OR 1.88) and E4F1 (OR 1.33; below its ATAC-top-N 1.55).
Dev vs held-out, Arm 2, Hi-C: dev median OR 1.026 (67% > 1, p 0.036), held-out 0.980 (43%, p 0.66); neither beats the matched null (p 0.45, 0.38).
Arm 1: all CIs/tests null (Hi-C median OR 1.023, 63% > 1, p 0.25).
**Verdict:** the literature ABC recipe with peak-based TF targets does not give TF-specific enrichment for knockdown responders. Targets overlap responders at about the rate of any accessible, H3K27ac-matched element set, and no better than the most accessible elements.
