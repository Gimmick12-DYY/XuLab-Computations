Group-level motif test (descriptive leg; no per-pair call may rest on it).
(1) First establish what the profile's matched percentile is computed against. If it is NOT random pairs matched on depth and promoter fraction, stop and report; nothing below runs.
(2) Known set = every pair in all_pairs_motif_profile.tsv with reference = True, PLUS MTF2-RBBP4 and RBBP4-ZIC2 (literature positives), fixed now. Its full membership is listed in this file before any percentile is read.
(3) For each of the 4 methods (peak500, peak2000, seqN20, seqN3k): one-sided Wilcoxon signed-rank of the known set's percentiles vs .5, plus the count above .5 with a binomial p. Holm over the 4 methods x 2 sets (known, AF3-strong).
(4) Repeat excluding any pair containing RBBP4 or RBBP7.
(5) Repeat on the AF3-strong set (af3 tier = strong in the triple-support table; novel pairs only).
(6) CO-BINDING CONFOUND: known complexes are co-bound, so they share peaks and therefore share motif content trivially. Compare the known set's percentiles with non-known pairs matched on co-binding correlation: for each known pair, the 5 nearest non-known pairs by Pearson co-binding correlation. Use a one-sided Mann-Whitney test. A group shift counts as 'shared motif beyond co-binding' ONLY if it also beats this correlation-matched comparison.
Predictions (design thread, recorded before the run): the known set shifts and holds without the WD40 pairs; the AF3-strong set does not. Either way the motif column stays descriptive.

Known set membership (step 1), fixed before any percentile is read for this test. all_pairs_motif_profile.tsv has no `reference` column, so
reference = True is taken from triple_support_20261109/triple_support.tsv (the 5 pairs hard-coded as REF in build_triple.py):
CTCF-STAG2, EZH2-MTF2, MAZ-ZNF143, MLX-SREBF2, RBBP4-RBBP7; plus the literature positives MTF2-RBBP4, RBBP4-ZIC2. Total 7.

Condition (1) unmet: matched on depth tertile + correlation bin, not promoter fraction. Stopped before any test.
