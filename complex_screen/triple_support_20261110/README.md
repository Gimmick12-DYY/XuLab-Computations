# triple_support_20261110

`motif_descriptive_v2.tsv` (current; built by `build_motif_descriptive_v2.py`): one row per pair in
`../interface_audit_20261110/triple_support_flagged_v2.tsv`, with its AF3 tier, number of matrices at >= .80, Boltz calls and, for each of the
four motif-profile methods (peak500, peak2000, seqN20, seqN3k), the top 3 shared motif names and the matched percentile. Motif values come from
`../pair_motif_composite/profile/novel_pairs_motif_profile.tsv` (commit a987c98), joined on the sorted pair name; the 3 pairs absent from it
(MTF2-RBBP4, RBBP4-RBBP7, EZH2-MTF2) are filled from `all_pairs_motif_profile.tsv` in the same folder. `source_table` records which table each row
came from. `motif_descriptive.tsv` (v1, `build_motif_descriptive.py`) is kept unchanged, with those 3 pairs as NA and its original status wording.

The motif values are description only (`motif_status` = "descriptive only; see README" on every row). They do not affect ranking,
which stays correlation + AF3.

Two comparisons. (1) Against random pairs (GC/CpG-matched test, 10 Nov), novel pairs pass at the random-pair rate: not detected in our data. (2) Against correlation-matched pairs (the profile's matched percentile), the comparison is uninformative, not negative, at high co-binding correlation. An uncontaminated negative set cannot be built where unrecognised complexes would sit.

Whether this reflects depth or the absence of a shared motif has not been tested. DDIT3-RBBP4's SP/KLF/NFY hit is promoter-concentrated shared signal, not pair grammar.

MTF2-RBBP4 is a known PRC2 pair (positive control) although triple_support.tsv has reference = False.

`matched_pct_<method>` (= `conc_<method>_matched_pct`, profile_score.py at a987c98) is the percentile of the pair's concordance, the Spearman
correlation of the two TFs' scores over all 2,921 motifs. It is computed against 200 random draws from panel pairs in the same cell-count tertile
pair (dbin) and the same co-binding correlation percentile bin (<= .6, .6-.8, .8-.9, > .9; pbin), falling back to the tertile pair alone if fewer
than 20 such pairs exist. It is not matched on promoter fraction. It describes the whole motif profile of the pair, not the motifs named in the
top-3 columns.

A few words dominate the top-3 lists across pairs and methods (PKNX1, ZNF460, ZNF575, ZNF160, FIZ1; ZNF460 is in a top 3 for 34 of 128 pairs). They are panel-wide words, not pair-specific.

v3 uses the masked rebuilt profile (3bfdaa3); v1/v2 numbers are superseded.
