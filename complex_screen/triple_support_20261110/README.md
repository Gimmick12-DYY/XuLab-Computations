# triple_support_20261110

`motif_descriptive.tsv`: one row per pair in `../interface_audit_20261110/triple_support_flagged_v2.tsv`, with its AF3 tier, number of matrices at
>= .80, Boltz calls, and, for each of the four motif-profile methods (peak500, peak2000, seqN20, seqN3k), the top 3 shared motif names and the
depth-matched percentile. Motif values come from `../pair_motif_composite/profile/novel_pairs_motif_profile.tsv` (commit a987c98), joined on the
sorted pair name. Built by `build_motif_descriptive.py`.

The motif values are description only (`motif_status` = "descriptive only; not detectable at this depth" on every row). They do not affect
ranking, which stays correlation + AF3. A pair absent from the profile table has NA in the motif columns.
