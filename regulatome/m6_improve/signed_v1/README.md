# signed_v1: TF sign annotation and signed readouts (SPEC_SIGNED.md)

- tf_sign_annotation.tsv: CANONICAL source A table (76 rows, sha256 7633e0f0d6fa7099d96ee944008757f8c63914e489dac822085c8ffd191f7f0f), transferred from the design thread with sign_readout_result.md (sha256 1e434858b5118ffdf8d969d1aacb77573e429c13083b9fa4e677a2773d72341a); its readout tf_sign_readout.tsv (sha256 b874072d...) stays untracked.
- tf_sign_annotation_rc79.tsv: the machine-built 79-TF table (build_sign_annotation.py, commit 1f1c9ce), kept for comparison; used only for the 3 panel TFs absent from the canonical table (GLYR1, SREBF2, ZNF367), flagged source=rc79.
- annotation_compare.tsv: genuine disagreements only (both tables give a direction, canonical row high/medium, directions differ). annotation_compare_policy.tsv: labelling-policy differences (dual/unknown/low vs a direction, coverage).

## Scoring rule (declared 2026-11-10, before any signed-z result is read for the edge layer)
Anything quantitative uses high- and medium-confidence rows only: the per-TF constant-sign baseline of the edge-sign layer and all agreement counts. Low-confidence rows are carried as hypotheses and never scored. Where rc79 says unknown and canonical says low, the TF is unknown for scoring. In edge-sign rule C, a TF whose canonical row is low confidence, dual or unknown has an UNSIGNED baseline; its elements can still be signed by element state or co-binding.
