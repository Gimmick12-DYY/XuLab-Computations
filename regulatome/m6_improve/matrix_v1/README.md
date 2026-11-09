# matrix_v1: full-matrix comparison of TF x gene scores vs Perturb-seq z (DESCRIPTIVE ONLY)
Spec: ../SPEC_MATRIX.md (sha256 f14eaf58dddb1fe2503dd041cd4cee631b0999695d1b1181c68401a089d83e39). Script matrix_compare.py, job 8712344 (commit d7ecf53). Held-out set is spent; nothing here is a test.

## Correction 1 (declared before output read), 2026-11-09 ~21:43 local commit
Job 8712344 has finished; none of its output files or log lines have been read when this section was written.
Double-centring z removes the gene main effect, and E (linked elements per gene) has no TF index, so E's correlation with DC z averages ~0 across TFs by construction; "S beats E" on the DC tables would be partly arithmetic. Therefore:
- For the DC variants (absz_DC, signed_DC) and the partial variants, the PRIMARY control is mis-paired S: TF t's z is correlated with another TF's S, using a permutation of TF labels on S with no fixed points (within arm and score), 5 draws with seed 20261109 + k (k = 0..4). Reported: min / median / max of rho_mispaired per TF, and delta = rho_S - median(rho_mispaired), per TF and in the summary. For DC variants the mis-paired S is the DC S row of the other TF; for partial variants it is the other TF's S with the same rank-residualisation on E and the gene's mean |z|.
- Uncentred E vs DC z stays as the SECONDARY control (already in matrix_compare.py outputs).
- RAW variants keep E as the primary control.
- Implemented in matrix_mispaired.py (same inputs, same universe and loader), run as its own job; outputs matrix_v1/mispaired_per_tf_DESCRIPTIVE.tsv and mispaired_summary_DESCRIPTIVE.tsv.
