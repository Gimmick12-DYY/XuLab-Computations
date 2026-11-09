# M6 improvement round: binding rules (copy)
Source: /mnt/project-files/regulome_validation/improvement_round_spec.md, sha256 635aa69f79c277ec1f47eb21e18e884a83530f7b7afb286eba38eff398b29fd8 (design thread, 2026-11-09).
This copy holds only the binding parts as relayed to this machine on 2026-11-09 20:41Z (the full file was not reachable here), so its sha256 differs.

1. Objective. The primary objective is ceiling-weighted own-vs-other specificity. Always report the own-KD rank, the B6 effect and the top-5% OR beside it, never the effect alone.
2. Controls. Every table carries three controls: element count with no binding, pooled other-TF binding, and accessibility (ATAC at the same elements). A candidate counts only if it beats the first two.
3. Held-out. The 24 held-out TFs stay sealed until tuning stops; then there is one scoring and one report.
4. NRF1. Report every pooled number both with and without NRF1.
5. Stopping rule. Stop when a candidate's weighted-specificity CI excludes 0 AND it beats element-count on the same statistic, or after two consecutive rounds with no gain. Then score held-out once and freeze as manifest v8.

Added candidate (next round): element-level normalisation of binding against the pooled panel signal at the same element (TF CPM relative to the mean of the other TFs at that element), plain ratio / log-ratio and residual, then summed over the window or links as usual.
