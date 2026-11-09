# M6 improvement round: binding rules (copy)
Source: /mnt/project-files/regulome_validation/improvement_round_spec.md, sha256 635aa69f79c277ec1f47eb21e18e884a83530f7b7afb286eba38eff398b29fd8 (design thread, 2026-11-09).
This copy holds only the binding parts as relayed to this machine on 2026-11-09 20:41Z (the full file was not reachable here), so its sha256 differs.

1. Objective. The primary objective is ceiling-weighted own-vs-other specificity. Always report the own-KD rank, the B6 effect and the top-5% OR beside it, never the effect alone.
2. Controls. Every table carries three controls: element count with no binding, pooled other-TF binding, and accessibility (ATAC at the same elements). A candidate counts only if it beats the first two.
3. Held-out. The 24 held-out TFs stay sealed until tuning stops; then there is one scoring and one report.
4. NRF1. Report every pooled number both with and without NRF1.
5. Stopping rule. Stop when a candidate's weighted-specificity CI excludes 0 AND it beats element-count on the same statistic, or after two consecutive rounds with no gain. Then score held-out once and freeze as manifest v8.

Added candidate (next round): element-level normalisation of binding against the pooled panel signal at the same element (TF CPM relative to the mean of the other TFs at that element), plain ratio / log-ratio and residual, then summed over the window or links as usual.

Addendum 1 (2026-11-09 ~20:50, before grid6 results): Held-out scoring, one run.
- Candidates: win100k raw @1000 (or grid6's winner if it beats win100k on weighted specificity at 1,000 random sets; named in writing before scoring) and ABC-linked raw sum (B6 list e).
- Controls: element count, pooled other-TF binding, ATAC, at matching windows.
- Statistics: ceiling-weighted own-vs-other specificity with bootstrap CI, own-KD rank, B6 effect, top-5% OR; both arms; every pooled number with and without NRF1.
- One run, one table, committed with the sha of the dev table the candidate was chosen from.
- Freeze decision rule: adopt the window rule if its held-out Arm 2 weighted specificity exceeds both ABC-sum and the element-count control (point estimates, even if CIs overlap); otherwise keep binding-only over ABC links (B6 list e) and state the window variant was not confirmed.
- Methods wording either way: benchmark-tuned on 52 TFs, confirmed on 24 held out.
