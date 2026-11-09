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

Addendum 2 (2026-11-09 ~21:00, after grid6 dev results, before net1b results and before any held-out scoring):

1. Named gene-score candidate: win100k raw @1000. Grid6 win100k resid exceeds raw by 0.03 Wspec, far inside the ~±0.2 Monte Carlo noise, and raw was the standing default before grid6.

2. Margin rule (symmetric, applies to all later choices): a challenger replaces the standing default only if it beats it by more than the Monte Carlo noise of the run. Under this rule the switch clause of Addendum 1 does not fire.

3. NRF1 is in development, so the 24 held-out TFs contain no NRF1. The held-out number is therefore an NRF1-free estimate, and it is the one we publish.

4. Pre-declared expectation: on dev without NRF1, win100k raw's margin over the element-count control was +0.11 [-0.24, 0.43], and ATAC ranked first. The expected held-out result is NO confirmed improvement. A null will be reported as the predicted result.

5. Held-out pass condition (final; no additions later): Arm 2, win100k raw, ceiling-weighted specificity CI excluding 0 AND its margin over the element-count control with CI excluding 0. Everything else in the table is descriptive: ABC-sum, the three controls, Arm 1, own-KD rank, B6 effect, top-5% OR and borrowing candidates. None of these can turn a fail into a pass.
