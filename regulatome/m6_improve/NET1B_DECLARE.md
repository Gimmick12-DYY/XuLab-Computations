# net1b declared choices (2026-11-09 ~20:58, before any net1/net1b benchmark output was read)
net1 (job 8700032) was cancelled after 45 s, during setup; it wrote graph.tsv and alpha_t.tsv only (binding data), no scores. The only log line seen was the binding-only setup line "tau2 0.1272 alpha_t median 0.5".
Fixes from the design thread, folded into net1b.py:
1. Reliability alpha_t: r_SB floored at 0.05 (so sigma2 = 1/r_SB - 1 <= 19); alpha_t = sigma2/(sigma2 + tau2), tau2 = panel median sigma2, capped at 0.7 (the sweep maximum). The alpha_t distribution and the number of TFs at the cap are printed and saved (alpha_t.tsv).
2. Primary three-way contrast = kNN5-smoothed lists: own KD vs kNN5 partners (real graph, with a dev KD) vs all other dev KDs. Dense W is secondary; for dense W, non-partners = TFs with W(t,u) < 0.5 (W = depth-matched Pearson pct x (1 + support)), reported as z_dense_nonpartner; the share of each TF's dense smoothing mass carried by non-partners is saved in dense_nonpartner_mass.tsv.
3. kNN5 degree-preserving double-edge swap: 5 draws (shufknn0-4); the summary reports min / median / max across draws. Dense weight permutation stays at 2 draws.
Everything else as net1.py / SPEC_NETWORK.md: dev TFs only, 1,000 random sets, base scores win100k raw and ABC-linked raw sum.
