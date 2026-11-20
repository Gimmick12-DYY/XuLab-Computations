"""Pair funnel v2 (2026-11-20): ALL 3,081 -> candidates_frozen_v2 (best of 5 >= .75, known removed; 1,198) -> specific shared motif with Codebook
artifacts dropped (pair_shared_motifs_v2_all3081.tsv, pair_motifs_v2.py) -> AF3 5-seed (rescore_5seed/tiers_5seed.tsv) -> triple support.
Fisher: motif vs AF3 5-seed strong within the 1,198 (and within the v2 candidates run at 5 seeds). AF3 was run only on >= .60 pairs.
usage: pair_funnel_v2.py out_prefix
"""
import sys
import pandas as pd
from scipy.stats import fisher_exact
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; out = sys.argv[1]
D = pd.read_csv(f"{C}/pair_motif_composite/motif_tables/pair_shared_motifs_v2_all3081.tsv", sep="\t")
T = pd.read_csv(f"{C}/af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv", sep="\t"); T = T[T.iloc[:, 0] == "DESCRIPTIVE"]
T["pair"] = T.panel_pair.str.replace("ZUP1", "ZUFSP"); T["pair"] = ["-".join(sorted(p.split("-"))) for p in T.pair]
D = D.merge(T[["pair", "seeds_strong", "ipsae_max_5seed", "iptm_top_5seed"]], on="pair", how="left"); assert len(D) == 3081
assert D.seeds_strong.notna().sum() == len(T), "5-seed pairs not all matched"
D["candidate_v2"] = D.candidate_v2.astype(str) == "True"; cand, mot = D.candidate_v2, D.motif_specific_noCB.astype(str) == "True"
S5, W5 = D.af3_5seed == "strong", D.af3_5seed == "weak"
rows = []
add = lambda s, n, note="": rows.append(dict(step=s, n_pairs=int(n), note=note))
add("1 ALL panel pairs", len(D)); add("2a best-of-5 percentile >= 0.75", (D.correlation_pct >= 0.75).sum()); add("2b candidates_frozen_v2 (known removed)", cand.sum())
add("3 MOTIF specific, Codebook artifacts dropped (all pairs)", mot.sum()); add("3 MOTIF ∩ candidates", (mot & cand).sum())
add("3 (MOTIF specific, Codebook kept) ∩ candidates", ((D.motif_specific_inclCB.astype(str) == "True") & cand).sum())
add("4 AF3 5-seed run, among candidates", (cand & (D.af3_5seed != "not_run")).sum(), "AF3 run only on >= .60 pairs")
for t in ("strong", "weak", "none"): add(f"4 AF3 5-seed {t} ∩ candidates", (cand & (D.af3_5seed == t)).sum())
add("5 candidates ∩ MOTIF ∩ AF3 strong (triple support)", (cand & mot & S5).sum()); add("5 candidates ∩ MOTIF ∩ AF3 strong-or-weak", (cand & mot & (S5 | W5)).sum())
F = pd.DataFrame(rows); F.to_csv(out + ".tsv", sep="\t", index=False); print(F.to_string(index=False))
fish = []
for scope, m in (("within candidates_frozen_v2 (1,198)", cand), ("within v2 candidates run at 5 seeds", cand & (D.af3_5seed != "not_run"))):
    for col in ("motif_specific_noCB", "motif_specific_inclCB"):
        x = D[m]; mm = x[col].astype(str) == "True"; s = x.af3_5seed == "strong"
        t = [[int((mm & s).sum()), int((mm & ~s).sum())], [int((~mm & s).sum()), int((~mm & ~s).sum())]]; o, p = fisher_exact(t)
        fish.append(dict(scope=scope, motif=col, motif_AF3strong=t[0][0], motif_notStrong=t[0][1], noMotif_AF3strong=t[1][0], noMotif_notStrong=t[1][1], odds_ratio=round(o, 3), fisher_p=round(p, 4)))
Fi = pd.DataFrame(fish); Fi.to_csv(out + "_fisher.tsv", sep="\t", index=False); print(Fi.to_string(index=False))
tri = D[cand & mot & (S5 | W5)].sort_values(["af3_5seed", "correlation_pct"], ascending=[True, False])
tri = tri[["pair", "correlation_pct", "best_matrix", "af3_5seed", "seeds_strong", "ipsae_max_5seed", "iptm_top_5seed", "n_shared_homer_specific_noCB", "n_shared_db_specific_noCB", "top_specific_noCB"]]
tri.to_csv(out + "_triple.tsv", sep="\t", index=False); print(tri.to_string(index=False))
print("AF3 strong in candidates without motif:", sorted(D[cand & S5 & ~mot].pair))
print("AF3 strong not in v2 candidates:", D[S5 & ~cand][["pair", "correlation_pct"]].to_string(index=False))
for p in ("RBBP4-ZNF644", "ADNP-MTF2", "DR1-NFYA"):
    r = D[D.pair == p].iloc[0]; print(p, "candidate_v2", r.candidate_v2, "pct", r.correlation_pct, r.best_matrix, "motif", r.motif_specific_noCB, r.top_specific_noCB, "AF3", r.af3_5seed, r.seeds_strong)
