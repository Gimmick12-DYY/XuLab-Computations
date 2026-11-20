"""Pair funnel with Yuyang's motif table as the MOTIF set (2026-11-20): pair_motif_lookup_candidates_final.tsv (114 pairs; uploaded copy
complex_screen/pair_motif_lookup/uploaded_candidates_final_20261120.tsv). MOTIF = pairs in that table; MOTIF_informative = pairs with >= 1 row
tier == pair_informative. ALL / CORR / AF3 as pair_funnel.py. Pairs matched order-independently, ZUFSP = ZUP1. usage: pair_funnel_uploaded.py out_prefix"""
import sys
import pandas as pd
from scipy.stats import fisher_exact
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; out = sys.argv[1]
norm = lambda t: "ZUP1" if t.strip().upper() == "ZUFSP" else t.strip().upper(); key = lambda a, b: tuple(sorted((norm(a), norm(b))))
A = pd.read_csv(f"{C}/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); A["k"] = [key(a, b) for a, b in zip(A.tf_a, A.tf_b)]; A["candidate"] = A.candidate.astype(str) == "True"
T = pd.read_csv(f"{C}/af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv", sep="\t"); T = T[T.iloc[:, 0] == "DESCRIPTIVE"]; T["k"] = [key(*p.split("-")) for p in T.panel_pair]
U = pd.read_csv(f"{C}/pair_motif_lookup/uploaded_candidates_final_20261120.tsv", sep="\t"); U["k"] = [key(a, b) for a, b in zip(U.a, U.b)]
inf = set(U[U.tier == "pair_informative"].k); top = U[U.tier == "pair_informative"].sort_values(["k", "rank_in_pair"]).groupby("k").first()
D = A.merge(T[["k", "tier_5seed", "seeds_strong", "ipsae_max_5seed", "iptm_top_5seed"]], on="k", how="left"); D["af3_5"] = D.tier_5seed.fillna("not_run")
D["motif"] = D.k.isin(set(U.k)); D["motif_inf"] = D.k.isin(inf); corr, S5, W5 = D.candidate, D.af3_5 == "strong", D.af3_5 == "weak"
rows = [("ALL panel pairs", len(D)), ("CORR >= 0.60", int((D.best_pct >= .6).sum())), ("CORR candidates (known removed)", int(corr.sum())),
        ("MOTIF (your table)", int(D.motif.sum())), ("MOTIF pair_informative", int(D.motif_inf.sum())), ("MOTIF ∩ CORR", int((D.motif & corr).sum())),
        ("AF3 5-seed strong", int(S5.sum())), ("AF3 strong ∩ CORR", int((S5 & corr).sum())), ("AF3 strong ∩ MOTIF", int((S5 & D.motif).sum())),
        ("CORR ∩ MOTIF ∩ AF3 strong", int((corr & D.motif & S5).sum())), ("CORR ∩ MOTIF ∩ AF3 strong-or-weak", int((corr & D.motif & (S5 | W5)).sum())),
        ("CORR ∩ MOTIF run at 5 seeds", int((corr & D.motif & (D.af3_5 != "not_run")).sum()))]
F = pd.DataFrame(rows, columns=["step", "n_pairs"]); F.to_csv(out + ".tsv", sep="\t", index=False); print(F.to_string(index=False))
fs = []
for scope, m in (("within CORR (1,732)", corr), ("within the 97 run at 5 seeds", corr & (D.af3_5 != "not_run"))):
    x = D[m]; t = [[int((x.motif & (x.af3_5 == "strong")).sum()), int((x.motif & (x.af3_5 != "strong")).sum())], [int((~x.motif & (x.af3_5 == "strong")).sum()), int((~x.motif & (x.af3_5 != "strong")).sum())]]
    o, p = fisher_exact(t); fs.append(dict(scope=scope, motif_strong=t[0][0], motif_not=t[0][1], nomotif_strong=t[1][0], nomotif_not=t[1][1], odds_ratio=round(o, 3), p=round(p, 4)))
Fi = pd.DataFrame(fs); Fi.to_csv(out + "_fisher.tsv", sep="\t", index=False); print(Fi.to_string(index=False))
tri = D[corr & D.motif & S5][["pair", "k", "best_pct", "best_matrix", "seeds_strong", "ipsae_max_5seed", "iptm_top_5seed"]].copy()
tri["top_motif"] = [top.consensus.get(k, "") + " | " + str(top.best_known_match.get(k, "")) if k in top.index else "" for k in tri.k]; tri["top_set"] = [top.set.get(k, "") for k in tri.k]
tri.drop(columns="k").to_csv(out + "_triple.tsv", sep="\t", index=False); print(tri.drop(columns="k").to_string(index=False))
print("AF3 strong NOT in your table:", sorted(D[S5 & ~D.motif].pair)); print("your pairs at 5 seeds:", D[D.motif & (D.af3_5 != "not_run")][["pair", "af3_5", "candidate"]].to_string(index=False))
