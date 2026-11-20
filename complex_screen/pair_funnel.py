"""Pair funnel 2026-11-20 (read-only on inputs): ALL panel pairs -> correlation candidates -> shared motif -> AF3, with overlaps.
Pairs matched order-independently; ZUFSP = ZUP1.
ALL   candidates_frozen_v1/candidates_all3081.tsv (3,081)
CORR  best-of-5 percentile >= .60 (1,784) and candidate=True (1,732, known pairs removed)
MOTIF pair_motif_composite/motif_tables/pair_shared_motifs_p005.tsv (pair_table_p005.py, 2026-11-12, built on the frozen 3,081/1,732 list):
      a motif shared at p <= .05 in BOTH TFs on BOTH chromosome halves (HOMER known + 2,528-motif DB).
      incl. flagged = any shared motif; specific = >= 1 shared motif not flagged generic (> 10 TFs) or Tn5 background;
      specific_noCB = specific after also dropping Codebook artifact motifs (codebook_artifacts.meme; flagged in the script header but not applied there).
AF3   af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv (97 pairs run at 5 seeds: strong / weak / none); 1-seed tier (af3_1seed_tier) for context.
Also: provenance/overlap of the uploaded pair_motif_lookup_candidates_final.tsv (114 pairs).
usage: pair_funnel.py out_prefix
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; out = sys.argv[1]
norm = lambda t: "ZUP1" if t.strip().upper() == "ZUFSP" else t.strip().upper()
key = lambda a, b: tuple(sorted((norm(a), norm(b))))
A = pd.read_csv(f"{C}/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); A["k"] = [key(a, b) for a, b in zip(A.tf_a, A.tf_b)]
A["candidate"] = A.candidate.astype(str) == "True"; A["corr60"] = A.best_pct >= 0.60
M = pd.read_csv(f"{C}/pair_motif_composite/motif_tables/pair_shared_motifs_p005.tsv", sep="\t"); M["k"] = [key(a, b) for a, b in zip(M.tf_a, M.tf_b)]
cbl = set(l.split()[2].upper() for l in open(f"{X}/motif_analysis/cache/motifdb/codebook_artifacts.meme") if l.startswith("MOTIF ") and len(l.split()) > 2)
def clean(s):
    return [x.strip() for x in str(s).split(";") if x.strip() and x.strip() != "nan"]
M["db_spec_noCB"] = [[x for x in clean(s) if x.upper() not in cbl] for s in M.shared_db_specific]
M["motif_any"] = (M.n_shared_homer + M.n_shared_db) > 0; M["motif_spec"] = M.any_specific_shared.astype(str) == "True"
M["motif_spec_noCB"] = (M.n_shared_homer_specific > 0) | (M.db_spec_noCB.str.len() > 0)
M["top_specific"] = [(clean(h) + clean(d))[0] if (clean(h) + clean(d)) else "" for h, d in zip(M.shared_homer_specific, M.shared_db_specific)]
M["top_any"] = [(clean(h) + clean(d))[0] if (clean(h) + clean(d)) else "" for h, d in zip(M.shared_homer_all, M.shared_db_all)]
T = pd.read_csv(f"{C}/af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv", sep="\t"); T = T[T.iloc[:, 0] == "DESCRIPTIVE"]
T["k"] = [key(*p.split("-")) for p in T.panel_pair]
D = A.merge(M[["k", "motif_any", "motif_spec", "motif_spec_noCB", "top_specific", "top_any"]], on="k", how="left").merge(
    T[["k", "tier_5seed", "seeds_strong", "ipsae_max_5seed", "iptm_top_5seed"]], on="k", how="left")
assert len(D) == 3081 and D.motif_any.notna().all()
D["af3_5"] = D.tier_5seed.fillna("not_run"); D["af3_1"] = D.af3_1seed_tier.fillna("not_run")
S5, W5 = D.af3_5 == "strong", D.af3_5 == "weak"; corr = D.candidate
rows = []
def add(step, n, note=""): rows.append(dict(step=step, n_pairs=int(n), note=note))
add("1 ALL panel pairs", len(D))
add("2a CORR best-of-5 percentile >= 0.60", D.corr60.sum())
add("2b CORR candidates (known pairs removed)", corr.sum(), "candidates_frozen_v1 candidate=True")
for lab, col in (("incl. generic/Tn5-flagged motifs", "motif_any"), ("specific (no generic/Tn5)", "motif_spec"), ("specific, Codebook artifacts also dropped", "motif_spec_noCB")):
    add(f"3 MOTIF {lab}", D[col].sum()); add(f"3 MOTIF {lab} ∩ CORR", (D[col] & corr).sum())
add("4 AF3 5-seed strong", S5.sum(), "of 97 pairs run at 5 seeds"); add("4 AF3 5-seed strong ∩ CORR", (S5 & corr).sum())
add("4 AF3 5-seed weak", W5.sum()); add("4 AF3 5-seed none", (D.af3_5 == "none").sum())
for t in ("strong", "weak", "none"): add(f"4 context: AF3 1-seed {t} (full sweep)", (D.af3_1 == t).sum())
for lab, col in (("specific", "motif_spec"), ("specific, no Codebook", "motif_spec_noCB"), ("incl. flagged", "motif_any")):
    add(f"5 CORR ∩ MOTIF {lab} ∩ AF3 5-seed strong", (corr & D[col] & S5).sum())
    add(f"5 CORR ∩ MOTIF {lab} ∩ AF3 5-seed strong-or-weak", (corr & D[col] & (S5 | W5)).sum())
F = pd.DataFrame(rows); F.to_csv(out + ".tsv", sep="\t", index=False); print(F.to_string(index=False))
fish = []
for scope, m in (("within CORR (1,732)", corr), ("within the 97 run at 5 seeds", corr & (D.af3_5 != "not_run"))):
    for col in ("motif_spec", "motif_spec_noCB"):
        x = D[m]; t = [[int((x[col] & (x.af3_5 == "strong")).sum()), int((x[col] & (x.af3_5 != "strong")).sum())],
                      [int((~x[col] & (x.af3_5 == "strong")).sum()), int((~x[col] & (x.af3_5 != "strong")).sum())]]
        o, p = fisher_exact(t); fish.append(dict(scope=scope, motif=col, motif_AF3strong=t[0][0], motif_notStrong=t[0][1], noMotif_AF3strong=t[1][0], noMotif_notStrong=t[1][1], odds_ratio=round(o, 3), fisher_p=round(p, 4)))
Fi = pd.DataFrame(fish); Fi.to_csv(out + "_fisher.tsv", sep="\t", index=False); print(Fi.to_string(index=False))
tri = D[corr & D.motif_spec & (S5 | W5)].copy().sort_values(["af3_5", "best_pct"], ascending=[True, False])
tri = tri[["pair", "best_pct", "best_matrix", "af3_5", "seeds_strong", "ipsae_max_5seed", "iptm_top_5seed", "motif_spec_noCB", "top_specific", "top_any"]]
tri.to_csv(out + "_triple.tsv", sep="\t", index=False); print(tri.to_string(index=False))
# uploaded table
U = pd.read_csv(f"{C}/pair_motif_lookup/uploaded_candidates_final_20261120.tsv", sep="\t"); U["k"] = [key(a, b) for a, b in zip(U.a, U.b)]
Up = U.groupby("k").af3_tier.first().reset_index(); Up = Up.merge(D[["k", "pair", "candidate", "known_reason", "best_pct", "af3_5", "af3_1"]], on="k", how="left")
Up.to_csv(out + "_uploaded114.tsv", sep="\t", index=False)
print("uploaded pairs", len(Up), "| in 3081:", Up.pair.notna().sum(), "| candidate True:", int((Up.candidate == True).sum()))
print("not candidate:", Up[Up.candidate != True][["k", "known_reason", "best_pct"]].to_string(index=False))
print("5-seed tiers among uploaded:", Up.af3_5.value_counts().to_dict()); print("1-seed tiers among uploaded not in 97:", Up[Up.af3_5 == "not_run"].af3_1.value_counts().to_dict())
for t in ("strong", "weak", "none"): print(f"uploaded 5-seed {t}:", sorted("-".join(k) for k in Up[Up.af3_5 == t].k))
for t in ("strong", "weak"): print(f"uploaded not in 97, 1-seed {t}:", sorted("-".join(k) for k in Up[(Up.af3_5 == "not_run") & (Up.af3_1 == t)].k))
print("uploaded, no AF3 at all:", sorted("-".join(k) for k in Up[(Up.af3_5 == "not_run") & (Up.af3_1 == "not_run")].k))
print("uploaded af3_tier vs 5-seed:", pd.crosstab(Up.af3_tier, Up.af3_5).to_string())
miss = D[S5 & ~D.k.isin(set(Up.k))]; print("5-seed strong absent from uploaded:", len(miss), sorted(miss.pair))
