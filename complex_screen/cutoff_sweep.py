"""Candidate-cutoff sweep 2026-11-20 (read-only; candidates_frozen_v1 unchanged).
Cutoffs 0.60/0.70/0.80/0.90 on the per-set percentile of TF-TF Pearson r (5 ChromHMM sets: all, non-quiescent, TSS, enhancer, repressive).
Per set: pairs >= cutoff (of 3,081) and after known-pair removal (same exclusion as the frozen list: known=True).
Combined frozen rule: best of 5 >= cutoff (must give 1,784 / 1,732 at 0.60). Strict reference: >= cutoff in all 5 sets.
AF3 5-seed strong (24, rescore_5seed/tiers_5seed.tsv) and triple-support (18, funnel_20261120/pair_funnel_20261120_triple.tsv) surviving;
MOTIF = specific shared motif with Codebook-artifact matches dropped (same code as pair_funnel.py); Fisher MOTIF vs AF3 strong within candidates.
AF3 was only run on >= 0.60 pairs, so AF3 counts can only shrink with the cutoff.
usage: cutoff_sweep.py out_prefix
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; out = sys.argv[1]
norm = lambda t: "ZUP1" if t.strip().upper() == "ZUFSP" else t.strip().upper()
key = lambda a, b: tuple(sorted((norm(a), norm(b))))
A = pd.read_csv(f"{C}/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); A["k"] = [key(a, b) for a, b in zip(A.tf_a, A.tf_b)]
A["candidate"] = A.candidate.astype(str) == "True"; A["known"] = A.known.astype(str) == "True"
M = pd.read_csv(f"{C}/pair_motif_composite/motif_tables/pair_shared_motifs_p005.tsv", sep="\t"); M["k"] = [key(a, b) for a, b in zip(M.tf_a, M.tf_b)]
from scipy.stats import norm as _n
FU = f"{C}/funnel_20261120"; PM = f"{C}/pair_motif_composite"
artH = set(pd.read_csv(f"{FU}/tomtom_homer_known.tsv", sep="\t").iloc[:, 0].str.replace(r"__\d+$", "", regex=True))
artD = set(pd.read_csv(f"{FU}/tomtom_known.tsv", sep="\t").iloc[:, 0])
alt = {}
for l in open(f"{PM}/db/known.meme"):
    if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
Fd = pd.read_csv(f"{PM}/motif_valid/scores_fisher_disc.tsv", sep="\t", index_col=0); Fv = pd.read_csv(f"{PM}/motif_valid/scores_fisher_val.tsv", sep="\t", index_col=0); zc = _n.isf(0.05)
dp = {norm(t): set(Fd.index[(Fd[t] >= zc) & (Fv[t] >= zc)]) for t in Fd.columns}; dgen = pd.Series([m for t in dp for m in dp[t]]).value_counts()
def clean(s):
    return [x.strip() for x in str(s).split(";") if x.strip() and x.strip() != "nan"]
def db_spec(a, b, drop_art):
    ms = [m for m in dp.get(norm(a), set()) & dp.get(norm(b), set()) if dgen.get(m, 0) <= 10 and not (drop_art and m in artD)]
    return sorted(ms, key=lambda m: -float(Fd.at[m, a] + Fd.at[m, b]))
M["h_spec"] = [clean(s) for s in M.shared_homer_specific]; M["h_spec_noCB"] = [[x for x in h if x not in artH] for h in M.h_spec]
M["d_spec"] = [db_spec(a, b, False) for a, b in zip(M.tf_a, M.tf_b)]; M["d_spec_noCB"] = [db_spec(a, b, True) for a, b in zip(M.tf_a, M.tf_b)]
assert (M.d_spec.str.len() == M.n_shared_db_specific).all(), "DB recompute does not reproduce the table"
M["motif_any"] = (M.n_shared_homer + M.n_shared_db) > 0; M["motif_spec"] = M.any_specific_shared.astype(str) == "True"
M["motif_spec_noCB"] = (M.h_spec_noCB.str.len() + M.d_spec_noCB.str.len()) > 0
M["top_specific"] = [(h + [alt.get(m, m) for m in d])[0] if (h + d) else "" for h, d in zip(M.h_spec_noCB, M.d_spec_noCB)]
M["top_specific_incl_CB"] = [(h + [alt.get(m, m) for m in d])[0] if (h + d) else "" for h, d in zip(M.h_spec, M.d_spec)]
M["n_specific_noCB"] = M.h_spec_noCB.str.len() + M.d_spec_noCB.str.len()
D = A.merge(M[["k", "motif_spec_noCB"]], on="k", how="left"); assert len(D) == 3081 and D.motif_spec_noCB.notna().all()
T = pd.read_csv(f"{C}/af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv", sep="\t"); T = T[T.iloc[:, 0] == "DESCRIPTIVE"]
S5 = {key(*p.split("-")) for p in T[T.tier_5seed == "strong"].panel_pair}; assert len(S5) == 24
TR = pd.read_csv(f"{C}/funnel_20261120/pair_funnel_20261120_triple.tsv", sep="\t"); TR = {key(*p.split("-")) for p in TR[TR.af3_5 == "strong"].pair}; assert len(TR) == 18
D["s5"] = D.k.isin(S5); D["tri"] = D.k.isin(TR)
SETS = [("All", "pct_all"), ("Non-quiescent", "pct_noquies"), ("TSS", "pct_tss"), ("Enhancer", "pct_enh"), ("Repressive", "pct_repr")]
assert np.allclose(D[[c for _, c in SETS]].max(axis=1), D.best_pct, atol=1e-4)
excl = D.known
assert ((D.best_pct >= 0.60) & ~excl).sum() == 1732 and (((D.best_pct >= 0.60) & ~excl) == D.candidate).all(), "known removal does not reproduce the frozen list"
FIG = {"RBBP4-ZNF644": key("RBBP4", "ZNF644"), "ADNP-MTF2": key("ADNP", "MTF2"), "DR1-NFYA": key("DR1", "NFYA")}
rows, fish = [], []
for c in (0.60, 0.70, 0.80, 0.90):
    r = dict(cutoff=f"{c:.2f}")
    for lab, col in SETS:
        p = D[col] >= c; r[f"{lab}_pairs"] = int(p.sum()); r[f"{lab}_candidates"] = int((p & ~excl).sum())
    best = D.best_pct >= c; cand = best & ~excl; allc = (D[[col for _, col in SETS]] >= c).all(axis=1)
    r["best_of_5_pairs"] = int(best.sum()); r["best_of_5_candidates"] = int(cand.sum()); r["all_5_candidates"] = int((allc & ~excl).sum())
    r["AF3_5seed_strong_of_24"] = int((cand & D.s5).sum()); r["triple_support_of_18"] = int((cand & D.tri).sum())
    r["MOTIF_and_candidates"] = int((cand & D.motif_spec_noCB).sum())
    for nm, k in FIG.items():
        x = D[D.k == k].iloc[0]; r[nm] = f"{'yes' if cand[x.name] else 'no'} ({x.best_pct:.2f} {x.best_matrix})"
    rows.append(r)
    x = D[cand]; t = [[int((x.motif_spec_noCB & x.s5).sum()), int((x.motif_spec_noCB & ~x.s5).sum())], [int((~x.motif_spec_noCB & x.s5).sum()), int((~x.motif_spec_noCB & ~x.s5).sum())]]
    o, pv = fisher_exact(t); fish.append(dict(cutoff=f"{c:.2f}", candidates=int(cand.sum()), motif_AF3strong=t[0][0], motif_notStrong=t[0][1], noMotif_AF3strong=t[1][0], noMotif_notStrong=t[1][1], odds_ratio=round(o, 3), fisher_p=round(pv, 4)))
R = pd.DataFrame(rows); Fi = pd.DataFrame(fish)
rem = D[(D.best_pct >= 0.60) & excl]; RE = rem.known_reason.fillna("").str.split(r"[;,|]").explode().str.strip().value_counts().rename_axis("known_reason").reset_index(name="pairs_removed_at_0.60")
print("removed 1,784 -> 1,732:", len(rem)); print(RE.to_string(index=False)); print(rem.known_reason.value_counts().to_string())
assert R.loc[0, "best_of_5_pairs"] == 1784 and R.loc[0, "MOTIF_and_candidates"] == 927 and R.loc[0, "AF3_5seed_strong_of_24"] == 24 and R.loc[0, "triple_support_of_18"] == 18
NOTE = "AF3 (5 seeds) was run only on pairs with best-of-5 percentile >= 0.60, so AF3 and triple-support counts can only shrink as the cutoff rises. Known pairs removed as in candidates_frozen_v1 (known=True). candidates_frozen_v1 is unchanged."
R.to_csv(out + ".tsv", sep="\t", index=False); Fi.to_csv(out + "_fisher.tsv", sep="\t", index=False)
with pd.ExcelWriter(out + ".xlsx") as w:
    R.to_excel(w, "sweep", index=False); Fi.to_excel(w, "fisher_motif_vs_AF3", index=False); RE.to_excel(w, "known_removed_at_0.60", index=False); pd.DataFrame({"note": [NOTE]}).to_excel(w, "note", index=False)
for c in (0.70, 0.80, 0.90):
    cand = (D.best_pct >= c) & ~excl; print(f"strong lost at {c}:", sorted(D[D.s5 & ~cand].pair), "| triple lost:", sorted(D[D.tri & ~cand].pair))
print(R.T.to_string()); print(Fi.to_string(index=False)); print(NOTE)
