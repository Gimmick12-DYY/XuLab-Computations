# Random-pair null for partner_motif_rank.py: same sets, same own-motif matching, same rule (target passes AND rank <= 10 AND n_pass <= 10).
# 500 random unordered panel pairs (seed 20261109) with >= 1 testable partner (own known motif in HOMER/AME libraries), excluding the T1-T3 and
# calibration pairs; both directions scored where testable, as in the main table. Existing motif results only.
# Writes partner_motif_null_rows.tsv and partner_motif_null_summary.tsv. usage: partner_motif_null.py
import os, sys, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.dirname(HERE); LK = f"{C}/pair_motif_lookup"
assert not os.path.exists(f"{HERE}/partner_motif_null_summary.tsv"), "output exists"
src = open(f"{LK}/build_pair_motif_lookup.py").read(); src = src[:src.index("# ---------------- de novo motifs")]
sys.argv = [f"{LK}/build_pair_motif_lookup.py", HERE]; g = {"__file__": f"{LK}/build_pair_motif_lookup.py"}; exec(src, g)
KNOWN, canon, tokens, cells = g["KNOWN"], g["canon"], g["tokens"], g["cells"]
ALIAS = dict(g["ALIAS"]); ALIAS.update({"DDIT3": {"DDIT3", "CHOP"}, "EPAS1": {"EPAS1", "HIF2A", "HIF2"}, "ZUFSP": {"ZUFSP", "ZUP1"}})
is_own = lambda tf, m: bool(ALIAS.get(tf, {tf}) & tokens(m))
SETS = ["A_raw_homer", "A_raw_ame", "B_imp_peaks_homer", "B_imp_peaks_homer_ocr", "B_imp_peaks_ame", "B_imp_peaks_ame_ocr", "C_imp_bins_homer", "C_imp_bins_ame"]
TAB = {}
for s in SETS:
    d = {}
    for tf, df in KNOWN[s].items():
        for m, p, ok in zip(df.motif, df.p, df["pass"]):
            if not isinstance(m, str): continue
            q = d.setdefault(m, {})
            if tf not in q or p < q[tf][0]: q[tf] = (float(p), bool(ok))
    TAB[s] = (d, sorted(KNOWN[s]))
PANEL = sorted(cells); OWN = {s: {t: [m for m in TAB[s][0] if is_own(t, m)] for t in PANEL} for s in SETS}
testable = {t for t in PANEL if any(OWN[s][t] for s in SETS)}
def score(D, L):
    out = []
    for s in SETS:
        d, tfs = TAB[s]; own = OWN[s][D]
        if not own or L not in tfs: continue
        best = None
        for m in own:
            pv = {t: d[m].get(t, (1.0, False)) for t in tfs}; pL = pv[L][0]; rk = 1 + sum(v[0] < pL for v in pv.values()); npass = sum(v[1] for v in pv.values())
            c = dict(motif_tf=D, target_tf=L, set=s, motif=m, p=pL, rank=rk, target_pass=pv[L][1], n_pass=npass, specific=bool(pv[L][1] and rk <= 10 and npass <= 10))
            if best is None or (c["specific"], c["target_pass"], -c["rank"], -c["p"]) > (best["specific"], best["target_pass"], -best["rank"], -best["p"]): best = c
        out.append(best)
    return out
T = pd.read_csv(f"{HERE}/triple_support.tsv", sep="\t", keep_default_na=False)
excl = {tuple(sorted(canon(x) for x in p.split("-"))) for p in T[T.tier.isin(["T1", "T2", "T3"])].pair} | {("CTCF", "STAG2"), ("EZH2", "MTF2"), ("MAZ", "ZNF143"), ("RBBP7", "SRCAP")}
cand = [p for p in itertools.combinations(PANEL, 2) if p not in excl and (p[0] in testable or p[1] in testable)]
rng = np.random.default_rng(20261109); pick = [cand[i] for i in rng.choice(len(cand), size=min(500, len(cand)), replace=False)]
rows = []
for a, b in pick:
    for D, L in ((a, b), (b, a)):
        if D in testable:
            for r in score(D, L): rows.append(dict(pair=f"{a}-{b}", **r))
R = pd.DataFrame(rows); R.to_csv(f"{HERE}/partner_motif_null_rows.tsv", sep="\t", index=False, float_format="%.3g")
S = []
for s in SETS:
    G = R[R.set == s]; S.append(dict(level="per_set_direction", set=s, n=len(G), n_specific=int(G.specific.sum()), rate=round(G.specific.mean(), 4) if len(G) else np.nan))
dirs = R.groupby(["pair", "motif_tf", "target_tf"]); pr = R.groupby("pair")
S.append(dict(level="per_direction_any_set", set="all", n=dirs.ngroups, n_specific=int(dirs.specific.any().sum()), rate=round(dirs.specific.any().mean(), 4)))
S.append(dict(level="per_pair_any_set", set="all", n=pr.ngroups, n_specific=int(pr.specific.any().sum()), rate=round(pr.specific.any().mean(), 4)))
S.append(dict(level="per_pair_any_raw_set", set="A_raw_*", n=pr.ngroups, n_specific=int(R[R.set.str.startswith("A_raw")].groupby("pair").specific.any().reindex(list(pr.groups), fill_value=False).sum()), rate=np.nan))
S.append(dict(level="per_pair_any_imputed_set", set="B/C", n=pr.ngroups, n_specific=int(R[~R.set.str.startswith("A_raw")].groupby("pair").specific.any().reindex(list(pr.groups), fill_value=False).sum()), rate=np.nan))
for name, pobs, nobs in (("RBBP4-ZIC2 like (rank 1, n_pass<=2, p<=2.83e-15)", 2.83e-15, 2), ("DDIT3-MTF2 like (rank 1, n_pass<=8, p<=6.95e-11)", 6.95e-11, 8), ("rank 1 & n_pass<=8 & pass (any p)", 1.0, 8)):
    hit = R.target_pass & (R["rank"] == 1) & (R.n_pass <= nobs) & (R.p <= pobs)
    S.append(dict(level="extreme_per_direction_any_set", set=name, n=dirs.ngroups, n_specific=int(hit.groupby([R.pair, R.motif_tf, R.target_tf]).any().sum()), rate=round(hit.groupby([R.pair, R.motif_tf, R.target_tf]).any().mean(), 4)))
S = pd.DataFrame(S); S.loc[S.rate.isna() & S.n.gt(0), "rate"] = (S.n_specific / S.n).round(4); S.to_csv(f"{HERE}/partner_motif_null_summary.tsv", sep="\t", index=False)
print("testable TFs", len(testable), "candidate pairs", len(cand), "drawn", len(pick)); pd.set_option("display.width", 200); print(S.to_string(index=False))
print(R[R.specific].sort_values("p").head(15).to_string(index=False))
