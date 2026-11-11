# Frozen candidate list (single source of candidates for all layers, per Yuyang 2026-11-11). Selection logic = af3_corr060_20261104/build_list.py
# (via jie_figures/summary_table_data.py), minus build_list's scheduling step (pairs already run in earlier AF3 rounds are NOT removed):
#   best_pct = max over the five element-set percentile matrices (all, no quiescent, TSS, enhancer, repressive);
#   known = CUR curated pair not labelled Novel (CUR_known / CUR_capselex), STRING physical >= .7 (r1_pairs_frozen string_physical non-missing),
#           CAP-SELEX called, STAG2-class pair, or a NEUROD4 pair; candidate = best_pct >= .60 and not known.
# Inputs are the exact files build_list.py read (identical md5 to the Mac copies). Annotation columns (AF3, Boltz) do not affect selection.
# Writes candidates_all3081.tsv and candidates_frozen_v1.tsv. usage: build_candidates.py
import ast, glob, os, re
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; O = f"{C}/candidates_frozen_v1"; S = f"{C}/af3_corr060_20261104"
PCT = {"all": f"{S}/align_corr_20261109/inputs/pct_all.tsv", "noquies": f"{S}/align_corr_20261109/inputs/pct_noquies.tsv", "tss": f"{S}/align_corr_20261109/inputs/pct_tss.tsv",
       "enh": f"{S}/align_corr_20261109/inputs/pct_enhancer_biv.tsv", "repr": f"{S}/align_corr_20261109/inputs/pct_repressive.tsv"}
NAME = {"all": "all", "noquies": "no quiescent", "tss": "TSS", "enh": "enhancer", "repr": "repressive"}
R1 = f"{X}/regulatome/r1_tftf/v2_final/r1_pairs_frozen.tsv"; CURF = f"{C}/jie_figures/summary_table_data.py"
def rd(f):
    m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
M = {k: rd(f) for k, f in PCT.items()}; tfs = list(M["all"].index)
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; pn = lambda t: ALIAS.get(t.upper(), t.upper()); key = lambda a, b: tuple(sorted((pn(a), pn(b))))
line = next(l for l in open(CURF) if l.startswith("CUR = ")); CUR = ast.literal_eval(line[len("CUR = "):].strip())
curcat = {}
for nm, (cat, ps) in CUR.items():
    for p in ps: curcat[key(*p)] = cat
P = pd.read_csv(R1, sep="\t", keep_default_na=False, na_values=[""])
strg = {key(r.a, r.b) for r in P[P.string_physical.notna()].itertuples()}; cap = {key(r.a, r.b) for r in P[P.capselex.astype(str).str.startswith("called")].itertuples()}
stag = {key(r.a, r.b) for r in P[P["class"].astype(str).str.startswith("STAG2")].itertuples()}
rk = {"strong": 2, "weak": 1, "none": 0}
s1 = {}
for f in glob.glob(f"{S}/scores_s1/*.tsv"):
    if os.path.getsize(f): x = open(f).read().split("\t"); a, b = x[0].split("__"); s1[key(a, b)] = x[4]
early = {}
for f in ("af3_local/af3_scores.tsv", "af3_novel_20261103/af3_scores.tsv", "af3_novel_20261103/af3_scores_weak.tsv", "af3_novel_20261103/af3_scores_srcap.tsv"):
    for r in pd.read_csv(f"{C}/{f}", sep="\t").itertuples():
        a, b = r.pair.split("__"); k = key(a, b)
        if k not in early or rk[r.tier] > rk[early[k][0]]: early[k] = (r.tier, f.split("/")[0])
s5all = {key(*l.strip().split("_", 1)) for l in open(f"{S}/seeds5/all.txt") if l.strip()}; s5 = {}
for f in glob.glob(f"{S}/scores_s5/*.tsv"):
    if os.path.getsize(f): x = open(f).read().split("\t"); a, b = x[0].split("__"); s5[key(a, b)] = "strong" if int(x[5]) >= 3 else ("weak" if float(x[2]) >= .1 else "none")
B = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t", keep_default_na=False); B = B[B["set"] != "homo"]; bz = {}
for r in B.itertuples():
    a, b = r.pair_id.split("__"); k = key(a, b); s = (float(r.ipsae_max), float(r.iptm_top))
    if k not in bz or s > bz[k][0]: bz[k] = (s, {"strong": "strong", "weak": "weak"}.get(r.hit, "none"))
rows = []
for i, a in enumerate(tfs):
    for b in tfs[i + 1:]:
        k = key(a, b); v = {m: float(M[m].loc[a, b]) for m in M}; bm = max(v, key=v.get); why = []
        c = curcat.get(k)
        if c == "Known": why.append("CUR_known")
        if c == "CAP-SELEX candidate": why.append("CUR_capselex")
        if k in strg: why.append("STRING_phys>=.7")
        if k in cap: why.append("CAPSELEX_called")
        if k in stag: why.append("STAG2_class")
        if "NEUROD4" in k: why.append("NEUROD4")
        if k in s1: t1, src = s1[k], "af3_corr060_20261104/scores_s1"
        elif k in early: t1, src = early[k]
        else: t1, src = "not_tested", ""
        t5 = s5.get(k, "pending" if k in s5all else "not_run")
        rows.append(dict(pair=f"{k[0]}-{k[1]}", tf_a=k[0], tf_b=k[1], **{f"pct_{m}": round(v[m], 4) for m in M}, best_pct=round(v[bm], 4), best_matrix=NAME[bm],
                         known=bool(why), known_reason=";".join(why), candidate=(v[bm] >= .60) and not why, af3_1seed_tier=t1, af3_1seed_source=src, af3_5seed_tier=t5,
                         boltz_tier=bz[k][1] if k in bz else "not_screened"))
A = pd.DataFrame(rows); A.to_csv(f"{O}/candidates_all3081.tsv", sep="\t", index=False)
F = A[A.best_pct >= .60]; F.to_csv(f"{O}/candidates_frozen_v1.tsv", sep="\t", index=False)
print("all", len(A), "| best>=.60", len(F), "| candidates", int(F.candidate.sum()), "| known among >=.60", int(F.known.sum()), F[F.known].known_reason.value_counts().to_dict())
print("candidates AF3 1-seed:", F[F.candidate].af3_1seed_tier.value_counts().to_dict(), "| sources", F[F.candidate].af3_1seed_source.value_counts().to_dict())
print("candidates AF3 5-seed:", F[F.candidate].af3_5seed_tier.value_counts().to_dict()); print("all 5-seed pairs found:", len(s5all), "in table", int(A.af3_5seed_tier.ne("not_run").sum()))
