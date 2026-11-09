# Descriptive check: does the 1-seed AF3 sweep (scores_s1) track the correlation percentiles within the >= 0.60 candidate set (list.tsv)?
# Tiers as in score_one.py (strong = ipSAE >= 0.3 & ipTM >= 0.4; weak = ipSAE >= 0.1). No AF3 control below 0.60, so this tests dose-response
# within the >= 0.60 set only. Inputs: list.tsv, scores_s1/, align_corr_20261109/inputs/pct_*.tsv (copies of the Mac findings_slides/jie
# data/corr_sets{,_extra} matrices), r1_pairs_frozen_v3.tsv (view1_score = depth-matched co-binding percentile, cicero_excess_z, hub_partner),
# screen/scores.tsv het rows (Boltz tier), cached AF3 MSA jsons (sequence lengths), earlier AF3 score tables. Writes align_corr_20261109/*.tsv.
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; O = f"{C}/af3_corr060_20261104"; OUT = f"{O}/align_corr_20261109"
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; pn = lambda t: ALIAS.get(t.upper(), t.upper()); key = lambda a, b: tuple(sorted((pn(a), pn(b))))
MATS = {"all": "pct_all", "noquies": "pct_noquies", "tss": "pct_tss", "enh": "pct_enhancer_biv", "rep": "pct_repressive"}; SETNAME = {"all": "all", "no quiescent": "noquies", "TSS": "tss", "enhancer": "enh", "repressive": "rep"}
def rd(f):
    m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
M = {k: rd(f"{OUT}/inputs/{v}.tsv") for k, v in MATS.items()}
L = pd.read_csv(f"{O}/list.tsv", sep="\t"); L["k"] = [key(a, b) for a, b in zip(L.a, L.b)]; L["name"] = L.a + "_" + L.b; big = set(open(f"{O}/pairs_big.txt").read().split())
for k, m in M.items(): L[f"pct_{k}"] = [float(m.loc[x, y]) for x, y in L.k]
pc = [f"pct_{k}" for k in MATS]; L["best_pct"] = L[pc].max(1); L["best_matrix"] = L[pc].idxmax(1).str[4:]; L["n_mat_ge080"] = (L[pc] >= .80).sum(1)
assert np.allclose(L.best_pct, L.value, atol=1e-3), "best-of-five pct does not reproduce list.tsv value"
R = pd.read_csv(f"{X}/regulatome/r1_tftf/v2_final_v3/r1_pairs_frozen_v3.tsv", sep="\t", keep_default_na=False, na_values=[""]); R.index = [key(a, b) for a, b in zip(R.a, R.b)]
L["cobind_depth_matched_pct"] = R.view1_score.reindex(L.k).values; L["cicero_excess_z"] = R.cicero_excess_z.reindex(L.k).values; L["hub_pair"] = (R.hub_partner.reindex(L.k) != ".").values
Sc = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t", keep_default_na=False); Sc = Sc[Sc["set"] == "het"]; BZ = {key(*p.split("__")): {"strong": "strong", "weak": "weak"}.get(h, "none") for p, h in zip(Sc.pair_id, Sc.hit)}; L["boltz_tier"] = [BZ.get(k, "not_screened") for k in L.k]
ln = {}
for d in ("af3_corr060_20261104", "af3_novel_20261103", "af3_local", "af3_dna"):
    for f in glob.glob(f"{C}/{d}/msa_out/*/*_data.json"):
        p = os.path.basename(os.path.dirname(f))
        if p not in ln: ln[p] = len(json.load(open(f))["sequences"][0]["protein"]["sequence"])
L["tokens"] = [ln.get(a.lower(), np.nan) + ln.get(b.lower(), np.nan) for a, b in zip(L.a, L.b)]
sc = {}
for f in glob.glob(f"{O}/scores_s1/*.tsv"):
    x = open(f).read().rstrip("\n").split("\t"); sc[os.path.basename(f)[:-4]] = (float(x[1]), float(x[2]), x[4])
L["scored"] = L.name.isin(sc); L["iptm"] = [sc.get(n, (np.nan,) * 3)[0] for n in L.name]; L["ipsae"] = [sc.get(n, (np.nan,) * 3)[1] for n in L.name]; L["tier"] = [sc.get(n, (np.nan, np.nan, "unscored"))[2] for n in L.name]
miss = L[~L.scored]; print(f"list {len(L)}, scored {L.scored.sum()}, unscored {len(miss)} (of them large pairs: {miss.name.isin(big).sum()} of {len(big)} large)"); print("unscored:", " ".join(miss.name))
S = L[L.scored].copy(); S["strong"] = (S.tier == "strong").astype(int); S["any_hit"] = S.tier.isin(["strong", "weak"]).astype(int)
# 1. hit rate by best-pct bin + Cochran-Armitage trend (scores = bin index)
edges = [.60, .70, .80, .90, .95, .99, 1.0001]; labs = ["0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", "0.95-0.99", ">=0.99"]; S["bin"] = pd.cut(S.best_pct, edges, right=False, labels=labs)
def cat(y, g):
    t = pd.crosstab(g, y).reindex(columns=[0, 1], fill_value=0); n = t.sum(1).values; r = t[1].values; s = np.arange(len(n)); N = n.sum(); p = r.sum() / N
    T = (r * (s - (n * s).sum() / N)).sum(); V = p * (1 - p) * ((n * s ** 2).sum() - (n * s).sum() ** 2 / N); z = T / np.sqrt(V); return z, 2 * stats.norm.sf(abs(z))
B = S.groupby("bin", observed=False).agg(n=("tier", "size"), strong=("strong", "sum"), strong_or_weak=("any_hit", "sum")); B["rate_strong"] = (B.strong / B.n).round(4); B["rate_strong_or_weak"] = (B.strong_or_weak / B.n).round(4)
zs, ps = cat(S.strong, S.bin); za, pa = cat(S.any_hit, S.bin); B.to_csv(f"{OUT}/hit_rate_by_bin.tsv", sep="\t"); print(B.to_string()); print(f"Cochran-Armitage trend: strong z={zs:.3f} p={ps:.3g}; strong+weak z={za:.3f} p={pa:.3g}")
pd.DataFrame([dict(outcome="strong", z=round(zs, 4), p=float(f"{ps:.3g}")), dict(outcome="strong_or_weak", z=round(za, 4), p=float(f"{pa:.3g}"))]).to_csv(f"{OUT}/trend_test.tsv", sep="\t", index=False)
# 2. Spearman of ipSAE (max) with each correlation measure, all scored pairs (pairwise complete)
rows = []
for c in ["best_pct"] + pc + ["n_mat_ge080", "cobind_depth_matched_pct", "cicero_excess_z"]:
    d = S[["ipsae", c]].dropna(); r, p = stats.spearmanr(d.ipsae, d[c]); r2, p2 = stats.spearmanr(S.dropna(subset=[c]).iptm, d[c]); rows.append(dict(measure=c, n=len(d), rho_ipsae=round(r, 4), p_ipsae=float(f"{p:.3g}"), rho_iptm=round(r2, 4), p_iptm=float(f"{p2:.3g}")))
Sp = pd.DataFrame(rows); Sp.to_csv(f"{OUT}/spearman.tsv", sep="\t", index=False); print(Sp.to_string(index=False))
# 3. logistic regression (Newton-Raphson, Wald 95% CI): hit ~ best_pct (per 0.1) + hub_pair + log(tokens)
def logit(y, Xm, names):
    Xm = np.column_stack([np.ones(len(y)), Xm]); b = np.zeros(Xm.shape[1])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-Xm @ b)); W = mu * (1 - mu); H = Xm.T @ (Xm * W[:, None]); st = np.linalg.solve(H, Xm.T @ (y - mu)); b += st
        if np.abs(st).max() < 1e-10: break
    se = np.sqrt(np.diag(np.linalg.inv(H))); z = b / se
    return [dict(term=n, coef=round(bb, 4), se=round(s, 4), OR=round(np.exp(bb), 4), OR_lo=round(np.exp(bb - 1.96 * s), 4), OR_hi=round(np.exp(bb + 1.96 * s), 4), p=float(f"{2 * stats.norm.sf(abs(zz)):.3g}")) for n, bb, s, zz in zip(["intercept"] + names, b, se, z)]
D = S.dropna(subset=["tokens"]); lg = []
for yname in ("strong", "any_hit"):
    for model, cols in (("pct_only", ["best_pct"]), ("pct+hub+logtokens", ["best_pct", "hub_pair", "tokens"])):
        Xm = np.column_stack([D.best_pct.values * 10 if c == "best_pct" else (np.log(D[c].values) if c == "tokens" else D[c].astype(float).values) for c in cols])
        for r in logit(D[yname].values.astype(float), Xm, ["best_pct_per_0.1" if c == "best_pct" else ("log_tokens" if c == "tokens" else c) for c in cols]): lg.append(dict(outcome=yname, model=model, n=len(D), events=int(D[yname].sum()), **r))
Lg = pd.DataFrame(lg); Lg.to_csv(f"{OUT}/logistic.tsv", sep="\t", index=False); print(Lg.to_string(index=False))
# 4. matrix source of the best pct: hits vs background (all scored pairs)
ms = pd.DataFrame({"all_scored": S.best_matrix.value_counts(), "strong": S[S.strong == 1].best_matrix.value_counts(), "strong_or_weak": S[S.any_hit == 1].best_matrix.value_counts()}).fillna(0).astype(int)
for c in ms: ms[c + "_share"] = (ms[c] / ms[c].sum()).round(3)
ch = stats.chi2_contingency(np.vstack([ms.strong_or_weak.values, (ms.all_scored - ms.strong_or_weak).values])); ms.to_csv(f"{OUT}/matrix_source.tsv", sep="\t"); print(ms.to_string()); print(f"chi2 hits vs non-hits across matrices (strong+weak): chi2={ch[0]:.2f} df={ch[2]} p={ch[1]:.3g}")
# 5. reference only: earlier AF3 runs (Boltz-selected) by best pct band, no test
ear = {}
for f in ("af3_local/af3_scores.tsv", "af3_novel_20261103/af3_scores.tsv", "af3_novel_20261103/af3_scores_weak.tsv", "af3_novel_20261103/af3_scores_srcap.tsv"):
    for r in pd.read_csv(f"{C}/{f}", sep="\t").itertuples():
        x, y = r.pair.split("__"); k = key(x, y)
        if k[0] in M["all"].index and k[1] in M["all"].index and k[0] != k[1]:
            rk = {"strong": 2, "weak": 1, "none": 0}
            if k not in ear or rk[r.tier] > rk[ear[k]]: ear[k] = r.tier
E = pd.DataFrame([(k, t, max(float(M[m].loc[k[0], k[1]]) for m in MATS)) for k, t in ear.items()], columns=["k", "tier", "best_pct"]); E["band"] = np.where(E.best_pct >= .60, ">=0.60", "<0.60")
Et = E.groupby("band").agg(n=("tier", "size"), strong=("tier", lambda t: (t == "strong").sum()), strong_or_weak=("tier", lambda t: t.isin(["strong", "weak"]).sum())); Et["rate_strong"] = (Et.strong / Et.n).round(3); Et["rate_strong_or_weak"] = (Et.strong_or_weak / Et.n).round(3)
Et.to_csv(f"{OUT}/earlier_af3_reference.tsv", sep="\t"); print("earlier AF3 runs (Boltz-selected, reference only):"); print(Et.to_string())
# 6. top 20 by ipSAE
T = S.sort_values(["ipsae", "iptm"], ascending=False).head(20)[["name", "tier", "ipsae", "iptm", "best_pct", "best_matrix"] + pc + ["n_mat_ge080", "cobind_depth_matched_pct", "cicero_excess_z", "hub_pair", "boltz_tier", "tokens"]]
T.to_csv(f"{OUT}/top20_by_ipsae.tsv", sep="\t", index=False, float_format="%.3f"); pd.set_option("display.width", 300); print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
L.drop(columns=["k"]).to_csv(f"{OUT}/pairs_annotated.tsv", sep="\t", index=False, float_format="%.4f")
