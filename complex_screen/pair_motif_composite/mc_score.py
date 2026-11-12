# Motif-centred read profiles, scoring and calibration (criteria fixed before results, same gates as motif_valid/mv_score.py):
# per motif m and library t: cf = log2(centre density / flank density) (+/- 50 bp vs 200-500 bp; background-free shape statistic) and
# occ = log2(centre reads per instance per million library reads in the bins) (level statistic). Each is panel-standardized per motif across the
# 79 libraries (OLS on log library reads removed; empirical-Bayes z with sampling variance). Gate 1: own motif rank <= 10 on odd AND even
# chromosomes for > 3 TFs. Gate 2: anchor A's best own motif (chosen on odd chr) at partner B's library, held-out percentile among the other 78
# libraries >= 0.95 for >= 2 of CTCF->STAG2, MAZ<->ZNF143 (either), NFYA->ZNF644, and not for CTCF->NFYA, NFYA->CTCF.
# Output motif_centred/{scores_<stat>_<part>.tsv, own_gate.tsv, calibration.tsv, anchor_controls.tsv}
import re, glob
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/motif_centred"
alt = {}
for l in open(f"{O}/db/known.meme"):
    if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
tfs = [l.strip() for l in open(f"{O}/recall/tfs.txt")]; C = {t: np.load(f"{G}/counts/{t}.npz") for t in tfs if glob.glob(f"{G}/counts/{t}.npz")}; tfs = list(C)
motifs = sorted(set.intersection(*[set(C[t]["motifs"]) for t in tfs])); mi = {m: k for k, m in enumerate(motifs)}; M = len(motifs)
depth = np.array([float(C[t]["reads_in_bins"]) for t in tfs]); P = ["disc", "val"]; V, E2 = {}, {}
for pt in P:
    cf = np.full((M, len(tfs)), np.nan); cfv = cf.copy(); oc = cf.copy(); ocv = cf.copy()
    for j, t in enumerate(tfs):
        d = C[t]; idx = np.array([mi.get(m, -1) for m in d["motifs"]]); k = idx >= 0
        ce, fl, ni = d[f"centre_{pt}"][k].astype(float), d[f"flank_{pt}"][k].astype(float), d[f"ninst_{pt}"][k].astype(float)
        cf[idx[k], j] = np.log2(((ce + 1) / 101) / ((fl + 1) / 602)); cfv[idx[k], j] = (1 / (ce + 1) + 1 / (fl + 1)) / np.log(2) ** 2
        oc[idx[k], j] = np.log2((ce + 1) / np.maximum(ni, 1) / depth[j] * 1e6); ocv[idx[k], j] = 1 / (ce + 1) / np.log(2) ** 2
    V[("cf", pt)], E2[("cf", pt)], V[("occ", pt)], E2[("occ", pt)] = cf, cfv, oc, ocv
def panel(Vm, Em):
    x = np.log(depth); Z = np.full_like(Vm, np.nan)
    for i in range(Vm.shape[0]):
        ok = np.isfinite(Vm[i])
        if ok.sum() < 20: continue
        A = np.c_[np.ones(ok.sum()), x[ok]]; r = Vm[i, ok] - A @ np.linalg.lstsq(A, Vm[i, ok], rcond=None)[0]; tau2 = max(r.var() - Em[i, ok].mean(), 0.01 * r.var()); Z[i, ok] = r / np.sqrt(tau2 + Em[i, ok])
    return Z
SC = {}
for st in ["cf", "occ"]:
    for pt in P: SC[(st, pt)] = V[(st, pt)]; SC[(st + "_panel", pt)] = panel(V[(st, pt)], E2[(st, pt)])
for (m, pt), Vm in SC.items(): pd.DataFrame(Vm, index=motifs, columns=tfs).round(3).to_csv(f"{G}/scores_{m}_{pt}.tsv", sep="\t")
METH = ["cf", "cf_panel", "occ", "occ_panel"]; RK = {k: pd.DataFrame(v, index=motifs, columns=tfs).rank(ascending=False, method="min") for k, v in SC.items()}
ALIAS = {"HP1": ["CBX3", "CBX1", "CBX5"], "ZUFSP": ["ZUP1"]}
def toks(s): return set(re.split(r"[.@:|/_\- ]+", s.upper()))
def own_ids(t): names = ALIAS.get(t, [t]); return [k for k, m in enumerate(motifs) if any(n in toks(m) | toks(alt.get(m, m)) for n in names)]
own, best = [], {}
for t in tfs:
    ids = own_ids(t)
    if not ids: continue
    for m in METH:
        rd = RK[(m, "disc")][t].iloc[ids]; rv = RK[(m, "val")][t].iloc[ids]; k = int(np.nanargmin(rd.values)); best[(m, t)] = ids[k]
        own.append(dict(tf=t, method=m, best_own=motifs[ids[k]], rank_disc=rd.min(), rank_val=rv.min()))
OW = pd.DataFrame(own); OW["strict"] = (OW.rank_disc <= 10) & (OW.rank_val <= 10); OW["loose"] = (OW.rank_disc <= 25) & (OW.rank_val <= 25); OW.to_csv(f"{G}/own_gate.tsv", sep="\t", index=False)
def anchor(m, A, B, pt):
    if (m, A) not in best or B not in tfs: return np.nan
    v = SC[(m, pt)][best[(m, A)]]; j = tfs.index(B); o = [i for i, t in enumerate(tfs) if t != A and np.isfinite(v[i])]
    return float(np.mean(v[o] < v[j]) + .5 * np.mean(v[o] == v[j]))
POS = [("CTCF", "STAG2"), ("MAZ", "ZNF143"), ("ZNF143", "MAZ"), ("NFYA", "ZNF644")]; NEG = [("CTCF", "NFYA"), ("NFYA", "CTCF")]; AC, CAL = [], []
for m in METH:
    for A, B in POS + NEG: AC.append(dict(method=m, anchor=A, partner=B, kind="positive" if (A, B) in POS else "negative", pct_disc=anchor(m, A, B, "disc"), pct_val=anchor(m, A, B, "val")))
    a = pd.DataFrame([r for r in AC if r["method"] == m]); ps = lambda A, B: bool((a[(a.anchor == A) & (a.partner == B)].pct_val >= .95).any())
    npos = int(ps("CTCF", "STAG2")) + int(ps("MAZ", "ZNF143") or ps("ZNF143", "MAZ")) + int(ps("NFYA", "ZNF644")); nneg = int(ps("CTCF", "NFYA")) + int(ps("NFYA", "CTCF"))
    o = OW[OW.method == m]; CAL.append(dict(method=m, n_tf_with_own=len(o), own_strict=int(o.strict.sum()), own_loose=int(o.loose.sum()), own_strict_tfs=",".join(o.tf[o.strict]), positives_passed=npos, negatives_passed=nneg,
                                          gate1=int(o.strict.sum()) > 3, gate2=(npos >= 2) and (nneg == 0)))
CA = pd.DataFrame(CAL); CA["usable"] = CA.gate1 & CA.gate2; CA.to_csv(f"{G}/calibration.tsv", sep="\t", index=False); AD = pd.DataFrame(AC); AD.to_csv(f"{G}/anchor_controls.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 150); print(len(motifs), "motifs;", len(tfs), "libraries"); print(CA.to_string(index=False)); print(AD.round(3).to_string(index=False))
K = ["CTCF", "NFYA", "NRF1", "MAZ", "ZNF143", "MLX", "ZIC2", "FOXK1", "SOX4", "E2F3", "ZBTB7A", "ZNF282", "ZNF768", "ZNF777", "SREBF2", "RELA", "HES1"]
s = OW[OW.tf.isin(K)]; print((s.pivot(index="tf", columns="method", values="rank_disc").astype("Int64").astype(str) + "/" + s.pivot(index="tf", columns="method", values="rank_val").astype("Int64").astype(str)).to_string())
