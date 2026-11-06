# Element-level co-binding enrichment for ALL panel TF pairs (no motifs). Same scoring as pilot_layer3_v2.py:
#  per TF, Poisson GLM over the active elements (ATAC, leave-one-out pooled panel, width, GC, GC^2, k50 mappability) ->
#  Pearson residual; eligible = >= 3 reads and positive residual within the set; top N = 5% of the set per TF.
#  obs = |top_a & top_b|, expected = N_a N_b / n_set, enrichment = obs/exp, hypergeometric p.
#  Sets: distal (non-promoter distal_active + bivalent) and promoter. Percentiles pooled and within library-group pair type.
import json, numpy as np, pandas as pd
from scipy.stats import hypergeom

X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pilot_pairs/cobind_allpairs"
import os; os.makedirs(O, exist_ok=True)
TOPF = 0.05
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None,
                names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values)
Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
V = f"{X}/regulatome/gate1/v2"
gc = pd.read_csv(f"{V}/gc.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
mp = pd.read_csv(f"{V}/map.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float)
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
raw = {t: np.load(f"{X}/regulatome/m1_binding/v2/counts/{t}.npz")["raw"][act].astype(float) for t in tfs}
meta = {t: json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json")) for t in tfs}
reads = {t: meta[t]["reads"] for t in tfs}
tot = sum(raw.values()); totr = sum(reads.values())
logw = np.log(Ea.end.values - Ea.start.values)

def glm_mu(y, off, Xd):
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)

def score(t):
    pool = (tot - raw[t]) / (totr - reads[t]) * 1e6
    Xd = np.column_stack([np.ones(nE), np.log1p(atac / atac.sum() * 1e6), np.log1p(pool), logw, gc, gc ** 2, mp])
    mu = glm_mu(raw[t], np.full(nE, np.log(reads[t] / 1e6)), Xd)
    return (raw[t] - mu) / np.sqrt(mu)

SETS = {"distal": Ea.cls.isin(["distal_active", "bivalent"]).values & (Ea.type != "promoter").values,
        "promoter": (Ea.type == "promoter").values}
PR = {t: score(t) for t in tfs}
grp = {}
v2 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t")
out = []
for setname, msk in SETS.items():
    ns = int(msk.sum()); N = int(round(TOPF * ns)); M = {}
    for t in tfs:
        pr = PR[t].copy(); pr[(raw[t] < 3) | ~msk | (PR[t] <= 0)] = -np.inf
        n = int(min(N, np.isfinite(pr).sum())); v = np.zeros(nE, bool); v[np.argsort(-pr)[:n]] = True; M[t] = v
    T = np.array([M[t] for t in tfs], dtype=np.float32); ov = T @ T.T; nt = T.sum(1)
    for i in range(len(tfs)):
        for j in range(i + 1, len(tfs)):
            o, e = int(ov[i, j]), nt[i] * nt[j] / ns
            out.append(dict(set=setname, a=tfs[i], b=tfs[j], n_set=ns, n_top_a=int(nt[i]), n_top_b=int(nt[j]), obs=o, expected=e,
                            enrichment=o / e if e else np.nan, p_hyper=float(hypergeom.sf(o - 1, ns, int(nt[i]), int(nt[j])))))
D = pd.DataFrame(out)
key = lambda a, b: tuple(sorted((a, b)))
g = {key(a, b): (gp, et) for a, b, gp, et in zip(v2.a, v2.b, v2.gpair, v2.edge_type)}
D["gpair"] = [g.get(key(a, b), (None, None))[0] for a, b in zip(D.a, D.b)]
D["r1v2_edge_type"] = [g.get(key(a, b), (None, None))[1] for a, b in zip(D.a, D.b)]
D["full_N_both"] = (D.n_top_a == (D.n_set * TOPF).round()) & (D.n_top_b == (D.n_set * TOPF).round())
D["enrich_pct_pooled"] = D.groupby("set").enrichment.rank(pct=True)
D["enrich_pct_stratum"] = D.groupby(["set", "gpair"]).enrichment.rank(pct=True)
D.to_csv(f"{O}/cobind_allpairs.tsv", sep="\t", index=False, float_format="%.5g")
pairs = [("HOXB8", "RBPJ"), ("MAZ", "ZNF143"), ("MLX", "SREBF2"), ("EZH2", "MTF2"), ("CTCF", "STAG2"), ("RBBP7", "SRCAP")]
K = D.assign(k=[key(a, b) for a, b in zip(D.a, D.b)])
print(K[K.k.isin([key(*p) for p in pairs])].drop(columns="k").round(4).to_string(index=False))
print(D.groupby("set").enrichment.describe(percentiles=[.25, .5, .75, .9, .95]).round(3).to_string())
for t in ("HOXB8", "RBPJ"):
    for s in SETS:
        S = D[(D.set == s) & ((D.a == t) | (D.b == t))].sort_values("enrichment", ascending=False)
        print(t, s, "top partners:", ", ".join(f"{(r.b if r.a == t else r.a)} {r.enrichment:.2f}" for r in S.head(8).itertuples()))
