# repro_v1 noise-floor test (README Amendment 5; SPEC_REPRO Amendment 5). Uninformative lists at matched depth and list length:
# (a) per-half element-label permutation of CPM then v8; (b) random genes matching the v8 list's linked-element-count histogram.
# 20 draws each. Writes noisefloor_per_draw.tsv, noisefloor_summary.tsv. DESCRIPTIVE. usage: noisefloor.py
import os
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; G1 = f"{R}/gate1/cellmats"; O = f"{R}/m6_improve/repro_v1"
assert not os.path.exists(f"{O}/noisefloor_summary.tsv")
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; NT = len(tfs); ND = 20
E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); Ea = E_[E_.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; ae = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int).values
GL = pd.Index(sorted(A.TargetGene.unique())); ag = GL.get_indexer(A.TargetGene.values); NG = len(GL)
cnt = np.bincount(ag, minlength=NG); EDGES = list(range(1, 16)) + [16, 21, 31, 51]
strat = np.array([min(c, 15) - 1 if c <= 15 else 15 + np.searchsorted([21, 31, 51], c, side="right") for c in cnt]); NS = strat.max() + 1
POOL = [np.flatnonzero(strat == s) for s in range(NS)]
def lsum(v): return np.bincount(ag, v[ae], NG)
def top(s, n): o = np.argsort(-s, kind="stable"); o = o[s[o] > 0][:n]; return frozenset(o)
RAW, NCELL, V8 = {}, {}, {}
for i, t in enumerate(tfs):
    M = sp.load_npz(f"{G1}/{t}.raw.npz").tocsr(); rd = np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float); n = M.shape[0]; NCELL[t] = n
    hh = np.random.default_rng(20261029 + i).permutation(n) < n // 2
    for k, m in ((0, hh), (1, ~hh)): RAW[t, k] = np.asarray(M[m].sum(0)).ravel() / rd[m].sum() * 1e6; V8[t, k] = top(lsum(RAW[t, k]), 1000)
for t in tfs:  # baseline must reproduce Run 2b
    for k in (0, 1): assert set(GL[sorted(V8[t, k])]) == set(x.strip() for x in open(f"{O}/run2b/lists/v8_1000_{t}_h{k + 1}.txt") if x.strip()), (t, k)
tert = pd.qcut(pd.Series(NCELL), 3, labels=["low", "mid", "high"]).astype(str)
jac = lambda a, b: len(a & b) / max(len(a | b), 1); rows = []; nshort = 0
for kind, kn in ((1, "perm"), (2, "countmatched")):
    for d in range(ND):
        L = {}
        for i, t in enumerate(tfs):
            for k in (0, 1):
                rng = np.random.default_rng(20261110 + 100000 * kind + 1000 * d + 2 * i + k)
                if kind == 1: L[t, k] = top(lsum(rng.permutation(RAW[t, k])), 1000)
                else:
                    need = np.bincount(strat[list(V8[t, k])], minlength=NS); pick = []
                    for s in range(NS):
                        m = need[s]
                        if not m: continue
                        take = min(m, len(POOL[s])); pick += list(rng.choice(POOL[s], take, replace=False))
                        if take < m:  # nearest-stratum fill, logged
                            nshort += m - take; used = set(pick)
                            for s2 in sorted(range(NS), key=lambda x: abs(x - s)):
                                av = [g for g in POOL[s2] if g not in used]; q = min(m - take, len(av))
                                if q: pick += list(rng.choice(av, q, replace=False)); take += q; used |= set(pick)
                                if take == m: break
                    L[t, k] = frozenset(pick)
        for t in tfs:
            jw = jac(L[t, 0], L[t, 1]); jb = np.mean([jac(L[t, k], L[u, k]) for u in tfs if u != t for k in (0, 1)])
            rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", kind=kn, draw=d, tf=t, cells=NCELL[t], depth_tertile=tert[t], n_list_h1=len(L[t, 0]), n_list_h2=len(L[t, 1]), J_within=jw, J_between=jb, diff=jw - jb))
        print(kn, d, flush=True)
P = pd.DataFrame(rows); P.to_csv(f"{O}/noisefloor_per_draw.tsv", sep="\t", index=False, float_format="%.5f")
S = []
for kn, G in P.groupby("kind"):
    for tr in ["all", "low", "mid", "high"]:
        H = G if tr == "all" else G[G.depth_tertile == tr]
        for stat in ("mean", "median"):
            D = H.groupby("draw")[["J_within", "J_between", "diff"]].agg(stat)
            S.append(dict(DESCRIPTIVE="DESCRIPTIVE", kind=kn, depth_tertile=tr, over_tfs=stat, n_tfs=H.tf.nunique(), n_draws=len(D),
                          **{f"{c}_{q}": v for c in D.columns for q, v in (("mean", D[c].mean()), ("lo95", np.percentile(D[c], 2.5)), ("hi95", np.percentile(D[c], 97.5)))}))
S = pd.DataFrame(S); low = S[(S.depth_tertile == "low") & (S.over_tfs == "mean")].set_index("kind").diff_hi95
survive = bool((low < 0.003).all() and (low < 0.013).all())
S["decision"] = "SURVIVES" if survive else "VOID"; S["count_matched_shortfall_genes"] = nshort
S.to_csv(f"{O}/noisefloor_summary.tsv", sep="\t", index=False, float_format="%.5f")
print(low.to_string(), "DECISION", "SURVIVES" if survive else "VOID", "shortfall", nshort)
