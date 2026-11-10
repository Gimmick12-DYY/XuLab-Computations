# repro_v1 Run 2b (README Amendment 2): identical to repro_run2.py (same code and seeds) but also writes the half lists (run2b/lists/) and adds J_between_partnerex
# (other TFs excluding the TF's known partners, run2_partners.tsv). Outputs methods_per_tf_2b.tsv, methods_summary_2b.tsv.
# repro_v1 Run 2 (README.md "Run 2 declaration", commit ca56a1c, declared before any Run 2 output). DESCRIPTIVE, no Perturb-seq.
# All 79 TFs at their real depths, r5_core-style half split; methods: v8 (ABC-linked raw CPM sum) at 250/500/1,000/2,000; +-50 kb and +-100 kb TSS
# windows; imputed binding (v8 rule); ratio and residual against the pooled panel signal of the same half (other 78 TFs). Per TF x method: J_within,
# J_between (vs the other 78 TFs' same-method same-half lists), difference. Writes methods_per_tf.tsv, methods_summary.tsv. usage: repro_run2.py
import os
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; G1 = f"{R}/gate1/cellmats"; O = f"{R}/m6_improve/repro_v1"
assert not os.path.exists(f"{O}/methods_per_tf_2b.tsv"); os.makedirs(f"{O}/run2b/lists", exist_ok=True)
PP = pd.read_csv(f"{O}/run2_partners.tsv", sep="\t"); PART = {}
for a_, b_ in zip(PP.a, PP.b): PART.setdefault(a_, set()).add(b_); PART.setdefault(b_, set()).add(a_)
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; NT = len(tfs)
E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); Ea = E_[E_.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; ae = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int).values
GL = pd.Index(sorted(A.TargetGene.unique())); ag = GL.get_indexer(A.TargetGene.values)
Gt = A.drop_duplicates("TargetGene")[["chr", "TargetGene", "TargetGeneTSS"]]; mid = ((Ea.start + Ea.end) // 2).values; WIN = {}
for W in (50000, 100000):
    gi_, ei_ = [], []
    for ch, G in Gt.groupby("chr"):
        ix = np.flatnonzero(Ea.chrom.values == ch); o = np.argsort(mid[ix]); ix = ix[o]; m = mid[ix]; lo = np.searchsorted(m, G.TargetGeneTSS.values - W); hi = np.searchsorted(m, G.TargetGeneTSS.values + W, side="right")
        for g, a_, b_ in zip(GL.get_indexer(G.TargetGene.values), lo, hi): gi_ += [g] * (b_ - a_); ei_ += list(ix[a_:b_])
    WIN[W] = (np.array(gi_), np.array(ei_, int))
RAW, IMP, NCELL = {}, {}, {}
for i, t in enumerate(tfs):
    M = sp.load_npz(f"{G1}/{t}.raw.npz").tocsr(); rd = np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float); n = M.shape[0]; NCELL[t] = n
    hh = np.random.default_rng(20261029 + i).permutation(n) < n // 2; I = sp.load_npz(f"{G1}/{t}.imp.npz").tocsr()
    for k, m in ((0, hh), (1, ~hh)):
        RAW[t, k] = np.asarray(M[m].sum(0)).ravel() / rd[m].sum() * 1e6; y = np.asarray(I[m].sum(0)).ravel(); IMP[t, k] = y / max(y.sum(), 1e-12) * 1e6
    print(t, n, flush=True)
TOT = {k: np.sum([RAW[t, k] for t in tfs], axis=0) for k in (0, 1)}
def lsum(v): return np.bincount(ag, v[ae], len(GL))
def wsum(v, W): g, e = WIN[W]; return np.bincount(g, v[e], len(GL))
def top(s, n): o = np.argsort(-s, kind="stable"); o = o[s[o] > 0][:n]; return frozenset(o)
METH = ["v8@250", "v8@500", "v8@1000", "v8@2000", "win50k@1000", "win100k@1000", "imputed@1000", "ratio_pool@1000", "resid_pool@1000"]
L = {}
for t in tfs:
    for k in (0, 1):
        c = RAW[t, k]; pool = (TOT[k] - c) / (NT - 1); b = np.dot(pool, c) / max(np.dot(pool, pool), 1e-12); s8 = lsum(c)
        for n in (250, 500, 1000, 2000): L["v8@%d" % n, t, k] = top(s8, n)
        L["win50k@1000", t, k] = top(wsum(c, 50000), 1000); L["win100k@1000", t, k] = top(wsum(c, 100000), 1000); L["imputed@1000", t, k] = top(lsum(IMP[t, k]), 1000)
        L["ratio_pool@1000", t, k] = top(lsum(c / (pool + 1)), 1000); L["resid_pool@1000", t, k] = top(lsum(np.maximum(c - b * pool, 0)), 1000)
for (mth, t, k), v in L.items(): open(f"{O}/run2b/lists/{mth.replace('@', '_')}_{t}_h{k + 1}.txt", "w").write("\n".join(GL[sorted(v)]) + "\n")
jac = lambda a, b: len(a & b) / max(len(a | b), 1); rows = []
for mth in METH:
    for t in tfs:
        jw = jac(L[mth, t, 0], L[mth, t, 1]); jb = np.mean([jac(L[mth, t, k], L[mth, u, k]) for u in tfs if u != t for k in (0, 1)])
        jbx = np.mean([jac(L[mth, t, k], L[mth, u, k]) for u in tfs if u != t and u not in PART.get(t, set()) for k in (0, 1)])
        rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", method=mth, tf=t, cells=NCELL[t], n_partners_excluded=len(PART.get(t, set())), J_within=jw, J_between=jb, diff=jw - jb, J_between_partnerex=jbx, diff_partnerex=jw - jbx))
P = pd.DataFrame(rows); P["depth_tertile"] = P.tf.map(pd.qcut(pd.Series(NCELL), 3, labels=["low", "mid", "high"]).astype(str)); P.to_csv(f"{O}/methods_per_tf_2b.tsv", sep="\t", index=False, float_format="%.5f")
DC = pd.read_csv(f"{O}/depth_curve.tsv", sep="\t"); bw, aw = np.polyfit(np.log10(DC.cells), DC.J_within, 1); dd = DC.dropna(subset=["diff_matched"]); bd, ad = np.polyfit(np.log10(dd.cells), dd.diff_matched, 1)
rng = np.random.default_rng(20261110); S = []
base = P[P.method == "v8@1000"].set_index("tf")
for mth in METH:
    for tert in ["all", "low", "mid", "high"]:
        G = P[(P.method == mth) & ((P.depth_tertile == tert) | (tert == "all"))].set_index("tf"); B = base.loc[G.index]
        dW, dB, dD = (G.J_within - B.J_within).values, (G.J_between - B.J_between).values, (G["diff"] - B["diff"]).values; dBx, dDx = (G.J_between_partnerex - B.J_between_partnerex).values, (G.diff_partnerex - B.diff_partnerex).values
        ci = lambda v: (np.percentile(v[rng.integers(0, len(v), (10000, len(v)))].mean(1), [2.5, 97.5]) if len(v) > 1 else [np.nan, np.nan])
        cw, cb, cd = ci(dW), ci(dB), ci(dD); cbx, cdx = ci(dBx), ci(dDx); medN = float(np.median(G.cells))
        eqW = 10 ** ((np.median(G.J_within) - aw) / bw) / 10 ** ((np.median(B.J_within) - aw) / bw)
        S.append(dict(DESCRIPTIVE="DESCRIPTIVE", method=mth, depth_tertile=tert, n_tfs=len(G), median_cells=medN, median_J_within=np.median(G.J_within), median_J_between=np.median(G.J_between), median_diff=np.median(G["diff"]),
                      dJ_within=dW.mean(), dJ_within_ci=f"[{cw[0]:.4f}, {cw[1]:.4f}]", dJ_between=dB.mean(), dJ_between_ci=f"[{cb[0]:.4f}, {cb[1]:.4f}]", d_diff=dD.mean(), d_diff_ci=f"[{cd[0]:.4f}, {cd[1]:.4f}]",
                      median_J_between_partnerex=np.median(G.J_between_partnerex), median_diff_partnerex=np.median(G.diff_partnerex), dJ_between_partnerex=dBx.mean(), dJ_between_partnerex_ci=f"[{cbx[0]:.4f}, {cbx[1]:.4f}]", d_diff_partnerex=dDx.mean(), d_diff_partnerex_ci=f"[{cdx[0]:.4f}, {cdx[1]:.4f}]",
                      rejected_both_up=bool(dW.mean() > 0 and dB.mean() > 0 and mth != "v8@1000"), equiv_cells_factor_Jwithin=eqW, diff_curve_slope=bd, equiv_cells_factor_diff=(10 ** (dD.mean() / bd) if abs(bd) > 1e-3 else np.nan)))
pd.DataFrame(S).to_csv(f"{O}/methods_summary_2b.tsv", sep="\t", index=False, float_format="%.5f")
