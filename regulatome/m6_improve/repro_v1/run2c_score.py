# repro_v1 Run 2c step 2 (README Amendment 4): lists and scoring for v8@1000 (baseline, asserted equal to Run 2b), peak_fe@1000,
# peak_mask@1000, nbr_k1@1000, nbr_k2@1000. Writes run2c/lists/, run2c_per_tf.tsv, run2c_summary.tsv. DESCRIPTIVE. usage: run2c_score.py
import os
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; G1 = f"{R}/gate1/cellmats"; O = f"{R}/m6_improve/repro_v1"; P2 = f"{O}/run2c"
assert not os.path.exists(f"{O}/run2c_per_tf.tsv"); os.makedirs(f"{P2}/lists", exist_ok=True)
PP = pd.read_csv(f"{O}/run2_partners.tsv", sep="\t"); PART = {}
for a_, b_ in zip(PP.a, PP.b): PART.setdefault(a_, set()).add(b_); PART.setdefault(b_, set()).add(a_)
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); Ea = E_[E_.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; ae = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int).values
GL = pd.Index(sorted(A.TargetGene.unique())); ag = GL.get_indexer(A.TargetGene.values)
mid = ((Ea.start + Ea.end) // 2).values; CH = {c: np.flatnonzero(Ea.chrom.values == c) for c in Ea.chrom.unique()}
for c in CH: CH[c] = CH[c][np.argsort(mid[CH[c]], kind="stable")]
def nbr(v, k):
    out = np.zeros_like(v)
    for c, ix in CH.items():
        m = mid[ix]; cs = np.concatenate([[0], np.cumsum(v[ix])]); lo = np.searchsorted(m, m - 1000 * k); hi = np.searchsorted(m, m + 1000 * k, side="right"); out[ix] = cs[hi] - cs[lo]
    return out
ES = {c: (Ea.start.values[ix], Ea.end.values[ix], ix) for c, ix in ((c, np.flatnonzero(Ea.chrom.values == c)) for c in Ea.chrom.unique())}
def peak_fe(t, k):
    """per element: max fold enrichment of overlapping peaks (any bp), plus overlap indicator"""
    P = pd.read_csv(f"{P2}/peaks/{t}_h{k + 1}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 2, 6]); fe = np.zeros(len(Ea)); NP[t, k] = len(P)
    for c, G in P.groupby(0):
        if c not in ES: continue
        s, e, ix = ES[c]; o = np.argsort(s); s, e, ix = s[o], e[o], ix[o]; me = np.maximum.accumulate(e)  # elements are non-overlapping; me guards anyway
        for ps, pe, f in zip(G[1].values, G[2].values, G[6].values):
            j = np.searchsorted(s, pe, side="left")  # elements with start < pe
            jj = j - 1
            while jj >= 0 and me[jj] > ps:
                if e[jj] > ps: fe[ix[jj]] = max(fe[ix[jj]], f)
                jj -= 1
    return fe
RAW, NCELL, NP = {}, {}, {}
for i, t in enumerate(tfs):
    M = sp.load_npz(f"{G1}/{t}.raw.npz").tocsr(); rd = np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float); n = M.shape[0]; NCELL[t] = n
    hh = np.random.default_rng(20261029 + i).permutation(n) < n // 2
    for k, m in ((0, hh), (1, ~hh)): RAW[t, k] = np.asarray(M[m].sum(0)).ravel() / rd[m].sum() * 1e6
def lsum(v): return np.bincount(ag, v[ae], len(GL))
def top(s, n): o = np.argsort(-s, kind="stable"); o = o[s[o] > 0][:n]; return frozenset(o)
METH = ["v8@1000", "peak_fe@1000", "peak_mask@1000", "nbr_k1@1000", "nbr_k2@1000"]; L = {}
for t in tfs:
    for k in (0, 1):
        c = RAW[t, k]; fe = peak_fe(t, k)
        L["v8@1000", t, k] = top(lsum(c), 1000); L["peak_fe@1000", t, k] = top(lsum(fe), 1000); L["peak_mask@1000", t, k] = top(lsum(np.where(fe > 0, c, 0.0)), 1000)
        L["nbr_k1@1000", t, k] = top(lsum(nbr(c, 1)), 1000); L["nbr_k2@1000", t, k] = top(lsum(nbr(c, 2)), 1000)
    print(t, NCELL[t], "peaks", NP[t, 0], NP[t, 1], flush=True)
for t in tfs:  # baseline must reproduce Run 2b exactly
    for k in (0, 1): assert set(GL[sorted(L["v8@1000", t, k])]) == set(x.strip() for x in open(f"{O}/run2b/lists/v8_1000_{t}_h{k + 1}.txt") if x.strip()), (t, k)
for (mth, t, k), v in L.items(): open(f"{P2}/lists/{mth.replace('@', '_')}_{t}_h{k + 1}.txt", "w").write("\n".join(GL[sorted(v)]) + "\n")
jac = lambda a, b: len(a & b) / max(len(a | b), 1); rows = []
for mth in METH:
    for t in tfs:
        jw = jac(L[mth, t, 0], L[mth, t, 1]); jb = np.mean([jac(L[mth, t, k], L[mth, u, k]) for u in tfs if u != t for k in (0, 1)])
        jbx = np.mean([jac(L[mth, t, k], L[mth, u, k]) for u in tfs if u != t and u not in PART.get(t, set()) for k in (0, 1)])
        rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", method=mth, tf=t, cells=NCELL[t], peaks_h1=NP[t, 0], peaks_h2=NP[t, 1], n_list_h1=len(L[mth, t, 0]), n_list_h2=len(L[mth, t, 1]),
                         n_partners_excluded=len(PART.get(t, set())), J_within=jw, J_between=jb, diff=jw - jb, J_between_partnerex=jbx, diff_partnerex=jw - jbx))
P = pd.DataFrame(rows); P["depth_tertile"] = P.tf.map(pd.qcut(pd.Series(NCELL), 3, labels=["low", "mid", "high"]).astype(str)); P.to_csv(f"{O}/run2c_per_tf.tsv", sep="\t", index=False, float_format="%.5f")
K = 0.058 / 0.24; rng = np.random.default_rng(20261110); S = []; base = P[P.method == "v8@1000"].set_index("tf")
ci = lambda v: (np.percentile(v[rng.integers(0, len(v), (10000, len(v)))].mean(1), [2.5, 97.5]) if len(v) > 1 else [np.nan, np.nan])
for mth in METH[1:]:
    for tert in ["all", "low", "mid", "high"]:
        G = P[(P.method == mth) & ((P.depth_tertile == tert) | (tert == "all"))].set_index("tf"); B = base.loc[G.index]
        dW, dB, dD = (G.J_within - B.J_within).values, (G.J_between - B.J_between).values, (G["diff"] - B["diff"]).values
        dBx, dDx = (G.J_between_partnerex - B.J_between_partnerex).values, (G.diff_partnerex - B.diff_partnerex).values
        cw, cb, cd, cbx, cdx = ci(dW), ci(dB), ci(dD), ci(dBx), ci(dDx); band = K * max(dW.mean(), 0)
        S.append(dict(DESCRIPTIVE="DESCRIPTIVE", method=mth, depth_tertile=tert, n_tfs=len(G), median_cells=float(np.median(G.cells)),
                      median_J_within=np.median(G.J_within), median_J_between=np.median(G.J_between), median_diff=np.median(G["diff"]),
                      dJ_within=dW.mean(), dJ_within_ci=f"[{cw[0]:.4f}, {cw[1]:.4f}]", dJ_between=dB.mean(), dJ_between_ci=f"[{cb[0]:.4f}, {cb[1]:.4f}]",
                      d_diff=dD.mean(), d_diff_ci=f"[{cd[0]:.4f}, {cd[1]:.4f}]", null_band=band, beats_depth=bool(dD.mean() > band and cd[0] > 0),
                      dJ_between_partnerex=dBx.mean(), dJ_between_partnerex_ci=f"[{cbx[0]:.4f}, {cbx[1]:.4f}]", d_diff_partnerex=dDx.mean(),
                      d_diff_partnerex_ci=f"[{cdx[0]:.4f}, {cdx[1]:.4f}]", beats_depth_partnerex=bool(dDx.mean() > band and cdx[0] > 0),
                      both_components_up_flag=bool(dW.mean() > 0 and dB.mean() > 0)))
pd.DataFrame(S).to_csv(f"{O}/run2c_summary.tsv", sep="\t", index=False, float_format="%.5f")
