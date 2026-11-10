# imputed_perhalf summary (README.md). Per TF x half: imputed bins -> elements (element_bin_map, as gate1_prep.py), summed over the half's cells,
# CPM, v8 rule top 1,000; paired raw v8 lists from Gate 1 cellmats on the same half. J_within, J_between (among the 6, same half), diff, Delta diff
# with bootstrap CI over the 6 TFs; outcome A/B/C per the pre-declared rules. Writes perhalf_per_tf.tsv, perhalf_summary.tsv, OUTCOME.txt.
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; I = f"{R}/m6_improve/repro_v1/imputed_perhalf"; G1 = f"{R}/gate1/cellmats"
TF6 = ["ZNF79", "PCGF6", "ZNF143", "E4F1", "SOX4", "XPA"]; tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); act = np.flatnonzero(E_.cls.isin(["promoter", "distal_active", "bivalent"]).values); Ea = E_.iloc[act].reset_index(drop=True)
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; ae = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int).values
GL = pd.Index(sorted(A.TargetGene.unique())); ag = GL.get_indexer(A.TargetGene.values)
MB = sp.load_npz(f"{R}/m1_binding/v2/counts/element_bin_map.npz").tocsr()[act]  # active elements x bins, as gate1_prep.py
def top(v, n=1000):
    s = np.bincount(ag, v[ae], len(GL)); o = np.argsort(-s, kind="stable"); o = o[s[o] > 0][:n]; return frozenset(GL[o])
def imputed_elements(wd):
    m = np.load(f"{wd}/impute/matrix_csr.npz"); Xb = sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=tuple(m["shape"]))  # bins x cells
    assert Xb.shape[0] == MB.shape[1], (wd, Xb.shape, MB.shape); return np.asarray(MB @ np.asarray(Xb.sum(1)).ravel()).ravel()
L = {}
for t in TF6:
    i = tfs.index(t); M = sp.load_npz(f"{G1}/{t}.raw.npz").tocsr(); rd = np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float); n = M.shape[0]; hh = np.random.default_rng(20261029 + i).permutation(n) < n // 2
    for k, m in ((1, hh), (2, ~hh)):
        L["v8", t, k] = top(np.asarray(M[m].sum(0)).ravel() / rd[m].sum() * 1e6)
        ye = imputed_elements(f"{I}/work/{t.lower()}_h{k}"); L["imputed", t, k] = top(ye / max(ye.sum(), 1e-12) * 1e6)
jac = lambda a, b: len(a & b) / max(len(a | b), 1); rows = []
for mth in ("v8", "imputed"):
    for t in TF6:
        jw = jac(L[mth, t, 1], L[mth, t, 2]); jb = np.mean([jac(L[mth, t, k], L[mth, u, k]) for u in TF6 if u != t for k in (1, 2)])
        rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", method=mth, tf=t, J_within=jw, J_between=jb, J_between_partnerex=jb, diff=jw - jb))
P = pd.DataFrame(rows); P.to_csv(f"{I}/perhalf_per_tf.tsv", sep="\t", index=False, float_format="%.5f")
W = P.pivot(index="tf", columns="method"); dW = (W.J_within.imputed - W.J_within.v8).values; dB = (W.J_between.imputed - W.J_between.v8).values; dD = (W["diff"].imputed - W["diff"].v8).values
rng = np.random.default_rng(20261110); bs = dD[rng.integers(0, 6, (10000, 6))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5]); m = dD.mean()
S = pd.DataFrame([dict(DESCRIPTIVE="DESCRIPTIVE", n_tfs=6, dJ_within=dW.mean(), dJ_between=dB.mean(), d_diff=m, d_diff_lo=lo, d_diff_hi=hi)]); S.to_csv(f"{I}/perhalf_summary.tsv", sep="\t", index=False, float_format="%.5f")
if m >= .05 and lo > 0 and dB.mean() < dW.mean(): out = "B PASS"
elif m < .05 and lo <= 0 <= hi: out = "A COLLAPSE"
else: out = "C INDETERMINATE"
open(f"{I}/OUTCOME.txt", "w").write(f"{out}: Delta diff {m:.4f} [{lo:.4f}, {hi:.4f}], dJ_within {dW.mean():.4f}, dJ_between {dB.mean():.4f}\n")
