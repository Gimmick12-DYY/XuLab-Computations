# m46 Amendment 2: raw vs imputed agreement per TF at matched sizes, TF x TF diagonal, lab peaks in the imputed ranking.
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; M = f"{X}/regulatome/m46_stage1_imputed"; O = f"{M}/out"
reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]; R = pd.DataFrame({"c": [r.split(":")[0] for r in reg], "s": [int(r.split(":")[1].split("-")[0]) for r in reg]})
mi = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz", allow_pickle=True)["bins"]; NB = len(mi); pos = np.full(len(R), -1); pos[mi] = np.arange(NB)
OFF = {c: (g.index.values[0], g.s.values[0], len(g)) for c, g in R.groupby("c", sort=False)}
def bidx(c, p):
    out = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in OFF: continue
        i0, s0, n = OFF[ch]; mm = c == ch; k = (p[mm] - s0) // 1000; ok = (k >= 0) & (k < n); r = np.full(mm.sum(), -1); r[ok] = i0 + k[ok]; out[mm] = r
    return out
KD = set(l.strip() for l in open(f"{M}/tfs63.txt") if l.strip()); TFS = sorted(d.upper() for d in os.listdir(U) if os.path.exists(f"{U}/{d}/impute/matrix_csr.npz"))
cpm = lambda v: v / max(v.sum(), 1) * 1e6; lg = lambda v: np.log1p(cpm(v)); RAW, IMP, rows = {}, {}, []
for t in TFS:
    tl = t.lower(); ri = [l.strip() for l in __import__("gzip").open(f"{U}/{tl}/mm/regions.tsv.gz", "rt")] if t == TFS[0] else None
    if ri is not None: assert ri == reg, "raw region order differs"
    r = sio.mmread(f"{U}/{tl}/mm/matrix.mtx.gz").tocsr(); r.data[:] = 1; rg = np.asarray(r.sum(1)).ravel().astype(float); del r
    m = sp.load_npz(f"{U}/{tl}/impute/matrix_csr.npz"); ig = np.asarray(m.sum(1)).ravel().astype(float); del m
    raw, imp = rg[mi], ig[mi]; only = np.clip(imp - raw, 0, None); RAW[t], IMP[t] = raw, imp
    row = {"TF": t, "knockdown TF": t in KD, "raw calls in ATAC bins": int(raw.sum()), "imputed calls in ATAC bins": int(imp.sum()),
           "Spearman raw vs imputed": spearmanr(raw, imp).correlation, "Pearson log1p CPM raw vs imputed": pearsonr(lg(raw), lg(imp))[0],
           "Spearman raw vs imputed-only": spearmanr(raw, only).correlation, "Pearson log1p CPM raw vs imputed-only": pearsonr(lg(raw), lg(only))[0]}
    fp = f"{U}/{tl}/impute/matrix_csr.npz.pre_openmask"
    if os.path.exists(fp):
        q = sp.load_npz(fp)
        if q.nnz >= rg.sum(): qg = np.asarray(q.sum(1)).ravel().astype(float); row["genome-wide Pearson log1p CPM raw vs imputed (pre-mask)"] = pearsonr(lg(rg), lg(qg))[0]
        del q
    for N in (1000, 5000, 20000):
        a, b = np.argsort(-raw, kind="stable")[:N], np.argsort(-imp, kind="stable")[:N]; row[f"top {N:,} raw vs imputed bins, share shared"] = len(np.intersect1d(a, b)) / N; row[f"top {N:,}: chance"] = N / NB
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]).sort_values("sig", ascending=False)
    P["b"] = bidx(P.c.values.astype(str), (P.s + P.off).values); P["a"] = np.where(P.b >= 0, pos[np.maximum(P.b, 0)], -1)
    rk = np.empty(NB, int); rk[np.argsort(-imp, kind="stable")] = np.arange(NB)
    for nm, S in (("top 5,000 lab peaks", P.iloc[:5000]), ("lab peaks q < 0.05", P[P.q >= -np.log10(0.05)])):
        a = S.a.values[S.a.values >= 0]; row[f"{nm}: n with summit in ATAC bins"] = len(a)
        if len(a):
            for N in (5000, 20000): row[f"{nm}: share in top {N:,} imputed bins"] = float((rk[a] < N).mean())
            row[f"{nm}: share with any imputed cell"] = float((imp[a] > 0).mean()); row[f"{nm}: share with any raw cell"] = float((raw[a] > 0).mean()); row[f"{nm}: share imputed >= raw cells"] = float((imp[a] >= raw[a]).mean())
    S = P.iloc[:5000]; S = S[S.a >= 0]
    if len(S) > 10: row["Spearman lab signalValue vs raw count at summit"] = spearmanr(S.sig, raw[S.a]).correlation; row["Spearman lab signalValue vs imputed count at summit"] = spearmanr(S.sig, imp[S.a]).correlation
    rows.append(row); print(t, flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{O}/5b_raw_vs_imputed_agreement.tsv", sep="\t", index=False, float_format="%.4g"); D.to_excel(f"{O}/5b_raw_vs_imputed_agreement.xlsx", index=False)
LR = np.vstack([lg(RAW[t]) for t in TFS]); LI = np.vstack([lg(IMP[t]) for t in TFS])
def cmat(A, B): A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True); B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True); return A @ B.T / A.shape[1]
C0 = cmat(LR, LI); C1 = cmat(LR - LR.mean(0), LI - LI.mean(0)); off = ~np.eye(len(TFS), dtype=bool); S2 = []
for nm, C in (("as is", C0), ("per-bin mean across TFs removed", C1)):
    pd.DataFrame(C, index=TFS, columns=TFS).to_csv(f"{O}/5c_raw_vs_imputed_matrix_{'asis' if C is C0 else 'centred'}.tsv", sep="\t", float_format="%.4g")
    S2.append({"version": nm, "diagonal is row maximum (of 79)": int((C.argmax(1) == np.arange(len(TFS))).sum()), "diagonal is column maximum (of 79)": int((C.argmax(0) == np.arange(len(TFS))).sum()), "median diagonal": float(np.median(np.diag(C))), "median off-diagonal": float(np.median(C[off]))})
S2 = pd.DataFrame(S2); S2.to_csv(f"{O}/5c_raw_vs_imputed_matrix_summary.tsv", sep="\t", index=False, float_format="%.4g")
P_ = rp.use(); plt.rcParams.update({"font.size": 9, "xtick.labelsize": 5, "ytick.labelsize": 5, "axes.titlesize": 10})
fig, axs = plt.subplots(1, 2, figsize=(14, 7.2))
for ax, C, ttl, cm, lim in ((axs[0], C0, "Raw (rows) vs imputed (columns), Pearson log1p CPM", P_.seq_cmap(), (np.percentile(C0, 1), 1)), (axs[1], C1, "Same, per-bin mean across TFs removed", P_.div_cmap(), (-1, 1))):
    im = ax.imshow(C, cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest"); ax.set_xticks(range(len(TFS))); ax.set_yticks(range(len(TFS))); ax.set_xticklabels(TFS, rotation=90); ax.set_yticklabels(TFS); ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig.suptitle("79 TFs, 104,738 HEK293T ATAC bins (same TF order on both axes)", fontsize=10); fig.tight_layout(); fig.savefig(f"{O}/imputed_vs_raw_correlation.png", dpi=200); plt.close(fig)
num = D.select_dtypes("number").columns
for nm, d in (("all 79", D), ("63 knockdown TFs", D[D["knockdown TF"]])): print("==", nm); print(d[num].median().round(3).to_string())
pd.set_option("display.width", 300); print(D[D.TF.isin(["NRF1", "E4F1", "CTCF", "NFYA", "RBBP4", "ZNF777", "MAZ", "ZNF503"])].set_index("TF")[num].T.round(3).to_string()); print(S2.round(3).to_string(index=False)); print("done")
