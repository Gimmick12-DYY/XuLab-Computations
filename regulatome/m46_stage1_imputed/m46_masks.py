# m46 Amendment 3, part 1: masks (Mzero, M5, M50, Mpeak, M0), pre-mask check, per-TF and across-TF summaries.
import os, gzip
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio
from scipy.stats import pearsonr
X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; M = f"{X}/regulatome/m46_stage1_imputed"; W = f"{M}/work"; O = f"{M}/out"
reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]; R = pd.DataFrame({"c": [r.split(":")[0] for r in reg], "s": [int(r.split(":")[1].split("-")[0]) for r in reg]}); NR = len(R)
OFF = {c: (g.index.values[0], g.s.values[0], len(g)) for c, g in R.groupby("c", sort=False)}
def bidx(c, p):
    out = np.full(len(p), -1)
    for ch in np.unique(c):
        if ch not in OFF: continue
        i0, s0, n = OFF[ch]; mm = c == ch; k = (p[mm] - s0) // 1000; ok = (k >= 0) & (k < n); r = np.full(mm.sum(), -1); r[ok] = i0 + k[ok]; out[mm] = r
    return out
A = np.load(f"{W}/atac_per_bin.npy"); mi = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz", allow_pickle=True)["bins"]
chrom_ok = R.c.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"]).values; bl = np.zeros(NR, bool)
B = pd.read_csv(f"{X}/downstream/cache/hg38-blacklist.v2.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
for c, s, e in zip(B.c, B.s, B.e):
    if c in OFF: i0, s0, n = OFF[c]; a, b = max(0, (s - s0) // 1000), min(n, (e - 1 - s0) // 1000 + 1); bl[i0 + a:i0 + b] = True
MAP = chrom_ok & ~bl; zero = MAP & (A == 0)
mz = MAP & (A > 0); cut5 = np.quantile(A[mz], 0.05); med = np.median(A[MAP])
MASK = {"M0": np.ones(NR, bool), "Mzero": mz, "M5": mz & (A > cut5), "M50": MAP & (A > med), "Mpeak": np.isin(np.arange(NR), mi)}
np.savez_compressed(f"{W}/masks.npz", **MASK)
MI = pd.DataFrame([{"mask": k, "bins kept": int(v.sum()), "share of all 1 kb bins": v.mean(), "share of mappable bins": (v & MAP).sum() / MAP.sum()} for k, v in MASK.items()])
MI["note"] = [f"mappable bins {MAP.sum():,}; zero-ATAC share of mappable {zero.sum() / MAP.sum():.3f}; M5 cut = ATAC count > {cut5:g}; M50 cut = > {med:g}"] + [""] * (len(MI) - 1)
MI.to_csv(f"{O}/10_masks.tsv", sep="\t", index=False, float_format="%.4g"); print(MI.to_string(index=False), flush=True)
TFS = sorted(d.upper() for d in os.listdir(U) if os.path.exists(f"{U}/{d}/impute/matrix_csr.npz.pre_openmask")); KD = set(l.strip() for l in open(f"{M}/tfs63.txt") if l.strip())
lg = lambda v: np.log1p(v / max(v.sum(), 1) * 1e6); rows, LR, LI = [], {}, {k: {} for k in MASK}; ok_tfs = []
for t in TFS:
    tl = t.lower(); bi = [l.strip() for l in open(f"{U}/{tl}/impute/barcodes.tsv")]; br = [l.strip() for l in gzip.open(f"{U}/{tl}/mm/barcodes.tsv.gz", "rt")]
    raw = sio.mmread(f"{U}/{tl}/mm/matrix.mtx.gz").tocsr(); raw.data[:] = 1; pre = sp.load_npz(f"{U}/{tl}/impute/matrix_csr.npz.pre_openmask").tocsr(); pre.data[:] = 1; cur = sp.load_npz(f"{U}/{tl}/impute/matrix_csr.npz").tocsr(); cur.data[:] = 1
    same_bc = bi == br
    only = (pre - pre.multiply(raw)).tocsr(); only.eliminate_zeros()
    rep = (only.multiply(MASK["Mpeak"][:, None]) + raw).tocsr(); rep.data[:] = 1; diff = abs(rep - cur).sum()
    r = {"TF": t, "knockdown TF": t in KD, "barcodes identical raw vs impute": same_bc, "raw calls": int(raw.nnz), "pre-mask imputed-only calls": int(only.nnz), "Mpeak re-applied equals current matrix (differing entries)": int(diff)}
    rv = np.asarray(raw.sum(1)).ravel().astype(float); orow = np.asarray(only.sum(1)).ravel().astype(float)
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]
    sb = bidx(P.c.values.astype(str), (P.s + P.off).values); sb = sb[sb >= 0]
    for k, mk in MASK.items():
        iv = rv + orow * mk; r[f"{k}: imputed-only calls kept"] = int((orow * mk).sum()); r[f"{k}: top-5,000 lab summits in mask"] = float(mk[sb].mean())
        r[f"{k}: Pearson log1p CPM raw vs imputed (mappable bins)"] = pearsonr(lg(rv)[MAP], lg(iv)[MAP])[0]; LI[k][t] = lg(iv)[MAP].astype(np.float32)
    LR[t] = lg(rv)[MAP].astype(np.float32); rows.append(r); ok_tfs.append(t) if diff == 0 and same_bc else None
    print(t, diff, same_bc, flush=True); del raw, pre, cur, only, rep
D = pd.DataFrame(rows); D.to_csv(f"{O}/11_masks_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
open(f"{W}/tfs_mask_ok.txt", "w").write("\n".join(t for t in ok_tfs if t in KD) + "\n")
def cmat(A_, B_): A_ = (A_ - A_.mean(1, keepdims=True)) / A_.std(1, keepdims=True); B_ = (B_ - B_.mean(1, keepdims=True)) / B_.std(1, keepdims=True); return A_ @ B_.T / A_.shape[1]
Rm = np.vstack([LR[t] for t in TFS]); off = ~np.eye(len(TFS), dtype=bool); S = []
for k in MASK:
    Im = np.vstack([LI[k][t] for t in TFS]); II = cmat(Im, Im); C0 = cmat(Rm, Im); C1 = cmat(Rm - Rm.mean(0), Im - Im.mean(0))
    S.append({"mask": k, "median pairwise Pearson among imputed tracks": float(np.median(II[off])), "raw vs imputed: diagonal row max (of %d)" % len(TFS): int((C0.argmax(1) == np.arange(len(TFS))).sum()), "median diagonal": float(np.median(np.diag(C0))), "median off-diagonal": float(np.median(C0[off])),
              "centred: diagonal row max": int((C1.argmax(1) == np.arange(len(TFS))).sum()), "centred: diagonal column max": int((C1.argmax(0) == np.arange(len(TFS))).sum()), "centred: median diagonal": float(np.median(np.diag(C1))), "centred: median off-diagonal": float(np.median(C1[off]))})
    del Im
S = pd.DataFrame(S); S.to_csv(f"{O}/12_masks_across_tfs.tsv", sep="\t", index=False, float_format="%.4g")
num = D.select_dtypes("number").columns; print(D[num].median().round(3).to_string()); print(S.round(3).to_string(index=False))
print("pre-mask check failed:", D[D["Mpeak re-applied equals current matrix (differing entries)"] > 0].TF.tolist(), "; barcodes differ:", D[~D["barcodes identical raw vs impute"]].TF.tolist()); print("done")
