# Peak calling on the imputed bin x cell matrices (unified/work/<tf>/impute/matrix_csr.npz: 1 kb bins x cells, binary; raw reads kept
# genome-wide, imputed calls only inside union ATAC). Per TF and bin: k = cells with signal, n = TF cells. Control = the other 78 TFs
# pooled (same imputation, same mask): p0 = (K - k + 1) / (N - n + 2). Score = -log10 one-sided binomial P(X >= k | n, p0); BH q over all
# 3.03 M bins; peaks = q < 0.05. Gate scores are written for the union-ATAC bins used by recall_pools.py (variant imp_binom).
# Output recall/imputed_scores_binom.npz (binom = -log10 p on mask bins, tfs), recall/imputed_binom_peaks.tsv (peaks per TF, in/out of ATAC)
import os
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import binom
X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"; R = f"{X}/complex_screen/pair_motif_composite/recall"
z = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz"); tfs = [t.upper() for t in z["tfs"]]; mi = z["bins"]
K = None; k_, n_ = {}, {}
for t in tfs:
    m = sp.load_npz(f"{U}/{t.lower()}/impute/matrix_csr.npz"); v = np.asarray(m.sum(1)).ravel().astype(np.float64); k_[t] = v; n_[t] = m.shape[1]
    K = v.copy() if K is None else K + v
Nt = sum(n_.values()); inmask = np.zeros(len(K), bool); inmask[mi] = True; out = np.zeros((len(tfs), len(mi)), np.float32); rows = []
for j, t in enumerate(tfs):
    k, n = k_[t], n_[t]; p0 = (K - k + 1) / (Nt - n + 2); lp = -binom.logsf(k - 1, n, p0) / np.log(10); lp[k == 0] = 0
    p = 10 ** -lp; o = np.argsort(p); r = np.empty_like(o); r[o] = np.arange(1, len(p) + 1); q = np.minimum.accumulate((p[o] * len(p) / r[o])[::-1])[::-1]; qq = np.empty_like(q); qq[o] = q
    out[j] = lp[mi]; pk = qq < 0.05; rows.append(dict(tf=t, cells=n, peaks_q05=int(pk.sum()), in_atac=int((pk & inmask).sum()), frac_in_atac=round((pk & inmask).sum() / max(pk.sum(), 1), 3)))
    print(rows[-1], flush=True)
np.savez_compressed(f"{R}/imputed_scores_binom.npz", binom=out, tfs=np.array(tfs)); pd.DataFrame(rows).to_csv(f"{R}/imputed_binom_peaks.tsv", sep="\t", index=False)
