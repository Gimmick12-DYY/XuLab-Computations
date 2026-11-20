#!/usr/bin/env python
"""Reconcile two raw-vs-imputed label checks (2026-11-20).
 m46 (2026-11-18, regulatome/m46_stage1_imputed/m46_raw_vs_imputed.py, commit 57e102b): raw = cells with >=1 read per ATAC
   bin (unified/work/<tf>/mm, binarised); imputed = cells positive per bin (impute/matrix_csr.npz summed); log1p CPM;
   for each RAW track, is its own imputed track the best of 79 (row max)? -> 77/79.
 r18 A2 (today): raw = 5' read counts per bin (ctcf_promoter_fp/counts/raw.npz); imputed = inputs/imp79.npz (same
   matrix_csr aggregate, mapped to the frame) / n cells; for each IMPUTED track, is its own raw track the best of 79? -> 2/79.
Here, on the same 79 TFs and the m46 ATAC bins (signal_mask, 104,738), compute all combinations:
   raw {cells (m46), reads (r18)} x imputed {all calls, imputed-only = imputed minus raw cells (calls the model added)}
   x {as is, per-bin mean across TFs removed}, and report own-is-best by row (per raw track) and by column (per imputed
   track), plus median diagonal / off-diagonal. The imputed matrix keeps the raw-positive cells (02_impute.py keep_raw),
   so "all calls" contains the raw signal by construction; "imputed-only" removes it."""
import gzip
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"
O = f"{X}/results_293T_analysis/fig2_qc/input_audit"
reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]
mi = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz", allow_pickle=True)["bins"]
TFS = sorted(d.upper() for d in os.listdir(U) if os.path.exists(f"{U}/{d}/impute/matrix_csr.npz"))
# raw read counts from the frame used in rounds 6/6b/6c (map m46 regions to frame bins)
C = f"{X}/results_293T_analysis/ctcf_promoter_fp/counts"
bi = pd.read_csv(f"{C}/bin_index.tsv", sep="\t"); off = dict(zip(bi.chrom, bi.offset))
stats = pd.read_csv(f"{C}/tf_stats.tsv", sep="\t"); rawc = sp.load_npz(f"{C}/raw.npz").tocsr()
rc = np.array([r.split(":")[0] for r in reg])[mi]; rs = np.array([int(r.split(":")[1].split("-")[0]) for r in reg])[mi]
fidx = np.array([off[c] for c in rc]) + rs // 1000
RC, RR, IA, IO = [], [], [], []
for t in TFS:
    tl = t.lower()
    r = sio.mmread(f"{U}/{tl}/mm/matrix.mtx.gz").tocsr(); r.data[:] = 1; rcell = np.asarray(r.sum(1)).ravel()[mi].astype(float); del r
    m = sp.load_npz(f"{U}/{tl}/impute/matrix_csr.npz"); imp = np.asarray(m.sum(1)).ravel()[mi].astype(float); del m
    RC.append(rcell); IA.append(imp); IO.append(np.clip(imp - rcell, 0, None))
    RR.append(rawc[list(stats.tf).index(t)].toarray().ravel()[fidx].astype(float))
    print(t, flush=True)
lg = lambda A: np.log1p(A / np.maximum(A.sum(1, keepdims=True), 1) * 1e6)
M = {"raw cells (m46)": lg(np.vstack(RC)), "raw reads (r18)": lg(np.vstack(RR))}
I = {"imputed all calls": lg(np.vstack(IA)), "imputed-only (model-added calls)": lg(np.vstack(IO))}


def cmat(A, B):
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)
    return A @ B.T / A.shape[1]


n = len(TFS); offd = ~np.eye(n, dtype=bool); rows = []
panel = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]
for rn, R in M.items():
    for iname, Im in I.items():
        for cen in ("as is", "per-bin mean removed"):
            A, B = (R, Im) if cen == "as is" else (R - R.mean(0), Im - Im.mean(0))
            Cm = cmat(A, B)   # rows raw, cols imputed
            rr = {"raw": rn, "imputed": iname, "version": cen,
                  "own imputed best for raw track (row max, of 79)": int((Cm.argmax(1) == np.arange(n)).sum()),
                  "own raw best for imputed track (col max, of 79)": int((Cm.argmax(0) == np.arange(n)).sum()),
                  "median diagonal": float(np.median(np.diag(Cm))), "median off-diagonal": float(np.median(Cm[offd]))}
            for t in panel:
                j = TFS.index(t)
                rr[f"{t}: rank of own raw for imputed (of 79)"] = int(1 + (Cm[:, j] > Cm[j, j]).sum())
                rr[f"{t}: best raw for imputed"] = TFS[int(Cm[:, j].argmax())]
            rows.append(rr)
            print(rr, flush=True)
T = pd.DataFrame(rows)
T.to_csv(f"{O}/imputed_label_reconcile.tsv", sep="\t", index=False, float_format="%.4g")
T.to_excel(f"{O}/imputed_label_reconcile.xlsx", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
print(T.iloc[:, :7].round(3).to_string())
print("done")
