#!/usr/bin/env python
"""Build Mdrop5-masked imputed per-bin aggregates for the Fig 2 frame (round 6b rerun, 2026-11-20).
Mask Mdrop5 (m46 Amendment 5, regulatome/m46_stage1_imputed/work/masks_extra.npz, 2026-11-18; the September definition:
1 kb bins with mean ATAC CPM > 0.0781542 over the two pooled ATAC bigwigs). Per TF: raw = cells with >= 1 read per bin
(unified/work/<tf>/mm); pre = genome-wide imputed calls before the open mask (impute/matrix_csr.npz.pre_openmask);
model-added = pre minus raw-positive cells. Outputs per bin (cells positive):
  masked = raw + model-added inside Mdrop5  (raw calls always kept, as in m46)
  added  = model-added inside Mdrop5 only   (no copied raw reads)
ZNF768 has no pre_openmask file; its current matrix_csr.npz (ATAC-union mask) is used and flagged.
Usage: r20a_imp_mdrop5.py <task index 0-78>  -> INP/imp_mdrop5_tmp/<TF>.npz ;  r20a_imp_mdrop5.py merge"""
import os
import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from fig2_common import INP, bin_index

X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"
TMPD = INP / "imp_mdrop5_tmp"; TMPD.mkdir(exist_ok=True)
fr = np.load(INP / "frame.npz", allow_pickle=True); tf79 = [str(t) for t in fr["tf79"]]
bi, off, N = bin_index()
mask = np.load(f"{X}/regulatome/m46_stage1_imputed/work/masks_extra.npz")["Mdrop5"]
reg_ref = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]
assert len(mask) == len(reg_ref)

if sys.argv[1] == "merge":
    A, B, flags = [], [], []
    for t in tf79:
        z = np.load(TMPD / f"{t}.npz"); A.append(sp.csr_matrix(z["masked"])); B.append(sp.csr_matrix(z["added"])); flags.append(str(z["note"]))
    sp.save_npz(INP / "imp79_mdrop5.npz", sp.vstack(A).tocsr()); sp.save_npz(INP / "imp79_mdrop5_added.npz", sp.vstack(B).tocsr())
    print({t: f for t, f in zip(tf79, flags) if f}); print("merged")
    sys.exit()

t = tf79[int(sys.argv[1])]; tl = t.lower()
reg = [l.strip() for l in open(f"{U}/{tl}/impute/regions.tsv")]
assert reg == reg_ref, f"{t} regions differ from nfya"
raw = sio.mmread(f"{U}/{tl}/mm/matrix.mtx.gz").tocsr(); raw.data[:] = 1
fp = f"{U}/{tl}/impute/matrix_csr.npz.pre_openmask"; note = ""
if not os.path.exists(fp):
    fp = f"{U}/{tl}/impute/matrix_csr.npz"; note = "no pre_openmask; ATAC-union-masked matrix used"
pre = sp.load_npz(fp).tocsr(); pre.data[:] = 1
if pre.shape != raw.shape:
    pre = pre.T.tocsr()
assert pre.shape == raw.shape, (pre.shape, raw.shape)
added = (pre - pre.multiply(raw)).tocsr(); added.eliminate_zeros()
rc = np.asarray(raw.sum(1)).ravel().astype(np.float32); ac = np.asarray(added.sum(1)).ravel().astype(np.float32) * mask
c = np.array([r.split(":")[0] for r in reg]); s = np.array([int(r.split(":")[1].split("-")[0]) for r in reg])
keep = np.array([x in off for x in c]); idx = np.array([off[x][0] if x in off else 0 for x in c]) + s // 1000
vm = np.zeros(N, np.float32); va = np.zeros(N, np.float32)
vm[idx[keep]] = (rc + ac)[keep]; va[idx[keep]] = ac[keep]
np.savez_compressed(TMPD / f"{t}.npz", masked=vm, added=va, note=note)
print(t, "raw calls", int(rc.sum()), "added in mask", int(ac.sum()), "added total", int(added.sum()), note, flush=True)
