# Gate 1 prep, per TF: cell x element matrices over the ACTIVE classes of elements_v2 (promoter, distal_active, bivalent)
#   raw : read 5' ends per element per cell (fragments on /vast, read-only)
#   imp : imputed calls per element per cell (1 kb bins x cells binary, overlap-weighted via the v2 element-bin map)
# Writes gate1/cellmats/<TF>.npz (raw, imp CSR cells x elements, barcodes, reads per cell)
import sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2")
from m1_common import load_elements
elements, tf, frag, imp_dir, bin_map, out = sys.argv[1:7]
E, _ = load_elements(elements)
act = np.flatnonzero(E["class"].isin(["promoter", "distal_active", "bivalent"]).values)
Ea = E.iloc[act].reset_index(drop=True)
idx = {c: (d.start.values, d.end.values, d.index.values) for c, d in Ea.groupby("chrom", sort=False)}
r = pd.read_csv(frag, sep="\t", header=None, usecols=[0, 1, 3], names=["chrom", "start", "bc"],
                dtype={"chrom": str, "start": np.int64, "bc": str})
bcs, cell = np.unique(r.bc.values, return_inverse=True)
reads_per_cell = np.bincount(cell, minlength=len(bcs))
I, J = [], []
for c in r.chrom.unique():
    if c not in idx: continue
    s, e, ix = idx[c]; m = (r.chrom.values == c)
    p = r.start.values[m]; k = np.searchsorted(s, p, side="right") - 1
    ok = (k >= 0) & (p < e[np.clip(k, 0, None)])
    I.append(cell[m][ok]); J.append(ix[k[ok]])
I, J = np.concatenate(I), np.concatenate(J)
raw = sp.csr_matrix((np.ones(len(I), np.float32), (I, J)), shape=(len(bcs), len(Ea)))
raw.sum_duplicates()
# imputed: bins x cells -> cells x active elements
m = np.load(f"{imp_dir}/matrix_csr.npz")
X = sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=tuple(m["shape"]))
ibc = pd.read_csv(f"{imp_dir}/barcodes.tsv", header=None, sep="\t")[0].astype(str).values
M = sp.load_npz(bin_map).tocsr()[act]  # active elements x bins
imp = (M @ X).T.tocsr().astype(np.float32)  # imputed cells x elements
pos = {b: i for i, b in enumerate(ibc)}
order = np.array([pos.get(b, -1) for b in bcs])
assert (order >= 0).mean() > 0.99, f"{tf}: only {(order >= 0).mean():.3f} of raw barcodes in imputed"
keep = order >= 0
raw, bcs, reads_per_cell, imp = raw[keep], bcs[keep], reads_per_cell[keep], imp[order[keep]]
sp.save_npz(f"{out}/{tf}.raw.npz", raw); sp.save_npz(f"{out}/{tf}.imp.npz", imp)
np.savez_compressed(f"{out}/{tf}.cells.npz", barcodes=bcs, reads=reads_per_cell)
print(tf, "cells", len(bcs), "reads", int(reads_per_cell.sum()), "in active", int(raw.sum()), "raw nnz", raw.nnz, "imp nnz", imp.nnz)
