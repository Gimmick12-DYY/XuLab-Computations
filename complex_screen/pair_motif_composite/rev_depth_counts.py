# NFY depth check (a), step 1: per TF, raw single-cell reads (read start) counted into the 104,738 union-ATAC 1 kb bins, after
# subsampling cells to a common count: N20 = cell count of the 20th-deepest TF (HOXB9, 5,903) and N3k = 3,000; 3 seeds each.
# TFs with fewer cells than N keep all cells (flagged in the summary). Output reversed/depth/counts/<TF>.npz (levels x seeds x bins).
import os, sys, gzip
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; D = f"{O}/reversed/depth"
FR = "/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"; t = sys.argv[1]
B = pd.read_csv(f"{O}/ref/bins1kb.bed", sep="\t", header=None, names=["c", "s", "e", "i"])
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
N20 = int(cells.sort_values(ascending=False).iloc[19]); LV = {"N20": N20, "N3k": 3000}
F = pd.read_csv(f"{FR}/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 3], names=["c", "p", "cell"])
bi = np.full(len(F), -1)
for c, g in B.groupby("c"):
    m = (F.c == c).values
    if not m.any(): continue
    s = g.s.values; k = np.searchsorted(s, F.p.values[m], side="right") - 1; ok = (k >= 0) & (F.p.values[m] < g.e.values[np.maximum(k, 0)])
    bi[np.flatnonzero(m)[ok]] = g.i.values[k[ok]]
cid, cu = pd.factorize(F.cell); nc = len(cu); rng = np.random.default_rng(20261110)
out = {}; info = dict(tf=t, n_cells=nc, n_reads=len(F), n_in_bins=int((bi >= 0).sum()))
for lv, n in LV.items():
    M = np.zeros((3, len(B)), np.float32)
    for sd in range(3):
        keep = np.zeros(nc, bool); keep[rng.choice(nc, min(n, nc), replace=False)] = True
        sel = keep[cid] & (bi >= 0); M[sd] = np.bincount(bi[sel], minlength=len(B))
    out[lv] = M; info[f"{lv}_cells"] = min(n, nc); info[f"{lv}_reads_mean"] = float(M.sum(1).mean())
np.savez_compressed(f"{D}/counts/{t}.npz", **out); pd.DataFrame([info]).to_csv(f"{D}/counts/{t}.info.tsv", sep="\t", index=False); print(info)
