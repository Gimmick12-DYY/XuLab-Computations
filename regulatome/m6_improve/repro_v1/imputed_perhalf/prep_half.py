# imputed_perhalf: subset a TF's unified bin x cell matrix (unified/work/<tf>/mm, read-only) to one Run 2b half and write it as <wd>/mm.
# usage: prep_half.py <TF> <half 1|2> <work_dir>
import gzip, os, sys
import numpy as np, scipy.io as sio, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; t, k, wd = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; i = tfs.index(t)
cb = np.load(f"{X}/regulatome/gate1/cellmats/{t}.cells.npz", allow_pickle=True)["barcodes"].astype(str); n = len(cb)
hh = np.random.default_rng(20261029 + i).permutation(n) < n // 2; keep = set(cb[hh] if k == 1 else cb[~hh])
src = f"{X}/unified/work/{t.lower()}/mm"; bc = np.array([l.strip() for l in gzip.open(f"{src}/barcodes.tsv.gz", "rt")])
cols = np.flatnonzero(np.isin(bc, list(keep))); M = sio.mmread(f"{src}/matrix.mtx.gz").tocsc()[:, cols]
os.makedirs(f"{wd}/mm", exist_ok=True); sio.mmwrite(f"{wd}/mm/matrix.mtx", sp.coo_matrix(M))
os.system(f"gzip -f {wd}/mm/matrix.mtx"); os.system(f"cp {src}/regions.tsv.gz {wd}/mm/")
with gzip.open(f"{wd}/mm/barcodes.tsv.gz", "wt") as fh: fh.write("\n".join(bc[cols]) + "\n")
if os.path.exists(f"{src}/matrix_info.tsv"): os.system(f"cp {src}/matrix_info.tsv {wd}/mm/")
print(t, k, "cells in half", len(keep), "matched in mm", len(cols), "of", len(bc), flush=True)
