# Per TF: raw read 5' ends per element, read/cell totals, and imputed cell calls per element (overlap-weighted)
import sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
from m1_common import load_elements, count_points

elements, tf, frag, imp_dir, regions_map, out = sys.argv[1:7]
E, idx = load_elements(elements)
n = len(E)
# raw: + strand records span the read (5' end = start); - strand records are the 1 bp 5' end
r = pd.read_csv(frag, sep="\t", header=None, usecols=[0, 1, 3, 5], names=["chrom", "start", "bc", "strand"],
                dtype={"chrom": str, "start": np.int64, "bc": str, "strand": str})
raw = count_points(idx, n, r.chrom.values, r.start.values)
tot = dict(tf=tf, reads=int(len(r)), cells=int(r.bc.nunique()), reads_in_elements=int(raw.sum()))
del r
# imputed: bins x cells binary CSR -> cells calling each bin -> elements by overlap fraction
m = np.load(f"{imp_dir}/matrix_csr.npz")
X = sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=tuple(m["shape"]))
binsum = np.asarray(X.sum(axis=1)).ravel()
M = sp.load_npz(regions_map)  # elements x bins, overlap fraction of each 1 kb bin inside the element
imp = M @ binsum
tot.update(imputed_cells=int(X.shape[1]), imputed_nnz=int(X.nnz), imputed_in_elements=float(imp.sum()))
np.savez_compressed(f"{out}/{tf}.npz", raw=raw.astype(np.int32), imp=imp.astype(np.float32))
json.dump(tot, open(f"{out}/{tf}.json", "w"))
print(tot)
