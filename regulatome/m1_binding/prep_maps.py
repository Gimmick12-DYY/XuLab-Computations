# One-off: elements x 1 kb bin overlap map (for imputed matrices) and OmniATAC Tn5 cut counts per element
import sys, gzip
import numpy as np, pandas as pd, scipy.sparse as sp
from m1_common import load_elements, count_points

elements, regions, out = sys.argv[1:4]
tagaligns = sys.argv[4:]
E, idx = load_elements(elements)
n = len(E)
reg = pd.read_csv(regions, header=None, names=["r"])
p = reg.r.str.extract(r"^(\w+):(\d+)-(\d+)$")
reg["chrom"], reg["s"] = p[0], p[1].astype(np.int64)
assert (reg.s % 1000 == 0).all()
row = {(c, s // 1000): i for i, (c, s) in enumerate(zip(reg.chrom.values, reg.s.values))}
I, J, W = [], [], []
for i, (c, s, e) in enumerate(zip(E.chrom.values, E.start.values, E.end.values)):
    for b in range(s // 1000, (e - 1) // 1000 + 1):
        j = row.get((c, b))
        if j is None: continue
        ov = min(e, (b + 1) * 1000) - max(s, b * 1000)
        I.append(i); J.append(j); W.append(ov / 1000.0)
M = sp.csr_matrix((W, (I, J)), shape=(n, len(reg)))
sp.save_npz(f"{out}/element_bin_map.npz", M)
print("bins mapped:", M.nnz, "elements with >=1 bin:", int((M.getnnz(axis=1) > 0).sum()), "of", n)
# ATAC Tn5 cut sites: + strand = start, - strand = end - 1 (tagAlign already Tn5-shifted)
atac = np.zeros(n, dtype=np.int64); total = 0
for f in tagaligns:
    for ch in pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 5], names=["chrom", "s", "e", "strand"],
                          dtype={"chrom": str}, chunksize=5_000_000):
        pos = np.where(ch.strand.values == "+", ch.s.values, ch.e.values - 1)
        atac += count_points(idx, n, ch.chrom.values, pos); total += len(ch)
np.savez_compressed(f"{out}/atac_counts.npz", atac=atac, total_cuts=total)
print("ATAC cuts:", total, "in elements:", int(atac.sum()))
