# Within-local-cluster percentile for the all-elements, no-quiescent and TSS sets, same definition as local_cluster.py. The new percentiles are checked against the earlier sets.
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
D_ = "data/corr_sets_pearson"
def rd(p):
    m = pd.read_csv(p, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
for k in ("all", "noquies", "tss"):
    R = rd(f"{D_}/pearson_{k}.tsv"); P = rd(f"{D_}/pct_{k}.tsv").loc[R.index, R.index]; O = rd(f"data/corr_sets/pct_{k}.tsv").loc[R.index, R.index]; print(k, "pct differs from earlier by at most", float(np.abs(P.values - O.values).max())); T = list(R.index); Dm = 1 - R.values; Dm = (Dm + Dm.T) / 2; np.fill_diagonal(Dm, 0); root = to_tree(linkage(squareform(Dm, checks=False), "average"))
    for a, b in (("EZH2", "MTF2"), ("MLX", "SREBF2")):
        ia, ib = T.index(a), T.index(b); node = root
        while True:
            for ch in (node.left, node.right):
                if ch is not None and not ch.is_leaf() and {ia, ib} <= set(ch.pre_order()): node = ch; break
            else: break
        L = node.pre_order(); sub = P.values[np.ix_(L, L)]; v = sub[np.triu_indices(len(L), 1)]; x = P.loc[a, b]; rk = float((v <= x + 1e-12).mean()); print(k, a, b, "pct", round(x, 4), "local", round(rk, 4), "TFs", len(L))
