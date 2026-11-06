# Shared helpers for M1 v2 (elements_v2.bed, 8 columns incl. class): element index and point counting
import numpy as np, pandas as pd

def load_elements(path):
    E = pd.read_csv(path, sep="\t", header=None,
                    names=["chrom", "start", "end", "element_id", "type", "parent_id", "parent_type", "class"])
    idx = {}
    for c, d in E.groupby("chrom", sort=False):
        idx[c] = (d.start.values, d.end.values, d.index.values)
    return E, idx

def count_points(idx, n, chrom, pos):
    """Count 0-based positions falling inside elements (elements are non-overlapping)."""
    out = np.zeros(n, dtype=np.int64)
    for c in np.unique(chrom):
        if c not in idx: continue
        s, e, ix = idx[c]
        p = pos[chrom == c]
        k = np.searchsorted(s, p, side="right") - 1
        ok = (k >= 0) & (p < e[np.clip(k, 0, None)])
        np.add.at(out, ix[k[ok]], 1)
    return out
