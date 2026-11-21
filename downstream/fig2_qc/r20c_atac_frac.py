#!/usr/bin/env python
"""Fraction of differential regions whose midpoint lies in the HEK293T ATAC union, per TF and set:
D-B (raw, round 6), D-I (round 6b, current imputed), D-I Mdrop5 masked, D-I Mdrop5 model-added only (top 1,000 / 5,000).
Writes OUT/round6_diffpeaks/imputed_mdrop5/atac_fraction.tsv."""
import numpy as np, pandas as pd
from fig2_common import ATAC_UNION, OUT, TFS
O6 = OUT / "round6_diffpeaks"; R = pd.read_csv(O6 / "rows.tsv", sep="\t")
A = pd.read_csv(ATAC_UNION, sep="\t", header=None, usecols=[0, 1, 2])
iv = {c: (g[1].values[np.argsort(g[1].values)], np.maximum.accumulate(g[2].values[np.argsort(g[1].values)])) for c, g in A.groupby(0)}
def inatac(ix):
    c = R.chrom.values[ix]; m = ((R.start.values[ix] + R.end.values[ix]) // 2)
    out = np.zeros(len(ix), bool)
    for ch in np.unique(c):
        if ch not in iv: continue
        s, e = iv[ch]; k = np.flatnonzero(c == ch); j = np.searchsorted(s, m[k], side="right") - 1
        out[k] = (j >= 0) & (e[np.clip(j, 0, None)] > m[k])
    return out
src = {"D-B raw (round 6)": (O6 / "diff_sets.npz", "D-B"), "D-I imputed current (6b)": (O6 / "imputed/DI_sets.npz", "D-I"),
       "D-I Mdrop5 masked": (O6 / "imputed_mdrop5/masked/DI_sets.npz", "D-I"), "D-I Mdrop5 model-added only": (O6 / "imputed_mdrop5/added/DI_sets.npz", "D-I")}
rows = []
for lab, (p, pre) in src.items():
    z = np.load(p)
    for n in (1000, 5000):
        for t in TFS:
            k = f"{pre}_{n}_{t}"
            if k in z.files and len(z[k]):
                rows.append(dict(set=lab, top=n, TF=t, n=len(z[k]), frac_in_ATAC=float(inatac(z[k]).mean())))
T = pd.DataFrame(rows); T.to_csv(O6 / "imputed_mdrop5/atac_fraction.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 200); print(T.pivot_table(index=["set", "top"], columns="TF", values="frac_in_ATAC").round(3).to_string())
