#!/usr/bin/env python
"""Noise ceiling for the region-level sc-vs-bulk correlation (P-b regions, Pb3_counts.npz).

Each TF's sc reads in the regions are split at random into two halves (binomial 0.5, 20 draws);
reliability = Pearson r between the halves, on log2(CPM+1) (plain) and after within-assay centring
(each region's mean across the 6 sc tracks removed, as in P-b V2). Spearman-Brown gives the
full-data reliability rel = 2r/(1+r). The maximum attainable sc-vs-bulk r is ~ sqrt(rel_sc * rel_bulk);
with rel_bulk <= 1 the ceiling is sqrt(rel_sc). Done at full depth and at ZNF777 depth.
Also reports mean sc reads per region.
Output OUT/panels/split_half_ceiling.tsv.
"""
import numpy as np
import pandas as pd

from fig2_common import DM_DEPTH, INP, NT, OUT, TFS

z = np.load(OUT / "panels/Pb3_counts.npz", allow_pickle=True)
S = z["S"]
tot = np.array([sum(len(v) for v in np.load(INP / "five" / f"{t}.npz").values()) for t in TFS])


def lc(C):
    return np.log2(C / C.sum(0, keepdims=True) * 1e6 + 1)


rows = []
for depth in ("full", "znf777"):
    rng = np.random.default_rng(5)
    Sx = S if depth == "full" else rng.binomial(S.astype(np.int64), np.minimum(1, DM_DEPTH / tot)).astype(float)
    acc = {k: [] for k in ("plain", "centred")}
    for _ in range(20):
        A = rng.binomial(Sx.astype(np.int64), 0.5).astype(float)
        Bh = Sx - A
        La, Lb = lc(A), lc(Bh)
        acc["plain"].append([np.corrcoef(La[:, i], Lb[:, i])[0, 1] for i in range(NT)])
        Ca, Cb = La - La.mean(1, keepdims=True), Lb - Lb.mean(1, keepdims=True)
        acc["centred"].append([np.corrcoef(Ca[:, i], Cb[:, i])[0, 1] for i in range(NT)])
    for k, v in acc.items():
        r = np.mean(v, 0)
        rel = 2 * r / (1 + r)
        for i, t in enumerate(TFS):
            rows.append(dict(depth=depth, version=k, tf=t, mean_reads_per_region=float(Sx[:, i].mean()),
                             frac_regions_zero=float((Sx[:, i] == 0).mean()), split_half_r=r[i],
                             reliability_sb=rel[i], max_r_ceiling=float(np.sqrt(max(rel[i], 0)))))
df = pd.DataFrame(rows)
df.to_csv(OUT / "panels/split_half_ceiling.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 200)
print(df.to_string())
