#!/usr/bin/env python
"""Round 9 DIAGNOSTIC (not part of the frozen spec): 6x6 with unshrunken MLE log2FC, the same restricted to peaks with
>= 10 total reads in the TF's replicates, and the CTCF sanity check at the top 1,000 CTCF ChIP summits."""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import OUT, ROOT, TFS  # noqa: E402

O9 = OUT / "round9_deseq2"
pk = pd.read_csv(O9 / "peaks.bed", sep="\t", header=None)
FC = pd.read_csv(O9 / "chip_log2fc.tsv.gz", sep="\t")[TFS].values
D = {t: pd.read_csv(O9 / f"diag_mle_{t}.tsv.gz", sep="\t") for t in TFS}


def perm(M):
    k = len(M); o = ~np.eye(k, dtype=bool); st = lambda A: A.diagonal().mean() - A[o].mean()
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


for lab, minc in (("MLE, all peaks", 0), ("MLE, peaks with >= 10 reads in the TF's replicates", 10)):
    M = np.zeros((6, 6))
    for i, t in enumerate(TFS):
        x = D[t].log2FC_mle.values; m = np.isfinite(x) & (D[t].tf_total.values >= minc)
        for j in range(6):
            M[i, j] = np.corrcoef(x[m], FC[m, j])[0, 1]
    obs, p = perm(M); _, p5 = perm(M[1:, 1:])
    print(f"== {lab}: own-best rows {sum(np.argmax(M[i]) == i for i in range(6))}/6, diag-off {obs:+.3f}, p {p:.3f}, no-CTCF p {p5:.3f}")
    print(pd.DataFrame(M, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).round(3).to_string())
    for t in TFS:
        print(" ", t, "peaks used", int((np.isfinite(D[t].log2FC_mle.values) & (D[t].tf_total.values >= minc)).sum()))
# CTCF sanity: consensus peaks containing the top 1,000 CTCF ChIP summits
P = pd.read_csv(ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed", sep="\t", header=None, usecols=range(10))
P = P.sort_values(6, ascending=False).head(1000); sm = P[1] + P[9]
hit = np.zeros(len(pk), bool)
for c, g in P.assign(sm=sm).groupby(0):
    q = pk[pk[0] == c]; i = np.searchsorted(q[1].values, g.sm.values, side="right") - 1
    ok = (i >= 0) & (g.sm.values < q[2].values[np.clip(i, 0, None)]); hit[q.index.values[i[ok]]] = True
x = D["CTCF"].log2FC_mle.values; f = np.isfinite(x)
a, b = x[hit & f], x[~hit & f]
print(f"CTCF sanity: MLE log2FC at {hit.sum()} peaks holding the top-1,000 CTCF summits mean {a.mean():+.3f} (median {np.median(a):+.3f})"
      f" vs other peaks {b.mean():+.3f} (median {np.median(b):+.3f}); Wilcoxon rank-sum p = {mannwhitneyu(a, b, alternative='greater').pvalue:.2e}")
