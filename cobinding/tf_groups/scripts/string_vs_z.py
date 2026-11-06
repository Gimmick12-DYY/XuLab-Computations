"""Pair-level test: do TF pairs with STRING support (combined >=0.4; experimental >0) have higher sharing z than other
TF pairs? Mann-Whitney + 10,000 label permutations (TF-label shuffle keeps each TF's z profile). args: indir tag min_items"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
IN = Path(sys.argv[1]); TAG = sys.argv[2]; MIN = int(sys.argv[3]); ROOT = Path("/work/users/d/y/dyy12/XuLab")
z = pd.read_csv(IN / f"null_{TAG}_z.tsv.gz", sep="\t", index_col=0); dep = pd.read_csv(IN / f"null_{TAG}_depth.tsv", sep="\t", header=None, index_col=0)[1]
k = dep.index[dep >= MIN]; z = z.loc[k, k]
st = pd.read_csv(ROOT / "cobinding/data_external/string_network.tsv", sep="\t"); m = pd.read_csv(ROOT / "cobinding/data_external/string_map.tsv", sep="\t")
inv = dict(zip(m.preferredName, m["query"])); S = {}
for a, b, s, e in zip(st.preferredName_A, st.preferredName_B, st.score, st.escore): S[frozenset((inv.get(a, a), inv.get(b, b)))] = (s, e)
n = len(k); iu = np.triu_indices(n, 1); zv = z.values[iu]; rng = np.random.default_rng(1)
for lab, f in [("STRING>=0.4", lambda s: s[0] >= 0.4), ("STRING>=0.7", lambda s: s[0] >= 0.7), ("STRING exp>0", lambda s: s[1] > 0)]:
    M = np.array([[f(S.get(frozenset((a, b)), (0, 0))) for b in k] for a in k]); y = M[iu]
    if y.sum() < 3: print(lab, "too few"); continue
    obs = np.median(zv[y]) - np.median(zv[~y]); null = []
    for _ in range(10000):
        p = rng.permutation(n); yp = M[np.ix_(p, p)][iu]; null.append(np.median(zv[yp]) - np.median(zv[~yp]))
    print(f"{TAG} {lab}: n={y.sum()} median z {np.median(zv[y]):.2f} vs {np.median(zv[~y]):.2f}; MW p={mannwhitneyu(zv[y], zv[~y], alternative='greater').pvalue:.3g}; perm p={(np.sum(np.array(null) >= obs) + 1) / 10001:.3g}")
