# Ceiling-weighted pooled TF specificity from grid3/per_tf.tsv (development TFs only); weights = b3b4 reliability_ceilings weight (r <= 0 -> 0).
# Bootstrap over TFs 10,000, seed 20261116. Writes grid3/pooled_weighted.tsv. usage: grid3_weighted.py
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m6_improve"
P = pd.read_csv(f"{O}/grid3/per_tf.tsv", sep="\t"); C = pd.read_csv(f"{O}/../m4_validation/b3b4/reliability_ceilings.tsv", sep="\t"); rng = np.random.default_rng(20261116); out = []
for a in ["arm2", "arm1"]:
    w = C[C.arm == a].groupby("panel_tf").weight.mean(); G = P[P.arm == a]; W = G.pivot(index="tf", columns="variant", values="specificity").dropna(); wt = w.reindex(W.index).fillna(0).values
    ix = rng.integers(0, len(W), (10000, len(W))); f = lambda e: (e[ix] * wt[ix]).sum(1) / np.maximum(wt[ix].sum(1), 1e-12)
    for v in W.columns:
        e = W[v].values; d = e - W["sum_cpm@1000"].values; bs, bd = f(e), f(d)
        out.append(dict(arm=a, variant=v, n_tfs=len(e), n_weighted=int((wt > 0).sum()), wspec=round(np.average(e, weights=wt), 3), lo=round(np.percentile(bs, 2.5), 3), hi=round(np.percentile(bs, 97.5), 3), d_vs_sum_cpm=round(np.average(d, weights=wt), 3), dlo=round(np.percentile(bd, 2.5), 3), dhi=round(np.percentile(bd, 97.5), 3)))
pd.DataFrame(out).sort_values(["arm", "wspec"], ascending=[True, False]).to_csv(f"{O}/grid3/pooled_weighted.tsv", sep="\t", index=False)
