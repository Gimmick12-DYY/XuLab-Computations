# Ceiling-weighted pooled TF specificity for an M6 result folder (per_tf.tsv from grid3/grid4/model1; development TFs only).
# weights = b3b4 reliability_ceilings weight (r <= 0 -> 0), mean per panel TF; bootstrap over TFs 10,000, seed 20261116. Also the unweighted mean.
# Writes <dir>/pooled_weighted.tsv. usage: weighted_spec.py <dir> <reference_variant>
import sys
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m6_improve"; dd, ref = sys.argv[1], sys.argv[2]
P = pd.read_csv(f"{O}/{dd}/per_tf.tsv", sep="\t"); C = pd.read_csv(f"{O}/../m4_validation/b3b4/reliability_ceilings.tsv", sep="\t"); rng = np.random.default_rng(20261116); out = []
for a in ["arm2", "arm1"]:
    w = C[C.arm == a].groupby("panel_tf").weight.mean(); G = P[P.arm == a]; W = G.pivot(index="tf", columns="variant", values="specificity").dropna(); wt = w.reindex(W.index).fillna(0).values
    ix = rng.integers(0, len(W), (10000, len(W))); f = lambda e: (e[ix] * wt[ix]).sum(1) / np.maximum(wt[ix].sum(1), 1e-12); fu = lambda e: e[ix].mean(1)
    for v in W.columns:
        e = W[v].values; d = e - W[ref].values; bs, bd, bu = f(e), f(d), fu(e)
        out.append(dict(arm=a, variant=v, n_tfs=len(e), n_weighted=int((wt > 0).sum()), wspec=round(np.average(e, weights=wt), 3), lo=round(np.percentile(bs, 2.5), 3), hi=round(np.percentile(bs, 97.5), 3),
                        unweighted=round(e.mean(), 3), u_lo=round(np.percentile(bu, 2.5), 3), u_hi=round(np.percentile(bu, 97.5), 3), d_vs_ref=round(np.average(d, weights=wt), 3), d_lo=round(np.percentile(bd, 2.5), 3), d_hi=round(np.percentile(bd, 97.5), 3)))
S = pd.DataFrame(out).sort_values(["arm", "wspec"], ascending=[True, False]); S.to_csv(f"{O}/{dd}/pooled_weighted.tsv", sep="\t", index=False); pd.set_option("display.width", 250); print("reference", ref); print(S.to_string(index=False))
