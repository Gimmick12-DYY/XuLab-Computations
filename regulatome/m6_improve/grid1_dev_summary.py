# Development-only summary of grid1/per_tf.tsv (split_v1.tsv); held-out rows sealed to heldout_SEALED.tsv, not summarised. usage: grid1_dev_summary.py
import os
import numpy as np, pandas as pd
from scipy import stats
O = "/work/users/d/y/dyy12/XuLab/regulatome/m6_improve"; G1 = f"{O}/grid1"; NB = 10000
assert not os.path.exists(f"{G1}/pooled_dev.tsv")
Sp = pd.read_csv(f"{O}/split_v1.tsv", sep="\t"); dev = set(Sp.tf[Sp.set == "development"]); ho = set(Sp.tf[Sp.set == "heldout"])
P = pd.read_csv(f"{G1}/per_tf.tsv", sep="\t"); assert set(P.tf) <= dev | ho, set(P.tf) - dev - ho
P[P.tf.isin(ho)].to_csv(f"{G1}/heldout_SEALED.tsv", sep="\t", index=False); D = P[P.tf.isin(dev)]; D.to_csv(f"{G1}/per_tf_dev.tsv", sep="\t", index=False)
rng = np.random.default_rng(20261109 + 7)
def bs(e):
    e = np.asarray(e, float); n = len(e); b = e[rng.integers(0, n, (NB, n))].mean(1); pos = int((e > 0).sum()); nz = int((e != 0).sum())
    return round(e.mean(), 3), round(np.percentile(b, 2.5), 3), round(np.percentile(b, 97.5), 3), pos, (float(f"{stats.binomtest(pos, nz, .5).pvalue:.3g}") if nz else 1.0)
S = []
for a, G in D.groupby("arm"):
    W = G.pivot(index="tf", columns="variant", values="effect").dropna(); Wo = G.pivot(index="tf", columns="variant", values="logor5").loc[W.index]
    for v in W.columns:
        e = bs(W[v]); o = bs(Wo[v]); d = bs(W[v] - W["sum_cpm@1000"]); da = bs(W[v] - W["atac_x_abc@1000"]); do = bs(Wo[v] - Wo["sum_cpm@1000"])
        S.append(dict(arm=a, variant=v, n_tfs=len(W), effect=e[0], effect_ci=f"[{e[1]}, {e[2]}]", logOR5=o[0], logOR5_ci=f"[{o[1]}, {o[2]}]",
                      d_vs_sum_cpm=d[0], d_vs_sum_cpm_ci=f"[{d[1]}, {d[2]}]", d_vs_sum_cpm_npos=d[3], d_vs_sum_cpm_sign_p=d[4], dlogOR_vs_sum_cpm=do[0], dlogOR_ci=f"[{do[1]}, {do[2]}]", d_vs_atac=da[0], d_vs_atac_ci=f"[{da[1]}, {da[2]}]"))
S = pd.DataFrame(S).sort_values(["arm", "effect"], ascending=[True, False]); S.to_csv(f"{G1}/pooled_dev.tsv", sep="\t", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(S.to_string(index=False)); print("dev TFs per arm:", D.groupby("arm").tf.nunique().to_dict(), "sealed rows:", int(P.tf.isin(ho).sum()))
