# Held-out split for M6 model tuning: by panel TF, 2/3 development, 1/3 held-out, stratified by Arm 2 KD reproducibility-ceiling tertile
# (b3b4/reliability_ceilings.tsv, mean per panel TF), seed 20261109; NRF1 forced to development; Arm 1 TFs inherit. Writes split_v1.tsv.
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/regulatome/m6_improve"; assert not os.path.exists(f"{O}/split_v1.tsv")
C = pd.read_csv(f"{X}/regulatome/m4_validation/b3b4/reliability_ceilings.tsv", sep="\t")
c2 = C[C.arm == "arm2"].groupby("panel_tf").ceiling.mean(); a1 = set(C[C.arm == "arm1"].panel_tf); a2 = set(c2.index)
T = pd.DataFrame(dict(tf=sorted(a1 | a2))); T["arm1_in"] = T.tf.isin(a1); T["arm2_in"] = T.tf.isin(a2); T["ceiling"] = T.tf.map(c2)
T["stratum"] = pd.qcut(T.ceiling, 3, labels=["low", "mid", "high"]).astype(str); T.loc[T.ceiling.isna(), "stratum"] = "na"
rng = np.random.default_rng(20261109); T["set"] = "development"
for s, G in T.groupby("stratum"):
    pool = [t for t in G.tf if t != "NRF1"]; k = int(round(len(G) / 3)); ho = rng.choice(pool, size=k, replace=False); T.loc[T.tf.isin(ho), "set"] = "heldout"
assert T.loc[T.tf == "NRF1", "set"].item() == "development"
T[["tf", "arm1_in", "arm2_in", "ceiling", "stratum", "set"]].to_csv(f"{O}/split_v1.tsv", sep="\t", index=False, float_format="%.4f")
print(T.groupby(["set", "stratum"]).size().to_string()); print(T.groupby("set")[["arm1_in", "arm2_in"]].sum().to_string())
