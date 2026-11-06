# 4 x 4 overlap of the per-TF promoter selections (MAZ, ZNF143, EZH2, MTF2) in pilot layer3_v2; diagonal = set size
import numpy as np, pandas as pd
L = "/work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs/layer3_v2"
a, b = np.load(f"{L}/MAZ_ZNF143/promoter/scatter.npz"), np.load(f"{L}/EZH2_MTF2/promoter/scatter.npz")
S = {"MAZ": set(a["top_a"]), "ZNF143": set(a["top_b"]), "EZH2": set(b["top_a"]), "MTF2": set(b["top_b"])}
n, k = int(a["setmask"].sum()), list(S)
T = pd.DataFrame([[len(S[x] & S[y]) for y in k] for x in k], index=k, columns=k)
T.to_csv(f"{L}/promoter_overlap_4tf.tsv", sep="\t")
m = len(S["MAZ"])
with open(f"{L}/promoter_overlap_4tf.tsv", "a") as f:
    f.write(f"# promoter set n = {n}; each TF selects {m} (5% cap); expected pairwise overlap = {m * m / n:.1f}; diagonal = set size\n")
print(open(f"{L}/promoter_overlap_4tf.tsv").read())
