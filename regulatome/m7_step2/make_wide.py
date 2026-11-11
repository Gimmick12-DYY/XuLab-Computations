# m7_step2 Amendment 7 (EXPLORATORY): widened gene universe = all ABC genes (power-law prediction TargetGenes) on primary chromosomes, same split rule as make_splits.py.
# usage: make_wide.py -> splits/gene_split_wide.tsv, wide_exploratory/wide_split_counts.tsv
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m7_step2"; W = f"{O}/wide_exploratory"; os.makedirs(W, exist_ok=True)
TEST = {"chr3", "chr11", "chr17", "chr20"}; VAL = {"chr5", "chr12"}; PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "TargetGene"]).drop_duplicates()
chrom = A.groupby("TargetGene").chr.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "")
G = pd.DataFrame(dict(gene=chrom.index, chromosome=chrom.values)); n0 = len(G); G = G[G.chromosome.isin(PRIMARY)].copy()
G["set"] = np.where(G.chromosome.isin(TEST), "test", np.where(G.chromosome.isin(VAL), "val", "train"))
d = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True)
mean = pd.Series(d["ntc_mean"].astype(float), index=d["genes"].astype(str)); mean = mean[~mean.index.duplicated(keep="last")]
det = pd.Series(d["ntc_detect"].astype(float), index=d["genes"].astype(str)); det = det[~det.index.duplicated(keep="last")]
G["in_matrix"] = G.gene.isin(mean.index); G["y_level"] = mean.reindex(G.gene).fillna(0.0).values; G["y_detect"] = (det.reindex(G.gene).fillna(0.0).values >= .1).astype(int)
G = G.sort_values("gene").reset_index(drop=True); G.to_csv(f"{O}/splits/gene_split_wide.tsv", sep="\t", index=False)
rows = [dict(label="EXPLORATORY (Amendment 7, post-hoc)", split=s, n_genes=len(H), frac_absent_set_to_0=round(float((~H.in_matrix).mean()), 4), frac_level_eq_0=round(float((H.y_level == 0).mean()), 4), frac_detected_ge10pct=round(float(H.y_detect.mean()), 4)) for s, H in [("all", G)] + list(G.groupby("set"))]
C = pd.DataFrame(rows); C.to_csv(f"{W}/wide_split_counts.tsv", sep="\t", index=False); print("abc symbols", n0, "dropped non-primary or multi-chromosome", n0 - len(G)); print(C.to_string(index=False))
