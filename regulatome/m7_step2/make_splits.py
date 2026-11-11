# m7_step2 step B: gene split (chromosome), TF folds, pair list, sha256. Written before any feature or fit. usage: make_splits.py
import hashlib, os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m7_step2"; os.makedirs(f"{O}/splits", exist_ok=True)
SEED = 20261111
TEST = {"chr3", "chr11", "chr17", "chr20"}; VAL = {"chr5", "chr12"}; PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
# gene split
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "TargetGene"]).drop_duplicates()
d2 = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); d1 = np.load(f"{R}/m4_validation/kd_response/out_v2/arm1_high_knockdown/kd_response.npz", allow_pickle=True)
g2 = set(d2["genes"][d2["ntc_detect"] >= .1].astype(str)); g1 = set(d1["genes"][d1["ntc_detect"] >= .1].astype(str))
n_multi = (A.groupby("TargetGene").chr.nunique() > 1).sum()
chrom = A.groupby("TargetGene").chr.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "")
uni = pd.Index(sorted(set(chrom.index) & (g1 | g2)))
G = pd.DataFrame(dict(gene=uni, chromosome=chrom.reindex(uni).values)); G["in_arm2"] = G.gene.isin(g2); G["in_arm1"] = G.gene.isin(g1)
no_chr = G.chromosome.eq("") | G.chromosome.isna(); nonprimary = ~G.chromosome.isin(PRIMARY) & ~no_chr
counts = dict(abc_genes=len(chrom), kd_genes_arm2=len(g2), kd_genes_arm1=len(g1), universe_before_drops=len(G), multi_chromosome_symbols_in_abc=int(n_multi), dropped_no_chromosome=int(no_chr.sum()), dropped_chrY_chrM_nonprimary=int(nonprimary.sum()))
counts["dropped_by_chrom"] = G.loc[nonprimary, "chromosome"].value_counts().to_dict()
G = G[~no_chr & ~nonprimary].copy(); G["set"] = np.where(G.chromosome.isin(TEST), "test", np.where(G.chromosome.isin(VAL), "val", "train"))
G.to_csv(f"{O}/splits/gene_split.tsv", sep="\t", index=False)
counts["universe_final"] = len(G); counts["set_sizes_arm2"] = G[G.in_arm2].set.value_counts().to_dict(); counts["set_sizes_arm1"] = G[G.in_arm1].set.value_counts().to_dict()
# TF folds
S = pd.read_csv(f"{R}/m6_improve/split_v1.tsv", sep="\t"); C = pd.read_csv(f"{R}/m4_validation/b3b4/reliability_ceilings.tsv", sep="\t"); c2 = C[C.arm == "arm2"].groupby("panel_tf").ceiling.mean()
F = pd.read_csv(f"{R}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf")
T = S[S.arm2_in & S.tf.isin(tfs)].copy(); assert set(T.tf) <= set(F.index)
T["ceiling"] = T.tf.map(c2)  # NaN ceiling (no reliable replicate split) -> own stratum "na"
T["cells"] = T.tf.map(F.cells); T["depth_tertile"] = T.tf.map(F.tertile); T["ceiling_tertile"] = pd.qcut(T.ceiling, 3, labels=["low", "mid", "high"]).astype(object).where(T.ceiling.notna(), "na").astype(str)
T["d"] = T.tf.map(F.d); T["claim_tier"] = T.d >= .04; T["split_v1_status"] = T["set"]; T = T.sort_values("tf").reset_index(drop=True)
rng = np.random.default_rng(SEED); T["fold"] = -1; k = 0
for (dt, ct), Sub in T.groupby(["depth_tertile", "ceiling_tertile"], sort=True):
    for i in rng.permutation(Sub.index.values): T.loc[i, "fold"] = k % 5; k += 1
assert (T.fold >= 0).all()
T.rename(columns={"tf": "TF"})[["TF", "fold", "depth_tertile", "ceiling_tertile", "split_v1_status", "cells", "ceiling", "d", "claim_tier", "arm1_in", "arm2_in"]].to_csv(f"{O}/splits/tf_folds.tsv", sep="\t", index=False, float_format="%.5g")
counts["testable_tfs"] = len(T); counts["claim_tier_tfs_testable"] = int(T.claim_tier.sum()); counts["fold_sizes"] = T.fold.value_counts().sort_index().to_dict(); counts["binding_tfs_not_testable"] = sorted(set(tfs) - set(T.tf)); counts["arm1_testable"] = int(T.arm1_in.sum())
# pair list
CTRL = ["MTF2-RBBP4", "RBBP4-ZIC2", "EZH2-MTF2", "CTCF-STAG2", "MAZ-ZNF143", "MLX-SREBF2"]
D = pd.read_csv(f"{X}/complex_screen/interface_audit_20261110/triple_support_flagged_v2.tsv", sep="\t")
P = D[D.af3_status.eq("af3_strong") | D.pair.isin(CTRL)][["pair", "reference", "best_pct", "af3_tier", "af3_ipsae", "af3_status", "tier"]].copy()
P["a"], P["b"] = zip(*P.pair.str.split("-")); P["control"] = P.pair.isin(CTRL)
bad = P[~(P.a.isin(tfs) & P.b.isin(tfs))]; counts["pairs_dropped_member_not_in_panel"] = bad.pair.tolist(); P = P[P.a.isin(tfs) & P.b.isin(tfs)]
assert all(c in set(P.pair) for c in CTRL); P.to_csv(f"{O}/pair_list.tsv", sep="\t", index=False, float_format="%.4g")
counts["pairs"] = len(P); counts["pairs_control"] = int(P.control.sum()); counts["pairs_af3_strong"] = int(P.af3_status.eq("af3_strong").sum())
print(counts)
with open(f"{O}/splits/splits.sha256", "w") as fh:
    for f in ("splits/gene_split.tsv", "splits/tf_folds.tsv", "pair_list.tsv"):
        fh.write(hashlib.sha256(open(f"{O}/{f}", "rb").read()).hexdigest() + "  " + f + "\n")
import json; json.dump(counts, open(f"{O}/splits/split_counts.json", "w"), indent=1, default=str)
