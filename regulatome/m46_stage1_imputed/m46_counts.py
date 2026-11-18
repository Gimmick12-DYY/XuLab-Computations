# m46: raw vs imputed amounts per TF: lab MACS2 peaks, ATAC bins with a lab summit, imputed-positive ATAC bins (>= 1 cell, >= 1%, >= 5% of cells),
# cell x bin calls (raw vs after imputation; imputer threshold sparsity_match:3 = about 3 x raw calls genome-wide before the open-chromatin mask),
# bin overlap lab vs imputed (>= 1%), and Stage 1 genes at padding 0 (lab, I1a, I1b) from 3_jaccard_imputed_vs_lab.tsv.
import json, os
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; O = f"{X}/regulatome/m46_stage1_imputed/out"
reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]; key = {r: k for k, r in enumerate(reg)}; mi = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz", allow_pickle=True)["bins"]
pos = np.full(len(reg), -1); pos[mi] = np.arange(len(mi)); W = int(reg[0].split(":")[1].split("-")[1]) - int(reg[0].split(":")[1].split("-")[0])
J = pd.read_csv(f"{O}/3_jaccard_imputed_vs_lab.tsv", sep="\t", dtype={"padding": str}); J = J[J.padding == "0"]; KD = set(J.TF)
rows = []
for t in sorted(d for d in os.listdir(U) if os.path.exists(f"{U}/{d}/impute/meta.json")):
    m = json.load(open(f"{U}/{t}/impute/meta.json")); T = t.upper(); f = f"{V}/macs2/TF.{T}/TF.{T}_peaks.narrowPeak"; r = {"TF": T, "knockdown TF (63)": T in KD}
    if os.path.exists(f) and os.path.getsize(f):
        P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 8, 9], names=["c", "s", "q", "off"]); sm = P.s + P.off
        r.update({"lab MACS2 peaks": len(P), "lab peaks q < 0.05": int((P.q >= -np.log10(0.05)).sum())})
        b0 = (sm // W) * W; idx = np.array([key.get(f"{c}:{a}-{a + W}", -1) for c, a in zip(P.c.values, b0.values)]); ob = pos[idx[idx >= 0]]; labbins = np.unique(ob[ob >= 0])
        r["ATAC bins with a lab summit"] = len(labbins)
    else: labbins = np.array([], int)
    x = sp.load_npz(f"{U}/{t}/impute/matrix_csr.npz"); v = np.asarray(x.sum(1)).ravel()[mi]; nc = x.shape[1]; del x
    imp1 = np.flatnonzero(v >= max(3, .01 * nc))
    r.update({"cells": nc, "imputed ATAC bins, >= 1 cell": int((v > 0).sum()), "imputed ATAC bins, >= 1% cells": len(imp1), "imputed ATAC bins, >= 5% cells": int((v >= max(3, .05 * nc)).sum()),
              "raw cell x bin calls": m["raw_nnz"], "cell x bin calls after imputation (open bins)": m["output_nnz"], "imputed calls before open mask": m.get("predicted_nnz_pre_mask"), "threshold mode": m["threshold_mode"],
              "lab-summit bins also imputed (>= 1%), share": len(np.intersect1d(labbins, imp1)) / max(len(labbins), 1), "imputed (>= 1%) bins with a lab summit, share": len(np.intersect1d(labbins, imp1)) / max(len(imp1), 1)})
    for lst, col in (("I1a", "Stage 1 genes, padding 0, imputed >= 1% (I1a)"), ("I1b", "Stage 1 genes, padding 0, imputed >= 5% (I1b)")):
        g = J[(J.TF == T) & (J.list == lst)]; r[col] = int(g["imputed genes"].iloc[0]) if len(g) else np.nan
        if len(g): r["Stage 1 genes, padding 0, lab peaks"] = int(g["lab genes"].iloc[0])
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv(f"{O}/5_raw_vs_imputed_amounts.tsv", sep="\t", index=False, float_format="%.4g"); D.to_excel(f"{O}/5_raw_vs_imputed_amounts.xlsx", index=False)
num = D.select_dtypes("number").columns
for nm, d in (("all 79", D), ("63 knockdown TFs", D[D["knockdown TF (63)"]])): print("==", nm); print(d[num].describe().T[["50%", "min", "max"]].round(3).to_string())
low = D.sort_values("lab MACS2 peaks").TF.iloc[0]; print("shallowest (fewest lab peaks):", low)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 30)
print(D[D.TF.isin(["NRF1", "E4F1", "CTCF", "NFYA", "RBBP4", "ZNF777", "MAZ", low])].round(3).T.to_string())
