# B0 ablation arms, built before any knockdown data is opened: promoter-only, imputed residual, blend. All 79 TFs, active elements, power-law ABC links as in R2.
# promoter-only: Btilde summed over the gene's self-promoter elements. Imputed residual: z of log2((imputed CPM + 1) / (leave-one-out mean imputed CPM + 1)), through ABC. Blend: w z(Btilde) + (1 - w) B_imp through ABC, w = cells / (cells + 10,000).
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2_scores.py")).read().split("y = Y[i];")[0]
sys.argv = ["r2_scores.py", "tf", "MBD3"]; exec(src)
OA = f"{X}/regulatome/m4_validation/arms"; G_ = f"{X}/regulatome/gate1/cellmats"
for a in ("promoter_only", "imputed_residual", "blend"): os.makedirs(f"{OA}/{a}", exist_ok=True)
A2 = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "isSelfPromoter"]); A2["e"] = (A2.chr + ":" + A2.start.astype(str) + "-" + A2.end.astype(str)).map(coord); A2 = A2.dropna(subset=["e"]); A2 = A2[A2.isSelfPromoter.astype(bool)]
gi = {g: k for k, g in enumerate(genes)}; Wp = sp.csr_matrix((np.ones(len(A2)), (A2.e.astype(int).values, A2.TargetGene.map(gi).values)), shape=(nE, len(genes)))
I = np.zeros((len(tfs), nE)); cells = np.zeros(len(tfs))
for i, q in enumerate(tfs):
    m = sp.load_npz(f"{G_}/{q}.imp.npz"); I[i] = np.asarray(m.sum(0)).ravel(); cells[i] = m.shape[0]; del m
Ic = I / I.sum(1, keepdims=True) * 1e6; z = lambda v: (v - v.mean()) / v.std(); rows = []
for i, q in enumerate(tfs):
    y = Y[i]; pc = (Y.sum(0) - y) / (rd.sum() - rd[i]) * 1e6; Xd = np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp]); mu = glm_mu(y, np.full(nE, np.log(rd[i] / 1e6)), Xd); B = np.log2((y + .5) / (mu + .5))
    loo = (Ic.sum(0) - Ic[i]) / (len(tfs) - 1); Bi = z(np.log2((Ic[i] + 1) / (loo + 1))); wt = cells[i] / (cells[i] + 10000); Bl = wt * z(B) + (1 - wt) * Bi; r = dict(tf=q, cells=int(cells[i]), blend_w=round(wt, 3))
    fr = pd.read_csv(f"{O}/per_tf/{q}.tsv.gz", sep="\t", usecols=["gene", "S_k1"]).set_index("gene").S_k1.reindex(genes); top_r = set(fr.nlargest(1000).index)
    for a, v in (("promoter_only", Wp.T @ B), ("imputed_residual", W.T @ Bi), ("blend", W.T @ Bl)):
        v = np.asarray(v).ravel(); T = pd.DataFrame(dict(gene=genes, S=v)); T["rank"] = T.S.rank(ascending=False, method="first").astype(int); t10 = T.nsmallest(1000, "rank"); t10.to_csv(f"{OA}/{a}/{q}.tsv", sep="\t", index=False, float_format="%.5g"); r[f"{a}_ovl_raw_top1000"] = len(set(t10.gene) & top_r); r[f"{a}_nonzero"] = int((v != 0).sum())
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{OA}/arms_summary.tsv", sep="\t", index=False); pd.set_option("display.width", 200); print(R.describe().round(1).loc[["min", "50%", "max"]].to_string())
