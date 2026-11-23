# m55 Amendment 4: per-gene score vs fold change (|log2FC|, pseudocount 0.1) over the Stage 1 pairs.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read(); exec(SRC50[:SRC50.index("# ---- responder counts")])  # TFS, UG, OK, L1, DEX
O = "/work/users/d/y/dyy12/XuLab/regulatome/m55_genes_per_tf_by_score/out"
F = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)
SC = pd.DataFrame(F["score"], index=F["tfs"].astype(str), columns=F["genes"].astype(str)).reindex(index=TFS, columns=UG).values; assert (np.isfinite(SC) == OK).all()
s = SC[OK]; f = L1[OK]; a = np.abs(s); af = np.abs(f); ex = np.broadcast_to(DEX[None, :], SC.shape)[OK]; terc = np.where(ex <= 2, "low", np.where(ex <= 6, "mid", "high"))
r1 = spearmanr(s, f).statistic; r2 = spearmanr(a, af).statistic; rt = {t: spearmanr(a[terc == t], af[terc == t]).statistic for t in ("low", "mid", "high")}
print("pairs", len(s), "| Spearman score vs log2FC", round(r1, 4), "| |score| vs |log2FC|", round(r2, 4), "| by expression tercile", {k: round(v, 4) for k, v in rt.items()}, flush=True)
pct = lambda x: 100 * (2 ** x - 1)
def row(lab, m):
    q = np.percentile(af[m], [25, 50, 75]) if m.any() else [np.nan] * 3
    d = {"|score|": lab, "pairs": int(m.sum()), "|log2FC| median": q[1], "|log2FC| IQR": f"{q[0]:.3f}-{q[2]:.3f}", "% change, median": pct(q[1]), "% change, IQR": f"{pct(q[0]):.1f}-{pct(q[2]):.1f}"}
    for t in ("low", "mid", "high"): mm = m & (terc == t); d[f"|log2FC| median, {t}-expression genes"] = float(np.median(af[mm])) if mm.any() else np.nan; d[f"pairs, {t}"] = int(mm.sum())
    return d
E = [0.05 * k for k in range(17)]
BAND = pd.DataFrame([row(f"{E[k]:.2f}-{E[k + 1]:.2f}", (a >= E[k]) & (a < E[k + 1])) for k in range(16)] + [row(">= 0.80", a >= 0.8)])
CUT = pd.DataFrame([row(f">= {c:.2f}", a >= c) for c in E[1:]])
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 20); print(BAND.round(3).to_string(index=False)); print(CUT.round(3).to_string(index=False))
with pd.ExcelWriter(f"{O}/score_vs_fold_change.xlsx") as w:
    pd.DataFrame([{"pairs": len(s), "Spearman score vs log2FC": r1, "Spearman |score| vs |log2FC|": r2, **{f"Spearman |score| vs |log2FC|, {k}-expression genes": v for k, v in rt.items()}}]).to_excel(w, sheet_name="correlation", index=False)
    BAND.to_excel(w, sheet_name="by |score| band", index=False); CUT.to_excel(w, sheet_name="by |score| cutoff", index=False)
    pd.DataFrame({"note": ["Score = paper per-gene score (per top-25% cell (log1p CP10k - own-GEM NTC mean) / own-GEM NTC SD, averaged over the TF's cells). log2FC = log2((mean CP10k KD + 0.1) / (GEM-weighted NTC mean CP10k + 0.1)), same cells. 63 TFs, own gene removed, Stage 1 universe.", "% change = 100 (2^|log2FC| - 1). Expression terciles from the NTC mean-expression deciles of the matched shuffle: low = deciles 1-3, mid = 4-7, high = 8-10."]}).to_excel(w, sheet_name="notes", index=False)
print("done")
