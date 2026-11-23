# m54: Stage 1 by direction with the signed within-TF percentile (top X % up, bottom X % down) beside top X % of |score|; one shuffle run (m50 stage1 code).
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read(); exec(SRC50[:SRC50.index("# ---- responder counts")]); exec(SRC50[SRC50.index("# ---- Stage 1"):SRC50.index("ST, LOO = stage1(")])
O = "/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out"
_F = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True); SC = pd.DataFrame(_F["score"], index=_F["tfs"].astype(str), columns=_F["genes"].astype(str)).reindex(index=TFS, columns=UG).values.astype(float); assert (np.isfinite(SC) == OK).all()
PA = np.full(SC.shape, np.nan); PS = np.full(SC.shape, np.nan)
for i in range(len(TFS)): ok = OK[i]; n = ok.sum(); PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / n; PS[i, ok] = 100 * (rankdata(SC[i, ok]) - 0.5) / n
three = np.array([t in ("NRF1", "E4F1", "DNMT1") for t in TFS]); br = np.array([t in BROAD for t in TFS])
SETS = {"all 63 TFs": np.ones(len(TFS), bool), "without broad TFs (55)": ~br, "broad TFs only (8)": br, "without NRF1, E4F1, DNMT1": ~three, "without both": ~br & ~three}
VS = []
for x in (10, 5, 2, 1): VS += [("all (top % of |score|)", f"top {x}% of genes", OK & (PA > 100 - x), SC), ("up (top % of signed score)", f"top {x}% up", OK & (PS > 100 - x), SC), ("down (bottom % of signed score)", f"bottom {x}% down", OK & (PS < x), SC)]
ST, _ = stage1(VS, 20261118, False); ST = ST[ST.direction == "all"].drop(columns="direction")
ST.to_csv(f"{O}/11_direction_split.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_percentile_direction.xlsx") as w:
    for p in ("10 kb", "0", "50 kb"): ST[(ST.padding == p)].to_excel(w, sheet_name=f"padding {p}", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20)
print(ST[(ST.padding == "10 kb") & (ST["TF set"] == "all 63 TFs")][["version", "cutoff", "TFs contributing", "Perturb-seq genes", "recall (%)", "shuffle mean (%)", "gap (points)", "observed / shuffle", "p (one-sided)", "precision (%)", "precision shuffle (%)"]].round(3).to_string(index=False)); print("done")
