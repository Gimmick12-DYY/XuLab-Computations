# m52: Stage 1 split by direction (down / up) with the score responders; |score| 0.2-0.5 (0.1 supplementary); one shuffle run (m50 stage1 code).
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read(); exec(SRC50[:SRC50.index("# ---- responder counts")]); exec(SRC50[SRC50.index("# ---- Stage 1"):SRC50.index("ST, LOO = stage1(")])
O = "/work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score/out"
SC = pd.read_csv("/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out/paper_score_tf_by_gene.tsv.gz", sep="\t", index_col=0).reindex(index=TFS, columns=UG).values.astype(float); assert (np.isfinite(SC) == OK).all()
A = np.abs(np.nan_to_num(SC)); three = np.array([t in ("NRF1", "E4F1", "DNMT1") for t in TFS]); br = np.array([t in BROAD for t in TFS])
SETS = {"all 63 TFs": np.ones(len(TFS), bool), "without broad TFs (55)": ~br, "broad TFs only (8)": br, "without NRF1, E4F1, DNMT1": ~three, "without both": ~br & ~three}
VS = [("paper per-gene score", f"|score| >= {c:g}", OK & (A >= c), SC) for c in (0.2, 0.3, 0.4, 0.5, 0.1)]
ST, _ = stage1(VS, 20261118, False); ST["main / supplementary"] = np.where(ST.cutoff == "|score| >= 0.1", "supplementary", "main")
ST.to_csv(f"{O}/11_direction_split.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_paper_score_direction.xlsx") as w:
    for p in ("10 kb", "0", "50 kb"): ST[(ST.padding == p)].to_excel(w, sheet_name=f"padding {p}", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20)
print(ST[(ST.padding == "10 kb") & (ST["TF set"] == "all 63 TFs")][["cutoff", "direction", "TFs contributing", "Perturb-seq genes", "recall (%)", "shuffle mean (%)", "gap (points)", "observed / shuffle", "p (one-sided)", "precision (%)", "precision shuffle (%)"]].round(3).to_string(index=False)); print("done")
