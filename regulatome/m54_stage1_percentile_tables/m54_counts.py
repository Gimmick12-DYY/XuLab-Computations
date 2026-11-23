# m54: responders per percentile cutoff (63 / 55 non-broad / 8 broad) with the typical fold change (|log2FC|, pseudocount 0.1, m50) and |score| of those pairs, and the padding table with a gap row.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read(); exec(SRC50[:SRC50.index("# ---- responder counts")])  # TFS, UG, OK, L1, BROAD
O = "/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out"
_F = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True); SC = pd.DataFrame(_F["score"], index=_F["tfs"].astype(str), columns=_F["genes"].astype(str)).reindex(index=TFS, columns=UG).values.astype(float); assert (np.isfinite(SC) == OK).all()
PA = np.full(SC.shape, np.nan)
for i in range(len(TFS)): ok = OK[i]; PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / ok.sum()
pct = lambda x: 100 * (2 ** x - 1); rows = []
for vn, sel in (("all 63 TFs", np.ones(len(TFS), bool)), ("without broad TFs (55)", np.array([t not in BROAD for t in TFS])), ("broad TFs only (8)", np.array([t in BROAD for t in TFS]))):
    for x in (10, 5, 2, 1, "bottom 50% (non-changing)"):
        M = OK & ((PA > 100 - x) if x != "bottom 50% (non-changing)" else (PA < 50)); M[~sel] = False; n = M.sum(1)[sel]; f = np.abs(L1[M]); s = np.abs(SC[M]); q = np.percentile(f, [25, 50, 75]); qs = np.percentile(s, [25, 50, 75])
        rows.append({"TFs": vn, "cutoff": f"top {x}% of genes" if x != "bottom 50% (non-changing)" else x, "Perturb-seq genes (TF-gene pairs)": int(n.sum()), "TFs with >= 1 gene": int((n >= 1).sum()), "TFs with >= 10 genes": int((n >= 10).sum()), "median Perturb-seq genes per TF": float(np.median(n)),
                     "down (%)": 100 * float((SC[M] < 0).mean()), "|log2FC| median": q[1], "|log2FC| IQR": f"{q[0]:.3f}-{q[2]:.3f}", "fold change median (% change)": pct(q[1]), "fold change IQR (% change)": f"{pct(q[0]):.1f}-{pct(q[2]):.1f}", "|score| median": qs[1], "|score| IQR": f"{qs[0]:.3f}-{qs[2]:.3f}"})
C = pd.DataFrame(rows); C[~C.cutoff.str.startswith("bottom")].to_csv(f"{O}/9_counts_per_cutoff.tsv", sep="\t", index=False, float_format="%.4g")
F = pd.read_csv(f"{O}/10_padding_full.tsv", sep="\t", dtype={"padding": str}); TB = pd.read_csv(f"{O}/10_padding_table.tsv", sep="\t", dtype=str); G = []
for _, r in TB.iterrows():
    G.append(r.to_dict()); g = F[F.padding == r["Padding"]].set_index("cutoff"); gr = {"Padding": f"{r['Padding']}: gap (points)", "Median genes per TF": ""}
    for x in (10, 5, 2, 1): gr[f"top {x}%"] = f"{g.loc[f'top {x}% of genes', 'gap (points)']:+.1f}"
    G.append(gr)
with pd.ExcelWriter(f"{O}/stage1_percentile_counts_and_padding.xlsx") as w:
    C[~C.cutoff.str.startswith("bottom")].to_excel(w, sheet_name="counts per cutoff", index=False); C[C.cutoff.str.startswith("bottom")].to_excel(w, sheet_name="non-changing reference", index=False)
    TB.to_excel(w, sheet_name="padding table", index=False); pd.DataFrame(G).to_excel(w, sheet_name="padding table with gap row", index=False); F.to_excel(w, sheet_name="padding full", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 20); print(C.round(3).to_string(index=False)); print(pd.DataFrame(G).to_string(index=False)); print("done")
