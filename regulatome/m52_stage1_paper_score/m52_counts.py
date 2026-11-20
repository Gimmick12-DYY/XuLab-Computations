# m52: responders per score cutoff (63 / 55 non-broad / 8 broad) with the typical fold change of those pairs (|log2FC|, pseudocount 0.1, m50), and the padding table with a gap row.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read(); exec(SRC50[:SRC50.index("# ---- responder counts")])  # TFS, UG, OK, L1 (log2FC pc 0.1), BROAD
O = "/work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score/out"
SC = pd.read_csv("/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out/paper_score_tf_by_gene.tsv.gz", sep="\t", index_col=0).reindex(index=TFS, columns=UG).values.astype(float); assert (np.isfinite(SC) == OK).all()
A = np.abs(np.nan_to_num(SC)); pct = lambda x: 100 * (2 ** x - 1); rows = []
for vn, sel in (("all 63 TFs", np.ones(len(TFS), bool)), ("without broad TFs (55)", np.array([t not in BROAD for t in TFS])), ("broad TFs only (8)", np.array([t in BROAD for t in TFS]))):
    for c in (0.1, 0.2, 0.3, 0.4, 0.5):
        M = OK & (A >= c); M[~sel] = False; n = M.sum(1)[sel]; f = np.abs(L1[M]); q = np.percentile(f, [25, 50, 75]) if M.any() else [np.nan] * 3
        rows.append({"TFs": vn, "cutoff": f"|score| >= {c:g}", "main / supplementary": "supplementary" if c == 0.1 else "main", "Perturb-seq genes (TF-gene pairs)": int(n.sum()), "TFs with >= 1 gene": int((n >= 1).sum()), "TFs with >= 10 genes": int((n >= 10).sum()),
                     "median Perturb-seq genes per TF": float(np.median(n)), "down (%)": 100 * float((SC[M] < 0).mean()) if M.any() else np.nan, "|log2FC| median": q[1], "|log2FC| IQR": f"{q[0]:.3f}-{q[2]:.3f}", "fold change median (% change)": pct(q[1]), "fold change IQR (% change)": f"{pct(q[0]):.1f}-{pct(q[2]):.1f}"})
C = pd.DataFrame(rows); C[C["main / supplementary"] == "main"].drop(columns="main / supplementary").to_csv(f"{O}/9_counts_per_cutoff.tsv", sep="\t", index=False, float_format="%.4g")
F = pd.read_csv(f"{O}/10_padding_full.tsv", sep="\t", dtype={"padding": str}); TB = pd.read_csv(f"{O}/10_padding_table.tsv", sep="\t", dtype=str); G = []
for _, r in TB.iterrows():
    G.append(r.to_dict()); g = F[F.padding == r["Padding"]].set_index("cutoff"); gr = {"Padding": f"{r['Padding']}: gap (points)", "Median genes per TF": ""}
    for c in (0.2, 0.3, 0.4, 0.5): gr[f"|score| ≥ {c:g}"] = f"{g.loc[f'|score| >= {c:g}', 'gap (points)']:+.1f}"
    G.append(gr)
with pd.ExcelWriter(f"{O}/stage1_paper_score_counts_and_padding.xlsx") as w:
    C[C["main / supplementary"] == "main"].drop(columns="main / supplementary").to_excel(w, sheet_name="counts per cutoff", index=False); C[C["main / supplementary"] == "supplementary"].to_excel(w, sheet_name="supplementary |score| 0.1", index=False)
    TB.to_excel(w, sheet_name="padding table", index=False); pd.DataFrame(G).to_excel(w, sheet_name="padding table with gap row", index=False); F.to_excel(w, sheet_name="padding full", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); print(C.round(3).to_string(index=False)); print(pd.DataFrame(G).to_string(index=False)); print("done")
