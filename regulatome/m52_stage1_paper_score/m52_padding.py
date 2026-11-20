# m52: generated from m48_padding.py (padding table 0 / 10 / 50 kb, top 10,000, from out47 table 1).
# m48 Amendment 1: padding table for the top 10,000 peaks (0 / 10 kb / 50 kb, 63 TFs), reformatted from the m47 run that feeds the peaks-per-TF slide (no new numbers).
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; I = "/work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score/out47"; O = f"{R}/m52_stage1_paper_score/out"
T = pd.read_csv(f"{I}/1_recall_and_shuffle.tsv", sep="\t", dtype={"padding": str}); T = T[(T["peak set"] == "top 10,000") & (T.TFs == "all 63 TFs")]
Ls = pd.read_csv(f"{I}/2_list_sizes.tsv", sep="\t", dtype={"padding": str}); Ls = Ls[(Ls["peak set"] == "top 10,000") & (Ls.TFs == "all 63 TFs")].set_index("padding")
def pf(p): return f"{p:.3f}" if p < 0.1 or round(p, 2) >= 1 else f"{p:.2f}"  # as on the peaks-per-TF slide
PADS = ["0", "10 kb", "50 kb"]; rows, full = [], []
for k in PADS:
    g = T[T.padding == k].set_index("cutoff"); r = {"Padding": k, "Median genes per TF": f"{int(Ls.loc[k, 'linked genes per TF, median']):,}"}
    for c in (2, 3, 4, 5):  # |score| x 10
        x = g.loc[f"|score| >= {c / 10:g}"]; r[f"|score| ≥ {c / 10:g}"] = f"{x['recall (%)']:.1f}% (p {pf(x['p (one-sided)'])})"
        full.append({"padding": k, "cutoff": f"|score| >= {c / 10:g}", "TFs left": int(x["TFs left"]), "median genes per TF": int(Ls.loc[k, "linked genes per TF, median"]), "total links": int(Ls.loc[k, "total links"]), "TFs > 4,000 genes": int(Ls.loc[k, "TFs with > 4,000 linked genes"]),
                     "recall (%)": x["recall (%)"], "shuffle mean (%)": x["shuffle mean (%)"], "gap (points)": x["gap (points)"], "observed / shuffle": x["recall (%)"] / x["shuffle mean (%)"], "shuffle 2.5-97.5 (%)": x["shuffle 2.5-97.5 (%)"], "p (one-sided)": x["p (one-sided)"]})
    rows.append(r)
TAB, FULL = pd.DataFrame(rows), pd.DataFrame(full); TAB.to_csv(f"{O}/10_padding_table.tsv", sep="\t", index=False); FULL.to_csv(f"{O}/10_padding_full.tsv", sep="\t", index=False, float_format="%.4g")
note = pd.DataFrame({"note": ["Top 10,000 lab peaks per TF (signalValue); link = summit in a promoter or canonical Hi-C ABC-linked element of the gene, plus summit within the padding of the TSS (0 = no padding); 63 TFs.",
  "Recall = share of Perturb-seq genes (|score| >= cutoff, top-25% cells, own gene removed) found in the Regulome, pooled over TFs. p = one-sided, matched gene shuffle within NTC expression x promoter ATAC deciles, 1,000 shuffles, seed 20261118 (same run as the peaks-per-TF table, m47 table 1). Gap = recall minus shuffle mean (points)."]})
with pd.ExcelWriter(f"{O}/stage1_top10k_padding_0_10_50kb.xlsx") as w:
    TAB.to_excel(w, sheet_name="table", index=False); FULL.to_excel(w, sheet_name="full", index=False); note.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 250); print(TAB.to_string(index=False)); print(FULL.round(3).to_string(index=False)); print("done")
