# m54: generated from m52_bulk_side.py by gen54.py (panel at top 5 % per TF).
# m48: bulk ChIP (HEK293T / HEK293) side table and panel for top 10,000, values from m47 tables 4 (per-list) and 5 (paired, as is). No new permutations.
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M47 = "/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out47"; M = f"{R}/m54_stage1_percentile_tables"; O = f"{M}/out"; PL = f"{M}/plots"
for d in ("slide", "paper"): os.makedirs(f"{PL}/{d}", exist_ok=True)
T4 = pd.read_csv(f"{M47}/4_bulk_vs_ours_per_tf.tsv", sep="\t", dtype={"padding": str}); T5 = pd.read_csv(f"{M47}/5_paired_bulk_vs_ours.tsv", sep="\t", dtype={"padding": str})
TF6 = ["CTCF", "ZNF282", "MAZ", "ZIC2", "ZNF777", "SETDB1"]; CELL = {"CTCF": "HEK293T", "ZNF282": "HEK293T", "MAZ": "HEK293", "ZIC2": "HEK293", "ZNF777": "HEK293", "SETDB1": "HEK293"}
rows = []
for pad in ("0", "10 kb", "50 kb"):
    for t in TF6:
        for c in ("top 10% of genes", "top 5% of genes", "top 2% of genes", "top 1% of genes"):
            b = T4[(T4.TF == t) & (T4.padding == pad) & (T4.cutoff == c) & (T4.list == "bulk ChIP top 10,000")].iloc[0]; o = T4[(T4.TF == t) & (T4.padding == pad) & (T4.cutoff == c) & (T4.list == "our HITAG top 10,000")].iloc[0]
            pr = T5[(T5.TF == t) & (T5.padding == pad) & (T5.cutoff == c) & (T5.pair == "top 10,000") & (T5.version == "as is")].iloc[0]; n = int(b["Perturb-seq genes"]); small = n < 10
            r = {"padding": pad, "TF": t, "bulk ChIP cell line": CELL[t] + (" (from file name)" if t == "SETDB1" else ""), "bulk peaks used": int(b["peaks used"]), "cutoff": c.replace(">=", "≥"), "Perturb-seq genes": n}
            for lab, x in (("bulk", b), ("ours", o)):
                r[f"{lab}: linked genes"] = int(x["linked genes"])
                for k, col in (("genes in both", "in both"), ("recall (%)", "recall (%)"), ("shuffle recall (%)", "shuffle mean recall (%)"), ("observed / shuffle", "observed / shuffle"), ("p", "p (one-sided)")):
                    r[f"{lab}: {k}"] = "n < 10" if small else x[col]
            r["ratio difference (ours - bulk)"] = "n < 10" if small else pr["ratio difference (ours - bulk)"]; r["paired p (two-sided)"] = "n < 10" if small else pr["p two-sided (ratio difference)"]
            rows.append(r)
D = pd.DataFrame(rows); sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
with pd.ExcelWriter(f"{O}/stage1_top10k_bulk_chip_side_table.xlsx") as w:
    for pad in ("10 kb", "0", "50 kb"): D[D.padding == pad].to_excel(w, sheet_name=f"padding {pad}", index=False)
    pd.DataFrame({"note": ["Bulk ChIP top 10,000 (ZNF282 8,514 and ZNF777 6,223 peaks: full file) vs our HITAG top 10,000; same Stage 1 path; ZBTB7A removed; no K562 / HepG2.", "observed / shuffle and p: per list, matched gene shuffle (m47 table 4). Paired p: one shared shuffle for both lists, difference ours - bulk (m47 table 5, as is).", "n < 10: fewer than 10 Perturb-seq genes at that cutoff."]}).to_excel(w, sheet_name="notes", index=False)
D.to_csv(f"{O}/9_bulk_chip_side_table.tsv", sep="\t", index=False, float_format="%.4g")
def panel(mode):
    P = rp.use(); s = mode == "slide"
    if s: plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    g = D[(D.padding == "10 kb") & (D.cutoff == "top 5% of genes")].set_index("TF").loc[TF6]; x = np.arange(len(TF6)); w = 0.38; CB, CO = P.neutral.dark, P.arms.raw
    fig, axs = plt.subplots(2, 1, figsize=(12.9, 5.9) if s else (7.2, 3.6), sharex=True, gridspec_kw=dict(hspace=0.12, left=0.1, right=0.99, bottom=0.14, top=0.9))
    for ax, k, yl in ((axs[0], "recall (%)", "recall (%)"), (axs[1], "observed / shuffle", "observed / shuffle")):
        ax.bar(x - w / 2, g[f"bulk: {k}"].astype(float), w, color=CB, label="bulk ChIP top 10,000"); ax.bar(x + w / 2, g[f"ours: {k}"].astype(float), w, color=CO, label="our HITAG top 10,000")
        ax.set_xticks(x); ax.set_xticklabels([f"{t} ({CELL[t]})" for t in TF6]); ax.set_ylabel(yl)
    axs[1].axhline(1, color=P.neutral.mid, lw=1.2 if s else 0.6, ls="--"); axs[1].set_ylim(0, max(1.6, float(np.nanmax(g[["bulk: observed / shuffle", "ours: observed / shuffle"]].astype(float).values)) * 1.1)); axs[0].set_ylim(0, 100)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.0))
    nm = "S6_bulk_chip_vs_ours_per_tf"
    if s: fig.savefig(f"{PL}/slide/{nm}.png", dpi=300); fig.savefig(f"{PL}/slide/{nm}.svg"); fig.savefig(f"{PL}/slide/{nm}.pdf")
    else: fig.savefig(f"{PL}/paper/{nm}.png", dpi=600); fig.savefig(f"{PL}/paper/{nm}.svg"); fig.savefig(f"{PL}/paper/{nm}.pdf")
    plt.close(fig)
panel("slide"); panel("paper"); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(D[D.padding == "10 kb"].to_string(index=False)); print("done")
