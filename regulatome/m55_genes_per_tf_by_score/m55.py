# m55 (SPEC.md + Amendment 1): genes left per TF at each |score| cutoff, and the |score| at each within-TF top-percentile boundary.
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
B = "/work/users/d/y/dyy12/XuLab/regulatome/m55_genes_per_tf_by_score"; O = f"{B}/out"; PL = f"{B}/plots"; os.makedirs(O, exist_ok=True); os.makedirs(PL, exist_ok=True)
F = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True); SC = F["score"]; TFS = F["tfs"].astype(str); OK = np.isfinite(SC)
D = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/plot_data.npz", allow_pickle=True); cells = pd.Series(D["cells"], index=D["tfs"].astype(str)).reindex(TFS).values.astype(int)
grp = np.where(cells <= 42, "25-42", np.where(cells <= 73, "43-73", "74-216")); A = np.where(OK, np.abs(np.nan_to_num(SC)), -1.0)
CUTS = [round(0.05 * k, 2) for k in range(1, 17)]; N = np.vstack([(A >= c).sum(1) for c in CUTS]).T  # TFs x cutoffs
T = pd.DataFrame([{"|score| cutoff": f"{c:.2f}", "mean genes left per TF": N[:, k].mean(), "median": float(np.median(N[:, k])), "min": int(N[:, k].min()), "25th pct": np.percentile(N[:, k], 25), "75th pct": np.percentile(N[:, k], 75), "max": int(N[:, k].max()),
                   "median, TFs with 25-42 cells": float(np.median(N[grp == "25-42", k])), "median, 43-73 cells": float(np.median(N[grp == "43-73", k])), "median, 74-216 cells": float(np.median(N[grp == "74-216", k]))} for k, c in enumerate(CUTS)])
PT = pd.DataFrame(N, columns=[f"|score| >= {c:.2f}" for c in CUTS]); PT.insert(0, "top-25% cells", cells); PT.insert(0, "TF", TFS); PT = PT.sort_values(["top-25% cells", "TF"]).reset_index(drop=True)
BND = []
for x in (10, 5, 2, 1):
    v = np.array([np.sort(np.abs(SC[i, OK[i]]))[::-1][int(round(OK[i].sum() * x / 100)) - 1] for i in range(len(TFS))])
    BND.append({"top % of genes per TF": f"top {x}%", "genes per TF": int(round(OK[0].sum() * x / 100)), "|score| at the boundary, mean": v.mean(), "median": float(np.median(v)), "min": v.min(), "max": v.max()})
BND = pd.DataFrame(BND)
OUTX = f"{O}/genes_left_per_TF_by_score_cutoff.xlsx"
with pd.ExcelWriter(OUTX) as w:
    T.to_excel(w, sheet_name="summary over 63 TFs", index=False); BND.to_excel(w, sheet_name="top % as |score|", index=False); PT.to_excel(w, sheet_name="per TF", index=False)
    pd.DataFrame({"note": ["Score: paper per-gene score (per top-25% cell, (log1p CP10k - own-GEM NTC mean) / own-GEM NTC SD, averaged over the TF's cells), full precision (m53); own gene removed; 11,226-11,227 genes per TF.", "Genes left = genes with |score| >= cutoff. Boundary |score| = the |score| of the last gene inside the top X % of that TF (genes per TF = round(n X / 100))."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(T.round(1).to_string(index=False)); print(BND.round(4).to_string(index=False), flush=True)
P = rp.use(); GREY, INK = P.neutral.light, P.neutral.ink; print("colours", GREY, INK, flush=True)
for mode in ("slide", "paper"):
    P = rp.use(); s = mode == "slide"
    if s: plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    fig, ax = plt.subplots(figsize=(12.9, 5.9) if s else (3.5, 2.6)); fig.subplots_adjust(left=0.08 if s else 0.17, right=0.97, bottom=0.14 if s else 0.18, top=0.88 if s else 0.84)
    for i in range(len(TFS)):
        y = N[i].astype(float); y[y == 0] = np.nan; ax.plot(CUTS, y, color=GREY, lw=1.0 if s else 0.4, label="each TF" if i == 0 else None)
    ax.plot(CUTS, N.mean(0), color=INK, lw=3.0 if s else 1.2, label="mean of 63 TFs")
    ax.set_yscale("log"); ax.set_ylim(0.8, 12000); ax.set_xlim(0.03, 0.82); ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]); ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"])
    ax.set_yticks([1, 10, 100, 1000, 10000]); ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"]); ax.set_xlabel("|score| cutoff"); ax.set_ylabel("genes left per TF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13 if s else 1.2), ncol=2, frameon=False, fontsize=None if s else 6)
    for ext in ("png", "pdf", "svg"): fig.savefig(f"{PL}/genes_left_per_TF_by_score_cutoff_{mode}.{ext}", dpi=300 if s else 600)
    plt.close(fig)
print("done")
