# m57 Amendment 12: box plots of linked genes per TF by peak q-value cutoff (linear and log10(y + 1)).
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
B = "/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff"; PL = f"{B}/plots"
P = pd.read_excel(f"{B}/out/linked_genes_per_TF_by_q_cutoff.xlsx", sheet_name="per_TF"); C = [c for c in P.columns if c.startswith("linked genes 10 kb")]; LAB = [c.split("<= ")[1] for c in C]; assert len(C) == 15 and len(P) == 63
DATA = [P[c].values.astype(float) for c in C]
for log in (False, True):
    for mode in ("slide", "paper"):
        Pp = rp.use(); sl = mode == "slide"
        if sl: plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
        fig, ax = plt.subplots(figsize=(12.9, 5.9) if sl else (7.2, 2.8)); fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14 if sl else 0.17, top=0.96)
        d = [np.log10(x + 1) for x in DATA] if log else DATA
        bp = ax.boxplot(d, whis=1.5, patch_artist=True, widths=0.6, medianprops=dict(color=Pp.neutral.ink, lw=1.8 if sl else 0.8), boxprops=dict(facecolor="#D98C1F", edgecolor=Pp.neutral.ink, lw=1.0 if sl else 0.5),
                        whiskerprops=dict(color=Pp.neutral.ink, lw=1.0 if sl else 0.5), capprops=dict(color=Pp.neutral.ink, lw=1.0 if sl else 0.5), flierprops=dict(marker="o", markersize=3 if sl else 1.2, markerfacecolor=Pp.neutral.mid, markeredgecolor="none"))
        ax.set_xticks(range(1, 16)); ax.set_xticklabels(LAB); ax.set_xlabel("peak q-value cutoff (q ≤)")
        if log:
            ax.set_ylabel("linked genes per TF"); tk = [0, 1, 10, 100, 1000, 10000]; ax.set_yticks(np.log10(np.array(tk) + 1)); ax.set_yticklabels([f"{t:,}" for t in tk]); ax.set_ylim(-0.1, np.log10(12000))
        else:
            ax.set_ylabel("linked genes per TF"); ax.set_ylim(-200, 11800); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        nm = "linked_genes_per_TF_by_q_cutoff_box_log" if log else "linked_genes_per_TF_by_q_cutoff_box"
        for ext in ("png", "pdf", "svg"): fig.savefig(f"{PL}/{nm}_{mode}.{ext}", dpi=300 if sl else 600)
        plt.close(fig)
print("done")
