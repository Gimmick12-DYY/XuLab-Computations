# m57 Amendment 12: peaks and linked genes per TF at each peak q-value cutoff, bar chart of the median.
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m57_stage1_q_cutoff/out"; PL = f"{R}/m57_stage1_q_cutoff/plots"; os.makedirs(PL, exist_ok=True); NT = len(TFS)
QS = [0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-15, 1e-20]; LAB = ["0.05", "0.01", "1e-3", "1e-4", "1e-5", "1e-6", "1e-8", "1e-10", "1e-15", "1e-20"]
NPK = np.zeros((NT, len(QS)), int); NG = np.zeros((NT, len(QS)), int)
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 8, 9], names=["c", "s", "q", "off"])
    for k, q in enumerate(QS):
        sub = P[P.q >= -np.log10(q)]; NPK[i, k] = len(sub)
        if len(sub): c5, p5 = sub.c.values.astype(str), (sub.s + sub.off).values; A, _ = rule_abc(c5, p5); NG[i, k] = int(((A | window(c5, p5, 10000)) & OK[i]).sum())
    print(t, flush=True)
assert NG[:, 0].sum() == 183470
T = pd.DataFrame([{"q cutoff": LAB[k], "linked genes per TF, median": float(np.median(NG[:, k])), "25th": np.percentile(NG[:, k], 25), "75th": np.percentile(NG[:, k], 75), "min": int(NG[:, k].min()), "max": int(NG[:, k].max()), "TFs with 0 linked genes": int((NG[:, k] == 0).sum()), "total links": int(NG[:, k].sum()),
                   "peaks per TF, median": float(np.median(NPK[:, k])), "peaks 25th": np.percentile(NPK[:, k], 25), "peaks 75th": np.percentile(NPK[:, k], 75), "TFs with 0 peaks": int((NPK[:, k] == 0).sum())} for k in range(len(QS))])
PT = pd.DataFrame({"TF": TFS, **{f"peaks, q <= {LAB[k]}": NPK[:, k] for k in range(len(QS))}, **{f"linked genes, q <= {LAB[k]}": NG[:, k] for k in range(len(QS))}})
with pd.ExcelWriter(f"{O}/linked_genes_per_TF_by_q.xlsx") as w:
    T.to_excel(w, sheet_name="summary over 63 TFs", index=False); PT.to_excel(w, sheet_name="per TF", index=False)
    pd.DataFrame({"note": ["Lab MACS2 peaks per TF kept if narrowPeak column 9 (-log10 q) >= -log10(cutoff). Linked genes = summit in the gene's promoter or a canonical Hi-C ABC-linked element (ABC >= 0.02), or within 10 kb of its TSS; 11,228-gene universe; 63 TFs."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); print(T.round(1).to_string(index=False), flush=True)
med = T["linked genes per TF, median"].values
for mode in ("slide", "paper"):
    P_ = rp.use(); sl = mode == "slide"
    if sl: plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    fig, ax = plt.subplots(figsize=(12.9, 5.9) if sl else (7.2, 2.6)); fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14 if sl else 0.18, top=0.95)
    x = np.arange(len(QS)); ax.bar(x, med, width=0.7, color=P_.arms.raw)
    for xi, v in zip(x, med): ax.text(xi, v + med.max() * 0.012, f"{int(v):,}", ha="center", va="bottom", fontsize=12 if sl else 5.5, color=P_.neutral.ink)
    ax.set_xticks(x); ax.set_xticklabels(LAB); ax.set_xlim(-0.6, len(QS) - 0.4); ax.set_ylim(0, med.max() * 1.08)
    ax.set_xlabel("peak q-value cutoff (q ≤)"); ax.set_ylabel("median linked genes per TF"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for ext in ("png", "pdf", "svg"): fig.savefig(f"{PL}/median_linked_genes_per_TF_by_q_{mode}.{ext}", dpi=300 if sl else 600)
    plt.close(fig)
print("done")
