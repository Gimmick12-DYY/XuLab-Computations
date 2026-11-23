# m57 Amendment 12: peaks and linked genes per TF at each peak q-value cutoff (descriptive), bar chart of the median linked genes.
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m57_stage1_q_cutoff/out"; PL = f"{R}/m57_stage1_q_cutoff/plots"; os.makedirs(PL, exist_ok=True); NTF = len(TFS)
QS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-15, 1e-20]; LAB = ["0.5", "0.2", "0.1", "0.05", "0.02", "0.01", "0.005", "1e-3", "1e-4", "1e-5", "1e-6", "1e-8", "1e-10", "1e-15", "1e-20"]
NPK = np.zeros((NTF, len(QS)), int); LG = {p: np.zeros((NTF, len(QS)), int) for p in ("10 kb", "0", "50 kb")}
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 8, 9], names=["c", "s", "q", "off"])
    for k, q in enumerate(QS):
        sub = P[P.q >= -np.log10(q)]; NPK[i, k] = len(sub)
        if len(sub):
            c5, p5 = sub.c.values.astype(str), (sub.s + sub.off).values; A, _ = rule_abc(c5, p5)
            LG["0"][i, k] = int((A & OK[i]).sum()); LG["10 kb"][i, k] = int(((A | window(c5, p5, 10000)) & OK[i]).sum()); LG["50 kb"][i, k] = int(((A | window(c5, p5, 50000)) & OK[i]).sum())
    print(t, flush=True)
k5, k1 = LAB.index("0.05"), LAB.index("0.01"); G = LG["10 kb"]
assert G[:, k5].sum() == 183470 and np.median(G[:, k5]) == 2196 and (G[:, k5] == 0).sum() == 7 and np.median(G[:, k1]) == 561, (G[:, k5].sum(), np.median(G[:, k5]), (G[:, k5] == 0).sum(), np.median(G[:, k1]))
rows = []
for k in range(len(QS)):
    r = {"q cutoff": LAB[k], "peaks per TF, median": float(np.median(NPK[:, k])), "peaks, mean": NPK[:, k].mean(), "peaks, 25th": np.percentile(NPK[:, k], 25), "peaks, 75th": np.percentile(NPK[:, k], 75), "peaks, min": int(NPK[:, k].min()), "peaks, max": int(NPK[:, k].max())}
    for p in ("10 kb", "0", "50 kb"):
        g = LG[p][:, k]; pre = "linked genes per TF" if p == "10 kb" else f"linked genes per TF ({p})"
        r.update({f"{pre}, median": float(np.median(g)), f"{pre}, mean": g.mean(), f"{pre}, 25th": np.percentile(g, 25), f"{pre}, 75th": np.percentile(g, 75), f"{pre}, min": int(g.min()), f"{pre}, max": int(g.max()), f"{pre}, TFs with 0": int((g == 0).sum()), f"{pre}, TFs with < 100": int((g < 100).sum()), f"{pre}, total links": int(g.sum())})
    rows.append(r)
T = pd.DataFrame(rows)
PT = pd.DataFrame({"TF": TFS, **{f"peaks, q <= {LAB[k]}": NPK[:, k] for k in range(len(QS))}, **{f"linked genes 10 kb, q <= {LAB[k]}": LG["10 kb"][:, k] for k in range(len(QS))}, **{f"linked genes 0 kb, q <= {LAB[k]}": LG["0"][:, k] for k in range(len(QS))}, **{f"linked genes 50 kb, q <= {LAB[k]}": LG["50 kb"][:, k] for k in range(len(QS))}})
with pd.ExcelWriter(f"{O}/linked_genes_per_TF_by_q_cutoff.xlsx") as w:
    T.to_excel(w, sheet_name="summary", index=False); PT.to_excel(w, sheet_name="per_TF", index=False)
    pd.DataFrame({"note": ["Lab MACS2 narrowPeak per TF; a peak is kept at cutoff q when column 9 (-log10 q) >= -log10(q). Linked genes = summit in the gene's promoter or a canonical Hi-C ABC-linked element (ABC >= 0.02), or within 10 kb (main; also 0 and 50 kb) of its TSS; 11,228-gene universe; 63 TFs. Descriptive (no Perturb-seq)."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(T[["q cutoff", "peaks per TF, median", "linked genes per TF, median", "linked genes per TF, mean", "linked genes per TF, 25th", "linked genes per TF, 75th", "linked genes per TF, min", "linked genes per TF, max", "linked genes per TF, TFs with 0", "linked genes per TF, TFs with < 100", "linked genes per TF, total links"]].round(1).to_string(index=False), flush=True)
med = T["linked genes per TF, median"].values
for mode in ("slide", "paper"):
    P_ = rp.use(); sl = mode == "slide"
    if sl: plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    fig, ax = plt.subplots(figsize=(12.9, 5.9) if sl else (7.2, 2.6)); fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14 if sl else 0.18, top=0.95)
    x = np.arange(len(QS)); ax.bar(x, med, width=0.7, color="#D98C1F")
    for xi, v in zip(x, med): ax.text(xi, v + med.max() * 0.012, f"{int(v):,}", ha="center", va="bottom", fontsize=12 if sl else 5.5, color=P_.neutral.ink)
    ax.set_xticks(x); ax.set_xticklabels(LAB); ax.set_xlim(-0.6, len(QS) - 0.4); ax.set_ylim(0, med.max() * 1.08)
    ax.set_xlabel("peak q-value cutoff (q ≤)"); ax.set_ylabel("median linked genes per TF"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for ext in ("png", "pdf", "svg"): fig.savefig(f"{PL}/median_linked_genes_per_TF_by_q_cutoff_{mode}.{ext}", dpi=300 if sl else 600)
    plt.close(fig)
print("done")
