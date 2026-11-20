# m48: Regulome Stage 1 panels for the top 10,000 peaks (m45_plots.py with paths, peak number text and axis ranges changed).
# m45: Regulome Stage 1 panels (slide: 12.9 x 5.9 in at 300 dpi, labels >= 12 pt; paper: Arial 7 pt, 183 mm). usage: m45_plots.py [slide|paper|both]
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M = f"{R}/m48_stage1_top10k"; T = f"{M}/out"; PL = f"{M}/plots"
for d in ("slide", "paper"): os.makedirs(f"{PL}/{d}", exist_ok=True)
T1 = pd.read_csv(f"{T}/1_recall_pooled.tsv", sep="\t", dtype={"padding": str}); T2 = pd.read_csv(f"{T}/2_recall_vs_background.tsv", sep="\t", dtype={"padding": str}); T3 = pd.read_csv(f"{T}/3_dose_response.tsv", sep="\t", dtype={"padding": str})
CUT = ["|z| >= 2", "|z| >= 3", "|z| >= 4", "|z| >= 5"]; CL = ["|z| ≥ 2", "|z| ≥ 3", "|z| ≥ 4", "|z| ≥ 5"]; PADS = ["0", "10 kb", "50 kb"]; PADL = {"0": "padding 0", "10 kb": "padding 10 kb", "50 kb": "padding 50 kb"}
ALL, NOB, BR = "all 63 TFs", "without broad TFs (55)", "broad TFs only (8)"
BINS = ["0-1", "1-2", "2-3", "3-4", "4-5", ">= 5"]; BL = ["0–1", "1–2", "2–3", "3–4", "4–5", "≥ 5"]
PADC = {"0": "#D98C1F", "10 kb": "#9A5B12", "50 kb": "#4A2C08"}
def style(mode):
    P = rp.use()
    if mode == "slide": plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "axes.titlesize": 14, "axes.linewidth": 0.9, "lines.linewidth": 2.0,
                                              "xtick.major.width": 0.9, "ytick.major.width": 0.9, "xtick.major.size": 4, "ytick.major.size": 4})
    return P
def save(fig, name, mode):
    if mode == "slide": fig.savefig(f"{PL}/slide/{name}.png", dpi=300); fig.savefig(f"{PL}/slide/{name}.svg"); fig.savefig(f"{PL}/slide/{name}.pdf")
    else: fig.savefig(f"{PL}/paper/{name}.png", dpi=600); fig.savefig(f"{PL}/paper/{name}.svg"); fig.savefig(f"{PL}/paper/{name}.pdf")
    plt.close(fig); print("saved", mode, name, flush=True)
SZ = lambda s, h=2.6: (12.9, 5.9) if s else (7.2, h)
g1 = lambda pad, v: T1[(T1.padding == pad) & (T1.TFs == v)].set_index("cutoff").loc[CUT]
g2 = lambda pad, v: T2[(T2.padding == pad) & (T2.TFs == v)].set_index("cutoff").loc[CUT]
g3 = lambda pad, v: T3[(T3.padding == pad) & (T3.TFs == v)].set_index("|z| bin").loc[BINS]
REC = "recall: share of Perturb-seq genes found in our Regulome (%)"

def schematic(mode):
    P = style(mode); s = mode == "slide"; fs = 14 if s else 7; sf = 12.5 if s else 6.3; INK, DARK, MID, PALE, FAINT = P.neutral.ink, P.neutral.dark, P.neutral.mid, P.neutral.pale, P.neutral.faint; OR = P.arms.raw; HL = P.highlight
    fig = plt.figure(figsize=SZ(s, 3.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 129); ax.set_ylim(0, 59); ax.axis("off")
    lw = 1.3 if s else 0.6
    def box(x, y, w, h, title, sub=None, fc="white", ec=INK, suby=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
        if sub: ax.text(x + w / 2, suby if suby is not None else y + h / 2 - 2.0, sub, ha="center", va="center", fontsize=sf, color=DARK, linespacing=1.3)
    def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw, mutation_scale=16 if s else 7, shrinkA=0, shrinkB=0))
    def dial(x, y, text):
        ax.add_patch(FancyBboxPatch((x, y), 27, 6.2, boxstyle="round,pad=0.25,rounding_size=3", fc=HL, ec=OR, lw=lw)); ax.text(x + 13.5, y + 3.1, text, ha="center", va="center", fontsize=sf, color=INK, fontweight="bold")
    def grid(x, y, seed):
        r = np.random.default_rng(seed).random((4, 7)) < 0.3; c = 1.25
        for i in range(4):
            for j in range(7): ax.add_patch(Rectangle((x + j * c, y + (3 - i) * c), c, c, fc=OR if r[i, j] else "white", ec=MID, lw=lw * 0.5))
    # row 1: binding
    yb, yp, h = 37, 8, 15
    box(1, yb, 25, h, "Our HITAG peaks", "top 10,000 per TF\n63 TFs, HEK293T"); arrow(26.6, yb + h / 2, 31.4, yb + h / 2)
    box(32, yb, 33, h, "Link rule", "summit in a promoter or an\nABC-linked element, + padding"); dial(35, yb - 8.2, "padding: 0 · 10 kb · 50 kb")
    arrow(65.6, yb + h / 2, 70.4, yb + h / 2); box(71, yb, 22, h, "Regulome", "binary TF × gene", suby=yb + h - 7.3); grid(77.6, yb + 1.0, 1)
    # row 2: Perturb-seq
    box(1, yp, 25, h, "Perturb-seq", "top-25% sgRNA cells\nz per TF × gene"); arrow(26.6, yp + h / 2, 31.4, yp + h / 2)
    box(32, yp, 33, h, "Cutoff", "gene changed if |z| ≥ cutoff\n(own gene removed)"); dial(35, yp - 7.4, "|z| ≥ 2 · 3 · 4 · 5")
    arrow(65.6, yp + h / 2, 70.4, yp + h / 2); box(71, yp, 22, h, "Perturb-seq genes", "binary TF × gene", suby=yp + h - 7.3); grid(77.6, yp + 1.0, 2)
    # overlap
    arrow(93.6, yb + h / 2, 100.4, 32.5); arrow(93.6, yp + h / 2, 100.4, 25.5)
    box(101, 17, 27, 24, "Overlap", None, fc=FAINT)
    ax.text(114.5, 31.0, "Recall =", ha="center", va="center", fontsize=fs, color=INK); ax.text(114.5, 26.3, "in both", ha="center", va="center", fontsize=sf, color=INK)
    ax.plot([106, 123], [24.4, 24.4], color=INK, lw=lw); ax.text(114.5, 22.0, "Perturb-seq genes", ha="center", va="center", fontsize=sf, color=INK)
    ax.text(114.5, 11.5, "judged against\nmatched background", ha="center", va="center", fontsize=sf, color=DARK, linespacing=1.3)
    ax.text(1, 56, "Two dials only (highlighted); everything else fixed", fontsize=sf, color=DARK, va="center")
    save(fig, "S1_schematic", mode)

def recall(mode):
    P = style(mode); s = mode == "slide"; fs = 13 if s else 6; ms = 9 if s else 3.5; DARK, LIGHT = P.neutral.dark, P.neutral.light; x = np.arange(4)
    fig = plt.figure(figsize=SZ(s)); gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.42, left=0.085, right=0.985, bottom=0.25, top=0.9)
    ax = fig.add_subplot(gs[0])
    for pad in PADS:
        g = g1(pad, ALL); ax.plot(x, g[REC], "-o", color=PADC[pad], ms=ms, label=PADL[pad])
        ax.text(3.12, g[REC].iloc[-1], f"{g[REC].iloc[-1]:.1f}%", va="center", fontsize=fs, color=PADC[pad])
    g = g1("0", ALL); ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{n:,} genes\n{t} TFs" for c, n, t in zip(CL, g["Perturb-seq genes"], g["TFs left (>= 1 Perturb-seq gene)"])])
    ax.set_ylabel("recall (%)"); ax.set_ylim(0, 80); ax.set_xlim(-0.3, 3.55); ax.legend(loc="upper left", frameon=False, ncol=3); ax.set_xlabel("Perturb-seq cutoff (Perturb-seq genes, TFs left)")
    ax2 = fig.add_subplot(gs[1]); med = [g1(p, ALL)["median our genes per TF"].iloc[0] for p in PADS]
    ax2.bar(range(3), med, color=[PADC[p] for p in PADS], width=0.65)
    for i, m in enumerate(med): ax2.text(i, m + 90, f"{m:,.0f}", ha="center", va="bottom", fontsize=fs, color=DARK)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(["0", "10 kb", "50 kb"]); ax2.set_xlabel("padding"); ax2.set_ylabel("our genes per TF (median)"); ax2.set_ylim(0, 6500)
    save(fig, "S2_recall_by_cutoff_and_padding", mode)

def background(mode):
    P = style(mode); s = mode == "slide"; fs = 12 if s else 5.5; C3 = [P.arms.raw, P.neutral.dark, P.neutral.light]; LB = ["changed genes (recall)", "matched background", "all non-changing genes"]; x = np.arange(4); w = 0.26
    fig, axs = plt.subplots(1, 2, figsize=SZ(s, 2.9), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.075, right=0.99, bottom=0.25, top=0.85))
    for ax, pad in zip(axs, ["0", "10 kb"]):
        g = g2(pad, ALL)
        for k, (col, c) in enumerate(zip(["recall, changed genes (%)", "matched background (%)", "all non-changing genes (%)"], C3)):
            ax.bar(x + (k - 1) * w, g[col], width=w, color=c, label=LB[k])
            if k == 0:
                for xi, v in zip(x, g[col]): ax.text(xi - w, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=fs, color=P.neutral.ink)
            if k == 1:
                for xi, v in zip(x, g[col]): ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=fs, color=P.neutral.dark)
        g1_ = g1(pad, ALL); ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{n:,} genes\n{t} TFs" for c, n, t in zip(CL, g1_["Perturb-seq genes"], g1_["TFs left (>= 1 Perturb-seq gene)"])]); ax.set_xlabel("Perturb-seq cutoff"); ax.set_title(PADL[pad], loc="left", fontweight="bold"); ax.set_ylim(0, 60)
    axs[0].set_ylabel("share of genes in our Regulome (%)"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.0))
    save(fig, "S3_recall_vs_background", mode)

def dose(mode):
    P = style(mode); s = mode == "slide"; ms = 8 if s else 3; fs = 12 if s else 5.5; x = np.arange(6)
    fig, axs = plt.subplots(1, 2, figsize=SZ(s), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.075, right=0.99, bottom=0.15, top=0.83))
    for ax, pad in zip(axs, ["0", "10 kb"]):
        for v, c, lab, mf in ((NOB, P.arms.raw, "55 TFs (broad TFs removed)", None), (BR, P.neutral.ink, "8 broad TFs", "white")):
            g = g3(pad, v); ax.plot(x, g["share bound (%)"], "-o", color=c, ms=ms, mfc=mf or c, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(BL); ax.set_xlabel("knockdown response, |z| bin"); ax.set_title(PADL[pad], loc="left", fontweight="bold"); ax.set_ylim(0, 80)
    axs[0].set_ylabel("share of genes in our Regulome (%)"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.0))
    save(fig, "S4_dose_response", mode)

def broad(mode):
    P = style(mode); s = mode == "slide"; ms = 8 if s else 3; x = np.arange(4)
    fig, axs = plt.subplots(1, 2, figsize=(12.9, 5.4) if s else SZ(s), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.075, right=0.99, bottom=0.17, top=0.83))
    for ax, pad in zip(axs, ["0", "10 kb"]):
        for v, c, lab, mf in ((ALL, P.neutral.mid, "all 63 TFs", None), (NOB, P.arms.raw, "55 TFs (broad removed)", None), (BR, P.neutral.ink, "8 broad TFs only", "white")):
            g = g1(pad, v); ax.plot(x, g[REC], "-o", color=c, ms=ms, mfc=mf or c, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(CL); ax.set_xlabel("Perturb-seq cutoff"); ax.set_title(PADL[pad], loc="left", fontweight="bold"); ax.set_ylim(0, 80)
    axs[0].set_ylabel("recall (%)"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.0))
    save(fig, "S5_broad_tfs", mode)

modes = ["slide", "paper"] if (len(sys.argv) < 2 or sys.argv[1] == "both") else [sys.argv[1]]
for m in modes:
    for f in (schematic, recall, background, dose, broad): f(m)
