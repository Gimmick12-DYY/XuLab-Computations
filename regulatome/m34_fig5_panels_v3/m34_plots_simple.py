# m34: Figure 5 panels c, h, i, j as plain overlap (no expected values). Top-25% sgRNA cells. usage: m34_plots_simple.py <letters>
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
R = "/work/users/d/y/dyy12/XuLab/regulatome"; PL = f"{R}/m34_fig5_panels_v3/plots"; S35 = f"{R}/m35_threshold_sweep/out"; O36 = f"{R}/m36_binary_overlap/out"
TP = pd.read_csv(f"{S35}/recovery_top25_per_tf.tsv", sep="\t"); PP = pd.read_csv(f"{S35}/recovery_top25_pooled.tsv", sep="\t"); OV = pd.read_csv(f"{O36}/overlap_per_tf.tsv", sep="\t"); OP = pd.read_csv(f"{O36}/overlap_pooled.tsv", sep="\t")
def style(mode):
    P = rp.use()
    if mode == "slide": plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.titlesize": 14, "axes.linewidth": 0.9, "lines.linewidth": 1.8})
    return P
def save(fig, name, mode):
    if mode == "slide": fig.savefig(f"{PL}/slide/{name}.png", dpi=300); fig.savefig(f"{PL}/slide/{name}.svg")
    else: fig.savefig(f"{PL}/paper/{name}.png", dpi=600); fig.savefig(f"{PL}/paper/{name}.svg"); fig.savefig(f"{PL}/paper/{name}.pdf")
    plt.close(fig); print("saved", mode, name, flush=True)
SZ = lambda s: (12.9, 5.8) if s else (7.0, 2.8)
def panel_c(mode):
    P = style(mode); s = mode == "slide"; fs = 12 if s else 6; OURS, BULK, PALE, DARK = P.arms.raw, P.neutral.ink, P.neutral.pale, P.neutral.dark
    order = ["NRF1", "NFYA", "E4F1", "CTCF", "MAZ", "ZNF777", "ZIC2", "ZBTB7A", "ZNF282", "SETDB1"]
    fig, axs = plt.subplots(1, 2, figsize=SZ(s), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.1, right=0.83 if s else 0.86, bottom=0.17, top=0.86))
    for ax, th, xl in ((axs[0], "abs2", "% of Perturb-seq genes (|z| ≥ 2) found"), (axs[1], "top200", "% of each TF's top 200 Perturb-seq genes found")):
        g0 = TP[TP.threshold == th]
        for yi, t in enumerate(order):
            g = g0[g0.TF == t]; o = g[g.source.str.startswith("our")]; b = g[g.source == "bulk ChIP"]; v = list(o.pct_found) + list(b.pct_found)
            ax.plot([min(v), max(v)], [yi, yi], color=PALE, lw=3 if s else 1.5, zorder=1); ax.scatter(b.pct_found, [yi] * len(b), s=90 if s else 14, facecolor="white", edgecolor=BULK, lw=1.4 if s else 0.7, zorder=3); ax.scatter(o.pct_found, [yi] * len(o), s=90 if s else 14, color=OURS, zorder=4)
        ax.set_xlabel(xl); ax.set_xlim(0, 100)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
    for yi, t in enumerate(order):
        b = TP[(TP.TF == t) & (TP.source == "bulk ChIP") & (TP.threshold == "abs2")]; axs[1].text(1.01, yi, " / ".join(x.split(" ")[0] for x in b.reference_cell_line), transform=axs[1].get_yaxis_transform(), va="center", fontsize=fs, color=DARK)
    axs[1].text(1.01, -0.75, "bulk ChIP cell line", transform=axs[1].get_yaxis_transform(), va="center", fontsize=fs, color=DARK, fontweight="bold")
    h1 = axs[0].scatter([], [], s=90 if s else 14, color=OURS, label="our HITAG (HEK293T)"); h2 = axs[0].scatter([], [], s=90 if s else 14, facecolor="white", edgecolor=BULK, label="bulk ChIP, same path")
    fig.legend(handles=[h1, h2], loc="upper left", bbox_to_anchor=(0.1, 0.995), ncol=2, frameon=False); save(fig, "Fig5c_ours_vs_bulk", mode)
def panel_h(mode):
    P = style(mode); s = mode == "slide"; fs = 12 if s else 6; OURS, BULK, DARK, MID = P.arms.raw, P.neutral.ink, P.neutral.dark, P.neutral.mid; TH = ["abs2", "abs3", "abs4", "abs5"]; XL = ["|z| ≥ 2", "|z| ≥ 3", "|z| ≥ 4", "|z| ≥ 5"]; x = np.arange(4)
    fig = plt.figure(figsize=SZ(s)); gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28, left=0.08, right=0.98, bottom=0.2, top=0.88)
    ax = fig.add_subplot(gs[0]); g = PP[PP.set.str.startswith("ours, all")].set_index("threshold").loc[TH]
    ax.plot(x, g.pct_found, "-o", color=OURS, ms=9 if s else 4)
    for xi, (pc, nt, nl, nf) in enumerate(zip(g.pct_found, g.n_tfs_ge10_links, g.links, g.found)): ax.text(xi, pc + 1.2, f"{nf:,} of {nl:,}\n({nt} TFs)", ha="center", va="bottom", fontsize=fs * 0.9, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(XL); ax.set_xlabel("knockdown threshold"); ax.set_ylabel("% of Perturb-seq genes found\nin our Regulome"); ax.set_ylim(0, 35); ax.set_xlim(-0.4, 3.4)
    ax.set_title("all 63 TFs, pooled (top-25% sgRNA cells)", loc="left", fontsize=fs * 1.05)
    ax2 = fig.add_subplot(gs[1]); o = PP[PP.set == "ours, 10 TFs with bulk ChIP"].set_index("threshold").loc[TH]; b = PP[PP.set.str.startswith("bulk")].set_index("threshold").loc[TH]
    ax2.plot(x, o.pct_found, "-o", color=OURS, ms=9 if s else 4, label="ours"); ax2.plot(x, b.pct_found, "-o", color=BULK, mfc="white", ms=9 if s else 4, label="bulk ChIP")
    for xi, nt in enumerate(o.n_tfs_ge10_links): ax2.text(xi, 3, f"{nt} TFs", ha="center", fontsize=fs * 0.9, color=DARK)
    ax2.set_xticks(x); ax2.set_xticklabels(XL); ax2.set_xlabel("knockdown threshold"); ax2.set_ylabel("% of Perturb-seq genes found"); ax2.set_ylim(0, 80); ax2.set_xlim(-0.4, 3.4); ax2.set_title("10 TFs with bulk ChIP", loc="left", fontsize=fs * 1.05); ax2.legend(loc="upper left", frameon=False)
    ax2.text(0.0, -0.2, "bulk references: CTCF HEK293T; NFYA, NRF1, E4F1 K562;\nMAZ, ZIC2, ZNF282, ZNF777, ZBTB7A, SETDB1 HEK293", transform=ax2.transAxes, fontsize=fs * 0.8, color=MID, ha="left", va="top")
    save(fig, "Fig5h_threshold_sweep", mode)
def panel_i(mode):
    P = style(mode); s = mode == "slide"; fs = 12 if s else 6; OURS, LIGHT, MID, DARK = P.arms.raw, P.neutral.light, P.neutral.mid, P.neutral.dark
    g = OV[OV.cutoff == "|z| >= 2"].sort_values("recovery_pct", ascending=False).reset_index(drop=True); x = np.arange(len(g))
    fig = plt.figure(figsize=SZ(s) if s else (7.0, 3.0)); gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.08, left=0.07, right=0.995, bottom=0.23, top=0.92)
    ax = fig.add_subplot(gs[0]); ax.bar(x, g.recovery_pct, color=OURS, width=0.8); ax.set_xlim(-0.7, len(g) - 0.3); ax.tick_params(axis="x", labelbottom=False, length=0); ax.set_ylabel("% of its Perturb-seq\ngenes found")
    ax.set_title("63 TFs, |z| ≥ 2, top-25% sgRNA cells", loc="left", fontsize=fs); ax.text(1.0, 1.02, "* CTCF: known exception (untested)", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs * 0.9, color=MID)
    ax2 = fig.add_subplot(gs[1], sharex=ax); ax2.bar(x, g.ours, color=LIGHT, width=0.8); ax2.invert_yaxis(); ax2.set_ylabel("Regulome\ngenes", fontsize=fs); ax2.spines["bottom"].set_visible(False); ax2.spines["top"].set_visible(True)
    ax2.set_xticks(x); ax2.set_xticklabels([t + "*" if t == "CTCF" else t for t in g.TF], rotation=90, fontsize=fs); ax2.tick_params(axis="x", length=0, pad=2)
    ax2.xaxis.set_ticks_position("bottom"); ax2.set_yticks([2500, 5000])
    save(fig, "Fig5i_high_confidence_per_tf", mode)
def panel_j(mode):
    P = style(mode); s = mode == "slide"; fs = 12 if s else 6; BOTH, OURS_O, PERT_O, MID = P.arms.raw, "#F2C48A", P.div[2], P.neutral.mid
    if s: plt.rcParams.update({"ytick.labelsize": 12})
    g = OV[OV.cutoff == "|z| >= 2"].nlargest(25, "both").iloc[::-1]
    fig = plt.figure(figsize=SZ(s) if s else (7.0, 3.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.38, left=0.09, right=0.98, bottom=0.11, top=0.9)
    ax = fig.add_subplot(gs[0]); y = np.arange(len(g))
    ax.barh(y, g.both, color=BOTH, height=0.75); ax.barh(y, g.ours_only, left=g.both, color=OURS_O, height=0.75); ax.barh(y, g.perturbseq_only, left=g.both + g.ours_only, color=PERT_O, height=0.75)
    tot = g.both + g.ours_only + g.perturbseq_only
    for yy, t_, r, p in zip(y, tot, g.recovery_pct, g.precision_pct): ax.text(t_ + 80, yy, f"{r:.0f}% / {p:.0f}%", va="center", fontsize=fs * 0.85)
    ax.set_yticks(y); ax.set_yticklabels([t + "*" if t == "CTCF" else t for t in g.TF]); ax.set_xlabel("genes (|z| ≥ 2)"); ax.set_xlim(0, tot.max() * 1.22)
    ax.set_title("25 TFs with the most genes in both · % of Perturb-seq genes found / % of our links that respond", loc="left", fontsize=fs * 0.95)
    ax.legend(handles=[Patch(color=BOTH, label="both"), Patch(color=OURS_O, label="our Regulome only"), Patch(color=PERT_O, label="Perturb-seq only")], loc="lower right", frameon=False, fontsize=fs * 0.9)
    ax.text(0.99, 0.2, "* CTCF: known exception (untested)", transform=ax.transAxes, ha="right", fontsize=fs * 0.8, color=MID)
    ax2 = fig.add_subplot(gs[1]); r = OP.set_index("cutoff").loc["|z| >= 2"]
    for k, (lab, tot_, col) in enumerate((("Perturb-seq\ngenes", r.perturbseq, PERT_O), ("our Regulome\nlinks", r.ours, OURS_O))):
        ax2.barh(k, r.both, color=BOTH, height=0.55); ax2.barh(k, tot_ - r.both, left=r.both, color=col, height=0.55)
        ax2.text(tot_ * 1.02, k, f"{int(r.both):,} of {int(tot_):,}\n= {100 * r.both / tot_:.1f}% in both", va="center", fontsize=fs * 0.9)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Perturb-seq\ngenes", "our Regulome\nlinks"]); ax2.invert_yaxis(); ax2.set_xlim(0, r.ours * 1.75); ax2.set_xlabel("TF–gene pairs, 63 TFs pooled")
    ax2.set_title("pooled, |z| ≥ 2", loc="left", fontsize=fs)
    r3 = OP.set_index("cutoff").loc["|z| >= 3"]; ax2.text(0.03, 0.5, f"|z| ≥ 3: {int(r3.both):,} of {int(r3.perturbseq):,} Perturb-seq genes\nin both = {r3.recovery_pct:.1f}%", transform=ax2.transAxes, fontsize=fs * 0.85, color=MID, va="center")
    save(fig, "Fig5j_binary_overlap", mode)
F = dict(c=panel_c, h=panel_h, i=panel_i, j=panel_j)
for p in sys.argv[1]:
    for m in ("paper", "slide"): F[p](m)
