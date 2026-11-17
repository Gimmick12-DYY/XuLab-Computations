# m36: Figure 5 panel j (binary overlap), slide and paper versions.
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{RG_}/m36_binary_overlap/out"; PL = f"{RG_}/m34_fig5_panels_v3/plots"
T = pd.read_csv(f"{O}/overlap_per_tf.tsv", sep="\t"); PO = pd.read_csv(f"{O}/overlap_pooled.tsv", sep="\t")
def run(mode):
    P = rp.use(); s = mode == "slide"; fs = 12 if s else 6
    if s: plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.linewidth": 0.9})
    BOTH, OURS_O, PERT_O, MID, INK = P.arms.raw, "#F2C48A", P.div[2], P.neutral.mid, P.neutral.ink
    g = T[T.cutoff == "|z| >= 2"].nlargest(25, "both").iloc[::-1]
    fig = plt.figure(figsize=(12.9, 5.8) if s else (7.0, 3.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35, left=0.09, right=0.98, bottom=0.11, top=0.9)
    ax = fig.add_subplot(gs[0]); y = np.arange(len(g))
    ax.barh(y, g.both, color=BOTH, height=0.75, label="both"); ax.barh(y, g.ours_only, left=g.both, color=OURS_O, height=0.75, label="ours only"); ax.barh(y, g.perturbseq_only, left=g.both + g.ours_only, color=PERT_O, height=0.75, label="Perturb-seq only")
    tot = g.both + g.ours_only + g.perturbseq_only
    for yy, t, r, e in zip(y, tot, g.recovery_pct, g.expected_recovery_pct): ax.text(t + 80, yy, f"{r:.0f}% (exp. {e:.0f}%)", va="center", fontsize=fs * 0.85)
    ax.set_yticks(y); ax.set_yticklabels([t + "*" if t == "CTCF" else t for t in g.TF], fontsize=fs); ax.set_xlabel("genes (|z| ≥ 2, top-25% sgRNA cells)"); ax.set_xlim(0, tot.max() * 1.32)
    ax.set_title("25 TFs with the most genes in both; recovery = both / Perturb-seq responders", loc="left", fontsize=fs)
    ax.legend(loc="lower right", frameon=False, fontsize=fs * 0.9); ax.text(0.99, 0.18, "* CTCF: known exception (untested)", transform=ax.transAxes, ha="right", fontsize=fs * 0.8, color=MID)
    ax2 = fig.add_subplot(gs[1]); r = PO.set_index("cutoff").loc["|z| >= 2"]
    bars = [("Perturb-seq\nresponders", r.perturbseq, r.both), ("our Regulome\nlinks", r.ours, r.both)]
    for k, (lab, tot_, both) in enumerate(bars):
        ax2.barh(k, both, color=BOTH, height=0.55); ax2.barh(k, tot_ - both, left=both, color=PERT_O if k == 0 else OURS_O, height=0.55)
        ax2.plot([r.expected_both] * 2, [k - 0.36, k + 0.36], color=INK, lw=2 if s else 1, ls="--")
        ax2.text(tot_ * 1.02, k, f"{int(both):,} of {int(tot_):,}\n= {100 * both / tot_:.1f}% (exp. {100 * r.expected_both / tot_:.1f}%)", va="center", fontsize=fs * 0.9)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels([b[0] for b in bars]); ax2.invert_yaxis(); ax2.set_xlim(0, r.ours * 1.9); ax2.set_xlabel("gene links, 63 TFs pooled")
    ax2.set_title("pooled 2×2 at |z| ≥ 2 (dashed = expected overlap)", loc="left", fontsize=fs)
    r3 = PO.set_index("cutoff").loc["|z| >= 3"]; ax2.text(0.03, 0.5, f"|z| ≥ 3: \n" + f" {int(r3.both):,} of {int(r3.perturbseq):,} responders found = {r3.recovery_pct:.1f}% (exp. {r3.expected_recovery_pct:.1f}%)", transform=ax2.transAxes, fontsize=fs * 0.85, color=MID, va="center")
    name = "Fig5j_binary_overlap"
    if s: fig.savefig(f"{PL}/slide/{name}.png", dpi=300); fig.savefig(f"{PL}/slide/{name}.svg")
    else: fig.savefig(f"{PL}/paper/{name}.png", dpi=600); fig.savefig(f"{PL}/paper/{name}.svg"); fig.savefig(f"{PL}/paper/{name}.pdf")
    plt.close(fig); print("saved", mode)
run("paper"); run("slide")
