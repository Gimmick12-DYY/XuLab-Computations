"""Supplementary to Figure 3:
S_pairs_loop_support      coordinated binding pairs > 25 kb per TF (TFs with >= 500 pairs; log scale) and the share on a Peakachu
                          0.7 loop (red); right: observed % vs the TF's own 10,000-shuffle null (same-chromosome shifts keeping each
                          pair's spacing; box = IQR, line = median, whiskers = 95 %).
S_module_edge_block_null  panel d bars and null (layout of panel d, middle/right) with the stricter null: each whole module shifted
                          as one block (TFs with >= 100 module edges > 25 kb; BH q over those TFs; filled = q < 0.05).
Inputs (source_data/): d_per_tf_null_summary.tsv (observed rates, p values), s_shuffle_null_percentiles.tsv, d_module_loop_coverage.tsv.gz
Outputs: supplementary/S_pairs_loop_support.{pdf,svg,png} (150 mm wide), supplementary/S_module_edge_block_null.* (112 x 90 mm).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import style  # noqa: E402
from module_edges_hic_support import LABS, bh, strata  # noqa: E402


def nullbox(b, i, r, N):
    b.plot([r.p2_5, r.p97_5], [i, i], color=N.mid, lw=0.7, solid_capstyle="butt")
    b.add_patch(__import__("matplotlib.pyplot").pyplot.Rectangle((r.p25, i - 0.28), r.p75 - r.p25, 0.56, color=N.light, lw=0))
    b.plot([r.p50, r.p50], [i - 0.28, i + 0.28], color=N.dark, lw=0.7)


def main(src, out):
    out = Path(out) / "supplementary"
    T = pd.read_csv(src / "d_per_tf_null_summary.tsv", sep="\t").set_index("tf")
    Q = pd.read_csv(src / "s_shuffle_null_percentiles.tsv", sep="\t").set_index(["test", "tf"])

    # ---- pairs ----
    P = style.use()
    import matplotlib.pyplot as plt
    N, RED = P.neutral, P.evidence.hic_loop
    W = T[T.pairs >= 500].sort_values("pairs")
    y = np.arange(len(W))
    fig = style.figure_mm(150, 2.6 * len(W) + 18)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.05)
    a, b = fig.add_subplot(gs[0]), None
    b = fig.add_subplot(gs[1], sharey=a)
    a.barh(y, W.pairs, color=N.pale, height=0.75, label="pairs >25 kb")
    a.barh(y, (W.pair_obs * W.pairs).round(), color=RED, height=0.75, label="… on a 0.7 Hi-C loop (±30 kb)")
    a.set_xscale("log")
    for i, (n, o) in enumerate(zip(W.pairs, W.pair_obs)):
        a.text(n * 1.08, i, f"{100 * o:.1f}%", va="center", fontsize=5, color=N.dark)
    a.set_yticks(y)
    a.set_yticklabels(W.index, fontsize=5.5)
    a.set_xlabel("coordinated binding pairs >25 kb (log scale)")
    a.set_xlim(right=W.pairs.max() * 3)
    a.legend(loc="lower right")
    a.set_title("Pairs with Hi-C loop support", loc="left")
    for i, tf in enumerate(W.index):
        r = Q.loc[("pairs", tf)]
        nullbox(b, i, r, N)
        b.plot([r.p50, W.pair_obs[tf] * 100], [i, i], color=N.light, lw=0.6, zorder=1)
        b.scatter([W.pair_obs[tf] * 100], [i], s=7, color=N.ink, lw=0, zorder=3)
    b.scatter([], [], s=7, color=N.ink, label="observed")
    b.add_patch(plt.Rectangle((0, 0), 0, 0, color=N.light, label="own shuffle null: box = IQR, line = median, whiskers = 95%"))
    b.legend(loc="upper center", bbox_to_anchor=(0.1, -0.075), ncol=2)
    b.set_xlim(12, max(W.pair_obs.max() * 100 + 1.5, 28))
    b.set_xlabel("% of pairs on a loop")
    b.set_title("Observed vs own null (10,000 shuffles)", loc="left")
    b.xaxis.grid(True, color=N.faint, lw=0.5)
    b.set_axisbelow(True)
    b.tick_params(axis="y", left=False, labelleft=False)
    b.spines["left"].set_visible(False)
    a.set_ylim(-0.7, len(W) - 0.3)
    nsig = int((bh(W.pair_p) < 0.05).sum())
    fig.text(0.99, -0.03, f"TFs with ≥500 pairs; same-chromosome shuffles keep each pair's spacing; "
             + (f"all {len(W)} TFs" if nsig == len(W) else f"{nsig} of {len(W)} TFs") + " BH q < 0.05",
             ha="right", fontsize=5.5, color=N.dark)
    style.save(fig, out, "S_pairs_loop_support", bbox_inches="tight")

    # ---- modules with the block-shuffle null (layout of panel d, middle and right) ----
    P = style.use()
    plt.rcParams["axes.linewidth"] = 0.6
    C = pd.read_csv(src / "d_module_loop_coverage.tsv.gz", sep="\t")
    C["stratum"] = [LABS[i] for i in strata(C.loop_edges.values, C.coverage.values)]
    Pt = C.groupby(["tf", "stratum"]).size().unstack(fill_value=0).reindex(columns=LABS, fill_value=0)
    Pt = Pt[Pt.sum(1) >= 20]
    Pt = Pt.loc[sorted(Pt.index, key=lambda t: (Pt.sum(1)[t], t))]
    E = T.loc[[t for t in Pt.index if t in T.index and T.edges[t] >= 100]].copy()
    E["q"] = bh(E.edge_p)
    cap = 2500
    fig = style.figure_mm(112, 90)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 0.001], wspace=0.05, left=0.12, right=0.99, top=0.995, bottom=0.075)
    ax = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1], sharey=ax)
    y = np.arange(len(Pt))
    left = np.zeros(len(Pt))
    scale = np.minimum(1.0, cap / Pt.sum(1).values)
    for l, c in zip(LABS, P.loop_strata):
        v = Pt[l].values.astype(float) * scale
        ax.barh(y, v, left=left, color=c, height=0.72, label=("No loop edge" if l == LABS[0] else l), edgecolor="#4D4D4D", linewidth=0.3)
        left += v
    tot = Pt.sum(1).values
    sup = tot - Pt[LABS[0]].values
    for i, (t, s_) in enumerate(zip(tot, sup)):
        ax.text(cap * 1.01 if t > cap else t + 25, i, (f"// {t:,} ({100 * s_ / t:.0f}%)" if t > cap else f"{100 * s_ / t:.0f}%"),
                va="center", fontsize=5, color=N.dark)
    ax.set_xlim(0, cap * 1.38)
    ax.set_yticks(y)
    ax.set_yticklabels(Pt.index, fontsize=5)
    ax.set_ylim(-0.7, len(Pt) - 0.3)
    ax.set_xlabel("Modules with ≥1 edge >25 kb")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.legend(title="Module edges on a loop", loc="lower right", fontsize=6, title_fontsize=6, handlelength=1.0, handleheight=0.8,
              labelspacing=0.25, borderaxespad=0.2, alignment="left")
    for i, tf in enumerate(Pt.index):
        if tf not in E.index:
            b.text(13.0, i, "n/a", va="center", fontsize=5, color=N.mid)
            continue
        r = Q.loc[("module_edges_block", tf)]
        b.plot([r.p2_5, r.p97_5], [i, i], color=N.mid, lw=0.6, solid_capstyle="butt")
        b.add_patch(plt.Rectangle((r.p25, i - 0.28), r.p75 - r.p25, 0.56, color=N.light, lw=0))
        b.plot([r.p50, r.p50], [i - 0.3, i + 0.3], color=N.dark, lw=0.6)
        b.plot([r.p50, E.edge_obs[tf] * 100], [i, i], color=N.light, lw=0.5, zorder=1)
        sig = E.q[tf] < 0.05
        b.scatter([E.edge_obs[tf] * 100], [i], s=5, color=N.ink if sig else "white", edgecolors=N.ink, linewidths=0 if sig else 0.5, zorder=3)
    b.set_xlim(12, max(E.edge_obs.max() * 100, max(Q.loc[("module_edges_block", t)].p97_5 for t in E.index)) + 1)
    b.set_xlabel("Module edges >25 kb on a loop (%)")
    b.xaxis.grid(True, color=N.faint, lw=0.4)
    b.set_axisbelow(True)
    b.tick_params(axis="y", left=False, labelleft=False)
    b.spines["left"].set_visible(False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=2.6, color=N.ink), plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=N.ink, mew=0.5),
         plt.Rectangle((0, 0), 1, 1, color=N.light)]
    b.legend(h, ["Observed, q < 0.05", "Observed, n.s.", "Shuffle null"], loc="lower right", fontsize=6, handlelength=1.0, labelspacing=0.25,
             borderaxespad=0.2, facecolor="white", frameon=True, framealpha=1, edgecolor="none")
    E[["edges", "edge_obs", "edge_p", "q"]].to_csv(out / "S_module_edge_block_null.tsv", sep="\t")
    print(f"S block null: {int((E.q < 0.05).sum())} of {len(E)} TFs q < 0.05")
    style.save(fig, out, "S_module_edge_block_null")

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
