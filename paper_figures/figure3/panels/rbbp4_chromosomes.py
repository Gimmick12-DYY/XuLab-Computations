"""Figure 3c: RBBP4 coordinated binding pairs and module edges along every chromosome, with the Hi-C loops that support them.

Above each chromosome: all RBBP4 first-step Cicero pairs > 25 kb (dark blue) and, on top, the module edges (pairs inside
higher-order coordinated binding modules, > 25 kb; yellow). Below: only the Peakachu score >= 0.7 loops (wild-type HEK293T,
5 kb + 10 kb) that support at least one drawn pair under the loop-support rule (_genomics.py), in red.

Inputs (source_data/): rbbp4_peak_edges.tsv.gz, rbbp4_modules.tsv, peakachu_{5000,10000}.loops.0.7.bedpe.gz
Output: rbbp4_chromosomes.{pdf,svg,png}, 180 x 150 mm (print size).
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import style  # noqa: E402
import _genomics as G  # noqa: E402

PAIRC, MODC, LOOPC, CHR, INK = "#1F3F8F", "#F2B705", "#C0392B", "#7F7F7F", "#1A1A1A"


def main(src, out):
    style.use(base_pt=7, small_pt=6)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    plt.rcParams["font.size"] = 6
    L = G.Loops(src / "peakachu_5000.loops.0.7.bedpe.gz", src / "peakachu_10000.loops.0.7.bedpe.gz")
    pairs = [p for p in G.load_pairs(src / "rbbp4_peak_edges.tsv.gz") if G.span_bp(p) > G.MIN_SPAN]
    edges = [e for e in G.load_module_edges(src / "rbbp4_modules.tsv") if G.span_bp(e) > G.MIN_SPAN]
    key = lambda lp: (lp["a"][0], lp["a"][1], lp["a"][2], lp["b"][1], lp["b"][2], lp["res"])
    sup_pair, sup_mod, n_on = {}, {}, 0
    for p in pairs:
        lp = L.assign(p)
        if lp is not None:
            sup_pair[key(lp)] = lp
            n_on += 1
    for e in edges:
        lp = L.assign(e)
        if lp is not None:
            sup_mod[key(lp)] = lp
    loops_drawn = [v for k, v in sup_pair.items() if k not in sup_mod] + list(sup_mod.values())
    n_sup = len(set(sup_pair) | set(sup_mod))
    print(f"c: pairs {len(pairs):,} (on a loop {n_on:,}); module edges {len(edges):,}; supporting loops {n_sup:,} of {L.n_total:,}")

    CH = list(G.HG38)
    y_of = {c: len(CH) - 1 - i for i, c in enumerate(CH)}

    def curve(c, a, b, side, hmax=0.42):
        x1, x2 = sorted((a / 1e6, b / 1e6))
        h = min(hmax, 0.07 + 0.2 * math.sqrt(max(x2 - x1, 1e-3)))
        t = np.linspace(0, 1, 13)
        return np.c_[x1 + (x2 - x1) * t, y_of[c] + side * h * np.sin(np.pi * t)]

    m = lambda p: (p[1] + p[2]) / 2
    fig, ax = plt.subplots(figsize=(180 * style.MM, 150 * style.MM))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.995, bottom=0.075)
    for c in CH:
        ax.plot([0, G.HG38[c] / 1e6], [y_of[c], y_of[c]], color=CHR, lw=1.4, solid_capstyle="round", zorder=4)
        ax.text(-3, y_of[c], c.replace("chr", ""), ha="right", va="center", fontsize=6, color=INK)
    ax.add_collection(LineCollection([curve(p["p1"][0], m(p["p1"]), m(p["p2"]), 1) for p in pairs if p["p1"][0] in y_of],
                                     colors=PAIRC, linewidths=0.3, alpha=0.55, zorder=1))
    ax.add_collection(LineCollection([curve(l["a"][0], m(l["a"]), m(l["b"]), -1) for l in loops_drawn if l["a"][0] in y_of],
                                     colors=LOOPC, linewidths=0.35, alpha=0.8, zorder=2))
    ax.add_collection(LineCollection([curve(e["p1"][0], m(e["p1"]), m(e["p2"]), 1) for e in edges if e["p1"][0] in y_of],
                                     colors=MODC, linewidths=0.7, zorder=5))
    ax.set_xlim(-8, max(G.HG38.values()) / 1e6 + 6)
    ax.set_ylim(-0.6, y_of[CH[0]] + 0.55)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(CHR)
    ax.set_xlabel("Position (Mb, hg38)")
    ax.tick_params(axis="x", colors=INK)
    ax.text(-7.5, y_of[CH[0]] + 0.42, "pairs ↑   loops ↓", fontsize=5.5, color="#555555", ha="left", va="center")
    ax.legend([Line2D([], [], color=c, lw=1.2) for c in (PAIRC, MODC, LOOPC)],
              [f"Coordinated binding pair (n = {len(pairs):,})", f"Multi-region Coordinated Cluster (MCC) edge (n = {len(edges):,})", f"Hi-C loop supporting a pair (n = {n_sup:,})"],
              loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=6, handlelength=1.4, labelspacing=0.35,
              title=f"Supporting loops: {n_sup:,} of {L.n_total:,} Peakachu loops", title_fontsize=6, alignment="left")
    style.save(fig, out, "rbbp4_chromosomes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
