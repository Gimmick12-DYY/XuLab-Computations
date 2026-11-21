"""Figure 3b (option 1): genome-wide circos of RBBP4 module edges with Hi-C loop support.

Every edge inside an RBBP4 higher-order coordinated binding module is a chord; edges on a wild-type HEK293T Peakachu loop
(score >= 0.7, both ends within +-30 kb of the anchors) are dark red with a wide light band underneath (5 kb pink, 10 kb blue);
modules with every edge on a loop are gold. Ticks inside the ring = module regions.

Inputs (source_data/):  rbbp4_modules.tsv, rbbp4_module_edges_on_loops.tsv
Output: b2_rbbp4_hic_circos.{pdf,svg,png}. Drawn on a 17 x 17 in canvas (large original; scale it down when placing).
"""
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

HG38 = {"chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
        "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
        "chr19": 58617616, "chr20": 64444167, "chr21": 46709983, "chr22": 50818468, "chrX": 156040895}
TF_SUP, TF_OTHER, LOOP5, LOOP10, GOLD, INK = "#7A0F3A", "#3B4A5C", "#F48FB1", "#90CAF9", "#F2B705", "#20242A"


def parse(p):
    c, xy = p.split(":")
    s, e = xy.split("-")
    return c, int(s), int(e)


def key(a, b):
    return (a, b) if a <= b else (b, a)


def main(src, out):
    style.use()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MP

    M = pd.read_csv(src / "rbbp4_modules.tsv", sep="\t")
    hits = pd.read_csv(src / "rbbp4_module_edges_on_loops.tsv", sep="\t")
    gap = math.radians(1.6)
    usable, genome = 2 * math.pi - gap * len(HG38), sum(HG38.values())
    spans, cur = {}, math.pi / 2
    for c, ln in HG38.items():
        w = usable * ln / genome
        spans[c] = (cur, cur - w)
        cur -= w + gap
    theta = lambda c, p: spans[c][0] + (spans[c][1] - spans[c][0]) * p / HG38[c]
    xy = lambda t, r: (r * math.cos(t), r * math.sin(t))

    res = {key(a, b): str(r) for a, b, r in zip(hits.peak1, hits.peak2, hits.resolution)}
    edges, regions = [], set()
    for regs in M.regions:
        peaks = regs.split(";")
        regions.update(peaks)
        edges.extend(key(a, b) for a, b in combinations(peaks, 2))
    edges = list(dict.fromkeys(edges))

    fig, ax = plt.subplots(figsize=(17, 17))
    for i, c in enumerate(HG38):
        s, e = spans[c]
        ts = np.linspace(s, e, 120)
        ax.plot(np.cos(ts), np.sin(ts), color=("#495867", "#8493A3")[i % 2], lw=8, solid_capstyle="butt", zorder=5)
        ax.text(*xy((s + e) / 2, 1.075), c[3:], ha="center", va="center", fontsize=9, fontweight="bold", color=INK)

    def chord(a, b, color, lw, alpha, z, r=0.93):
        (ca, sa, ea), (cb, sb, eb) = parse(a), parse(b)
        if ca not in spans or cb not in spans:
            return
        t1, t2 = theta(ca, (sa + ea) / 2), theta(cb, (sb + eb) / 2)
        mid = math.atan2(math.sin(t1) + math.sin(t2), math.cos(t1) + math.cos(t2))
        ax.add_patch(PathPatch(MP([xy(t1, r), xy(mid, 0.45), xy(t2, r)], [MP.MOVETO, MP.CURVE3, MP.CURVE3]),
                               facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round", zorder=z))

    for a, b in edges:                                   # loop bands underneath
        if (a, b) in res:
            chord(a, b, LOOP5 if res[(a, b)] == "5000" else LOOP10, 6.5, 0.55, 1)
    for a, b in edges:
        if (a, b) not in res:
            chord(a, b, TF_OTHER, 0.9, 0.55, 3)
    full, n_full = set(), 0
    for regs in M.regions:
        es = [key(a, b) for a, b in combinations(regs.split(";"), 2)]
        if es and all(e in res for e in es):
            full.update(es)
            n_full += 1
    for a, b in edges:
        if (a, b) in res and (a, b) not in full:
            chord(a, b, TF_SUP, 1.6, 0.95, 4)
    for a, b in edges:
        if (a, b) in full:
            chord(a, b, "#111111", 3.4, 0.9, 5)
            chord(a, b, GOLD, 2.2, 1.0, 6)
    for p in sorted(regions):
        c, s, e = parse(p)
        if c in spans:
            t = theta(c, (s + e) / 2)
            ax.plot(*zip(xy(t, 0.93), xy(t, 0.975)), color=TF_SUP, lw=0.8, alpha=0.9, zorder=6)

    n_sup = sum(1 for k in edges if k in res)
    ax.text(0, 0.08, "RBBP4", ha="center", va="center", fontsize=26, fontweight="bold", color=INK)
    ax.text(0, -0.04, f"{len(M):,} modules · {len(edges):,} coordinated binding edges", ha="center", fontsize=13, color="#5C626B")
    ax.text(0, -0.12, f"{n_sup:,} edges on a Hi-C loop", ha="center", fontsize=13, color="#5C626B")
    ax.text(0, -0.20, f"{n_full:,} modules with every edge on a loop", ha="center", fontsize=13, color="#B38600")
    ax.legend(handles=[Line2D([0], [0], color=TF_OTHER, lw=1.2, alpha=0.7, label="RBBP4 module edge"),
                       Line2D([0], [0], color=TF_SUP, lw=2.2, label="RBBP4 module edge on a loop"),
                       Line2D([0], [0], color=GOLD, lw=3, label=f"module with every edge on a loop ({n_full})"),
                       Line2D([0], [0], color=LOOP5, lw=7, alpha=0.7, label="Peakachu 5 kb loop (underneath)"),
                       Line2D([0], [0], color=LOOP10, lw=7, alpha=0.7, label="Peakachu 10 kb loop (underneath)")],
              loc="lower right", fontsize=11)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    print(f"b2: {len(M)} modules, {len(edges)} edges, {n_sup} on a loop, {n_full} fully supported")
    style.save(fig, out, "b2_rbbp4_hic_circos", bbox_inches=style.content_bbox(fig, 0.05))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
