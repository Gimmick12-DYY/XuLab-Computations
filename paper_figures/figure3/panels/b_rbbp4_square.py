"""Figure 3b (square slot option): RBBP4 pairs as a two-ring donut.

Inner ring: element classes of the two ends of each unique RBBP4 pair (p <= 0.05, > 25 kb): promoter-promoter, promoter-enhancer,
enhancer-enhancer, promoter-repressive, repressive-repressive, other combinations. Outer ring: within each class, pairs on a
Peakachu 0.7 loop (dark) vs not (light tint). Table: % of pairs and % on a loop per class. Palette C colours.

Inputs: as b_rbbp4_characterization.py (via _rbbp4.py).   Output: b_rbbp4_square.{pdf,svg,png}, 60 x 60 mm (print size).
Also writes b_rbbp4_square_classes.tsv (n, on loop, in a module edge, per class).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import style  # noqa: E402
import _rbbp4  # noqa: E402

GREY = "#D9D9D9"
CL6 = [("promoter-promoter", "P–P", "#1F3F8F"), ("promoter-enhancer", "P–E", "#7A9CCB"), ("enhancer-enhancer", "E–E", "#E8821E"),
       ("promoter-repressive", "P–R", "#6A51A3"), ("repressive-repressive", "R–R", "#55606E"), ("rest", "Rest", GREY)]


def main(src, out):
    P = style.use(base_pt=6, small_pt=6)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    N = P.neutral
    D = _rbbp4.load(src)
    uniq = {}
    for k, pq, on, c in zip(D["keys"], D["pairs"], D["pair_on_loop"], D["pair_class"]):
        uniq[k] = (on, c if c in {x[0] for x in CL6[:5]} else "rest")
    T = pd.DataFrame([(c, on, k in D["module_edges"]) for k, (on, c) in uniq.items()], columns=["cls", "loop", "module"])
    SUM = T.groupby("cls").agg(n=("loop", "size"), on_loop=("loop", "sum"), module=("module", "sum")).reindex([c[0] for c in CL6])
    SUM["pct"] = 100 * SUM.n / SUM.n.sum()
    SUM["pct_loop"] = 100 * SUM.on_loop / SUM.n
    SUM.to_csv(Path(out) / "b_rbbp4_square_classes.tsv", sep="\t", float_format="%.3f")
    NAMES = ["Promoter–promoter", "Promoter–enhancer", "Enhancer–enhancer", "Promoter–repressive", "Repressive–repressive", "Other combinations"]
    col = [c[2] for c in CL6]
    tint = lambda c, f=.55: tuple(np.array(to_rgb(c)) + (1 - np.array(to_rgb(c))) * f)
    n, on, pct, loop = SUM.n.values, SUM.on_loop.values, SUM.pct.values, SUM.on_loop.values / SUM.n.values

    fig = style.figure_mm(60, 60)
    ax = fig.add_axes([0.0, 0.40, 1.0, 0.59])
    ax.set_aspect("equal")
    ax.axis("off")
    ek = dict(edgecolor="white", linewidth=0.6)
    ax.pie(n, colors=col, startangle=90, counterclock=False, radius=0.84, wedgeprops=dict(width=0.32, **ek))
    outer, oc = [], []
    for c, k, o in zip(col, n, on):
        outer += [o, k - o]
        oc += [c if c != GREY else "#B3B3B3", tint(c) if c != GREY else "#EFEFEF"]
    ax.pie(outer, colors=oc, startangle=90, counterclock=False, radius=1.0, wedgeprops=dict(width=0.15, **ek))
    ang = 90
    for i, p in enumerate(pct):
        span = p * 3.6
        m = np.deg2rad(ang - span / 2)
        ang -= span
        if p > 8:
            ax.text(0.68 * np.cos(m), 0.68 * np.sin(m), f"{p:.0f}%", ha="center", va="center", color="white" if i < 2 else N.ink,
                    fontsize=6, fontweight="bold")
    ax.text(0, 0.07, f"{int(n.sum()):,}", ha="center", va="center", fontsize=7.5, color=N.ink)
    ax.text(0, -0.12, "RBBP4 pairs", ha="center", va="center", fontsize=6, color=N.dark)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    tx = fig.add_axes([0.04, 0.0, 0.94, 0.34])
    tx.axis("off")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    y0, dy = 0.93, 0.135
    tx.text(0.70, y0, "Pairs", ha="right", va="center", color=N.dark, fontsize=6)
    tx.text(0.98, y0, "On loop", ha="right", va="center", color=N.dark, fontsize=6)
    for i, (nm, c, p, l) in enumerate(zip(NAMES, col, pct, loop)):
        y = y0 - dy * (i + 1)
        tx.add_patch(plt.Rectangle((0.0, y - 0.045), 0.035, 0.09, color=c, lw=0))
        tx.text(0.055, y, nm, va="center", color=N.ink)
        tx.text(0.70, y, f"{p:.1f}%", ha="right", va="center", color=N.ink)
        tx.text(0.98, y, f"{l * 100:.0f}%", ha="right", va="center", color=N.ink)
    fig.text(0.5, 0.375, "outer ring: dark = on a Hi-C loop, light = not on a loop", ha="center", color=N.dark, fontsize=6)
    print(SUM.round(1).to_string())
    style.save(fig, out, "b_rbbp4_square")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
