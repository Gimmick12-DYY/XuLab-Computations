"""Figure 3i: network illustration ("flower") of the chr19 super-enhancer hub (SEdb SE_02_1951, HEK293T).

All Cicero edges of the hub (union over TFs): edges from the SE are radial spokes to its partner elements on a ring
(ordered by position); partner-partner edges bulge outward as petals. Colour = class of the partner element / edge
(promoter blue, enhancer vermillion, other grey; lighter shades between partners); solid = on a Peakachu 0.7 Hi-C loop,
dashed = Cicero only. Partner dots coloured by class; promoter and enhancer partners labelled.

Inputs (source_data/):  se_hub_edges.tsv (hub edges), se_partners.tsv (partner class), se_enhancer_names.tsv
Output: se_flower.{pdf,svg,png}, 89 mm wide (print size); labels 6 pt, "SE" 7 pt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

W_IN, H_IN = 9.4, 6.2                   # design canvas (data units = design inches)
S = 89.0 / (W_IN * 25.4)                # print scale: the canvas is printed 89 mm wide
L0 = 430000                             # left end of the modelled window (chr19), 2 kb beads


def mid(r):
    return sum(map(int, r.split(":")[1].split("-"))) // 2


def main(src, out):
    P = style.use()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, PathPatch, RegularPolygon
    from matplotlib.path import Path as MP

    EC, NODE = style.SE_EDGE, style.SE_NODE
    H = pd.read_csv(src / "se_hub_edges.tsv", sep="\t")
    for c in ("label_a", "label_b"):
        H[c] = H[c].str.replace("RNU6-2/TMEM259", "TMEM259")
    H["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(H.cls_a, H.cls_b, H.from_se, H.type)]
    nodes = {}
    for r in H.itertuples():
        for k, l in ((r.a, r.label_a), (r.b, r.label_b)):
            if str(k) != "SE":
                nodes[str(k)] = l
    Pn = pd.DataFrame([dict(key=k, label=v) for k, v in nodes.items()])
    Pn["keyn"] = Pn.key.astype(int)
    Pn = Pn.sort_values("keyn").reset_index(drop=True)
    T = pd.read_csv(src / "se_partners.tsv", sep="\t")
    T["bin"] = [mid(x) // 10000 for x in T.partner]
    assert T.bin.is_unique and set(T.bin) == set(Pn.keyn)
    cls = dict(zip(T.bin, T.cls))
    names = pd.read_csv(src / "se_enhancer_names.tsv", sep="\t").set_index("bin_10kb").name.to_dict()
    Pn["cls"] = [cls[k] for k in Pn.keyn]
    Pn["label"] = [names.get(k, l) for k, l in zip(Pn.keyn, Pn.label)]

    fig = plt.figure(figsize=(W_IN * S, H_IN * S))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_IN)
    ax.set_ylim(0, H_IN)
    ax.set_axis_off()
    cx, cy, Rg, PIE = 4.85, 3.1, 2.05, 0.075
    n = len(Pn)
    ang = {r.key: np.pi / 2 - 2 * np.pi * (i + 0.5) / n for i, r in Pn.iterrows()}
    pt = lambda k: np.array([cx + Rg * np.cos(ang[k]), cy + Rg * np.sin(ang[k])])
    order = H.assign(o_=H.type.map(dict(other=0, enhancer=1, promoter=2))).sort_values(["from_se", "on_loop", "o_"])
    for r in order.itertuples():
        c_ = EC[r.ek]
        ls = "-" if r.on_loop == 1 else (0, (2.5, 2))
        al = 1 if r.on_loop == 1 else 0.8 if r.from_se else 0.55
        lw = (2.6 if (r.from_se and r.on_loop == 1) else 1.2 if r.on_loop == 1 else 1.1 if r.from_se else 0.5) * S
        if r.from_se:                                      # spoke from the SE
            q = pt(str(r.b if r.a == "SE" else r.a))
            ax.plot([cx, q[0]], [cy, q[1]], color=c_, lw=lw, ls=ls, alpha=al, zorder=3 + r.on_loop)
        else:                                              # partner-partner petal (cubic Bezier bulging outward)
            q0, q1 = pt(str(r.a)), pt(str(r.b))
            t0, t1 = ang[str(r.a)], ang[str(r.b)]
            dt = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
            um = np.array([np.cos(t0 + dt / 2), np.sin(t0 + dt / 2)])
            C0 = np.array([cx, cy])
            u0, u1 = (q0 - C0) / Rg, (q1 - C0) / Rg
            d0, d1 = (u0 + um) / np.linalg.norm(u0 + um), (u1 + um) / np.linalg.norm(u1 + um)
            rpk = Rg + Rg * (0.08 + 0.18 * abs(dt) / np.pi)
            Lc = (rpk - Rg * np.cos(dt / 2)) * 8 / (3 * float(np.dot(d0 + d1, um)))
            ax.add_patch(PathPatch(MP([q0, q0 + Lc * d0, q1 + Lc * d1, q1], [MP.MOVETO, MP.CURVE4, MP.CURVE4, MP.CURVE4]),
                                   fc="none", ec=c_, lw=lw, ls=ls, alpha=al, zorder=1 + r.on_loop + 0.1 * r.o_))
    tips = []
    for r in Pn.itertuples():
        q = pt(r.key)
        ax.add_patch(Circle(q, PIE, fc=NODE[r.cls], ec="none", zorder=6.2))
        if r.cls != "other":
            tips.append((q[0], q[1], r.label, r.cls, ang[r.key]))
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=0.34, fc=NODE.se_fill, ec=NODE.se_edge, lw=1.6 * S, zorder=7))
    ax.text(cx, cy, "SE", ha="center", va="center", fontweight="bold", color="#2B2100", zorder=8, fontsize=7)
    rout = Rg + PIE + Rg * 0.27 + 0.08
    for side, sub in ((1, [t for t in tips if t[0] >= cx]), (-1, [t for t in tips if t[0] < cx])):
        sub = sorted(sub, key=lambda t: t[1])
        ys = [cy + rout * np.sin(t[4]) for t in sub]
        lx = cx + side * (rout + 0.3)
        for _ in range(100):                               # spread labels to >= 0.3 apart
            for i in range(1, len(ys)):
                if ys[i] < ys[i - 1] + 0.3:
                    d = (ys[i - 1] + 0.3 - ys[i]) / 2
                    ys[i] += d
                    ys[i - 1] -= d
        for t, ty in zip(sub, ys):
            ax.plot([cx + (Rg + PIE) * np.cos(t[4]), cx + rout * np.cos(t[4]), lx],
                    [cy + (Rg + PIE) * np.sin(t[4]), cy + rout * np.sin(t[4]), ty], color="#AAAAAA", lw=0.5 * S, zorder=0)
            ax.text(lx + side * 0.04, ty, t[2], ha="left" if side > 0 else "right", va="center", fontsize=6, fontstyle="italic",
                    color=NODE.enhancer if t[3] == "enhancer" else "#222222")
    style.save(fig, out, "se_flower", bbox_inches=style.content_bbox(fig, 0.01))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
