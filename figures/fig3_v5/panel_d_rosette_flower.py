"""Fig 3d v5 (179 x 48 mm): the chr19 super-enhancer hub of panel c in 3D (left) and as a flower (right).
Left: Hi-C-constrained structure #9102 (14 of 18 SE loop-edges closed), rendered by rosette_render.py (camera and crop of
se_sedb/scripts/47_rosette_own_beads.py; no TF spheres): grey chromatin, gold SE, black promoters, orange enhancers,
light-grey other sites. Labels: promoters and SE only.
Right: flower of the 31 Cicero edges from the SE (partner-partner edges left out): SE hexagon in the centre, partners on a
ring in genomic order (clockwise from chr19:0.51 Mb), one petal per SE edge; outline red solid = on a Hi-C loop,
blue dashed = Cicero only, light fill of the same hue. Node fill = element type (black promoter, orange enhancer, light grey
other); TF pies dropped (cleaner at this size). Labels: promoter genes and enhancers E1-E6 (numbering of enhancer_map.tsv).
Inputs (inputs/): rosette9102_v5_render.png, rosette9102_v5_nodes.tsv, flower_tf_binding.tsv, hub_edges_all.tsv, enhancer_map.tsv."""
import numpy as np, pandas as pd
from PIL import Image
from matplotlib.patches import Circle, RegularPolygon, Polygon
from fig3common import INP, RED, RED_FILL, BLUE, BLUE_FILL, INK, GREY, TYPE_COL, SE_GOLD, SE_EDGE, FS, fig_mm, save, plt

W, H = 179.0, 48.0
fig = fig_mm(W, H); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.set_aspect("equal")
EM = pd.read_csv(INP / "enhancer_map.tsv", sep="\t").set_index("old_label").number.to_dict()
short = lambda l: (f"E{EM[l]}" if l in EM else "E" + l.split()[1] if l.startswith("enhancer ") else l.replace("RNU6-2/TMEM259", "TMEM259"))


def spread(ys, gap, lo, hi):
    ys = list(ys)
    for _ in range(300):
        for i in range(1, len(ys)):
            if ys[i] < ys[i - 1] + gap: d_ = (ys[i - 1] + gap - ys[i]) / 2; ys[i] += d_; ys[i - 1] -= d_
        ys = [min(max(v, lo), hi) for v in ys]
    return ys


# ---------------- d1: rosette ----------------
im = Image.open(INP / "rosette9102_v5_render.png").convert("RGBA"); A = np.asarray(im)[..., 3]; ys, xs = np.where(A > 5)
c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.02)
x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); w = im.size[0]
IW = 44.0; IX, IY = 22.0, 3.8
ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="lanczos", zorder=0)
T = pd.read_csv(INP / "rosette9102_v5_nodes.tsv", sep="\t"); T["x"] = IX + (T.px - x0) / w * IW; T["y"] = IY + IW - (T.py - y0) / w * IW
TL = T[T.cls.isin(["promoter", "se"])]; midx = IX + IW / 2
for side, sub in ((-1, TL[TL.x < midx]), (1, TL[TL.x >= midx])):
    sub = sub.sort_values("y"); yy = spread(sub.y.values, 2.4, IY + 1.0, IY + IW - 1.0)
    lx = IX - 0.5 if side < 0 else IX + IW + 0.5
    for r, ty in zip(sub.itertuples(), yy):
        se = r.cls == "se"
        ax.plot([r.x, lx], [r.y, ty], color=INK, lw=0.3, zorder=3)
        ax.text(lx + (-0.4 if side < 0 else 0.4), ty, "SE" if se else r.label, ha="right" if side < 0 else "left", va="center", fontsize=FS,
                fontstyle="normal" if se else "italic", fontweight="bold" if se else "normal", color=SE_EDGE if se else INK, zorder=4)
ly, lx = 1.8, IX - 9.0   # one-row node key for d1 and d2
for k, (c_, ec_, t_) in enumerate(((SE_GOLD, SE_EDGE, "SE"), (TYPE_COL["promoter"], "none", "Promoter"), (TYPE_COL["enhancer"], "none", "Enhancer"), (TYPE_COL["other"], "none", "Other"))):
    xx = lx + [0, 8.0, 22.0, 36.0][k]
    ax.add_patch(Circle((xx + 0.9, ly), 0.9, fc=c_, ec=ec_, lw=0.4)); ax.text(xx + 2.4, ly, t_, va="center", fontsize=FS)

# ---------------- d2: flower ----------------
H_ = pd.read_csv(INP / "hub_edges_all.tsv", sep="\t"); S_ = H_[H_.from_se == 1].copy(); assert len(S_) == 31
S_["k"] = [str(b if str(a) == "SE" else a) for a, b in zip(S_.a, S_.b)]
B = pd.read_csv(INP / "flower_tf_binding.tsv", sep="\t").sort_values("bin_10kb").reset_index(drop=True); assert len(B) == 31
assert set(S_.k) == set(B.bin_10kb.astype(str))
cx, cy, R, NR = 135.0, 24.5, 18.5, 1.05
n = len(B); ang = {str(k): np.pi / 2 - 2 * np.pi * (i + 0.5) / n for i, k in enumerate(B.bin_10kb)}
pt = lambda k, r=R: np.array([cx + r * np.cos(ang[k]), cy + r * np.sin(ang[k])])
for r in S_.sort_values("on_loop").itertuples():
    loop = r.on_loop == 1; col, fill = (RED, RED_FILL) if loop else (BLUE, BLUE_FILL)
    t = np.linspace(0, 1, 60); L = R - NR; wd = R * np.sin(np.pi / n) * 0.62; yv = wd * np.sin(np.pi * t ** 1.6)
    th = ang[r.k]; u = np.array([np.cos(th), np.sin(th)]); v = np.array([-u[1], u[0]])
    pts = np.vstack([np.c_[t * L, yv], np.c_[t[::-1] * L, -yv[::-1]]]); xy = np.array([cx, cy]) + pts[:, :1] * u + pts[:, 1:] * v
    ax.add_patch(Polygon(xy, closed=True, fc=fill, ec="none", alpha=0.6, zorder=2 + loop))
    ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.6 if loop else 0.5, ls="-" if loop else (0, (2.0, 1.4)), zorder=2.5 + loop)
for rr in B.itertuples():
    ax.add_patch(Circle(pt(str(rr.bin_10kb)), NR, fc=TYPE_COL[rr.cls], ec="white", lw=0.3, zorder=6))
ax.add_patch(RegularPolygon((cx, cy), 6, radius=2.4, fc=SE_GOLD, ec=SE_EDGE, lw=0.5, zorder=7))
ax.text(cx, cy, "SE", ha="center", va="center", fontsize=FS, fontweight="bold", color=INK, zorder=8)
tips = [(pt(str(r.bin_10kb)), short(r.label), ang[str(r.bin_10kb)]) for r in B.itertuples() if r.cls != "other"]
for side in (1, -1):
    sub = sorted([t for t in tips if (t[0][0] >= cx) == (side > 0)], key=lambda t: t[0][1]); yy = spread([t[0][1] for t in sub], 2.35, 1.5, H - 1.5)
    lx_ = cx + side * (R + NR + 1.6)
    for t, ty in zip(sub, yy):
        e = t[0] + (NR + 0.2) * np.array([np.cos(t[2]), np.sin(t[2])])
        ax.plot([e[0], lx_], [e[1], ty], color=GREY, lw=0.25, zorder=0.5)
        ax.text(lx_ + side * 0.4, ty, t[1], ha="left" if side > 0 else "right", va="center", fontsize=FS, fontstyle="normal" if t[1].startswith("E") and t[1][1:].isdigit() else "italic")
print("SE petals", len(S_), "on loop", int(S_.on_loop.sum()), "labels", [t[1] for t in tips])
save(fig, "fig3d_se_hub_3d_flower")
