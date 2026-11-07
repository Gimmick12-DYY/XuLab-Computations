"""Fig 3d (179 x 53 mm): the chr19 super-enhancer hub of panel c in 3D and as a flower.
Left: Hi-C-constrained structure #9102 (most-closed: 14 of 18 SE loop-edges closed; own-bead rosette from
results_3d_model/se_sedb/scripts/47_rosette_own_beads.py, PyMOL render, true-depth gold SE sphere, grey chromatin, orange
enhancers), re-laid out at panel size without the title and footnote; promoter (blue), enhancer (vermillion) and SE labels.
Right: flower of all 135 Cicero edges in the hub (redrawn from se_sedb/scripts/42_chord_tf.py, flower_final): SE hexagon in
the centre, the 31 partners on a ring in genomic order (clockwise from chr19:0.51 Mb), TF pies on the nodes, ring = partner
class. Arcs reversed relative to the old chord drawing: every SE edge is a petal from the SE out to its partner (filled,
translucent, outline in the edge colour); edges between partners are thin inner chords (with --pp outer they bulge outward
from the ring as outer petals instead; that version, fig3d_se_hub_3d_flower_outer, was too busy at this size). Edge colour =
what it connects (as in panel c); solid = on a Hi-C loop, dashed = Cicero only.
Inputs (inputs/): rosette9102_render.png, rosette9102_nodes.tsv, rosette9102_tf_binding.tsv, flower_tf_binding.tsv,
hub_edges_all.tsv."""
import argparse
import numpy as np, pandas as pd
from PIL import Image
from matplotlib.patches import Circle, Wedge, RegularPolygon, PathPatch, Polygon
from matplotlib.path import Path as MP
from fig3common import P, INP, fig_mm, save, plt

ap = argparse.ArgumentParser(); ap.add_argument("--pp", default="inner", choices=["outer", "inner"]); a = ap.parse_args()
N = P.neutral
EC = dict(se_promoter="#0072B2", se_enhancer="#D55E00", se_other="#999999", pp_promoter="#56B4E9", pp_enhancer="#F0A35E", pp_other="#C8C8C8")
RING = dict(promoter="#0072B2", enhancer="#D55E00", other="#8C8C8C")
TC = dict(se="#8A6A00", promoter="#0072B2", enhancer="#D55E00")
PT = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]
PC = dict(zip(PT, ["#E69F00", "#009E73", "#F0E442", "#CC79A7", "#117733", "#DDCC77", "#44AA99", "#999933", "#AA4499", "#A6761D"])); OTHER = "#BDBDBD"
W, H = 179.0, 53.0
fig = fig_mm(W, H); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.set_aspect("equal")

# ---------------- d1: rosette ----------------
im = Image.open(INP / "rosette9102_render.png").convert("RGBA"); A = np.asarray(im)[..., 3]; ys, xs = np.where(A > 5)
c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.02)
x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); w = im.size[0]
IW = 46.5; IX, IY = 17.5, 5.8
ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="lanczos", zorder=0)
T = pd.read_csv(INP / "rosette9102_nodes.tsv", sep="\t"); T["x"] = IX + (T.px - x0) / w * IW; T["y"] = IY + IW - (T.py - y0) / w * IW
TL = T[T.cls != "other"]; midx = IX + IW / 2
for side, sub in ((-1, TL[TL.x < midx]), (1, TL[TL.x >= midx])):
    sub = sub.sort_values("y"); yy = list(sub.y.values); gap = 2.4
    for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + gap)
    if yy and yy[-1] > IY + IW - 1: yy = [v - (yy[-1] - (IY + IW - 1)) for v in yy]
    for i in range(len(yy) - 2, -1, -1): yy[i] = min(yy[i], yy[i + 1] - gap)
    lx = IX - 0.3 if side < 0 else IX + IW + 0.3
    for r, ty in zip(sub.itertuples(), yy):
        ax.plot([r.x, lx], [r.y, ty], color=N.dark, lw=0.3, zorder=3)
        ax.text(lx + (-0.5 if side < 0 else 0.5), ty, r.label, ha="right" if side < 0 else "left", va="center", fontsize=6,
                fontstyle="normal" if r.cls == "se" else "italic", fontweight="bold" if r.cls == "se" else "normal", color=TC[r.cls], zorder=4)
ly = 2.6; lx = 3.0
ax.plot([lx, lx + 3.5], [ly, ly], color="#A9A9A9", lw=2.2, solid_capstyle="round"); ax.text(lx + 4.6, ly, "Chromatin", va="center", fontsize=6)
for k, (c_, ec_, t_) in enumerate((("#FFC20A", "#5A4500", "SE"), ("#222222", "none", "Promoter / other"), ("#D55E00", "none", "Enhancer"))):
    xx = lx + 17.5 + k * [0, 8.5, 26][k] / max(k, 1) if k else lx + 17.5
    xx = [lx + 17.5, lx + 26.0, lx + 47.5][k]
    ax.add_patch(Circle((xx + 0.9, ly), 0.9, fc=c_, ec=ec_, lw=0.4)); ax.text(xx + 2.4, ly, t_, va="center", fontsize=6)
ax.add_patch(Circle((lx + 61.0, ly), 0.5, fc=PC["SOX4"], lw=0)); ax.text(lx + 62.2, ly, "TF peak", va="center", fontsize=6)

# ---------------- d2: flower ----------------
H_ = pd.read_csv(INP / "hub_edges_all.tsv", sep="\t"); H_["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(H_.cls_a, H_.cls_b, H_.from_se, H_.type)]
B = pd.read_csv(INP / "flower_tf_binding.tsv", sep="\t").fillna({"named": ""}).sort_values("bin_10kb").reset_index(drop=True)
B["label"] = B.label.str.replace("RNU6-2/TMEM259", "TMEM259")
cx, cy, R, PIE = 121.5, 27.5, 15.0, 1.3
n = len(B); ang = {str(k): np.pi / 2 - 2 * np.pi * (i + 0.5) / n for i, k in enumerate(B.bin_10kb)}
pt = lambda k, r=R: np.array([cx + r * np.cos(ang[k]), cy + r * np.sin(ang[k])])
order = H_.assign(o_=H_.type.map(dict(other=0, enhancer=1, promoter=2))).sort_values(["on_loop", "o_"])


def style(r):
    loop = r.on_loop == 1
    return dict(ls="-" if loop else (0, (2.2, 1.6)), alpha_f=0.20 if loop else 0.06, lw=(0.6 if r.from_se else 0.45) if loop else (0.45 if r.from_se else 0.3), z=3 + loop)


def petal_se(k, col, st):
    """Teardrop from the SE centre to the partner node: half-width w * sin(pi * t^1.6), widest at ~65% of the way out."""
    t = np.linspace(0, 1, 60); L = R - PIE * 0.2; wd = R * np.sin(np.pi / n) * 0.95; yv = wd * np.sin(np.pi * t ** 1.6)
    th = ang[k]; u = np.array([np.cos(th), np.sin(th)]); v = np.array([-u[1], u[0]])
    up = np.c_[t * L, yv]; dn = np.c_[t[::-1] * L, -yv[::-1]]; pts = np.vstack([up, dn]); xy = np.array([cx, cy]) + pts[:, :1] * u + pts[:, 1:] * v
    ax.add_patch(Polygon(xy, closed=True, fc=col, ec="none", alpha=st["alpha_f"], zorder=st["z"]))
    ax.plot(xy[:, 0], xy[:, 1], color=col, lw=st["lw"], ls=st["ls"], zorder=st["z"] + 0.5)


def petal_pp_outer(k1, k2, col, st):
    """Outward loop between two ring nodes: cubic Bezier leaving each node radially, bulge grows with angular separation."""
    a1, a2 = ang[k1], ang[k2]; d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    h = R * (0.22 + 0.55 * abs(d) / np.pi)
    p1, p2 = pt(k1), pt(k2); u1 = np.array([np.cos(a1), np.sin(a1)]); u2 = np.array([np.cos(a2), np.sin(a2)])
    q1, q2 = p1 + h * 1.35 * u1, p2 + h * 1.35 * u2
    path = MP([p1, q1, q2, p2], [MP.MOVETO, MP.CURVE4, MP.CURVE4, MP.CURVE4])
    ax.add_patch(PathPatch(path, fc=col, ec="none", alpha=st["alpha_f"] * 0.6, zorder=1))
    ax.add_patch(PathPatch(path, fc="none", ec=col, lw=st["lw"], ls=st["ls"], zorder=1.5))


def chord_inner(k1, k2, col, st):
    q0, q1 = pt(k1), pt(k2); ctr = np.array([cx, cy]) + 0.35 * ((q0 + q1) / 2 - np.array([cx, cy]))
    ax.add_patch(PathPatch(MP([q0, ctr, q1], [MP.MOVETO, MP.CURVE3, MP.CURVE3]), fc="none", ec=col, lw=st["lw"], ls=st["ls"], alpha=0.8, zorder=1.5))


for r in order.itertuples():
    st = style(r); col = EC[r.ek]
    if r.from_se: petal_se(str(r.b if r.a == "SE" else r.a), col, st)
    elif a.pp == "outer": petal_pp_outer(str(r.a), str(r.b), col, st)
    else: chord_inner(str(r.a), str(r.b), col, st)
for rr in B.itertuples():
    q = pt(str(rr.bin_10kb)); tl = [t for t in PT if t in rr.named.split(",")]; no = rr.n_tfs - len(tl); tot = rr.n_tfs
    ax.add_patch(Circle(q, PIE, fc="white", ec="none", zorder=6)); s0 = 90.0
    for t in tl + (["other"] if no else []):
        wd = 360.0 * (no if t == "other" else 1) / tot; ax.add_patch(Wedge(q, PIE, s0, s0 + wd, color=PC.get(t, OTHER), lw=0, zorder=6.1)); s0 += wd
    ax.add_patch(Circle(q, PIE, fc="none", ec=RING[rr.cls], lw=0.7, zorder=6.2))
ax.add_patch(RegularPolygon((cx, cy), 6, radius=2.3, fc="#FFC20A", ec="#5A4500", lw=0.6, zorder=7))
ax.text(cx, cy, "SE", ha="center", va="center", fontsize=6, fontweight="bold", color="#2B2100", zorder=8)
# labels: promoters and enhancers, two side columns with leader lines
OUTR = R + PIE + 0.6 + (R * 0.77 * 1.0 if a.pp == "outer" else 0)
tips = [(pt(str(r.bin_10kb)), r.label, r.cls, ang[str(r.bin_10kb)]) for r in B.itertuples() if r.cls != "other"]
xmax_ext = R * (1.0 + (0.77 * 1.35 * 0.75 if a.pp == "outer" else 0.0)) + PIE
for side in (1, -1):
    sub = sorted([t for t in tips if (t[0][0] >= cx) == (side > 0)], key=lambda t: t[0][1]); ys_ = [t[0][1] for t in sub]
    for _ in range(200):
        for i in range(1, len(ys_)):
            if ys_[i] < ys_[i - 1] + 2.3: d_ = (ys_[i - 1] + 2.3 - ys_[i]) / 2; ys_[i] += d_; ys_[i - 1] -= d_
    lx_ = cx + side * (xmax_ext + 1.0)
    for t, ty in zip(sub, ys_):
        e = t[0] + (PIE + 0.2) * np.array([np.cos(t[3]), np.sin(t[3])])
        ax.plot([e[0], lx_], [e[1], ty], color=N.mid, lw=0.25, zorder=0.5)
        ax.text(lx_ + side * 0.4, ty, t[1], ha="left" if side > 0 else "right", va="center", fontsize=6, fontstyle="italic", color=TC[t[2]])
# legend (right column): node classes + TF pie colours
lx, y = 158.0, H - 3.0
ax.text(lx, y, "Partner", fontsize=6, va="center"); y -= 2.5
for c_, nm in ((RING["promoter"], "promoter"), (RING["enhancer"], "enhancer"), (RING["other"], "other")):
    ax.add_patch(Circle((lx + 0.9, y), 0.8, fc="white", ec=c_, lw=0.7)); ax.text(lx + 2.4, y, nm, va="center", fontsize=6); y -= 2.4
y -= 0.6; ax.text(lx, y, "TF peaks (pies,", fontsize=6, va="center"); y -= 2.3; ax.text(lx, y, "small spheres)", fontsize=6, va="center"); y -= 2.6
seen = set(",".join(B.named).split(",")) - {""}
for t in [t for t in PT if t in seen] + ["other"]:
    ax.add_patch(Circle((lx + 0.9, y), 0.6, fc=PC.get(t, OTHER), lw=0)); ax.text(lx + 2.4, y, t if t != "other" else "other TFs", va="center", fontsize=6, fontstyle="italic" if t != "other" else "normal"); y -= 2.3
print("pp mode", a.pp, "edges", len(H_), "SE petals", int(H_.from_se.sum()), "legend bottom", round(y, 1), "flower x extent", round(cx - xmax_ext, 1), round(cx + xmax_ext, 1))
save(fig, "fig3d_se_hub_3d_flower" + ("" if a.pp == "inner" else "_outer"))
