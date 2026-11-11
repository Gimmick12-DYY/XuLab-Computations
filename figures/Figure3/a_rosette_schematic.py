"""Fig 3 panel a (schematic): what a higher-order co-binding module is, drawn as a possible rosette.
Left: linear genome, five elements (promoters blue, enhancer vermillion) bound by TFs (small coloured ovals), joined by module edges
(arcs, all pairs = fully connected module). Right: the same elements meeting at one hub in 3D, the chromatin between them forming
loops (petals), TFs clustered at the hub. Illustration only, not data. Final size 120 x 62 mm, Arial 6-7 pt, Palette C-type colours;
SVG text as explicit Arial attributes (Illustrator-safe).   python a_rosette_schematic.py <outdir>"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, FancyBboxPatch, Ellipse, Circle, FancyArrowPatch
from matplotlib.path import Path as MP
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "svg.fonttype": "none", "pdf.fonttype": 42})
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."); MM = 1 / 25.4; W, H = 120.0, 52.0
INK, FIB, FIBE = "#1A1A1A", "#C9CDD3", "#8C929B"; PROM, ENH = "#0072B2", "#D55E00"; EDGE = "#B83A2E"
TF = {"TF A": "#E69F00", "TF B": "#009E73", "TF C": "#CC79A7"}
fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
def txt(x, y, s, **k): ax.text(x, y, s, color=k.pop("color", INK), **k)
# ---------- left: linear view ----------
y0 = 21.0; xs = [6, 15.5, 25, 34.5, 44]; kind = ["P", "E", "P", "P", "P"]; tfs = [["TF A", "TF B"], ["TF A", "TF C"], ["TF B", "TF C"], ["TF A", "TF B", "TF C"], ["TF B"]]
ax.plot([2, 48], [y0, y0], color=FIBE, lw=1.6, solid_capstyle="round", zorder=1)
mod = [0, 1, 2, 3]   # the module: 4 elements, all pairs linked
for i in range(len(mod)):
    for j in range(i + 1, len(mod)):
        a, b = xs[mod[i]], xs[mod[j]]; h = 2.5 + 0.26 * (b - a); t = np.linspace(0, 1, 60)
        ax.plot(a + (b - a) * t, y0 + 2.6 + h * np.sin(np.pi * t), color=EDGE, lw=0.7, zorder=2)
for x, k, tl in zip(xs, kind, tfs):
    ax.add_patch(FancyBboxPatch((x - 1.7, y0 - 1.1), 3.4, 2.2, boxstyle="round,pad=0,rounding_size=0.5", fc=PROM if k == "P" else ENH, ec="none", zorder=3))
    for n, t in enumerate(tl):
        ax.add_patch(Ellipse((x - (len(tl) - 1) * 0.95 + n * 1.9, y0 + 1.95), 1.7, 1.25, fc=TF[t], ec="white", lw=0.3, zorder=4))
txt(25, y0 - 4.2, "Linear genome", ha="center", fontsize=6.5, color="#555555")
txt(xs[1] + 4.5, y0 + 15.5, "Module edges", ha="left", fontsize=6, color=EDGE)
txt(2, 9.5, "Higher-order co-binding module:", fontsize=6, fontweight="bold")
txt(2, 6.4, "≥3 regions, every pair co-bound across cells", fontsize=6)
ax.add_patch(FancyArrowPatch((50.5, 30), (59.5, 30), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#555555"))
txt(55, 32.2, "3D", ha="center", fontsize=6, color="#555555")
# ---------- right: rosette ----------
cx, cy, r0 = 89.0, 32.0, 3.6; n = 4; ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n; pts = [np.array([cx + r0 * np.cos(a), cy + r0 * np.sin(a)]) for a in ang]
ax.add_patch(Circle((cx, cy), r0 + 2.4, fc="#F3F4F6", ec="none", zorder=0))
L = [15.0, 18.0, 14.0, 17.0]   # petal lengths (loop sizes)
verts, codes = [pts[0]], [MP.MOVETO]
for i in range(n):
    p, q = pts[i], pts[(i + 1) % n]; am = (ang[i] + np.pi / n); u = np.array([np.cos(am), np.sin(am)]); side = np.array([-u[1], u[0]])
    d = lambda a: np.array([np.cos(a), np.sin(a)]); c1 = p + L[i] * d(am - 0.75); c2 = q + L[i] * d(am + 0.75)   # fanned control points -> round petal
    verts += [c1, c2, q]; codes += [MP.CURVE4] * 3
fib = MP(verts, codes)
ax.add_patch(PathPatch(fib, fc="none", ec=FIBE, lw=4.4, capstyle="round", joinstyle="round", zorder=1))
ax.add_patch(PathPatch(fib, fc="none", ec=FIB, lw=3.2, capstyle="round", joinstyle="round", zorder=1.1))
kinds = ["P", "E", "P", "P"]
for i, (p, k) in enumerate(zip(pts, kinds)):
    ax.add_patch(Circle(p, 1.35, fc=PROM if k == "P" else ENH, ec="white", lw=0.5, zorder=3))
rng = np.random.default_rng(3)
for t, (dx, dy) in zip(["TF A", "TF B", "TF C", "TF A", "TF B", "TF C", "TF B"], [(-1.3, 0.7), (0.3, 1.4), (1.4, -0.2), (-0.4, -1.3), (-1.6, -0.9), (0.5, 0.0), (1.2, 1.2)]):
    ax.add_patch(Ellipse((cx + dx, cy + dy), 1.7, 1.25, angle=rng.uniform(0, 180), fc=TF[t], ec="white", lw=0.3, zorder=4))
txt(cx, 8.5, "Co-bound elements meet at a hub;\nthe chromatin between them forms loops", ha="center", va="center", fontsize=6, color="#555555", linespacing=1.1)
txt(cx + 15.5, cy + 17.5, "Chromatin loop\n(Hi-C)", ha="left", va="center", fontsize=6, color="#555555", linespacing=1.0)
ax.annotate("", xy=(cx + 9.6, cy + 11.6), xytext=(cx + 15.2, cy + 16.5), arrowprops=dict(arrowstyle="-", color="#9E9E9E", lw=0.5))
txt(cx - 14.0, cy - 17.0, "TF hub", ha="right", va="center", fontsize=6, color="#555555")
ax.annotate("", xy=(cx - 2.0, cy - 1.5), xytext=(cx - 13.6, cy - 16.4), arrowprops=dict(arrowstyle="-", color="#9E9E9E", lw=0.5))
# ---------- legend ----------
lx, ly = 2.0, 1.8
for c, l in ((PROM, "Promoter"), (ENH, "Enhancer")):
    ax.add_patch(FancyBboxPatch((lx, ly - 0.8), 2.4, 1.6, boxstyle="round,pad=0,rounding_size=0.4", fc=c, ec="none")); txt(lx + 3.2, ly, l, va="center"); lx += 14
for t, c in TF.items():
    ax.add_patch(Ellipse((lx + 0.9, ly), 1.7, 1.25, fc=c, ec="none")); txt(lx + 2.4, ly, t, va="center"); lx += 11
PS = {("normal", "normal"): "ArialMT", ("bold", "normal"): "Arial-BoldMT"}
def illustrator_text(svg):
    def one(m):
        tag = m.group(0); sm = re.search(r'style="([^"]*)"', tag)
        if not sm: return tag
        keep, size, wt = [], None, "normal"
        for d in [d.strip() for d in sm.group(1).split(";") if d.strip()]:
            k, _, v = d.partition(":"); k, v = k.strip(), v.strip()
            if k == "font":
                mm_ = re.match(r"(?:(?:italic|oblique)\s+)?(?:(bold|[1-9]00)\s+)?([\d.]+)px", v)
                if mm_: wt = "bold" if mm_.group(1) in ("bold", "700") else wt; size = mm_.group(2)
            else: keep.append(f"{k}: {v}")
        if size is None: return tag
        tag = tag.replace(sm.group(0), f'style="{"; ".join(keep)}"' if keep else "")
        return tag[:-1].rstrip() + f' font-family="Arial, {PS[(wt, "normal")]}, Helvetica, sans-serif" font-size="{float(size):.3f}"' + (' font-weight="bold"' if wt == "bold" else "") + ">"
    return re.sub(r"<text\b[^>]*>", one, svg)
stem = OUT / "a_coordinated_binding_rosette"
fig.savefig(f"{stem}.svg"); fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png", dpi=300)
p = Path(f"{stem}.svg"); p.write_text(illustrator_text(p.read_text())); print("written", stem)
