"""RBBP4 square, two-ring donut in palette C (2026-11-21; coordinator draft). Inner ring = pair class of our 19,524 RBBP4 pairs (>25 kb);
outer ring = within each class, on a Peakachu 0.7 Hi-C loop (dark) vs not (light tint). n and on-loop counts from rbbp4_square_A.py
(class table printed there). Text raised to >= 6 pt (lab minimum).
  python Figure3/scripts/rbbp4_square_donut.py <out path without extension>   (Mac)"""
import sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import to_rgb
fam = "Arial" if any("Arial" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
INK, DARK, MID = "#1A1A1A", "#4D4D4D", "#8C8C8C"
names = ["Promoter–promoter", "Promoter–enhancer", "Enhancer–enhancer", "Promoter–repressive", "Repressive–repressive", "Other combinations"]
n = np.array([3761, 1828, 308, 502, 16, 13109]); on_loop = np.array([1112, 516, 94, 114, 5, 3348]); loop = on_loop / n
col = ["#1F3F8F", "#7A9CCB", "#E8821E", "#6A51A3", "#55606E", "#D9D9D9"]
def tint(c, f=.55): r = np.array(to_rgb(c)); return tuple(r + (1 - r) * f)
pct = n / n.sum() * 100
mm = 1 / 25.4; fig = plt.figure(figsize=(60 * mm, 60 * mm))
ax = fig.add_axes([0.0, 0.40, 1.0, 0.59]); ax.set_aspect("equal"); ax.axis("off")
ek = dict(edgecolor="white", linewidth=0.6)
ax.pie(n, colors=col, startangle=90, counterclock=False, radius=0.84, wedgeprops=dict(width=0.32, **ek))
outer = []; oc = []
for c, k, l in zip(col, n, loop):
    on = k * l; outer += [on, k - on]
    oc += [c if c != "#D9D9D9" else "#B3B3B3", tint(c) if c != "#D9D9D9" else "#EFEFEF"]
ax.pie(outer, colors=oc, startangle=90, counterclock=False, radius=1.0, wedgeprops=dict(width=0.15, **ek))
ang = 90
for i, (p, c) in enumerate(zip(pct, col)):
    span = p * 3.6; mid = np.deg2rad(ang - span / 2); ang -= span
    if p > 8:
        r = 0.68; ax.text(r * np.cos(mid), r * np.sin(mid), f"{p:.0f}%", ha="center", va="center", color="white" if i < 2 else INK, fontsize=6, fontweight="bold")
ax.text(0, 0.07, "19,524", ha="center", va="center", fontsize=7.5, color=INK)
ax.text(0, -0.12, "RBBP4 pairs", ha="center", va="center", fontsize=6, color=DARK)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
tx = fig.add_axes([0.04, 0.0, 0.94, 0.34]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(0, 1)
y0 = 0.93; dy = 0.135
tx.text(0.70, y0, "Pairs", ha="right", va="center", color=DARK, fontsize=6)
tx.text(0.98, y0, "On loop", ha="right", va="center", color=DARK, fontsize=6)
for i, (nm, c, p, l) in enumerate(zip(names, col, pct, loop)):
    y = y0 - dy * (i + 1)
    tx.add_patch(plt.Rectangle((0.0, y - 0.045), 0.035, 0.09, color=c, lw=0))
    tx.text(0.055, y, nm, va="center", color=INK)
    tx.text(0.70, y, f"{p:.1f}%", ha="right", va="center", color=INK)
    tx.text(0.98, y, f"{l * 100:.0f}%", ha="right", va="center", color=INK)
fig.text(0.5, 0.375, "outer ring: dark = on a Hi-C loop, light = not on a loop", ha="center", color=DARK, fontsize=6)
out = sys.argv[1] if len(sys.argv) > 1 else "rbbp4_donut2"
for ext in ("png", "svg", "pdf"): fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
