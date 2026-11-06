"""v2 step 8 (Mac): 80 x 80 mm Illustrator panels for the hub-flower rosette + 2D schematic (Arial 7 pt, live SVG text).

  flower_render(_ghost).svg/.pdf   render (600 dpi PNG, transparent) + vector labels: hub cores (italic, leader lines),
                                   petals P1.. at their tips with span, legend, Cicero annotation; *_linked.svg link the PNG
  flower_schematic.svg/.pdf        TF core as a circle, petals as loops labelled with span (kb) and coordinates, hub cores
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_flower"; R = O / "raster"; R.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600})
MM = 1 / 25.4; PT = 72 / 25.4; W = H = 80.0
Z = np.load(O / "flower_pick.npz")
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""})
pet = pd.read_csv(O / "flower_petals.tsv", sep="\t"); pj = pd.read_csv(O / "flower_projection.tsv", sep="\t")
anc = pd.read_csv(O / "flower_anchors.tsv", sep="\t").fillna({"hub_label": ""}); aset = set(anc.bead)
CP = pd.read_csv(Bd / "inputs/pairs_cicero.tsv", sep="\t"); HE = CP[CP.set == "hub_edges"]
he_in = HE[HE.b1.isin(aset) & HE.b2.isin(aset)]
ft = pd.read_csv(O / "flower_table.tsv", sep="\t")
n_tf = int((Z["LT"] == int(Z["hc"])).sum())
hubs = pj[pj.kind == "hub"]; petp = pj[pj.kind == "petal"]

def save(fig, stem, tr=True):
    for e in ("svg", "pdf"): fig.savefig(O / f"{stem}.{e}", transparent=tr)
    fig.savefig(O / f"{stem}_preview.png", dpi=250, transparent=False, facecolor="white"); plt.close(fig)

for ver in ("render_flower", "render_flower_ghost"):
    im = np.asarray(Image.open(O / f"{ver}.png").convert("RGBA"))
    h0, w0 = im.shape[:2]
    if ver == "render_flower":
        ys, xs = np.where(im[..., 3] > 5); c = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2)
        half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * 1.08)
        box = (max(0, c[0] - half), max(0, c[1] - half), min(w0, c[0] + half), min(h0, c[1] + half))
    cr = im[box[1]:box[3], box[0]:box[2]]; ch, cw = cr.shape[:2]
    IMW = 62.0; IMH = IMW * ch / cw; IX = (W - IMW) / 2 + 4; IY = 9.5
    dpi = cw / (IMW * MM)
    Image.fromarray(cr, "RGBA").save(R / f"{ver}.png", dpi=(round(dpi), round(dpi)))
    tox = lambda px: IX + (px - box[0]) / cw * IMW; toy = lambda py: IY + IMH - (py - box[1]) / ch * IMH
    fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.imshow(cr, extent=(IX, IX + IMW, IY, IY + IMH), interpolation="none", zorder=0)
    # hub-core labels: column at the upper left, leader lines to the cores
    hs = hubs.assign(x=tox(hubs.px), y=toy(hubs.py)).sort_values("y", ascending=False)
    for k, r in enumerate(hs.itertuples()):
        ty = H - 6 - k * 3.4
        ax.plot([r.x, 17.4], [r.y, ty], color="#333333", lw=0.4, zorder=3)
        ax.text(1, ty, r.label, ha="left", va="center", fontstyle="italic", zorder=4)
    cx0, cy0 = tox(hubs.px.mean()), toy(hubs.py.mean())
    for r in petp.itertuples():
        p = pet[pet.petal == r.label].iloc[0]; x, y = tox(r.px), toy(r.py); v = np.array([x - cx0, y - cy0]); v /= np.linalg.norm(v)
        ax.text(min(max(x + 4.5 * v[0], 9), W - 9), min(max(y + 3.5 * v[1], 12), H - 14), f"{r.label} {p.span_kb} kb", ha="center",
                va="center", fontsize=6, color=p.colour, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.add_patch(Circle((3, 7), 1.0, color="#D55E00", lw=0)); ax.text(4.6, 7, f"TF core ({n_tf} TFs)", va="center", fontsize=6)
    ax.add_patch(Circle((24, 7), 0.6, color="#A6A6A6", lw=0)); ax.text(25.2, 7, "anchor bead", va="center", fontsize=6)
    ax.add_patch(Circle((40, 7), 1.0, color="#222222", lw=0)); ax.text(41.6, 7, "hub core", va="center", fontsize=6)
    ax.text(53, 7, "tube = petal loop", va="center", fontsize=6)
    ax.text(W - 1, H - 1, f"conformation {int(Z['conf'])} of 10,000 (16 kT)\n{len(pet)} petals, {len(anc)} anchor beads\n"
            f"Cicero hub edges inside: {len(he_in)} of {len(HE)}", ha="right", va="top", fontsize=6)
    ax.text(1, 0.8, "typical flower: median hub-core (3) and petal (5) counts among the\n43.5% of conformations with ≥3 hub cores and ≥4 petals",
            fontsize=5.5, va="bottom")
    save(fig, ver)
    t = open(O / f"{ver}.svg").read(); m = next(re.finditer(r"<image\b.*?/>", t, re.S))
    new = (f'<image xlink:href="raster/{ver}.png" x="{IX * PT:.4f}" y="{(H - IY - IMH) * PT:.4f}" width="{IMW * PT:.4f}" '
           f'height="{IMH * PT:.4f}" preserveAspectRatio="none"/>')
    open(O / f"{ver}_linked.svg", "w").write(t[:m.start()] + new + t[m.end():])
    print(ver, f"render {cw}x{ch} px over {IMW} mm = {dpi:.0f} dpi")

# 2D schematic
fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_axis_off()
ax.set_aspect("equal")
ax.add_patch(Circle((0, 0), 7, color="#D55E00", lw=0))
ax.text(0, 0, f"TF core\n{n_tf} TFs", ha="center", va="center", color="white", fontsize=6)
n = len(pet); maxkb = pet.span_kb.max()
for k, p in enumerate(pet.itertuples()):
    th = np.pi / 2 + 2.3 - 4.6 * k / max(1, n - 1); L = 11 + 11 * np.sqrt(p.span_kb / maxkb); wdt = 0.32
    t = np.linspace(-1, 1, 200)
    r = 7 + L * np.sin(np.pi * (t + 1) / 2); ang = th + wdt * t * (1 + 0.4 * (1 - np.abs(t)))
    ax.plot(r * np.cos(ang), r * np.sin(ang), color=p.colour, lw=2.2, solid_capstyle="round")
    tx, ty = (7 + L + 5.5) * np.cos(th), (7 + L + 4.5) * np.sin(th); tx = min(max(tx, -30), 30)
    ax.text(tx, ty, f"{p.petal}: {p.span_kb} kb\n{p.start/1e6:.2f}–{p.end/1e6:.2f} Mb", ha="center", va="center", fontsize=6, color=p.colour)
hb = anc[anc.hub_label != ""]
for k, r in enumerate(hb.itertuples()):
    ang = -np.pi / 2 + (k - (len(hb) - 1) / 2) * 0.42; x, y = 7 * np.cos(ang), 7 * np.sin(ang)
    ax.add_patch(Circle((x, y), 1.4, color="#222222", lw=0, zorder=3))
    ax.text(0, -12.5 - 3.2 * k, r.hub_label, ha="center", va="center", fontstyle="italic", zorder=4)
ax.text(0, -12.5 - 3.2 * len(hb), "hub cores (dark anchors)", ha="center", va="center", fontsize=6, color="#555555")
ax.text(-39, -38.5, f"conformation {int(Z['conf'])}; anchors = beads within 2σ of core TFs; \npetals = loops of 20–200 kb between consecutive anchors;\n"
        f"Cicero hub edges with both ends in the flower: {len(he_in)} of {len(HE)}", fontsize=5.5, va="bottom")
save(fig, "flower_schematic")
print("hub edges inside:", len(he_in), he_in[["b1", "b2", "group"]].values.tolist())
print(f"flower frequency: >=3 hub cores in the hub cluster {(ft.n_hub >= 3).mean()*100:.1f}%; plus >=4 petals {((ft.n_hub >= 3) & (ft.n_petals >= 4)).mean()*100:.1f}%")
