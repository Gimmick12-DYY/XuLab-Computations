"""v2 step 11 (Mac): Illustrator panels tying the chr16p11.2 browser hub to its 3D flower (Arial 7 pt, live SVG text).

  hub_flow(_ghost).svg/.pdf   render + browser gene labels (italic, leader lines) + legend, and underneath a linear
                              mini-track on the browser x-axis (29.73-31.55 Mb): viridis position strip, petals as bars in
                              their position colour, flower anchors (grey ticks), hub elements (dark ticks), and the 24
                              browser shared edges as arcs (in the flower: browser colours; not in the flower: light grey)
  hub_flow_schematic.svg/.pdf core + petals (span, coordinates) + hub genes in the flower
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_hub_flow"; R = O / "raster"; R.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600})
MM = 1 / 25.4; PT = 72 / 25.4; VIR = plt.get_cmap("viridis"); LOOPC, EDGE = "#C0392B", "#4A6FA5"
Z = np.load(O / "hub_flow_pick.npz")
EL = pd.read_csv(O / "hub_elements.tsv", sep="\t"); HE = pd.read_csv(O / "hub_edges_beads.tsv", sep="\t")
pet = pd.read_csv(O / "hub_flow_petals.tsv", sep="\t"); pj = pd.read_csv(O / "hub_flow_projection.tsv", sep="\t")
anc = pd.read_csv(O / "hub_flow_anchors.tsv", sep="\t"); aset = set(anc.bead)
L0, L1, N = 29.73, 31.55, 910
mb = lambda b: L0 + (np.asarray(b) + 0.5) * 0.002
inside = HE.b1.isin(aset) & HE.b2.isin(aset)
n_tf = int((Z["LT"] == int(Z["hc"])).sum())
hubs = pj[pj.kind == "hub"]
W, IMW = 100.0, 68.0
def save(fig, stem):
    for e in ("svg", "pdf"): fig.savefig(O / f"{stem}.{e}", transparent=True)
    fig.savefig(O / f"{stem}_preview.png", dpi=250, facecolor="white"); plt.close(fig)

box = None
for ver in ("render_hub_flow", "render_hub_flow_ghost"):
    im = np.asarray(Image.open(O / f"{ver}.png").convert("RGBA")); h0, w0 = im.shape[:2]
    if box is None:
        ys, xs = np.where(im[..., 3] > 5); c = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2)
        half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * 1.04)
        box = (max(0, c[0] - half), max(0, c[1] - half), min(w0, c[0] + half), min(h0, c[1] + half))
    cr = im[box[1]:box[3], box[0]:box[2]]; ch, cw = cr.shape[:2]
    IMH = IMW * ch / cw; TRK = 30.0; H = TRK + IMH + 8; IX = 15.0; IY = TRK + 2
    dpi = cw / (IMW * MM); Image.fromarray(cr, "RGBA").save(R / f"{ver}.png", dpi=(round(dpi), round(dpi)))
    tox = lambda px: IX + (px - box[0]) / cw * IMW; toy = lambda py: IY + IMH - (py - box[1]) / ch * IMH
    fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.imshow(cr, extent=(IX, IX + IMW, IY, IY + IMH), interpolation="none", zorder=0)
    # labels: two columns (left / right of the render), de-overlapped
    hs = hubs.assign(x=tox(hubs.px), y=toy(hubs.py)); cx = hs.x.median()
    for side, sub in ((-1, hs[hs.x < cx]), (1, hs[hs.x >= cx])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for k in range(1, len(yy)): yy[k] = max(yy[k], yy[k - 1] + 2.9)
        if yy and yy[-1] > H - 3: yy = [v - (yy[-1] - (H - 3)) for v in yy]
        tx = IX + 2 if side < 0 else IX + IMW - 2
        for r, ty in zip(sub.itertuples(), yy):
            lx = IX - 0.5 if side < 0 else IX + IMW + 0.5
            ax.plot([r.x, lx], [r.y, ty], color="#333333", lw=0.35, zorder=3)
            ax.text(lx - 0.4 if side < 0 else lx + 0.4, ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", fontsize=6.5, zorder=4)
    ax.text(W - 1, H - 1, f"conformation {int(Z['conf'])} of 10,000 (16 kT)\n{len(hubs)}/23 hub elements, {int(inside.sum())}/24 shared edges in one flower\n"
            f"core {n_tf} TFs; {len(pet)} petals", ha="right", va="top", fontsize=6)
    # mini-track
    t = fig.add_axes([IX / W, 4 / H, IMW / W, (TRK - 9) / H]); t.set_xlim(L0, L1); t.set_ylim(-1.6, 3.2); t.set_axis_off()
    for k in range(200): t.add_patch(Rectangle((L0 + k * (L1 - L0) / 200, -1.5), (L1 - L0) / 200, 0.35, color=VIR(k / 199), lw=0))
    for r in pet.itertuples():
        x0_, x1_ = mb(r.bead1), mb(r.bead2); mc = VIR(((r.bead1 + r.bead2) / 2) / (N - 1))
        t.add_patch(Rectangle((x0_, -0.9), x1_ - x0_, 0.45, color=mc, lw=0))
        t.text((x0_ + x1_) / 2, -1.0, r.petal, ha="center", va="top", fontsize=5.5)
    t.vlines(mb(anc.bead), -0.25, 0.05, color="#AAAAAA", lw=0.3)
    t.vlines(mb(EL.bead), -0.3, 0.25, color="#BBBBBB", lw=0.6)
    t.vlines(mb(EL.bead[EL.bead.isin(aset)]), -0.3, 0.25, color="#222222", lw=0.9)
    for r, ins in zip(HE.itertuples(), inside):
        x1_, x2_ = sorted((mb(r.b1), mb(r.b2))); xs = np.linspace(x1_, x2_, 60); hgt = 0.35 + 2.6 * (x2_ - x1_) / (L1 - L0)
        col = (LOOPC if r._5 == 1 else EDGE) if ins else "#D0D0D0"
        t.plot(xs, 0.3 + hgt * np.sin(np.pi * (xs - x1_) / max(x2_ - x1_, 1e-9)), color=col, lw=0.8 if ins else 0.5, zorder=3 if ins else 1)
    fig.text(IX / W, 2.2 / H, "29.73", fontsize=6, va="center"); fig.text((IX + IMW) / W, 2.2 / H, "31.55 Mb", fontsize=6, va="center", ha="right")
    fig.text((IX + IMW / 2) / W, 2.2 / H, "chr16 (same axis as the browser view)", fontsize=6, va="center", ha="center")
    lx = 1.5; ly = TRK - 1.5
    ax.add_patch(Circle((lx + 1, ly), 0.9, color="#D55E00", lw=0)); ax.text(lx + 2.5, ly, "TF core", va="center", fontsize=6)
    ax.add_patch(Circle((lx + 15, ly), 0.9, color="#222222", lw=0)); ax.text(lx + 16.5, ly, "hub element (browser)", va="center", fontsize=6)
    ax.plot([lx + 40, lx + 44], [ly] * 2, color=LOOPC, lw=1); ax.plot([lx + 45, lx + 49], [ly] * 2, color=EDGE, lw=1)
    ax.text(lx + 50, ly, "shared edge, on / not on 0.7 loop", va="center", fontsize=6)
    ax.plot([lx + 40, lx + 49], [ly - 3.2] * 2, color="#D0D0D0", lw=1); ax.text(lx + 50, ly - 3.2, "shared edge not in this flower", va="center", fontsize=6)
    save(fig, ver.replace("render_", ""))
    t_ = open(O / f"{ver.replace('render_', '')}.svg").read(); m = next(re.finditer(r"<image\b.*?/>", t_, re.S))
    new = (f'<image xlink:href="raster/{ver}.png" x="{IX * PT:.4f}" y="{(H - IY - IMH) * PT:.4f}" width="{IMW * PT:.4f}" '
           f'height="{IMH * PT:.4f}" preserveAspectRatio="none"/>')
    open(O / f"{ver.replace('render_', '')}_linked.svg", "w").write(t_[:m.start()] + new + t_[m.end():])
    print(ver, f"{W} x {H:.1f} mm; render {cw}x{ch} px over {IMW} mm = {dpi:.0f} dpi")

# schematic
Ws = 90.0
fig = plt.figure(figsize=(Ws * MM, Ws * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_axis_off(); ax.set_aspect("equal")
ax.add_patch(Circle((0, 0), 8, color="#D55E00", lw=0)); ax.text(0, 0, f"TF core\n{n_tf} TFs", ha="center", va="center", color="white", fontsize=6.5)
n = len(pet); maxkb = pet.span_kb.max()
for k, p in enumerate(pet.itertuples()):
    th = np.pi / 2 + 1.95 - 3.9 * k / max(1, n - 1); Lp = 10 + 11 * np.sqrt(p.span_kb / maxkb); tt = np.linspace(-1, 1, 200)
    rr = 8 + Lp * np.sin(np.pi * (tt + 1) / 2); ang = th + 0.27 * tt * (1 + 0.4 * (1 - np.abs(tt)))
    ax.plot(rr * np.cos(ang), rr * np.sin(ang), color=VIR(((p.bead1 + p.bead2) / 2) / (N - 1)), lw=2.4, solid_capstyle="round")
    tx, ty = (8 + Lp + 6) * np.cos(th), (8 + Lp + 4.5) * np.sin(th)
    ax.text(min(max(tx, -36), 36), ty, f"{p.petal}: {p.span_kb} kb\n{p.start/1e6:.2f}–{p.end/1e6:.2f} Mb", ha="center", va="center", fontsize=6)
gl = EL[EL.bead.isin(aset)].sort_values("bead")
for k, r in enumerate(gl.itertuples()):
    ang = -np.pi / 2 + (k - (len(gl) - 1) / 2) * (1.1 / max(1, len(gl) - 1)); ax.add_patch(Circle((7.2 * np.cos(ang), 7.2 * np.sin(ang)), 1.0, color="#222222", lw=0, zorder=3))
cols = 3; per = int(np.ceil(len(gl) / cols))
for k, r in enumerate(gl.itertuples()):
    ax.text(-27 + 27 * (k // per), -20 - 2.8 * (k % per), r.label, ha="center", va="center", fontstyle="italic", fontsize=6.5)
ax.text(0, -20 - 2.8 * per - 1, "hub elements in the core (browser gene names)", ha="center", va="center", fontsize=6, color="#555555")
ax.text(-44, -44, f"conformation {int(Z['conf'])}; anchors = beads within 2σ of core TFs; petals = 20–200 kb loops between\n"
        f"consecutive anchors; colour = genomic position (as in the mini-track).\nSelected example: {Z['level_frac']*100:.2f}% of conformations reach this level.",
        fontsize=5.5, va="bottom")
save(fig, "hub_flow_schematic")
print("inside edges", int(inside.sum()), "hub elements", gl.label.tolist())
