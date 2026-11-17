"""Fig 4d step 2 (Mac): assemble the three PyMOL renders (d_render_af3.py) into one panel with chain keys and AF3 numbers.
Numbers: strong seeds of 5 and medians over the 5 seeds (per seed: ipTM of the top-ranked sample, best ipSAE), from scores_s5 per-seed
strings (as rescore_5seed/tiers_5seed.tsv); confident interface residues per side = median over the 5 seeds (heavy atom within 5 A,
pLDDT >= 70; interface_audit logic). Palette C, Arial 7 pt, editable SVG text, 183 mm wide.
usage: d_compose.py render_dir palette_dir out_prefix
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

rd, paldir, out = sys.argv[1:4]
sys.path.insert(0, paldir)
import regulome_palette as rp
P = rp.use(); plt.rcParams.update({"savefig.dpi": 300})
CA, CB = "#7A9CCB", "#F4A582"          # chain A / chain B cartoon colours (same hex as d_render_af3.py)
PAIRS = [  # pair, chain A, chain B, strong seeds, median ipSAE, median ipTM, median confident interface residues A, B, extra line
    ("EPAS1_HES1", "EPAS1", "HES1", 5, 0.56, 0.41, 24, 39, "partner retained with DNA (AF3 + DNA test)"),
    ("ADNP_MTF2", "ADNP", "MTF2", 4, 0.46, 0.48, 31, 59, ""),
    ("DR1_NFYA", "DR1", "NFYA", 4, 0.45, 0.52, 13, 9, ""),
]
mm = 1 / 25.4; W, H = 183, 76
fig = plt.figure(figsize=(W * mm, H * mm))
cw, top, imh = W / 3, 66.5, 45
for n, (p, a, b, ns, ips, ipt, ra, rb, extra) in enumerate(PAIRS):
    im = Image.open(f"{rd}/{p}.png").convert("RGB"); bb = Image.eval(im, lambda v: 255 - v).getbbox(); im = im.crop(bb)
    w, h = im.size; s = min((cw - 4) / w, imh / h); iw, ih = w * s, h * s
    x0 = n * cw + (cw - iw) / 2; y0 = 18 + (imh - ih) / 2
    ax = fig.add_axes([x0 / W, y0 / H, iw / W, ih / H]); ax.imshow(im); ax.axis("off")
    cx = (n * cw + cw / 2) / W
    fig.text(cx, (top + 2.6) / H, f"{a}–{b}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    # chain key
    for k, (name, col) in enumerate(((a, CA), (b, CB))):
        kx = n * cw + cw / 2 - 13 + k * 15
        fig.add_artist(Rectangle((kx / W, (top - 1.6) / H), 2.2 / W, 2.2 / H, color=col, transform=fig.transFigure, lw=0))
        fig.text((kx + 3) / W, (top - 0.5) / H, name, ha="left", va="center", fontsize=6)
    lines = [f"{ns}/5 strong seeds", f"median ipSAE {ips:.2f} · ipTM {ipt:.2f}", f"confident interface: {a} {ra} · {b} {rb} residues"]
    if extra: lines.append(extra)
    fig.text(cx, 16 / H, "\n".join(lines), ha="center", va="top", fontsize=6, linespacing=1.35)
fig.text(0.5, 0.2 / H, "AF3 model: top-ranked sample of the strong seed with the highest ipSAE; sticks, interface residues (≤5 Å, pLDDT ≥ 70); residues with pLDDT < 50 hidden",
         ha="center", va="bottom", fontsize=5.5, color=P.neutral.dark)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
