"""Fig 4d step 2 (Mac): CAP-SELEX-style layout (Xie et al. 2025 Fig. 4d). Per pair: full complex (d_render_af3.py *_full.png) with the
contact region marked by a translucent grey circle, two thin black lines to a square close-up inset below (*_zoom.png) carrying dashed key
contacts with distances (A) and sparse residue labels; chain names beside the full view; AF3 numbers underneath.
Label positions come from panels.json (3D anchors projected to pixels by d_render_af3.py). Numbers: strong seeds of 5 and medians over the
5 seeds (per seed: ipTM of the top-ranked sample, best ipSAE), as rescore_5seed/tiers_5seed.tsv.
Palette C, Arial 7 pt (5-6 pt labels), editable SVG text, 183 mm. usage: d_compose.py render_dir palette_dir out_prefix
"""
import sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

rd, paldir, out = sys.argv[1:4]
sys.path.insert(0, paldir)
import regulome_palette as rp
P = rp.use(); plt.rcParams.update({"savefig.dpi": 300})
CA_, CB_ = "#2166AC", "#6E6E6E"
J = json.load(open(f"{rd}/panels.json"))
PAIRS = [("EPAS1_HES1", "EPAS1", "HES1", 5, 0.56, 0.41, "partner retained with DNA (AF3 + DNA test)"),
         ("ADNP_MTF2", "ADNP", "MTF2", 4, 0.46, 0.48, ""), ("DR1_NFYA", "DR1", "NFYA", 4, 0.45, 0.52, "")]
mm = 1 / 25.4; W, H = 183, 88
fig = plt.figure(figsize=(W * mm, H * mm))
fax = fig.add_axes([0, 0, 1, 1]); fax.set_xlim(0, W); fax.set_ylim(0, H); fax.axis("off"); fax.set_zorder(10); fax.patch.set_alpha(0)    # mm overlay above images
cw = W / 3; FB = (36, 34); ZS = 31; ZY = []
for n, (p, a, b, ns, ips, ipt, extra) in enumerate(PAIRS):
    m = J[p]; x0 = n * cw; cx = x0 + cw / 2
    # full view: trim white, fit into the box, keep the pixel -> mm mapping
    im = Image.open(f"{rd}/{m['full']['img']}").convert("RGB"); bb = Image.eval(im, lambda v: 255 - v).getbbox(); im = im.crop(bb)
    w, h = im.size; s = min((cw - 8) / w, FB[1] / h); iw, ih = w * s, h * s; fx = cx - iw / 2; ftop = H - 8; fy = ftop - ih
    ax = fig.add_axes([fx / W, fy / H, iw / W, ih / H]); ax.imshow(im); ax.axis("off")
    tomm = lambda px: (fx + (px[0] - bb[0]) * s, ftop - (px[1] - bb[1]) * s)
    c = tomm(m["full"]["circle"]); r = max(3.2, min(7.5, m["full"]["radius"] * s * 1.15))
    fax.add_patch(Circle(c, r, color=P.neutral.mid, alpha=0.28, lw=0, zorder=5))
    fax.add_patch(Circle(c, r, fill=False, ec=P.neutral.dark, lw=0.4, zorder=5))
    # chain names beside the full view, at the height of each chain's centre, outside the structure
    for nm, key, col, side in ((a, "anchorA", CA_, -1), (b, "anchorB", CB_, 1)):
        q = tomm(m["full"][key]); yq = min(max(q[1], fy + 2), ftop - 2)
        fax.text(x0 + 1.5 if side < 0 else x0 + cw - 1.5, yq, nm, ha="left" if side < 0 else "right", va="center", fontsize=6.5, color=col, fontweight="bold", zorder=6)
    fax.text(cx, H - 4.5, f"{a}–{b}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    # inset square below, lines from the circle to the inset's top corners
    zx, ztop = cx - ZS / 2, fy - 3.5; zy = ztop - ZS
    zim = Image.open(f"{rd}/{m['zoom']['img']}").convert("RGB"); zw = zim.size[0]; zs = ZS / zw
    zax = fig.add_axes([zx / W, zy / H, ZS / W, ZS / H]); zax.imshow(zim); zax.set_xticks([]); zax.set_yticks([])
    for sp in zax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color(P.neutral.ink)
    ang = np.arctan2(ztop - c[1], 0)
    fax.plot([c[0] - r, zx], [c[1], ztop], color=P.neutral.ink, lw=0.5, zorder=4)
    fax.plot([c[0] + r, zx + ZS], [c[1], ztop], color=P.neutral.ink, lw=0.5, zorder=4)
    ztomm = lambda px: (zx + px[0] * zs, ztop - px[1] * zs)
    # labels: distances at the dash midpoints, residues near their CA, pushed apart so none overlap (thin leaders if moved)
    items, done = [], set()
    for L in m["zoom"]["labels"]:
        items.append(dict(t=f"{L['dist']:.1f} Å", a=ztomm(L["mid"]), col=P.neutral.ink, bold=False, dist=True))
        for key, res, nm, col in (("labA", L["resA"], a, CA_), ("labB", L["resB"], b, CB_)):
            if (nm, res) not in done: done.add((nm, res)); items.append(dict(t=f"{nm} {res}", a=ztomm(L[key]), col=col, bold=True, dist=False))
    zc = np.array([zx + ZS / 2, zy + ZS / 2])
    for it in items:
        v = np.array(it["a"]) - zc; v = v / (np.linalg.norm(v) + 1e-9); it["p"] = np.array(it["a"]) + (np.array([1.9, 1.2]) if it["dist"] else v * 2.0)   # distances beside, not on, the dash
        it["w"], it["h"] = 0.62 * 5.5 * 0.3528 * len(it["t"]) + 0.4, 5.5 * 0.3528 * 1.25
    for _ in range(300):
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                A_, B_ = items[i], items[j]; d = B_["p"] - A_["p"]
                ox, oy = (A_["w"] + B_["w"]) / 2 - abs(d[0]), (A_["h"] + B_["h"]) / 2 - abs(d[1])
                if ox > 0 and oy > 0:
                    sh = np.array([0.0, (oy / 2 + 0.05) * (1 if d[1] >= 0 else -1)]); A_["p"] = A_["p"] - sh; B_["p"] = B_["p"] + sh; moved = True
        for it in items:
            it["p"][0] = min(max(it["p"][0], zx + it["w"] / 2 + 0.5), zx + ZS - it["w"] / 2 - 0.5); it["p"][1] = min(max(it["p"][1], zy + it["h"] / 2 + 0.4), ztop - it["h"] / 2 - 0.4)
        if not moved: break
    for it in items:
        if not it["dist"] and np.linalg.norm(it["p"] - np.array(it["a"])) > 1.6:
            fax.plot([it["a"][0], it["p"][0]], [it["a"][1], it["p"][1]], color=P.neutral.mid, lw=0.3, zorder=6)
        fax.text(it["p"][0], it["p"][1], it["t"], ha="center", va="center", fontsize=5.5, color=it["col"], fontweight="bold" if it["bold"] else "normal", zorder=7,
                 bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none", alpha=0.65))
    lines = [f"{ns}/5 strong seeds · median ipSAE {ips:.2f} · ipTM {ipt:.2f}"] + ([extra] if extra else [])
    ZY.append(zy); fax.text(cx, zy - 1.8, "\n".join(lines), ha="center", va="top", fontsize=6, linespacing=1.35)
fax.text(W / 2, min(ZY) - 9.5, "AF3 model: top-ranked sample of the strong seed with the highest ipSAE; residues with pLDDT < 50 hidden. Insets: contacting side chains "
         "(≤ 5 Å, pLDDT ≥ 70); dashed lines, key contacts with heavy-atom distances.", ha="center", va="bottom", fontsize=5.5, color=P.neutral.dark)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
