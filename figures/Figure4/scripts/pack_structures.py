"""Figure 4 PDF pack, AF3 structures: per pair (a) PLAIN whole complex and (b) ZOOMED interface close-up, as Illustrator-editable PDFs
(pdf.fonttype 42, Arial; labels are live vector text over the embedded ray-traced PyMOL PNG; image >= 600 dpi at placed size, asserted).
Same models, renders and colours as Fig 4d (d_render_af3.py -> work_d/*_full.png, *_zoom.png, panels.json; chain A #2166AC, chain B #B83A2E):
RBBP4-ZNF644 seed 1 sample 3, ADNP-MTF2 seed 4 sample 0, DR1-NFYA seed 1 sample 1 (top-ranked sample of the strong seed with the highest ipSAE).
Numbers: strong seeds of 5 and medians over the 5 seeds (per seed: ipTM of the top-ranked sample, best ipSAE), as rescore_5seed/tiers_5seed.tsv.
Zoomed: contacting side chains (<= 5 A, pLDDT >= 70) as sticks; the 2 strongest contacts (salt bridges first, then shortest) dashed with
heavy-atom distances; residue labels pushed apart with thin leaders (label layout as d_compose.py).
usage: pack_structures.py render_dir palette_dir out_dir
"""
import sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

rd, paldir, out = sys.argv[1:4]
sys.path.insert(0, paldir)
import regulome_palette as rp
P = rp.use(); plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
CA_, CB_ = "#2166AC", "#B83A2E"
J = json.load(open(f"{rd}/panels.json"))
PAIRS = [("RBBP4_ZNF644", "RBBP4", "ZNF644", 4, 0.53, 0.77, "c1", "One-sided interface: median 12 confident interface residues on ZNF644 (109 on RBBP4)."),
         ("ADNP_MTF2", "ADNP", "MTF2", 4, 0.46, 0.48, "c2", "ADNP side mostly in annotated DNA-binding domains (79% of its interface residues); both key contacts in the homeobox."),
         ("DR1_NFYA", "DR1", "NFYA", 4, 0.45, 0.52, "c3", "DNA-facing test not applicable: DR1 has no annotated DNA-binding domain.")]
mm = 1 / 25.4; MIN_DPI = 600; rows = []

def overlay(fig, W, H):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_zorder(10); ax.patch.set_alpha(0); return ax

for p, a, b, ns, ips, ipt, tag, note in PAIRS:
    m = J[p]; stats = f"AF3 · seed {m['seed']} sample {m['sample']} · {ns}/5 strong seeds · median ipSAE {ips:.2f} · ipTM {ipt:.2f}"
    # (a) plain
    im = Image.open(f"{rd}/{m['full']['img']}").convert("RGB"); bb = Image.eval(im, lambda v: 255 - v).getbbox(); im = im.crop(bb); w, h = im.size
    W, H = 120, 100; BW, BH = 84, 80; s = min(BW / w, BH / h); iw, ih = w * s, h * s; fx = (W - iw) / 2; ftop = H - 9; fy = ftop - ih
    dpi = w / (iw * mm); assert dpi >= MIN_DPI, (p, dpi)
    fig = plt.figure(figsize=(W * mm, H * mm)); ax = fig.add_axes([fx / W, fy / H, iw / W, ih / H]); ax.imshow(im, interpolation="none"); ax.axis("off"); fax = overlay(fig, W, H)
    tomm = lambda px: (fx + (px[0] - bb[0]) * s, ftop - (px[1] - bb[1]) * s)
    fax.text(W / 2, H - 2.5, f"{a}–{b}", ha="center", va="top", fontsize=8, fontweight="bold")
    for nm, key, col, side in ((a, "anchorA", CA_, -1), (b, "anchorB", CB_, 1)):
        q = tomm(m["full"][key]); yq = min(max(q[1], fy + 3), ftop - 3)
        fax.text(fx - 2 if side < 0 else fx + iw + 2, yq, nm, ha="right" if side < 0 else "left", va="center", fontsize=7, color=col, fontweight="bold")
    fax.text(W / 2, fy - 3, stats, ha="center", va="top", fontsize=6)
    fig.savefig(f"{out}/{tag}a_{a}_{b}_AF3_plain.pdf"); fig.savefig(f"{out}/{tag}a_{a}_{b}_AF3_plain.png", dpi=150); plt.close(fig)
    rows.append(dict(file=f"{tag}a_{a}_{b}_AF3_plain.pdf", render=m["full"]["img"], placed_mm=f"{iw:.1f} x {ih:.1f}", image_dpi=round(dpi)))
    # (b) zoomed
    zim = Image.open(f"{rd}/{m['zoom']['img']}").convert("RGB"); zw = zim.size[0]
    W, H, ZS = 120, 122, 96; zx, ztop = (W - ZS) / 2, H - 9; zy = ztop - ZS; zs = ZS / zw; dpi = zw / (ZS * mm); assert dpi >= MIN_DPI, (p, dpi)
    fig = plt.figure(figsize=(W * mm, H * mm)); zax = fig.add_axes([zx / W, zy / H, ZS / W, ZS / H]); zax.imshow(zim, interpolation="none"); zax.set_xticks([]); zax.set_yticks([])
    for sp in zax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color(P.neutral.ink)
    fax = overlay(fig, W, H); ztomm = lambda px: (zx + px[0] * zs, ztop - px[1] * zs)
    fax.text(W / 2, H - 2.5, f"{a}–{b} interface", ha="center", va="top", fontsize=8, fontweight="bold")
    LAB = m["zoom"]["labels"][:2]; items, done, obst = [], set(), []
    for L in LAB:
        (x1, y1), (x2, y2) = ztomm(L["pA"]), ztomm(L["pB"])
        fax.plot([x1, x2], [y1, y2], color=P.neutral.ink, lw=1.2, ls=(0, (2.4, 1.5)), solid_capstyle="butt", zorder=6.5)
        for t_ in np.linspace(0, 1, 7): obst.append(np.array([x1 + t_ * (x2 - x1), y1 + t_ * (y2 - y1)]))     # dash points: labels keep off them
        d_ = np.array([x2 - x1, y2 - y1]); nrm = np.array([-d_[1], d_[0]]) / (np.linalg.norm(d_) + 1e-9)
        items.append(dict(t=f"{L['dist']:.1f} Å", a=ztomm(L["mid"]), col=P.neutral.ink, bold=False, dist=True, nrm=nrm, fs=7))
        for key, res, nm, col in (("labA", L["resA"], a, CA_), ("labB", L["resB"], b, CB_)):
            if (nm, res) not in done: done.add((nm, res)); items.append(dict(t=f"{nm} {res}", a=ztomm(L[key]), col=col, bold=True, dist=False, fs=7))
    zc = np.array([zx + ZS / 2, zy + ZS / 2])
    for it in items:
        v = np.array(it["a"]) - zc; v = v / (np.linalg.norm(v) + 1e-9)
        it["p"] = np.array(it["a"]) + ((it["nrm"] * np.sign(it["nrm"] @ v + 1e-9)) * 4.0 if it["dist"] else v * 4.5)
        it["w"], it["h"] = 0.62 * it["fs"] * 0.3528 * len(it["t"]) + 0.8, it["fs"] * 0.3528 * 1.3
    for _ in range(400):
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                A_, B_ = items[i], items[j]; d = B_["p"] - A_["p"]
                ox, oy = (A_["w"] + B_["w"]) / 2 - abs(d[0]), (A_["h"] + B_["h"]) / 2 - abs(d[1])
                if ox > 0 and oy > 0:
                    sh = np.array([0.0, (oy / 2 + 0.1) * (1 if d[1] >= 0 else -1)]); A_["p"] = A_["p"] - sh; B_["p"] = B_["p"] + sh; moved = True
        for it in items:                                          # push labels off the dashed lines (fixed 1.2 mm points)
            for q in obst:
                d = it["p"] - q; ox, oy = it["w"] / 2 + 0.6 - abs(d[0]), it["h"] / 2 + 0.6 - abs(d[1])
                if ox > 0 and oy > 0:
                    if oy <= ox: it["p"][1] += (oy + 0.1) * (1 if d[1] >= 0 else -1)
                    else: it["p"][0] += (ox + 0.1) * (1 if d[0] >= 0 else -1)
                    moved = True
        for it in items:
            it["p"][0] = min(max(it["p"][0], zx + it["w"] / 2 + 1), zx + ZS - it["w"] / 2 - 1); it["p"][1] = min(max(it["p"][1], zy + it["h"] / 2 + 1), ztop - it["h"] / 2 - 1)
        if not moved: break
    for it in items:
        if np.linalg.norm(it["p"] - np.array(it["a"])) > 2.5:
            fax.plot([it["a"][0], it["p"][0]], [it["a"][1], it["p"][1]], color=P.neutral.dark, lw=0.4, zorder=6)
        fax.text(it["p"][0], it["p"][1], it["t"], ha="center", va="center", fontsize=it["fs"], color=it["col"], fontweight="bold" if it["bold"] else "normal", zorder=7,
                 bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75))
    fax.text(W / 2, zy - 2.5, stats, ha="center", va="top", fontsize=6)
    fax.text(W / 2, zy - 6.2, note, ha="center", va="top", fontsize=6)
    fax.text(W / 2, zy - 9.9, "Sticks: contacting side chains (≤ 5 Å, pLDDT ≥ 70). Dashed: the two strongest contacts, heavy-atom distance.", ha="center", va="top", fontsize=5.5, color=P.neutral.dark)
    fig.savefig(f"{out}/{tag}b_{a}_{b}_AF3_interface_zoom.pdf"); fig.savefig(f"{out}/{tag}b_{a}_{b}_AF3_interface_zoom.png", dpi=150); plt.close(fig)
    rows.append(dict(file=f"{tag}b_{a}_{b}_AF3_interface_zoom.pdf", render=m["zoom"]["img"], placed_mm=f"{ZS} x {ZS}", image_dpi=round(dpi),
                     contacts="; ".join(f"{a} {L['resA']}–{b} {L['resB']} {L['kind']} {L['dist']:.2f} Å" for L in LAB)))
    rows[-2]["model"] = rows[-1]["model"] = m["model"]
import pandas as pd
R = pd.DataFrame(rows); R.to_csv(f"{out}/structures_check.tsv", sep="\t", index=False); print(R.to_string(index=False))
