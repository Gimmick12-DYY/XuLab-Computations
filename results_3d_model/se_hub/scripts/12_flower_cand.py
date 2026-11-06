"""SE hub partial flower: the same representative structure as 09_render_3d.py, drawn along the real chain. Chromatin between the SE and each partner closed
in this structure (< 2 sigma) is a red tube (petal; petals on the same side of the SE nest and share chain); the rest of the chain is
not shown. Closed partners: dominant-TF colour; open partners: faded, thin dashed spoke to the SE.
  render : python 10_partial_flower.py --sim <traj dir> --conf N    (pymol-open-source)
  figure : python 10_partial_flower.py --sim <traj dir> --conf N --figure
"""
import argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--conf", type=int, required=True); ap.add_argument("--figure", action="store_true"); ap.add_argument("--out", default="figure_3d_v2b"); ap.add_argument("--size", type=int, default=2600); a = ap.parse_args()
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / a.out; N = 910
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]
COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, RED, SEC, GHOST, FADE = "#BDBDBD", "#C0392B", "#4D4D4D", "#E4E4E4", "#D0D0D0"
EL = pd.read_csv(Bd / "model/inputs/hub_elements.tsv", sep="\t"); core = int(EL.bead[EL.is_core].iloc[0]); SP = pd.read_csv(Bd / "figure_3d_v2/partners_3d.tsv", sep="\t")
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t_, k_ = divmod(a.conf - 1, 50); X = np.load(files[t_])["X"][k_, :N].astype(float)
SP["dist_sigma"] = np.linalg.norm(X[SP.bead] - X[core], axis=1); SP["closed"] = SP.dist_sigma < 2.0; SP["cls"] = ["element" if re.search(r" [+-]\d+ kb$", l) else "promoter" for l in SP.label]; SP.to_csv(O / "petals_3d.tsv", sep="\t", index=False)
if not a.figure:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    Y = (X - X[core]) * 10.0; pet = np.zeros(N, bool)
    for b in SP.bead[SP.closed]: pet[min(b, core):max(b, core) + 1] = True
    _, _, Vt = np.linalg.svd(Y[pet] - Y[pet].mean(0), full_matrices=False)   # face the plane of the petals
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T; cmd.reinitialize()
    def path(b_a, b_b):   # real chain from bead a to bead b, Gaussian-smoothed along the chain (sd 6 kb), ends pinned to the beads
        idx = np.arange(b_a, b_b + (1 if b_b > b_a else -1), 1 if b_b > b_a else -1); Z = gaussian_filter1d(Y[idx], 3.0, axis=0, mode="nearest")
        tp = np.minimum(np.minimum(np.arange(len(idx)), np.arange(len(idx))[::-1]) / 6.0, 1.0)[:, None]; return tp * Z + (1 - tp) * Y[idx]
    cl = sorted(SP.bead[SP.closed]); pts = []; o = []
    for side in ([b for b in cl if b < core][::-1], [b for b in cl if b > core]):   # petals on one side nest: alternate reds along the shared chain
        prev = core
        for k, b in enumerate(side):
            q = path(prev, b); pts.append(q); c = rgb(RED if k % 2 == 0 else "#8E2A20")
            for u, v in zip(q[:-1], q[1:]): o += [cgo.CYLINDER, *u, *v, 1.1, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 1.1]
            prev = b
    cmd.load_cgo(o, "petals"); pts = np.vstack(pts + [Y[SP.bead.values] * 1.0, np.zeros((1, 3))])
    sp, c = [], rgb(FADE)
    for r in SP[~SP.closed].itertuples():
        P = Y[r.bead]; L = np.linalg.norm(P)
        if r.cls == "promoter": sp += [cgo.CYLINDER, 0, 0, 0, *P, 0.3, *c, *c]
        else:
            for s in np.arange(0, L, 2.4): sp += [cgo.CYLINDER, *(P * s / L), *(P * min(s + 1.0, L) / L), 0.4, *c, *c]
    cmd.load_cgo(sp, "open_spokes")
    def octa(c0, r, col):   # diamond for non-promoter elements
        V = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]; o = [cgo.COLOR, *col]
        for a_ in V[:2]:
            for b_ in V[2:4]:
                for z_ in V[4:]:
                    o += [cgo.NORMAL, *((a_ + b_ + z_) / np.sqrt(3))] + sum(([cgo.VERTEX, *(c0 + r * v)] for v in (a_, b_, z_)), [])
        return o
    oc = [cgo.BEGIN, cgo.TRIANGLES]
    an = [cgo.COLOR, *rgb(SEC), cgo.SPHERE, 0, 0, 0, 5.5]
    for r in SP.itertuples():
        dt = r.dominant_tf if isinstance(r.dominant_tf, str) else ""
        col = rgb(COL.get(dt, OTHER) if r.closed else FADE); rad = 4.0 if r.closed else 2.4
        if r.cls == "promoter": an += [cgo.COLOR, *col, cgo.SPHERE, *Y[r.bead], rad]
        else: oc += octa(Y[r.bead], rad * 1.5, col)
    cmd.load_cgo(an, "anchors"); cmd.load_cgo(oc + [cgo.END], "elements")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.004, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.5, specular=0.1, direct=0.5,
                     reflect=0.25, orthoscopic=1, antialias=2, cgo_lighting=1, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k, v)
    w = h = a.size; cmd.viewport(w // 4, h // 4); cmd.reset()
    for i, p in enumerate(pts[::5]): cmd.pseudoatom("frame", pos=[float(x) for x in p], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=2, complete=1); cmd.delete("frame")
    view = cmd.get_view(); cmd.png(str(O / "render.png"), width=w, height=h, dpi=600, ray=1); cmd.save(str(O / "se_partial_flower.pse"))
    R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); idx = np.r_[core, SP.bead.values]; vc = (Y[idx] - org) @ R_ + cam
    hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=idx, px=w / 2 * (1 + vc[:, 0] / hh), py=h / 2 * (1 - vc[:, 1] / hh))).to_csv(O / "projection.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
    im = Image.open(O / "render.png").convert("RGBA"); PJ = pd.read_csv(O / "projection.tsv", sep="\t")
    aa = np.asarray(im)[..., 3]; ys, xs = np.where(aa > 30); c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.01)
    x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); PJ["px"] -= x0; PJ["py"] -= y0; w = im.size[0]
    W, H = 330.0, 104.0; IW = 100.0; IX = (W - IW) / 2 - 45; IY = 2.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="none", zorder=0)
    PJ["x"] = IX + PJ.px / w * IW; PJ["y"] = IY + IW - PJ.py / w * IW; Q = PJ.iloc[1:].copy(); Q["label"] = SP.label.values; Q["closed"] = SP.closed.values; Q["span"] = SP.span_kb.values; mid = IX + IW / 2
    for side, sub in ((-1, Q[Q.x < mid]), (1, Q[Q.x >= mid])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 6.0)
        if yy and yy[-1] > 98: yy = [v - (yy[-1] - 98) for v in yy]
        lx = IX - 3 if side < 0 else IX + IW + 3
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx], [r.y, ty], color="#777777" if r.closed else "#CCCCCC", lw=0.35, zorder=3)
            ax.text(lx + (-0.8 if side < 0 else 0.8), ty, f"{r.label} ({r.span:.0f} kb)" if r.closed else r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", fontsize=7,
                    color="#222222" if r.closed else "#9A9A9A", fontweight="bold" if r.closed else "normal")
    ax.text(PJ.x[0], PJ.y[0], "SE\ncand.", ha="center", va="center", fontsize=5.5, linespacing=0.95, color="white", fontweight="bold", zorder=4)
    lx, ly, dy = IX + IW + 52, 92, 5.0
    ax.text(lx - 2, ly + 5, "Partner colour: dominant TF (closed loops)", fontsize=7, fontweight="bold", va="center")
    shown = [t for t in TFS if t in set(SP.dominant_tf[SP.closed].dropna())]
    for i, (c, l) in enumerate([(COL[t], t) for t in shown]):
        ax.plot(lx, ly - dy * i, "o", ms=6, color=c, mec="none"); ax.text(lx + 3, ly - dy * i, l, va="center", fontstyle="italic")
    y2 = ly - dy * (len(shown) + 0.6) - 2; ax.text(lx - 2, y2, "Structure", fontsize=7, fontweight="bold", va="center")
    ax.plot(lx, y2 - dy, "o", ms=9, color=SEC, mec="none"); ax.text(lx + 3, y2 - dy, "super-enhancer candidate core", va="center")
    ax.plot([lx - 2.5, lx + 2], [y2 - 2 * dy] * 2, color=RED, lw=2.5); ax.text(lx + 3, y2 - 2 * dy, f"petal: chromatin of a loop-edge closed here (< 2\u03c3), {int(SP.closed.sum())} of 12", va="center")
    ax.plot(lx - 1.2, y2 - 3 * dy, "o", ms=4.5, color="#777777", mec="none"); ax.plot(lx + 1.2, y2 - 3 * dy, "D", ms=4, color="#777777", mec="none"); ax.text(lx + 3, y2 - 3 * dy, "sphere = promoter (TSS within 2.5 kb); diamond = other element", va="center")
    ax.plot(lx, y2 - 4 * dy, "o", ms=4, color=FADE, mec="none"); ax.text(lx + 3, y2 - 4 * dy, "pale = partner whose loop is open in this structure", va="center")
    ax.plot([lx - 2.5, lx + 2], [y2 - 5 * dy] * 2, color=FADE, lw=0.8); ax.text(lx + 3, y2 - 5 * dy, "open loop-edge to a promoter", va="center")
    ax.plot([lx - 2.5, lx + 2], [y2 - 6 * dy] * 2, color=FADE, lw=1, ls=(0, (1, 1))); ax.text(lx + 3, y2 - 6 * dy, "open loop-edge to an element", va="center")
    ax.text(lx - 2, y2 - 7 * dy, "Petals on one side of the SE candidate nest and share chain, so the\npartner class is shown by the node, not the tube (reds change at each\npartner). Chain smoothed (about 10 kb); other chromatin is not shown.", fontsize=6, va="top", color="#444444")
    for e in ("svg", "pdf"): fig.savefig(O / f"se_partial_flower.{e}", transparent=True)
    fig.savefig(O / "se_partial_flower.png", dpi=600, facecolor="white"); fig.savefig(O / "se_partial_flower_preview.png", dpi=220, facecolor="white")
print(pd.crosstab(SP.closed, SP.cls)); print(SP[["label", "cls", "bead", "span_kb", "dist_sigma", "closed"]].round(2).to_string(index=False))
