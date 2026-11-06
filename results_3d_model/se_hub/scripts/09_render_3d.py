"""SE hub 3D view, one representative structure of the Hi-C-first ensemble (rosette style of tf_polymer_hicfirst/figure_simple): light grey spline
tube at 10 kb, SE as a large sphere, the 12 partners as spheres coloured by their dominant TF (highest MACS2 signalValue with a summit in the
partner's 2 kb bead, among the 10 v2-coloured TFs; grey otherwise), straight red SE-partner spokes (thin and dashed when Hi-C O/E < 1).
  render : python 09_render_3d.py --sim <traj dir> --peaks <peak dir> --conf N    (pymol-open-source)
  figure : python 09_render_3d.py --sim <traj dir> --peaks <peak dir> --conf N --figure
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--peaks", required=True); ap.add_argument("--conf", type=int, required=True)
ap.add_argument("--figure", action="store_true"); ap.add_argument("--size", type=int, default=2600); a = ap.parse_args()
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_3d_v2"; N = 910; L0 = 430000
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]
COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, RED, SEC, TUBE = "#BDBDBD", "#C0392B", "#4D4D4D", "#DEDEDE"
EL = pd.read_csv(Bd / "model/inputs/hub_elements.tsv", sep="\t"); SP = pd.read_csv(Bd / "figure_v2/spokes.tsv", sep="\t")
core = int(EL.bead[EL.is_core].iloc[0]); SP = SP.sort_values("bead").reset_index(drop=True); SP["num"] = np.arange(1, len(SP) + 1)
best = {b: ("", -1.0) for b in SP.bead}
for t in TFS:
    for ln in open(Path(a.peaks) / f"TF.{t}_peaks.q005.narrowPeak"):
        x = ln.split("\t")
        if x[0] != "chr19": continue
        b = (int(x[1]) + int(x[9]) - L0) // 2000
        if b in best and float(x[6]) > best[b][1]: best[b] = (t, float(x[6]))
SP["dominant_tf"] = [best[b][0] for b in SP.bead]; SP["dominant_signal"] = [best[b][1] for b in SP.bead]
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t_, k_ = divmod(a.conf - 1, 50); X = np.load(files[t_])["X"][k_, :N].astype(float)
SP["dist_sigma"] = np.linalg.norm(X[SP.bead] - X[core], axis=1); SP.to_csv(O / "partners_3d.tsv", sep="\t", index=False)
if not a.figure:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    Y = (X - X[core]) * 10.0
    _, _, Vt = np.linalg.svd(Y[SP.bead], full_matrices=False)   # face the plane in which SE and partners spread most
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T; cmd.reinitialize()
    b0, b1 = SP.bead.min() // 5, SP.bead.max() // 5 + 1   # chain only between the outermost partners, 10 kb means
    S10 = Y[: N // 5 * 5].reshape(-1, 5, 3).mean(1)[b0:b1]; tck, _ = splprep(S10.T, s=len(S10) * 0.5, k=3); pts = np.array(splev(np.linspace(0, 1, len(S10) * 10), tck)).T
    c = rgb(TUBE); tb = []
    for u, v in zip(pts[:-1], pts[1:]): tb += [cgo.CYLINDER, *u, *v, 0.95, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 0.95]
    cmd.load_cgo(tb, "chromatin")
    sp, c = [], rgb(RED)
    for r in SP.itertuples():
        P = Y[r.bead]; L = np.linalg.norm(P)
        if r.oe >= 1: sp += [cgo.CYLINDER, *[0, 0, 0], *P, 0.45 + 0.3 * min(r.oe, 4), *c, *c]
        else:
            for s in np.arange(0, L, 4.5): sp += [cgo.CYLINDER, *(P * s / L), *(P * min(s + 2.6, L) / L), 0.5, *c, *c]
    cmd.load_cgo(sp, "spokes")
    an = [cgo.COLOR, *rgb(SEC), cgo.SPHERE, 0, 0, 0, 5.5]
    for r in SP.itertuples(): an += [cgo.COLOR, *rgb(COL.get(r.dominant_tf, OTHER)), cgo.SPHERE, *Y[r.bead], 3.6]
    cmd.load_cgo(an, "anchors")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.004, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.5, specular=0.1, direct=0.5,
                     reflect=0.25, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k, v)
    w = h = a.size; cmd.viewport(w // 4, h // 4); cmd.reset()
    for i, p in enumerate(pts[::5]): cmd.pseudoatom("frame", pos=[float(x) for x in p], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=2, complete=1); cmd.delete("frame")
    view = cmd.get_view(); cmd.png(str(O / "render.png"), width=w, height=h, dpi=600, ray=1); cmd.save(str(O / "se_3d.pse"))
    R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); idx = np.r_[core, SP.bead.values]; vc = (Y[idx] - org) @ R_ + cam
    hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=idx, px=w / 2 * (1 + vc[:, 0] / hh), py=h / 2 * (1 - vc[:, 1] / hh))).to_csv(O / "projection.tsv", sep="\t", index=False)
if a.figure:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
    im = Image.open(O / "render.png").convert("RGBA"); PJ = pd.read_csv(O / "projection.tsv", sep="\t"); aa = np.asarray(im)[..., 3]; ys, xs = np.where(aa > 30); c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.01)
    x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); PJ["px"] -= x0; PJ["py"] -= y0; w = im.size[0]
    W, H = 330.0, 104.0; IW = 100.0; IX = (W - IW) / 2 - 45; IY = 2.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="none", zorder=0)
    PJ["x"] = IX + PJ.px / w * IW; PJ["y"] = IY + IW - PJ.py / w * IW; Q = PJ.iloc[1:].copy(); Q["label"] = SP.label.values; mid = IX + IW / 2
    for side, sub in ((-1, Q[Q.x < mid]), (1, Q[Q.x >= mid])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 6.0)
        if yy and yy[-1] > 98: yy = [v - (yy[-1] - 98) for v in yy]
        lx = IX - 3 if side < 0 else IX + IW + 3
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx], [r.y, ty], color="#777777", lw=0.35, zorder=3)
            ax.text(lx + (-0.8 if side < 0 else 0.8), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", fontsize=7)
    ax.text(PJ.x[0], PJ.y[0], "SE", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=4)
    lx, ly, dy = IX + IW + 52, 92, 5.0
    ax.text(lx - 2, ly + 5, "Partner colour: dominant TF", fontsize=7, fontweight="bold", va="center")
    shown = [t for t in TFS if t in set(SP.dominant_tf)]
    for i, (c, l) in enumerate([(COL[t], t) for t in shown] + [(OTHER, "none of the 10 coloured TFs")]):
        ax.plot(lx, ly - dy * i, "o", ms=6, color=c, mec="none"); ax.text(lx + 3, ly - dy * i, l, va="center", fontstyle="italic" if l in TFS else "normal")
    y2 = ly - dy * (len(shown) + 1) - 2; ax.text(lx - 2, y2, "Structure", fontsize=7, fontweight="bold", va="center")
    ax.plot(lx, y2 - dy, "o", ms=9, color=SEC, mec="none"); ax.text(lx + 3, y2 - dy, "super-enhancer core", va="center")
    ax.plot([lx - 2.5, lx + 2], [y2 - 2 * dy] * 2, color=RED, lw=2); ax.text(lx + 3, y2 - 2 * dy, "SE\u2013partner loop-edge, width ~ Hi-C O/E", va="center")
    ax.plot([lx - 2.5, lx + 2], [y2 - 3 * dy] * 2, color=RED, lw=0.8, ls=(0, (2, 1.5))); ax.text(lx + 3, y2 - 3 * dy, "loop-edge with Hi-C O/E < 1", va="center")
    ax.plot([lx - 2.5, lx + 2], [y2 - 4 * dy] * 2, color=TUBE, lw=2.5); ax.text(lx + 3, y2 - 4 * dy, "chromatin between outermost partners (10 kb)", va="center")
    for e in ("svg", "pdf"): fig.savefig(O / f"se_3d_star.{e}", transparent=True)
    fig.savefig(O / "se_3d_star.png", dpi=600, facecolor="white"); fig.savefig(O / "se_3d_star_preview.png", dpi=220, facecolor="white")
print(SP[["num", "label", "oe", "dominant_tf", "dominant_signal", "dist_sigma"]].round(2).to_string(index=False))
