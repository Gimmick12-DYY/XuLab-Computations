"""Simple rosette figure: one representative Hi-C-constrained structure of the chr16p11.2 promoter hub (Hi-C-first arm,
no TF attraction). Shows only the hub region: the browser hub elements (promoters) that are linked within 2 sigma in this
conformation, and the chain between consecutive linked elements when <= 200 kb apart (the loops). Encodes three things:
grey chromatin tube (one soft colour per loop), dark promoter spheres (gene labels), red chords = shared edges on a 0.7 loop.
Run with a Python that has pymol-open-source (render) and then again with --figure (matplotlib overlay).
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/tf_polymer_hicfirst/hicfirst")
ap.add_argument("--sim", required=True)
ap.add_argument("--conf", type=int, default=3338)
ap.add_argument("--figure", action="store_true")
ap.add_argument("--size", type=int, default=2600)
a = ap.parse_args()
Bd = Path(a.base); O = Bd.parent / "figure_simple"; (O / "raster").mkdir(parents=True, exist_ok=True)
N = 910; S = 10.0
EL = pd.read_csv(Bd / "figure_hub_flow/hub_elements.tsv", sep="\t"); HE = pd.read_csv(Bd / "figure_hub_flow/hub_edges_beads.tsv", sep="\t")
RED = HE[HE["on_loop_0.7"] == 1]
PAL = ["#8FB3D9", "#A8D5A2", "#F2C48D", "#C9A9DC", "#F4A7A7", "#9ED9D3", "#D9C58F", "#B8B8E0", "#E3B5CF", "#B5D9A8"]

def load():
    files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t, k = divmod(a.conf - 1, 50)
    return np.load(files[t])["X"][k, :N].astype(float)

if not a.figure:
    P = load(); hb = EL.bead.values
    pr = cKDTree(P[hb]).query_pairs(2.0, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(hb), len(hb))), directed=False)
    mem = np.sort(hb[lab == np.bincount(lab).argmax()])
    loops = [(int(p), int(q)) for p, q in zip(mem[:-1], mem[1:]) if q - p <= 100]
    show = np.zeros(N, bool)
    for p, q in loops: show[p:q + 1] = True
    show[mem] = True
    cen = P[mem].mean(0); Y = (P - cen) * S
    _, _, Vt = np.linalg.svd(Y[show] - Y[show].mean(0), full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    cmd.reinitialize()
    for k, (p, q) in enumerate(loops):
        seg = np.arange(p, q + 1); pts = Y[seg]
        if len(seg) >= 4:
            tck, _ = splprep(pts.T, s=len(seg) * 0.08, k=3); pts = np.array(splev(np.linspace(0, 1, len(seg) * 8), tck)).T
        c = rgb(PAL[k % len(PAL)]) if q - p >= 5 else rgb("#B0B0B0")
        o = []
        for u, v in zip(pts[:-1], pts[1:]): o += [cgo.CYLINDER, *u, *v, 2.0, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 2.0]
        cmd.load_cgo(o, f"loop{k}")
    ch = []
    for r in RED.itertuples():
        if r.b1 in set(mem) and r.b2 in set(mem):
            c = rgb("#C0392B"); ch += [cgo.CYLINDER, *Y[r.b1], *Y[r.b2], 0.9, *c, *c]
    if ch: cmd.load_cgo(ch, "red_edges")
    sp = [cgo.COLOR, *rgb("#222222")]
    for b in mem: sp += [cgo.SPHERE, *Y[b], 5.5]
    cmd.load_cgo(sp, "promoters")
    for k_, v in dict(ray_trace_mode=1, ray_trace_gain=0.006, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15,
                      ambient=0.45, specular=0.1, direct=0.55, reflect=0.3, orthoscopic=1, antialias=2, cgo_lighting=1,
                      ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k_, v)
    w = h = a.size; cmd.viewport(w // 4, h // 4); cmd.reset()
    for i, p in enumerate(Y[show]): cmd.pseudoatom("frame", pos=[float(x) for x in p], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=3, complete=1); cmd.delete("frame")
    view = cmd.get_view(); cmd.png(str(O / "rosette_render.png"), width=w, height=h, dpi=600, ray=1); cmd.save(str(O / "rosette.pse"))
    R = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); vc = (Y - org) @ R + cam
    hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    px = w / 2 * (1 + vc[:, 0] / hh); py = h / 2 * (1 - vc[:, 1] / hh)
    lab_of = dict(zip(EL.bead, EL.label)); deg = pd.concat([RED.b1, RED.b2]).value_counts()
    pd.DataFrame(dict(bead=mem, label=[lab_of[b] for b in mem], degree=[int(deg.get(b, 0)) for b in mem], px=px[mem], py=py[mem])).to_csv(O / "rosette_promoters.tsv", sep="\t", index=False)
    json.dump(dict(conf=a.conf, n_promoters=len(mem), loops=[(p, q) for p, q in loops], n_red_inside=len(ch) // 13), open(O / "rosette_meta.json", "w"))
    print(f"conf {a.conf}: {len(mem)} promoters linked, {len(loops)} segments, red chords {len(ch) // 13}")
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
    MM = 1 / 25.4; PT = 72 / 25.4
    im = np.asarray(Image.open(O / "rosette_render.png").convert("RGBA")); ys, xs = np.where(im[..., 3] > 5)
    cx_, cy_ = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; half = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.04)
    x0, y0 = max(0, cx_ - half), max(0, cy_ - half); cr = im[y0:cy_ + half, x0:cx_ + half]; ch_, cw_ = cr.shape[:2]
    W, H = 110.0, 100.0; IMW = 70.0; IMH = IMW * ch_ / cw_; IX = 20.0; IY = 12.0
    Image.fromarray(cr, "RGBA").save(O / "raster/rosette_render.png", dpi=(round(cw_ / (IMW * MM)),) * 2)
    T = pd.read_csv(O / "rosette_promoters.tsv", sep="\t")
    T["x"] = IX + (T.px - x0) / cw_ * IMW; T["y"] = IY + IMH - (T.py - y0) / ch_ * IMH
    lab = T.sort_values(["degree", "bead"], ascending=[False, True]).head(10)
    fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.imshow(cr, extent=(IX, IX + IMW, IY, IY + IMH), interpolation="none", zorder=0)
    mid = IX + IMW / 2
    for side, sub in ((-1, lab[lab.x < mid]), (1, lab[lab.x >= mid])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for k in range(1, len(yy)): yy[k] = max(yy[k], yy[k - 1] + 3.4)
        if yy and yy[-1] > IY + IMH: yy = [v - (yy[-1] - (IY + IMH)) for v in yy]
        lx = IX - 1 if side < 0 else IX + IMW + 1
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx], [r.y, ty], color="#444444", lw=0.4, zorder=3)
            ax.text(lx - 0.5 * (side < 0) + 0.5 * (side > 0), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", zorder=4)
    ax.text(W / 2, H - 3, "Hi-C-constrained 3D model of the chr16p11.2 promoter hub", ha="center", va="top", fontsize=8, fontweight="bold")
    ly = 7.0
    ax.plot([8, 14], [ly, ly], color="#9AB8D8", lw=3, solid_capstyle="round"); ax.text(15.5, ly, "chromatin loop", va="center")
    ax.add_patch(Circle((44, ly), 1.2, color="#222222", lw=0)); ax.text(46, ly, "hub promoter (loop anchor)", va="center")
    ax.plot([80, 86], [ly, ly], color="#C0392B", lw=1); ax.text(87.5, ly, "Hi-C loop edge", va="center")
    ax.text(W / 2, 2.6, "Polymer model restrained by WT HEK293T Hi-C (10 kb O/E) and Peakachu 0.7 loops;\none example structure from 9,850 simulated",
            ha="center", va="center", fontsize=6, color="#444444")
    for e in ("svg", "pdf"): fig.savefig(O / f"rosette.{e}", transparent=True)
    fig.savefig(O / "rosette_preview.png", dpi=300, facecolor="white"); plt.close(fig)
    import re
    t = open(O / "rosette.svg").read(); m = next(re.finditer(r"<image\b.*?/>", t, re.S))
    open(O / "rosette_linked.svg", "w").write(t[:m.start()] + f'<image xlink:href="raster/rosette_render.png" x="{IX * PT:.3f}" y="{(H - IY - IMH) * PT:.3f}" '
                                              f'width="{IMW * PT:.3f}" height="{IMH * PT:.3f}" preserveAspectRatio="none"/>' + t[m.end():])
    print(f"{W} x {H} mm; render {cw_} px over {IMW} mm = {cw_ / (IMW * MM):.0f} dpi; labels {lab.label.tolist()}")
