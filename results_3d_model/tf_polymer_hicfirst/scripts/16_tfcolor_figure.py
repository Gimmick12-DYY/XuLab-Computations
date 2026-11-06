"""TF-colour version of the simple rosette (conformation 3338, Hi-C-first model, no TF term).

Anchors and loops are exactly those of figure_simple (15_simple_figure.py): the browser hub elements linked within 2 sigma,
and the chain between consecutive linked elements <= 200 kb apart. TF binding per anchor = TFs whose MACS2 q<0.05
narrowPeak overlaps the anchor's 2 kb bead (measured binding; the model has no TF term). Colours: the 10 TFs named in
the hub browser figure (ZNF507, SOX4, NFYA, GLYR1, NME2, ZNF282, ADNP, DR1, MTF2, E4F1, in that order) get distinct
colour-blind-safe colours that avoid the red (Hi-C-supported edge) and blue (Cicero-only) hues; all other TFs are grey.
A loop counts as closed by Hi-C if a Peakachu loop or red-edge restraint that is ON in this trajectory has its anchors
within +-3 beads of the loop's two ends.
  (render)   python 16_tfcolor_figure.py --peaks <dir> --sim <hicfirst traj dir>          (needs pymol-open-source)
  (figure)   python 16_tfcolor_figure.py --peaks <dir> --sim <dir> --figure
"""
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/tf_polymer_hicfirst")
ap.add_argument("--peaks", required=True)
ap.add_argument("--sim", required=True)
ap.add_argument("--figure", action="store_true")
ap.add_argument("--size", type=int, default=2600)
a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_tfcolor"; (O / "raster").mkdir(parents=True, exist_ok=True)
FS = Bd / "figure_simple"; meta = json.load(open(FS / "rosette_meta.json")); CONF = meta["conf"]
N, L0 = 910, 29730000
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]
COL = dict(zip(TFS, ["#E69F00", "#009E73", "#F0E442", "#CC79A7", "#117733", "#DDCC77", "#44AA99", "#999933", "#AA4499", "#A6761D"]))
OTHER, INK, RED = "#BDBDBD", "#222222", "#C0392B"
EL = pd.read_csv(Bd / "hicfirst/figure_hub_flow/hub_elements.tsv", sep="\t"); lab_of = dict(zip(EL.bead, EL.label))
HE = pd.read_csv(Bd / "hicfirst/figure_hub_flow/hub_edges_beads.tsv", sep="\t"); REDE = HE[HE["on_loop_0.7"] == 1]
PR = pd.read_csv(FS / "rosette_promoters.tsv", sep="\t"); mem = PR.bead.values
loops = [tuple(x) for x in meta["loops"]]

# measured binding per anchor bead
bind = {b: set() for b in mem}
for f in sorted(glob.glob(os.path.join(a.peaks, "TF.*_peaks.q005.narrowPeak"))):
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.split("\t")
        if x[0] != "chr16": continue
        s, e = int(x[1]), int(x[2])
        for b in mem:
            bs = L0 + 2000 * b
            if s < bs + 2000 and e > bs: bind[b].add(tf)
BT = pd.DataFrame([dict(bead=b, label=lab_of[b], n_tfs=len(bind[b]), named=",".join(t for t in TFS if t in bind[b]),
                        n_other=len(bind[b] - set(TFS)), all_tfs=",".join(sorted(bind[b]))) for b in mem])
BT.to_csv(O / "anchor_tf_binding.tsv", sep="\t", index=False)

# loops closed by an ON Hi-C loop / red-edge restraint in this trajectory
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t, k = divmod(CONF - 1, 50); z = np.load(files[t])
RS = pd.read_csv(Bd / "hicfirst/inputs/restraints.tsv", sep="\t"); on = z["loop_on"].astype(bool)
LL = RS[on & RS.kind.isin(["loop", "red_edge"]).values]
closed = [bool(((np.abs(LL.b1 - p) <= 3) & (np.abs(LL.b2 - q) <= 3)).any()) for p, q in loops]
PT = pd.DataFrame([dict(loop=f"L{i + 1}", bead1=p, bead2=q, gene1=lab_of[p], gene2=lab_of[q], span_kb=(q - p) * 2,
                        start=L0 + 2000 * p, end=L0 + 2000 * (q + 1), closed_by_hic_loop=c) for i, ((p, q), c) in enumerate(zip(loops, closed))])
PT.to_csv(O / "loops.tsv", sep="\t", index=False)

if not a.figure:
    P = z["X"][k, :N].astype(float)
    show = np.zeros(N, bool)
    for p, q in loops: show[p:q + 1] = True
    show[mem] = True
    Y = (P - P[mem].mean(0)) * 10.0
    _, _, Vt = np.linalg.svd(Y[show] - Y[show].mean(0), full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    cmd.reinitialize(); g = rgb("#A9A9A9")
    for i, (p, q) in enumerate(loops):
        seg = np.arange(p, q + 1); pts = Y[seg]
        if len(seg) >= 4:
            tck, _ = splprep(pts.T, s=len(seg) * 0.08, k=3); pts = np.array(splev(np.linspace(0, 1, len(seg) * 8), tck)).T
        o = []
        for u, v in zip(pts[:-1], pts[1:]): o += [cgo.CYLINDER, *u, *v, 1.8, *g, *g, cgo.COLOR, *g, cgo.SPHERE, *v, 1.8]
        cmd.load_cgo(o, f"loop{i}")
    ch = []
    for r in REDE.itertuples():
        if r.b1 in set(mem) and r.b2 in set(mem): c = rgb(RED); ch += [cgo.CYLINDER, *Y[r.b1], *Y[r.b2], 0.8, *c, *c]
    if ch: cmd.load_cgo(ch, "hic_edges")
    gold = np.pi * (3 - np.sqrt(5))
    core, sat = [cgo.COLOR, *rgb(INK)], []
    for b in mem:
        core += [cgo.SPHERE, *Y[b], 4.0]
        tfl = [t_ for t_ in TFS if t_ in bind[b]] + ["other"] * min(len(bind[b] - set(TFS)), 12)
        n = len(tfl)
        for j, t_ in enumerate(tfl):
            zz = 1 - 2 * (j + 0.5) / n; rr = np.sqrt(1 - zz * zz); ph = j * gold
            d = np.array([rr * np.cos(ph), rr * np.sin(ph), zz]); c = rgb(COL.get(t_, OTHER))
            sat += [cgo.COLOR, *c, cgo.SPHERE, *(Y[b] + 5.6 * d), 1.6 if t_ in COL else 1.0]
    cmd.load_cgo(core, "anchors"); cmd.load_cgo(sat, "tf_binding")
    for k_, v in dict(ray_trace_mode=1, ray_trace_gain=0.006, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15,
                      ambient=0.45, specular=0.1, direct=0.55, reflect=0.3, orthoscopic=1, antialias=2, cgo_lighting=1,
                      ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k_, v)
    w = h = a.size; cmd.viewport(w // 4, h // 4); cmd.reset()
    for i, p in enumerate(Y[show]): cmd.pseudoatom("frame", pos=[float(x) for x in p], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=4, complete=1); cmd.delete("frame")
    view = cmd.get_view(); cmd.png(str(O / "tfcolor_render.png"), width=w, height=h, dpi=600, ray=1); cmd.save(str(O / "tfcolor.pse"))
    R = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); vc = (Y - org) @ R + cam
    hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=mem, px=w / 2 * (1 + vc[mem, 0] / hh), py=h / 2 * (1 - vc[mem, 1] / hh))).to_csv(O / "anchor_projection.tsv", sep="\t", index=False)
    print("rendered; anchors", len(mem), "red chords", len(ch) // 13)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
    MM = 1 / 25.4; PTS = 72 / 25.4
    FOOT = ("One example structure (conformation 3338 of 9,850, chosen as the most compact rosette); Hi-C-constrained polymer model\n"
            "(Hi-C O/E pixels + Peakachu 0.7 loops), no TF input; TF colours show measured binding (MACS2 peaks), not modelled TF clusters.")
    def legend(ax, x, y, dy=3.0, cols=2, colw=24):
        items = [(COL[t_], t_) for t_ in TFS] + [(OTHER, "other TFs"), (INK, "hub promoter"), (RED, "Hi-C-supported edge")]
        per = int(np.ceil(len(items) / cols))
        for i, (c, l) in enumerate(items):
            xx, yy = x + colw * (i // per), y - dy * (i % per)
            if l == "Hi-C-supported edge": ax.plot([xx - 1.2, xx + 1.2], [yy, yy], color=c, lw=1.0)
            else: ax.add_patch(Circle((xx, yy), 0.9 if l != "hub promoter" else 1.1, color=c, lw=0))
            ax.text(xx + 2.0, yy, l, va="center", fontstyle="italic" if l in TFS else "normal")
    # ---- 3D panel
    im = np.asarray(Image.open(O / "tfcolor_render.png").convert("RGBA")); ys, xs = np.where(im[..., 3] > 5)
    cx_, cy_ = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; half = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.04)
    x0, y0 = max(0, cx_ - half), max(0, cy_ - half); cr = im[y0:cy_ + half, x0:cx_ + half]; chh, cww = cr.shape[:2]
    W, H = 160.0, 100.0; IMW = 78.0; IMH = IMW * chh / cww; IX = 20.0; IY = 13.0
    Image.fromarray(cr, "RGBA").save(O / "raster/tfcolor_render.png", dpi=(round(cww / (IMW * MM)),) * 2)
    PJ = pd.read_csv(O / "anchor_projection.tsv", sep="\t").merge(BT, on="bead")
    PJ["x"] = IX + (PJ.px - x0) / cww * IMW; PJ["y"] = IY + IMH - (PJ.py - y0) / chh * IMH
    deg = pd.concat([REDE.b1, REDE.b2]).value_counts(); PJ["deg"] = [int(deg.get(b, 0)) for b in PJ.bead]
    labs = PJ.sort_values(["deg", "n_tfs"], ascending=False).head(10)
    fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.imshow(cr, extent=(IX, IX + IMW, IY, IY + IMH), interpolation="none", zorder=0)
    midx = IX + IMW / 2
    for side, sub in ((-1, labs[labs.x < midx]), (1, labs[labs.x >= midx])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 3.4)
        if yy and yy[-1] > IY + IMH: yy = [v - (yy[-1] - (IY + IMH)) for v in yy]
        lx = IX - 1 if side < 0 else IX + IMW + 1
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx], [r.y, ty], color="#555555", lw=0.4, zorder=3)
            ax.text(lx + (-0.5 if side < 0 else 0.5), ty, f"{r.label} ({r.n_tfs})", ha="right" if side < 0 else "left", va="center", fontstyle="italic")
    ax.text(W / 2, H - 2.5, "chr16p11.2 promoter hub: Hi-C-constrained structure with measured TF binding", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(IX + IMW + 26, H - 12, "TF peaks at each promoter\n(small spheres; n = TFs\nwith a MACS2 peak)", va="top", fontsize=6.5)
    legend(ax, IX + IMW + 27, H - 20, cols=1)
    ax.text(W / 2, 4.0, FOOT, ha="center", va="center", fontsize=5.5, color="#444444")
    for e in ("svg", "pdf"): fig.savefig(O / f"tfcolor_3d.{e}", transparent=True)
    fig.savefig(O / "tfcolor_3d_preview.png", dpi=300, facecolor="white"); plt.close(fig)
    t_ = open(O / "tfcolor_3d.svg").read(); m = next(re.finditer(r"<image\b.*?/>", t_, re.S))
    open(O / "tfcolor_3d_linked.svg", "w").write(t_[:m.start()] + f'<image xlink:href="raster/tfcolor_render.png" x="{IX * PTS:.3f}" y="{(H - IY - IMH) * PTS:.3f}" width="{IMW * PTS:.3f}" height="{IMH * PTS:.3f}" preserveAspectRatio="none"/>' + t_[m.end():])
    # ---- 2D schematic: anchors on a ring in genomic order, petal between consecutive anchors, length ~ log10(span)
    W2 = 150.0; fig = plt.figure(figsize=(W2 * MM, 110 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-75, 75); ax.set_ylim(-62, 52); ax.set_aspect("equal"); ax.set_axis_off()
    Rr = 16.0; n = len(mem); ang = {b: np.pi / 2 - 2 * np.pi * i / n for i, b in enumerate(mem)}
    ax.add_patch(Circle((-25, 0), Rr, fill=False, ec="#DDDDDD", lw=0.8, zorder=0))
    cx0 = -25.0
    for r in PT.itertuples():
        a1, a2 = ang[r.bead1], ang[r.bead2]
        if a2 > a1: a2 -= 2 * np.pi
        m_ = (a1 + a2) / 2; half_ = min(0.16, (a1 - a2) / 2 * 0.55); Lp = 5 + 10 * np.log10(r.span_kb / 2 + 1)
        tt = np.linspace(-1, 1, 200); rr = Rr + Lp * np.sin(np.pi * (tt + 1) / 2); aa = m_ + half_ * tt
        ax.plot(cx0 + rr * np.cos(aa), rr * np.sin(aa), color="#8C8C8C", lw=1.6, ls="-" if r.closed_by_hic_loop else (0, (3, 2)))
        ax.text(cx0 + (Rr + Lp + 3.2) * np.cos(m_), (Rr + Lp + 3.2) * np.sin(m_), f"{r.span_kb} kb", ha="center", va="center", fontsize=6)
    for r in REDE.itertuples():
        if r.b1 in ang and r.b2 in ang:
            ax.plot([cx0 + Rr * np.cos(ang[r.b1]), cx0 + Rr * np.cos(ang[r.b2])], [Rr * np.sin(ang[r.b1]), Rr * np.sin(ang[r.b2])], color=RED, lw=0.6, alpha=0.8, zorder=1)
    for b in mem:
        tfl = [t_ for t_ in TFS if t_ in bind[b]]; no = len(bind[b] - set(TFS)); tot = len(tfl) + no
        x, y = cx0 + Rr * np.cos(ang[b]), Rr * np.sin(ang[b]); st = 90.0
        for t_ in tfl + (["other"] if no else []):
            w_ = 360.0 * (no if t_ == "other" else 1) / max(tot, 1)
            ax.add_patch(Wedge((x, y), 2.0, st, st + w_, color=COL.get(t_, OTHER), lw=0, zorder=3)); st += w_
        ax.add_patch(Circle((x, y), 0.8, color=INK, lw=0, zorder=4))
        ca, sa = np.cos(ang[b]), np.sin(ang[b]); lx_, ly_ = cx0 + (Rr + 3.4) * ca, (Rr + 3.4) * sa
        ax.text(lx_, ly_, f"{lab_of[b]} ({tot})", ha="left" if ca > 0.08 else ("right" if ca < -0.08 else "center"),
                va="bottom" if sa > 0.5 else ("top" if sa < -0.5 else "center"), fontsize=5.5, fontstyle="italic", zorder=5,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
    ax.text(-25, 48, "Flower schematic of the same structure (anchors in genomic order)", ha="center", va="center", fontsize=7.5, fontweight="bold")
    legend(ax, 22, 38, dy=3.2, cols=1)
    ax.plot([21, 25], [-6, -6], color="#8C8C8C", lw=1.6); ax.text(27, -6, "loop closed by a Hi-C loop / edge", va="center")
    ax.plot([21, 25], [-9.5, -9.5], color="#8C8C8C", lw=1.6, ls=(0, (3, 2))); ax.text(27, -9.5, "loop formed in the model only", va="center")
    ax.text(21, -14, "pie = TFs with a MACS2 peak at the promoter\n(n) = number of TFs with a peak\npetal length grows with log10(span); label = true span", va="top", fontsize=6)
    ax.text(0, -55, FOOT, ha="center", va="center", fontsize=5.5, color="#444444")
    for e in ("svg", "pdf"): fig.savefig(O / f"tfcolor_schematic.{e}", transparent=True)
    fig.savefig(O / "tfcolor_schematic_preview.png", dpi=300, facecolor="white"); plt.close(fig)
    print(BT[["label", "n_tfs", "named", "n_other"]].to_string(index=False)); print(PT[["loop", "gene1", "gene2", "span_kb", "closed_by_hic_loop"]].to_string(index=False))
