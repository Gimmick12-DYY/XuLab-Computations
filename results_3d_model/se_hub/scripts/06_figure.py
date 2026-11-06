"""SE-hub figure (4E-H8b style): 3D render of the representative Hi-C-first structure, SE-centred flower schematic and a
Hi-C vs model split-matrix inset. Palette C + v2 TF colours. Petals = the hub loop-edges (SE core <-> partner); each petal is the
chromatin between the SE and that partner, drawn with its partner promoter at the petal base next to the SE; solid = closed
(< 2 sigma) in the shown structure, dashed = open. Length grows with log10(span).
  render : python 06_figure.py --base <se_hub> --sim <traj dir>        (pymol-open-source)
  figure : python 06_figure.py --base <se_hub> --sim <traj dir> --figure
"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--sim", required=True)
ap.add_argument("--figure", action="store_true"); ap.add_argument("--size", type=int, default=2600); a = ap.parse_args()
Bd = Path(a.base); M = Bd / "model"; I = M / "inputs"; A = M / "analysis"; O = Bd / "figure"; (O / "raster").mkdir(parents=True, exist_ok=True)
N = 910; S_ = json.load(open(A / "summary.json")); CONF = S_["rep_conf"]
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]
COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, INK, RED, DRED, GHOST, SEC = "#BDBDBD", "#222222", "#C0392B", "#8E2A20", "#D9D9D9", "#4D4D4D"
B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"tfs": "", "hub_label": ""}); EL = pd.read_csv(I / "hub_elements.tsv", sep="\t"); EL.loc[EL.is_core, "label"] = "SE core"
HE = pd.read_csv(I / "hub_edges.tsv", sep="\t"); RS = pd.read_csv(I / "restraints.tsv", sep="\t"); RE = RS[RS.kind == "red_edge"].reset_index(drop=True)
EC = pd.read_csv(A / "edge_closure.tsv", sep="\t")
core = int(EL.bead[EL.is_core].iloc[0]); part = EL[~EL.is_core].reset_index(drop=True)
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t, k = divmod(CONF - 1, 50); X = np.load(files[t])["X"][k, :N].astype(float)
dist = {(min(r.b1, r.b2), max(r.b1, r.b2)): np.linalg.norm(X[r.b1] - X[r.b2]) for r in RE.itertuples()}
def tfset(b): return set(x for x in B.tfs[max(0, b - 0):b + 1].str.cat(sep=",").split(",") if x)
part["span_kb"] = (part.bead - core).abs() * 2; part["closed"] = [dist.get((min(core, b), max(core, b)), 99) < 2.0 for b in part.bead]
part["freq"] = [float(EC[(EC[["b1", "b2"]].min(1) == min(core, b)) & (EC[["b1", "b2"]].max(1) == max(core, b))].closure_freq.iloc[0]) for b in part.bead]
part.to_csv(O / "petals.tsv", sep="\t", index=False)
if not a.figure:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    Y = (X - X[EL.bead].mean(0)) * 10.0
    seg = np.zeros(N, bool)
    for b in part.bead:
        if abs(b - core) <= 30: seg[min(b, core):max(b, core) + 1] = True
    _, _, Vt = np.linalg.svd(Y[seg] - Y[seg].mean(0), full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T; cmd.reinitialize()
    tck, _ = splprep(Y.T, s=N * 0.08, k=3); tt = np.linspace(0, 1, N * 6); pts = np.array(splev(tt, tck)).T; bo = np.minimum((tt * (N - 1)).round().astype(int), N - 1)
    gh, pe = [cgo.ALPHA, 0.10], []
    for u, v, b in zip(pts[:-1], pts[1:], bo[1:]):
        if seg[b]: c = rgb(RED); pe += [cgo.CYLINDER, *u, *v, 1.4, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 1.4]
        else: c = rgb(GHOST); gh += [cgo.CYLINDER, *u, *v, 0.8, *c, *c]
    cmd.load_cgo(gh, "ghost"); cmd.load_cgo(pe, "petals")
    ln = []
    for b in part.bead:
        if abs(b - core) > 30: c = rgb(RED); ln += [cgo.CYLINDER, *Y[core], *Y[b], 0.9, *c, *c]
    if ln: cmd.load_cgo(ln, "long_edges")
    gold = np.pi * (3 - np.sqrt(5)); sp, sat = [], []
    for b, is_se in [(core, True)] + [(b, False) for b in part.bead]:
        sp += [cgo.COLOR, *rgb(SEC if is_se else INK), cgo.SPHERE, *Y[b], 5.0 if is_se else 3.0]
        tf = tfset(b); tfl = [x for x in TFS if x in tf] + ["other"] * min(len(tf - set(TFS)), 14); n = len(tfl); rad = 7.0 if is_se else 4.4
        for j, x in enumerate(tfl):
            zz = 1 - 2 * (j + 0.5) / n; rr = np.sqrt(1 - zz * zz); ph = j * gold
            sat += [cgo.COLOR, *rgb(COL.get(x, OTHER)), cgo.SPHERE, *(Y[b] + rad * np.array([rr * np.cos(ph), rr * np.sin(ph), zz])), 1.3 if x in COL else 0.8]
    cmd.load_cgo(sp, "anchors"); cmd.load_cgo(sat, "tf_binding")
    for k_, v in dict(ray_trace_mode=1, ray_trace_gain=0.006, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.45,
                      specular=0.1, direct=0.55, reflect=0.3, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3, ray_opaque_background=0,
                      ray_shadows=0, depth_cue=0).items(): cmd.set(k_, v)
    w = h = a.size; cmd.viewport(w // 4, h // 4); cmd.reset()
    for i, p in enumerate(Y[np.isin(np.arange(N), EL.bead)]): cmd.pseudoatom("frame", pos=[float(x) for x in p], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=45, complete=1); cmd.delete("frame")
    view = cmd.get_view(); cmd.png(str(O / "render.png"), width=w, height=h, dpi=600, ray=1); cmd.save(str(O / "se_hub.pse"))
    R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); vc = (Y - org) @ R_ + cam
    hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=EL.bead, label=EL.label, px=w / 2 * (1 + vc[EL.bead, 0] / hh), py=h / 2 * (1 - vc[EL.bead, 1] / hh))).to_csv(O / "projection.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge, RegularPolygon, Rectangle
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
    MM = 1 / 25.4
    def pie(ax, x, y, b, r):
        tf = tfset(b); tl = [t_ for t_ in TFS if t_ in tf]; no = len(tf - set(TFS)); tot = max(1, len(tl) + no); st = 90.0
        for t_ in tl + (["other"] if no else []):
            w_ = 360.0 * (no if t_ == "other" else 1) / tot; ax.add_patch(Wedge((x, y), r, st, st + w_, color=COL.get(t_, OTHER), lw=0, zorder=4)); st += w_
    # ---- schematic
    fig = plt.figure(figsize=(170 * MM, 120 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-75, 95); ax.set_ylim(-60, 60); ax.set_aspect("equal"); ax.set_axis_off()
    cx, cy, Rse = -18.0, 0.0, 7.5; n = len(part); P = part.sort_values("bead").reset_index(drop=True)
    for i, r in P.iterrows():
        th = np.pi / 2 - 2 * np.pi * (i + 0.5) / n; Lp = 9 + 10 * np.log10(r.span_kb / 10 + 1)
        base = np.array([cx + (Rse + 4.0) * np.cos(th), cy + (Rse + 4.0) * np.sin(th)]); w = 0.17
        pol = lambda rr, aa: np.array([cx + rr * np.cos(aa), cy + rr * np.sin(aa)])
        P0, P3 = pol(Rse, th - 0.12), base; P1, P2 = pol(Rse + 1.4 * Lp, th - w), pol(Rse + 1.4 * Lp, th + w)
        tt = np.linspace(0, 1, 200)[:, None]; Bz = (1 - tt) ** 3 * P0 + 3 * (1 - tt) ** 2 * tt * P1 + 3 * (1 - tt) * tt ** 2 * P2 + tt ** 3 * P3
        ax.fill(Bz[:, 0], Bz[:, 1], color="#F3C9C4", alpha=0.45 if r.closed else 0.18, lw=0, zorder=1)
        ax.plot(Bz[:, 0], Bz[:, 1], color=RED, lw=0.9, ls="-" if r.closed else (0, (2, 1.2)), zorder=2)
        pie(ax, *base, r.bead, 2.1); ax.add_patch(Circle(base, 0.8, color=INK, lw=0, zorder=5))
        tip = Bz[np.argmax(np.hypot(Bz[:, 0] - cx, Bz[:, 1] - cy))]; u = (tip - [cx, cy]) / np.hypot(*(tip - [cx, cy]))
        ax.text(tip[0] + 4.5 * u[0], tip[1] + 3.4 * u[1], f"{r.span_kb:.0f} kb\n{r.label}", ha="center", va="center", fontsize=4.8, color="#333333")
    pie(ax, cx, cy, core, Rse + 1.6)
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=Rse, orientation=0, fc="#F4F4F4", ec=SEC, lw=1.4, zorder=5))
    ax.text(cx, cy + 1.6, "SE", ha="center", va="center", fontsize=9, fontweight="bold", color=SEC, zorder=6)
    se = EL[EL.is_core].iloc[0]; ax.text(cx, cy - 2.4, f"{se.se.replace('HEK293_', '')}\n{len(tfset(core))} TFs", ha="center", va="center", fontsize=4.8, zorder=6)
    lx, ly, dy = 37, 52, 2.8
    ax.text(lx - 1.5, ly + 2.6, "TF with a MACS2 peak", fontsize=6, fontweight="bold", va="center")
    for i, (c, l) in enumerate([(COL[t_], t_) for t_ in TFS] + [(OTHER, "other TFs")]):
        ax.add_patch(Circle((lx, ly - dy * i), 0.9, color=c, lw=0)); ax.text(lx + 2.2, ly - dy * i, l, va="center", fontstyle="italic" if l in TFS else "normal")
    y2 = ly - dy * 11 - 2; ax.text(lx - 1.5, y2, "Structure", fontsize=6, fontweight="bold", va="center")
    ax.add_patch(RegularPolygon((lx, y2 - dy), 6, radius=1.3, fc="#F4F4F4", ec=SEC, lw=1)); ax.text(lx + 2.2, y2 - dy, "super-enhancer (ROSE, ENCODE H3K27ac)", va="center")
    ax.add_patch(Circle((lx, y2 - 2 * dy), 1.0, color=INK, lw=0)); ax.text(lx + 2.2, y2 - 2 * dy, "linked promoter / element", va="center")
    ax.plot([lx - 1.5, lx + 1.5], [y2 - 3 * dy] * 2, color=RED, lw=1); ax.text(lx + 2.2, y2 - 3 * dy, "loop-edge, closed in this structure", va="center")
    ax.plot([lx - 1.5, lx + 1.5], [y2 - 4 * dy] * 2, color=RED, lw=1, ls=(0, (2, 1.2))); ax.text(lx + 2.2, y2 - 4 * dy, "loop-edge, open in this structure", va="center")
    ax.text(lx - 1.5, y2 - 5.6 * dy, "petal = chromatin between the SE and a linked\nelement; length grows with log10(span)", fontsize=5.4, va="top")
    ax.text(cx, 56, "Super-enhancer hub, chr19p13.3", ha="center", va="center", fontsize=8, fontweight="bold")
    for e in ("svg", "pdf"): fig.savefig(O / f"se_schematic.{e}", transparent=True)
    fig.savefig(O / "se_schematic.png", dpi=600, facecolor="white"); fig.savefig(O / "se_schematic_preview.png", dpi=250, facecolor="white"); plt.close(fig)
    # ---- 3D + inset
    im = np.asarray(Image.open(O / "render.png").convert("RGBA")); ys, xs = np.where(im[..., 3] > 30)
    c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; half = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.03)
    x0, y0 = max(0, c0 - half), max(0, c1 - half); cr = im[y0:c1 + half, x0:c0 + half]; chh, cww = cr.shape[:2]
    W, H = 185.0, 105.0; IMW = 80.0; IMH = IMW * chh / cww; IX, IY = 22.0, 12.0
    Image.fromarray(cr, "RGBA").save(O / "raster/render.png", dpi=(round(cww / (IMW * MM)),) * 2)
    PJ = pd.read_csv(O / "projection.tsv", sep="\t"); PJ["x"] = IX + (PJ.px - x0) / cww * IMW; PJ["y"] = IY + IMH - (PJ.py - y0) / chh * IMH
    fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.imshow(cr, extent=(IX, IX + IMW, IY, IY + IMH), interpolation="none", zorder=0); mid = IX + IMW / 2
    for side, sub in ((-1, PJ[PJ.x < mid]), (1, PJ[PJ.x >= mid])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 3.2)
        if yy and yy[-1] > IY + IMH: yy = [v - (yy[-1] - (IY + IMH)) for v in yy]
        lx_ = IX - 1 if side < 0 else IX + IMW + 1
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx_], [r.y, ty], color="#555555", lw=0.4, zorder=3)
            ax.text(lx_ + (-0.5 if side < 0 else 0.5), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", fontsize=6,
                    fontweight="bold" if r.label.startswith("SE") else "normal")
    ax.text(W / 2, H - 2.5, f"One Hi-C-constrained structure (#{CONF}): SE hub, chr19:0.43–2.25 Mb", ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.text(IX + IMW + 32, 30, "red lines: the 12 loop-edges (SE core to linked element)\ngrey: chromatin of the 1.8 Mb window\nlarge grey sphere: SE core; dark: linked elements\nsmall spheres: TFs with a MACS2 peak", fontsize=5.5, va="top")
    # inset split matrix
    Hh = np.load(I / "hic_10kb.npz"); ice, keep = Hh["ice"], Hh["keep"]; Sm = np.load(A / "sim_10kb.npy"); n10 = ice.shape[0]
    def oe(Mx):
        E = np.full_like(Mx, np.nan, dtype=float)
        for d in range(n10):
            i = np.arange(n10 - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(Mx[i, j])
            if ok.any():
                mu = Mx[i[ok], j[ok]].mean()
                if mu > 0: E[i, j] = E[j, i] = Mx[i, j] / mu
        return E
    with np.errstate(divide="ignore", invalid="ignore"): Ho, So = np.log2(oe(ice)), np.log2(oe(Sm))
    Z = np.full((n10, n10), np.nan); iu = np.triu_indices(n10, 1); Z[iu] = Ho[iu]; Z[(iu[1], iu[0])] = So[(iu[1], iu[0])]; Z[~np.isfinite(Z)] = np.nan
    axi = fig.add_axes([(IX + IMW + 32) / W, 38 / H, 45 / W, 45 / H]); cm_ = plt.get_cmap("RdBu_r").copy(); cm_.set_bad("#EEEEEE")
    axi.imshow(np.ma.masked_invalid(Z), cmap=cm_, vmin=-1.5, vmax=1.5, interpolation="nearest")
    for r in RE.itertuples():
        i, j = r.b1 // 5, r.b2 // 5
        for (px, py) in ((j, i), (i, j)): axi.add_patch(Circle((px, py), 2.2, fill=False, ec=RED, lw=0.6))
    axi.set_xticks([]); axi.set_yticks([]); axi.set_title("Hi-C (upper) vs model (lower), O/E", fontsize=5.5, pad=2)
    for e in ("svg", "pdf"): fig.savefig(O / f"se_3d.{e}", transparent=True)
    fig.savefig(O / "se_3d.png", dpi=600, facecolor="white"); fig.savefig(O / "se_3d_preview.png", dpi=250, facecolor="white"); plt.close(fig)
    print(P[["label", "span_kb", "closed", "freq"]].to_string(index=False))
