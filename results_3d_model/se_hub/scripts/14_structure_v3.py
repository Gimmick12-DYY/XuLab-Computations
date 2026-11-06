"""SE candidate structure slides, hub-style. 3D: red tube = chromatin of the loops closed in this structure (< 2 sigma), pale chain = rest, TF spheres around each closed partner, sphere = promoter, diamond = other element. Flower: all 12 loops around the SE candidate.
  render : python 14_structure_v3.py --sim <traj dir> --conf N --out s3362     (pymol-open-source)
  figure : python 14_structure_v3.py --sim <traj dir> --conf N --out s3362 --figure"""
import argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--conf", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--figure", action="store_true"); ap.add_argument("--size", type=int, default=2800); a = ap.parse_args()
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_3d_v3" / a.out; N = 910
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, INK, RED, SEC, GHOST, FADE = "#BDBDBD", "#222222", "#C0392B", "#4D4D4D", "#D9D9D9", "#BEBEBE"
EL = pd.read_csv(Bd / "model/inputs/hub_elements.tsv", sep="\t"); core = int(EL.bead[EL.is_core].iloc[0]); S = pd.read_csv(Bd / "figure_v2_cand/spokes.tsv", sep="\t").sort_values("pos").reset_index(drop=True); B = pd.read_csv(Bd / "model/inputs/beads.tsv", sep="\t").fillna({"tfs": ""})
EC = pd.read_csv(Bd / "model/analysis/edge_closure.tsv", sep="\t"); S["freq"] = [float(EC[(EC[["b1", "b2"]].min(1) == min(core, b)) & (EC[["b1", "b2"]].max(1) == max(core, b))].closure_freq.iloc[0]) for b in S.bead]
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t_, k_ = divmod(a.conf - 1, 50); X = np.load(files[t_])["X"][k_, :N].astype(float); S["dist"] = np.linalg.norm(X[S.bead] - X[core], axis=1); S["closed"] = S.dist < 2.0; S.to_csv(O / "partners.tsv", sep="\t", index=False)
tfset = lambda b: set(x for x in B.tfs[b].split(",") if x)
if not a.figure:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    Y = (X - X[core]) * 10.0; pet = np.zeros(N, bool)
    for b in S.bead[S.closed]: pet[min(b, core):max(b, core) + 1] = True
    _, _, Vt = np.linalg.svd(Y[pet] - Y[pet].mean(0), full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T; cmd.reinitialize(); b0, b1 = S.bead.min(), S.bead.max() + 1
    Z = gaussian_filter1d(Y, 2.5, axis=0, mode="nearest"); idx = np.arange(b0, b1); P = Z[idx]
    for b in np.r_[core, S.bead.values]: P[b - b0] = Y[b]
    Pd = np.array([np.interp(np.linspace(0, len(P) - 1, len(P) * 4), np.arange(len(P)), P[:, d]) for d in range(3)]).T; bo = b0 + np.minimum(np.round(np.linspace(0, len(P) - 1, len(Pd))).astype(int), len(P) - 1)
    g, r_ = rgb(GHOST), rgb(RED); gh, pe = [cgo.ALPHA, 0.25], []
    for u, v, b in zip(Pd[:-1], Pd[1:], bo[1:]):
        if pet[b]: pe += [cgo.CYLINDER, *u, *v, 1.25, *r_, *r_, cgo.COLOR, *r_, cgo.SPHERE, *v, 1.25]
        else: gh += [cgo.CYLINDER, *u, *v, 0.55, *g, *g]
    cmd.load_cgo(gh, "other"); cmd.load_cgo(pe, "petals")
    def octa(c0, r, col):
        V = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]; o = [cgo.COLOR, *col]
        for a_ in V[:2]:
            for b_ in V[2:4]:
                for z_ in V[4:]:
                    o += [cgo.NORMAL, *((a_ + b_ + z_) / np.sqrt(3))] + sum(([cgo.VERTEX, *(c0 + r * v)] for v in (a_, b_, z_)), [])
        return o
    sp, oc, sat = [cgo.COLOR, *rgb(SEC), cgo.SPHERE, 0, 0, 0, 5.2], [cgo.BEGIN, cgo.TRIANGLES], []; gold = np.pi * (3 - np.sqrt(5))
    for r in S.itertuples():
        col = rgb(INK if r.closed else FADE); rad = 3.0 if r.closed else 1.9; p_ = Y[r.bead]
        if r.cls == "promoter": sp += [cgo.COLOR, *col, cgo.SPHERE, *p_, rad]
        else: oc += octa(p_, rad * 1.45, col)
        if not r.closed: continue
        tf = tfset(r.bead); tl = [x for x in TFS if x in tf] + ["other"] * min(len(tf - set(TFS)), 10); n = max(len(tl), 1)
        for j, x in enumerate(tl):
            zz = 1 - 2 * (j + 0.5) / n; rr = np.sqrt(1 - zz * zz); ph = j * gold; sat += [cgo.COLOR, *rgb(COL.get(x, OTHER)), cgo.SPHERE, *(p_ + 4.6 * np.array([rr * np.cos(ph), rr * np.sin(ph), zz])), 1.25 if x in COL else 0.75]
    tf = tfset(core); tl = [x for x in TFS if x in tf] + ["other"] * min(len(tf - set(TFS)), 12); n = len(tl)
    for j, x in enumerate(tl):
        zz = 1 - 2 * (j + 0.5) / n; rr = np.sqrt(1 - zz * zz); ph = j * gold; sat += [cgo.COLOR, *rgb(COL.get(x, OTHER)), cgo.SPHERE, *(7.0 * np.array([rr * np.cos(ph), rr * np.sin(ph), zz])), 1.3 if x in COL else 0.8]
    cmd.load_cgo(sp, "anchors"); cmd.load_cgo(oc + [cgo.END], "elements"); cmd.load_cgo(sat, "tfs")
    op = []
    for r in S[~S.closed].itertuples():
        p_ = Y[r.bead]; L = np.linalg.norm(p_); c = rgb(FADE)
        for s in np.arange(5, L, 3.2): op += [cgo.CYLINDER, *(p_ * s / L), *(p_ * min(s + 1.5, L) / L), 0.35, *c, *c]
    if op: cmd.load_cgo(op, "open_links")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.005, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.5, specular=0.1, direct=0.5, reflect=0.25, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k, v)
    w = a.size; cmd.viewport(w // 4, w // 4); cmd.reset()
    for i, q in enumerate(Y[np.r_[core, S.bead.values, np.where(pet)[0][::4]]]): cmd.pseudoatom("frame", pos=[float(x) for x in q], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=9, complete=1); cmd.delete("frame"); view = cmd.get_view(); cmd.png(str(O / "render.png"), width=w, height=w, dpi=600, ray=1); cmd.save(str(O / "render.pse"))
    R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); idx = np.r_[core, S.bead.values]; vc = (Y[idx] - org) @ R_ + cam; hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=idx, px=w / 2 * (1 + vc[:, 0] / hh), py=w / 2 * (1 - vc[:, 1] / hh))).to_csv(O / "projection.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42})
    typ = a.conf == 3362; nc = int(S.closed.sum())
    # 3D panel
    W, H = 7.10, 5.75; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.text(0.06, H - 0.2, f"One typical structure: {nc} of 12 loops closed" if typ else f"The fullest structure: {nc} of 12 loops closed", fontsize=14, fontweight="bold", va="center")
    im = Image.open(O / "render.png").convert("RGBA"); PJ = pd.read_csv(O / "projection.tsv", sep="\t"); aa = np.asarray(im)[..., 3]; ys, xs = np.where(aa > 25); c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.01); x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); PJ["px"] -= x0; PJ["py"] -= y0; w = im.size[0]
    IW = 4.25; IX, IY = (W - IW) / 2, 1.08; ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="lanczos", zorder=0); ax.set_xlim(0, W); ax.set_ylim(0, H)
    PJ["x"] = IX + PJ.px / w * IW; PJ["y"] = IY + IW - PJ.py / w * IW; Q = PJ.iloc[1:].copy(); Q["label"] = S.label.values; Q["closed"] = S.closed.values; mid = IX + IW / 2
    for side, sub in ((-1, Q[Q.x < mid]), (1, Q[Q.x >= mid])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 0.3)
        if yy and yy[-1] > H - 0.5: yy = [v - (yy[-1] - (H - 0.5)) for v in yy]
        for i in range(len(yy) - 2, -1, -1): yy[i] = min(yy[i], yy[i + 1] - 0.3)
        lx = IX - 0.05 if side < 0 else IX + IW + 0.05
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx], [r.y, ty], color="#777777" if r.closed else "#CCCCCC", lw=0.6, zorder=3); ax.text(lx + (-0.04 if side < 0 else 0.04), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", fontsize=12, color=INK if r.closed else "#8A8A8A", fontweight="bold" if r.closed else "normal")
    ax.text(PJ.x[0], PJ.y[0], "SE", ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=4)
    y = 0.86; ax.plot([0.1, 0.42], [y, y], color=RED, lw=4, solid_capstyle="butt"); ax.text(0.5, y, "closed loop", va="center"); ax.plot(1.65, y, "o", ms=9, color=INK, mec="none"); ax.text(1.78, y, "promoter", va="center"); ax.plot(2.75, y, "D", ms=8, color=INK, mec="none"); ax.text(2.88, y, "other element", va="center"); ax.plot(4.2, y, "o", ms=6, color=FADE, mec="none"); ax.text(4.32, y, "open (not closed here)", va="center")
    items = [(COL[t], t) for t in TFS] + [(OTHER, "other TFs")]
    for k, (c_, l) in enumerate(items):
        col, row = k % 6, k // 6; x_, y_ = 0.18 + col * 1.15, 0.52 - row * 0.28; ax.add_patch(Circle((x_, y_), 0.07, color=c_, lw=0)); ax.text(x_ + 0.12, y_, l, va="center", fontstyle="italic" if l in TFS else "normal")
    for e in ("svg", "pdf"): fig.savefig(O / f"structure_3d.{e}", dpi=600, transparent=True)
    fig.savefig(O / "structure_3d.png", dpi=600, facecolor="white"); fig.savefig(O / "structure_3d_check.png", dpi=144, facecolor="white"); plt.close(fig)
    # flower
    W, H = 5.85, 5.75; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    ax.text(0.06, H - 0.2, "Across 10,000 structures: each loop closes only sometimes" if typ else "Which loops are closed in this rare structure", fontsize=14 if not typ else 13.5, fontweight="bold", va="center")
    cx, cy, Rs = W / 2, 3.08, 0.36; n = len(S)
    for i, r in S.iterrows():
        th = np.pi / 2 - 2 * np.pi * i / n; Lp = 0.45 + 0.42 * np.log10(r.span_kb / 40); pol = lambda rr, aa: np.array([cx + rr * np.cos(aa), cy + rr * np.sin(aa)]); w_ = 0.2
        P0, P3 = pol(Rs * 0.8, th - 0.2), pol(Rs * 0.8, th + 0.2); P1, P2 = pol(Rs + 1.5 * Lp, th - w_), pol(Rs + 1.5 * Lp, th + w_); tt = np.linspace(0, 1, 200)[:, None]; Bz = (1 - tt) ** 3 * P0 + 3 * (1 - tt) ** 2 * tt * P1 + 3 * (1 - tt) * tt ** 2 * P2 + tt ** 3 * P3
        if typ: ax.fill(Bz[:, 0], Bz[:, 1], color=RED, alpha=float(np.clip(r.freq / 0.65, 0.08, 1)), lw=0, zorder=1); ax.plot(Bz[:, 0], Bz[:, 1], color=RED, lw=1.2, zorder=2, ls="-" if r.cls == "promoter" else (0, (1.2, 1.2)))
        else: ax.fill(Bz[:, 0], Bz[:, 1], color=RED if r.closed else "#FFFFFF", alpha=0.75 if r.closed else 1, lw=0, zorder=1); ax.plot(Bz[:, 0], Bz[:, 1], color=RED if r.closed else "#AAAAAA", lw=1.2, zorder=2, ls="-" if r.closed else (0, (3, 2)))
        tip = Bz[np.argmax(np.hypot(Bz[:, 0] - cx, Bz[:, 1] - cy))]; u = (tip - [cx, cy]) / np.hypot(*(tip - [cx, cy])); txt = f"{r.label}\n{r.freq * 100:.0f}%" if typ else r.label; ha = "left" if u[0] > 0.2 else "right" if u[0] < -0.2 else "center"
        ax.text(tip[0] + 0.1 * u[0], tip[1] + 0.2 * u[1] if abs(u[0]) <= 0.2 else tip[1] + 0.08 * u[1], txt, ha=ha, va="center", fontsize=12, fontstyle="italic", linespacing=0.95, color=INK if (typ or r.closed) else "#8A8A8A", fontweight="normal" if typ or not r.closed else "bold")
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=Rs, fc="#F4F4F4", ec=SEC, lw=1.5, zorder=5)); ax.text(cx, cy, "SE\ncand.", ha="center", va="center", fontsize=12, fontweight="bold", color=SEC, linespacing=0.9, zorder=6)
    msg = "Darker petal = closes more often (% of structures); longer = longer span.\nA typical structure closes 4 of 12; none closes all 12." if typ else "Filled = closed here; dashed = open. Longer petal = longer span.\nOnly 20 of 10,000 structures (0.2%) reach 10; a typical one closes 4."
    ax.text(W / 2, 0.35, msg, ha="center", va="center", fontsize=12, linespacing=1.15)
    for e in ("svg", "pdf"): fig.savefig(O / f"flower.{e}", dpi=600, transparent=True)
    fig.savefig(O / "flower.png", dpi=600, facecolor="white"); fig.savefig(O / "flower_check.png", dpi=144, facecolor="white")
print(S[["label", "cls", "span_kb", "freq", "closed"]].round(2).to_string(index=False))
