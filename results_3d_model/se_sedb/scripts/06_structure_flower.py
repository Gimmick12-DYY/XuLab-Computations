"""Exact SEdb SE, chr16-hub format. 3D: one typical structure; closed loop-edges <= 400 kb as red chromatin petals, longer closed ones as red chords, open loop-edges as thin dashed red lines, Cicero-only edges as thin dashed blue lines; partners coloured by class (teal promoter, gold enhancer, grey other), SE orange, TF satellites at closed partners and the SE. Flower: all 31 edges as petals around the SE (red = on a Hi-C loop, blue = Cicero only), length ~ log10(span), node colour = class. Each figure at its exact frame size.
  render : python 06_structure_flower.py --sim <traj dir> --conf N   (pymol-open-source)
  figure : python 06_structure_flower.py --sim <traj dir> --conf N --figure"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--conf", type=int, required=True); ap.add_argument("--figure", action="store_true"); ap.add_argument("--size", type=int, default=3000); a = ap.parse_args()
R0 = Path(__file__).resolve().parents[1]; O = R0 / "figure"; I = R0 / "model/inputs"; N = 910; L0 = 430000
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, INK, RED, BLUE, GHOST = "#BDBDBD", "#222222", "#C0392B", "#4A6FA5", "#D9D9D9"; CC = dict(se="#E07B00", promoter="#0F7C72", enhancer="#D9B000", other="#8C8C8C"); TC = dict(se="#B85F00", promoter="#0F7C72", enhancer="#8F7400", other="#666666")
E = pd.read_csv(I / "hub_edges.tsv", sep="\t"); B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"tfs": ""}); Cl = pd.read_csv(R0 / "model/analysis/edge_closure_all.tsv", sep="\t"); mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000
core = tb((1248039 + 1254347) // 2); E["a"] = [tb(mid(x)) for x in E.region1]; E["p"] = [tb(mid(x)) for x in E.region2]; E["pos"] = [mid(x) for x in E.region2]; E["freq"] = Cl.closure.values; E["loop"] = E["on_loop_0.7"] == 1; E = E.sort_values("pos").reset_index(drop=True); E["label"] = E.label.str.replace("RNU6-2/TMEM259", "TMEM259")
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t_, k_ = divmod(a.conf - 1, 50); X = np.load(files[t_])["X"][k_, :N].astype(float); E["dist"] = np.linalg.norm(X[E.a] - X[E.p], axis=1); E["closed"] = E.dist < 2.0; E.to_csv(O / "edges_in_structure.tsv", sep="\t", index=False); print(E[E.closed][["label", "cls", "loop", "kb"]].to_string(index=False)); tfset = lambda b: set(x for x in B.tfs[b].split(",") if x)
if not a.figure:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    Y = (X - X[core]) * 10.0; pet = np.zeros(N, bool)
    for r in E[E.closed & E.loop & (E.kb <= 400)].itertuples(): pet[min(r.a, r.p):max(r.a, r.p) + 1] = True
    frm = np.r_[np.where(pet)[0], E.p.values, core]; _, _, Vt = np.linalg.svd(Y[E.p[E.closed | E.loop].values], full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T; cmd.reinitialize(); b0, b1 = E.p.min(), E.p.max() + 1; Z = gaussian_filter1d(Y, 2.5, axis=0, mode="nearest"); P = Z[b0:b1].copy()
    for b in np.r_[core, E.p.values]: P[b - b0] = Y[b]
    Pd = np.array([np.interp(np.linspace(0, len(P) - 1, len(P) * 4), np.arange(len(P)), P[:, d]) for d in range(3)]).T; bo = b0 + np.minimum(np.round(np.linspace(0, len(P) - 1, len(Pd))).astype(int), len(P) - 1); g, r_ = rgb(GHOST), rgb(RED); gh, pe = [cgo.ALPHA, 0.22], []
    for u, v, b in zip(Pd[:-1], Pd[1:], bo[1:]):
        if pet[b]: pe += [cgo.CYLINDER, *u, *v, 1.3, *r_, *r_, cgo.COLOR, *r_, cgo.SPHERE, *v, 1.3]
        else: gh += [cgo.CYLINDER, *u, *v, 0.5, *g, *g]
    cmd.load_cgo(gh, "other"); cmd.load_cgo(pe, "petals"); ln = []
    for r in E.itertuples():
        q = Y[r.p]; s0 = Y[r.a]; L = np.linalg.norm(q - s0); u_ = (q - s0) / L; c = r_ if r.loop else rgb(BLUE)
        if r.loop and r.closed and r.kb > 400: ln += [cgo.CYLINDER, *s0, *q, 0.9, *c, *c]
        elif not (r.loop and r.closed):
            for s in np.arange(4, L, 3.0): ln += [cgo.CYLINDER, *(s0 + u_ * s), *(s0 + u_ * min(s + 1.4, L)), 0.3, *c, *c]
    cmd.load_cgo(ln, "links"); sp = [cgo.COLOR, *rgb(CC["se"]), cgo.SPHERE, *Y[core], 5.0]; sat = []; gold = np.pi * (3 - np.sqrt(5))
    def sats(b, rad):
        tf = tfset(b); tl = [x for x in TFS if x in tf] + ["other"] * min(len(tf - set(TFS)), 10); n_ = max(len(tl), 1); o = []
        for j, x in enumerate(tl):
            zz = 1 - 2 * (j + 0.5) / n_; rr = np.sqrt(1 - zz * zz); ph = j * gold; o += [cgo.COLOR, *rgb(COL.get(x, OTHER)), cgo.SPHERE, *(Y[b] + rad * np.array([rr * np.cos(ph), rr * np.sin(ph), zz])), 1.2 if x in COL else 0.7]
        return o
    for r in E.itertuples():
        on = r.loop and r.closed; sp += [cgo.COLOR, *rgb(CC[r.cls]), cgo.SPHERE, *Y[r.p], 3.0 if on else 1.9 if r.loop else 1.3]
        if on: sat += sats(r.p, 4.4)
    sat += sats(core, 6.6); cmd.load_cgo(sp, "anchors"); cmd.load_cgo(sat, "tfs")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.005, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.5, specular=0.1, direct=0.5, reflect=0.25, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k, v)
    w = a.size; cmd.viewport(w // 4, w // 4); cmd.reset()
    for i, q in enumerate(Y[np.r_[np.where(pet)[0][::3], E.p[E.loop].values, core]]): cmd.pseudoatom("frame", pos=[float(x) for x in q], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=8, complete=1); cmd.delete("frame"); view = cmd.get_view(); cmd.png(str(O / "render.png"), width=w, height=w, dpi=600, ray=1); cmd.save(str(O / "render.pse")); R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); idx = np.r_[core, E.p.values]; vc = (Y[idx] - org) @ R_ + cam; hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=idx, px=w / 2 * (1 + vc[:, 0] / hh), py=w / 2 * (1 - vc[:, 1] / hh))).to_csv(O / "projection.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42}); Mt = pd.read_csv(O / "conformation_metrics.tsv", sep="\t"); me = Mt[Mt.conf == a.conf].iloc[0]; S_ = json.load(open(R0 / "model/analysis/summary.json"))
    W, H = 12.95, 5.75; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.text(0.08, H - 0.2, f"One typical Hi-C-constrained structure: {int(E.closed[E.loop].sum())} of {int(E.loop.sum())} loop-edges closed", fontsize=14, fontweight="bold", va="center")
    im = Image.open(O / "render.png").convert("RGBA"); PJ = pd.read_csv(O / "projection.tsv", sep="\t"); aa = np.asarray(im)[..., 3]; ys, xs = np.where(aa > 25); c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.01); x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); PJ["px"] -= x0; PJ["py"] -= y0; w = im.size[0]
    IW = 5.25; IX, IY = 1.75, 0.05; ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="lanczos", zorder=0); ax.set_xlim(0, W); ax.set_ylim(0, H); PJ["x"] = IX + PJ.px / w * IW; PJ["y"] = IY + IW - PJ.py / w * IW; Q = PJ.iloc[1:].copy().reset_index(drop=True); Q = pd.concat([Q, E[["label", "cls", "loop", "closed"]]], axis=1); Q = Q[Q.loop]; midx = IX + IW / 2
    for side, sub in ((-1, Q[Q.x < midx]), (1, Q[Q.x >= midx])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 0.3)
        if yy and yy[-1] > H - 0.45: yy = [v - (yy[-1] - (H - 0.45)) for v in yy]
        for i in range(len(yy) - 2, -1, -1): yy[i] = min(yy[i], yy[i + 1] - 0.3)
        lx = IX - 0.05 if side < 0 else IX + IW + 0.05
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx], [r.y, ty], color="#777777" if r.closed else "#CCCCCC", lw=0.6, zorder=3); ax.text(lx + (-0.04 if side < 0 else 0.04), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", color=TC[r.cls], fontweight="bold" if r.closed else "normal")
    lx, y = 8.75, 5.05; ax.text(lx, y + 0.28, "Super-enhancer (SEdb, HEK293T)", fontweight="bold", color=TC["se"], va="center")
    for c_, l_ in (("promoter", "promoter partner"), ("enhancer", "enhancer partner"), ("other", "other partner")):
        ax.plot(lx + 0.1, y, "o", ms=10, color=CC[c_], mec="none"); ax.text(lx + 0.28, y, l_, va="center"); y -= 0.28
    y -= 0.06
    ax.plot([lx, lx + 0.25], [y, y], color=RED, lw=4, solid_capstyle="butt"); ax.text(lx + 0.33, y, "loop-edge, closed here (bold name)", va="center"); y -= 0.28; ax.plot([lx, lx + 0.25], [y, y], color=RED, lw=1, ls=(0, (2, 1.5))); ax.text(lx + 0.33, y, "loop-edge, open here", va="center"); y -= 0.28; ax.plot([lx, lx + 0.25], [y, y], color=BLUE, lw=1, ls=(0, (2, 1.5))); ax.text(lx + 0.33, y, "Cicero only, not on a Hi-C loop", va="center"); y -= 0.28; ax.plot([lx, lx + 0.25], [y, y], color=GHOST, lw=1.5); ax.text(lx + 0.33, y, "other chromatin", va="center"); y -= 0.34
    ax.text(lx, y, "TF spheres (partners closed here, SE):", va="center"); y -= 0.26
    for k, (c_, l_) in enumerate([(COL[t], t) for t in TFS] + [(OTHER, "other")]):
        xx, yy_ = lx + 0.1 + (k % 3) * 1.25, y - (k // 3) * 0.25; ax.add_patch(Circle((xx, yy_), 0.07, color=c_, lw=0)); ax.text(xx + 0.12, yy_, l_, va="center", fontstyle="italic" if l_ in TFS else "normal")
    y = y - 4 * 0.25 - 0.02; ax.text(lx, y, f"Structure #{a.conf}: median closure (7 of 18); Rg {me.rg_pct:.0f}th,\ncrumpling {me.crumple_pct:.0f}th percentile; no self-overlap.\nLoop-edges are restraints; Cicero-only edges are not.", va="top", color="#444444", linespacing=1.1, fontsize=11)
    for e in ("svg", "pdf"): fig.savefig(O / f"structure_3d.{e}", dpi=600, transparent=True)
    fig.savefig(O / "structure_3d.png", dpi=600, facecolor="white"); fig.savefig(O / "structure_3d_check.png", dpi=144, facecolor="white"); plt.close(fig)
    # flower
    W, H = 12.95, 5.75; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.text(0.08, H - 0.2, "All Cicero edges from the super-enhancer", fontsize=14, fontweight="bold", va="center"); cx, cy, Rs = 4.75, 2.72, 0.34; n = len(E); tips = []
    for i, r in E.iterrows():
        th = np.pi / 2 - 2 * np.pi * (i + 0.5) / n; Lp = 0.5 + 0.55 * np.log10(r.kb / 40); pol = lambda rr, aa: np.array([cx + rr * np.cos(aa), cy + rr * np.sin(aa)]); w_ = 0.075; P0, P3 = pol(Rs * 0.85, th - w_), pol(Rs * 0.85, th + w_); P1, P2 = pol(Rs + 1.4 * Lp, th - w_ * 1.05), pol(Rs + 1.4 * Lp, th + w_ * 1.05); tt = np.linspace(0, 1, 120)[:, None]; Bz = (1 - tt) ** 3 * P0 + 3 * (1 - tt) ** 2 * tt * P1 + 3 * (1 - tt) * tt ** 2 * P2 + tt ** 3 * P3; c_ = RED if r.loop else BLUE; ax.fill(Bz[:, 0], Bz[:, 1], color=c_, alpha=0.32 if r.loop else 0.2, lw=0, zorder=1); ax.plot(Bz[:, 0], Bz[:, 1], color=c_, lw=1.0, zorder=2); rad = Rs + 0.75 * 1.4 * Lp + 0.07; tip = pol(rad, th); ax.plot(*tip, "o", ms=6.5, color=CC[r.cls], mec="white", mew=0.5, zorder=4); tips.append((i, tip[0], tip[1], r.label, r.cls))
    for side, sub in ((1, [t for t in tips if t[1] >= cx]), (-1, [t for t in tips if t[1] < cx])):
        sub = sorted(sub, key=lambda t: t[2]); m_ = len(sub); ys_ = np.linspace(cy - 2.42, cy + 2.42, m_) if m_ > 1 else [cy]; lx_ = cx + side * 2.45
        for t, ty in zip(sub, ys_): ax.plot([t[1], lx_], [t[2], ty], color="#BBBBBB", lw=0.4, zorder=0); ax.text(lx_ + side * 0.04, ty, t[3], ha="left" if side > 0 else "right", va="center", fontsize=10.5, fontstyle="italic", color=TC[t[4]])
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=Rs, fc="#FDF1E2", ec=CC["se"], lw=2, zorder=6)); ax.text(cx, cy, "SE", ha="center", va="center", fontweight="bold", color=TC["se"], zorder=7, fontsize=13)
    lx, y = 8.95, 4.75; ax.text(lx, y + 0.3, "Super-enhancer (SEdb, HEK293T)", fontweight="bold", color=TC["se"], va="center"); ax.add_patch(plt.Rectangle((lx, y - 0.08), 0.3, 0.16, color=RED, alpha=0.5, lw=0)); ax.text(lx + 0.38, y, f"on a Hi-C loop ({int(E.loop.sum())})", va="center"); y -= 0.3; ax.add_patch(plt.Rectangle((lx, y - 0.08), 0.3, 0.16, color=BLUE, alpha=0.35, lw=0)); ax.text(lx + 0.38, y, f"Cicero only ({int((~E.loop).sum())})", va="center"); y -= 0.38
    for c_, l_ in (("promoter", "promoter"), ("enhancer", "enhancer"), ("other", "other")):
        nn = int((E.cls == c_).sum()); ax.plot(lx + 0.15, y, "o", ms=9, color=CC[c_], mec="none"); ax.text(lx + 0.38, y, f"{l_} ({nn})", va="center"); y -= 0.3
    y -= 0.1; ax.text(lx, y, "Petals run clockwise in genomic order, from\nchr19:0.51 Mb at the top; a longer petal is a\nlonger genomic span.", va="top", color="#444444", linespacing=1.15, fontsize=11)
    for e in ("svg", "pdf"): fig.savefig(O / f"flower_all_edges.{e}", dpi=600, transparent=True)
    fig.savefig(O / "flower_all_edges.png", dpi=600, facecolor="white"); fig.savefig(O / "flower_all_edges_check.png", dpi=144, facecolor="white")
