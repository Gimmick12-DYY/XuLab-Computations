"""SE hub with edges coloured by what they connect (teal = promoter-promoter, amber = an enhancer end, the SE included; purple = the rest), solid = on a Hi-C loop, dashed = Cicero only; nodes neutral, SE orange. Closure of every hub edge across the ensemble against same-separation random pairs (partner-partner edges were not forced). 3D: structure --conf; closed edges only, loop-edges <= 400 kb as chromatin petals, others as chords. Flower: all edges, SE in the centre, partners on a ring in genomic order.
  render : python 10_typed_structure_flower.py --sim <traj dir> --conf N   (pymol-open-source)
  figure : python 10_typed_structure_flower.py --sim <traj dir> --conf N --figure"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--conf", type=int, required=True); ap.add_argument("--figure", action="store_true"); ap.add_argument("--size", type=int, default=3000); a = ap.parse_args()
R0 = Path(__file__).resolve().parents[1]; O = R0 / "figure_typed"; I = R0 / "model/inputs"; N = 910; L0 = 430000; files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
TY = dict(promoter="#0F8B8D", enhancer="#D08A00", other="#8A63B8"); TN = dict(promoter="promoter–promoter", enhancer="to an enhancer or the SE", other="other"); OTHER, INK, GHOST, NODE, SEC = "#BDBDBD", "#222222", "#D9D9D9", "#555555", "#E07B00"
H = pd.read_csv(R0 / "hub_edges_all.tsv", sep="\t"); B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"tfs": ""}); mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000; H["b1"] = [tb(mid(x)) for x in H.reg_a]; H["b2"] = [tb(mid(x)) for x in H.reg_b]; H["kb"] = (H.b1 - H.b2).abs() * 2; se_b = tb((1248039 + 1254347) // 2); H["label_a"] = H.label_a.str.replace("RNU6-2/TMEM259", "TMEM259"); H["label_b"] = H.label_b.str.replace("RNU6-2/TMEM259", "TMEM259"); tfset = lambda b: set(x for x in B.tfs[b].split(",") if x)
Pn = {}
for r in H.itertuples():
    for k, l, b in ((r.a, r.label_a, r.b1), (r.b, r.label_b, r.b2)):
        if str(k) != "SE": Pn[str(k)] = (l, b)
P = pd.DataFrame([dict(key=k, label=v[0], bead=v[1]) for k, v in Pn.items()]); P["keyn"] = P.key.astype(int); P = P.sort_values("keyn").reset_index(drop=True); t_, k_ = divmod(a.conf - 1, 50); X = np.load(files[t_])["X"][k_, :N].astype(float); H["closed"] = np.linalg.norm(X[H.b1] - X[H.b2], axis=1) < 2.0; H.to_csv(O / "hub_edges_in_structure.tsv", sep="\t", index=False)
if not (O / "edge_closure_typed.tsv").exists():
    rng = np.random.default_rng(37); nu = []; lp = H[H.on_loop == 1]
    for r in H.itertuples():
        d = abs(r.b1 - r.b2); c = rng.integers(0, N - d, 200); nu.append(np.c_[c, c + d])
    nu = np.array(nu); cl = np.zeros(len(H)); cn = np.zeros(len(H)); m = 0; ncl = []
    for f in files:
        Xs = np.load(f)["X"][:, :N]
        for Q in Xs: u = np.linalg.norm(Q[H.b1] - Q[H.b2], axis=1) < 2.0; cl += u; cn += (np.linalg.norm(Q[nu[..., 0]] - Q[nu[..., 1]], axis=2) < 2.0).mean(1); m += 1; ncl.append(u)
    H["closure"] = cl / m; H["null"] = cn / m; H.to_csv(O / "edge_closure_typed.tsv", sep="\t", index=False); ncl = np.array(ncl); pd.DataFrame(ncl[:, (H.from_se == 0) & (H.on_loop == 1)].sum(1), columns=["n_pp_loop_closed"]).to_csv(O / "pp_closed_per_structure.tsv", sep="\t", index=False)
    Sm = H.groupby(["from_se", "on_loop", "type"]).agg(n=("closure", "size"), closure=("closure", "mean"), null=("null", "mean")); Sm["ratio"] = Sm.closure / Sm.null; print(Sm.round(3).to_string())
print("closed in structure", H.closed.sum(), pd.crosstab([H.from_se, H.on_loop], H.closed).to_string())
if not a.figure:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    Y = (X - X[se_b]) * 10.0; cl = H[H.closed]; lo = cl[(cl.on_loop == 1) & (cl.kb <= 400)].copy(); lo = lo.sort_values("kb", ascending=False); colb = {}
    for r in lo.itertuples():
        for b in range(min(r.b1, r.b2), max(r.b1, r.b2) + 1): colb[b] = r.type
    idx = np.array(sorted(colb)); _, _, Vt = np.linalg.svd(Y[np.r_[P.bead.values]], full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T; cmd.reinitialize(); b0, b1 = P.bead.min(), P.bead.max() + 1; Z = gaussian_filter1d(Y, 2.5, axis=0, mode="nearest"); Pz = Z[b0:b1].copy()
    for b in np.r_[se_b, P.bead.values]: Pz[b - b0] = Y[b]
    Pd = np.array([np.interp(np.linspace(0, len(Pz) - 1, len(Pz) * 4), np.arange(len(Pz)), Pz[:, d]) for d in range(3)]).T; bo = b0 + np.minimum(np.round(np.linspace(0, len(Pz) - 1, len(Pd))).astype(int), len(Pz) - 1); g = rgb(GHOST); gh = [cgo.ALPHA, 0.22]; pe = []
    for u, v, b in zip(Pd[:-1], Pd[1:], bo[1:]):
        if b in colb: c = rgb(TY[colb[b]]); pe += [cgo.CYLINDER, *u, *v, 1.25, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 1.25]
        else: gh += [cgo.CYLINDER, *u, *v, 0.5, *g, *g]
    cmd.load_cgo(gh, "other"); cmd.load_cgo(pe, "petals"); ln = []
    for r in cl.itertuples():
        q0, q1 = Y[r.b1], Y[r.b2]; L = np.linalg.norm(q1 - q0); c = rgb(TY[r.type])
        if r.on_loop == 1 and r.kb <= 400: continue
        if r.on_loop == 1: ln += [cgo.CYLINDER, *q0, *q1, 0.8, *c, *c]
        else:
            u_ = (q1 - q0) / max(L, 1e-6)
            for s in np.arange(0, L, 2.2): ln += [cgo.CYLINDER, *(q0 + u_ * s), *(q0 + u_ * min(s + 1.1, L)), 0.5, *c, *c]
    if ln: cmd.load_cgo(ln, "chords")
    on = set(cl.a.astype(str)) | set(cl.b.astype(str)); sp = [cgo.COLOR, *rgb(SEC), cgo.SPHERE, *Y[se_b], 5.0]; sat = []; gold = np.pi * (3 - np.sqrt(5))
    def sats(b, rad):
        tf = tfset(b); tl = [x for x in TFS if x in tf] + ["other"] * min(len(tf - set(TFS)), 10); n_ = max(len(tl), 1); o = []
        for j, x in enumerate(tl):
            zz = 1 - 2 * (j + 0.5) / n_; rr = np.sqrt(1 - zz * zz); ph = j * gold; o += [cgo.COLOR, *rgb(COL.get(x, OTHER)), cgo.SPHERE, *(Y[b] + rad * np.array([rr * np.cos(ph), rr * np.sin(ph), zz])), 1.2 if x in COL else 0.7]
        return o
    for r in P.itertuples():
        act = str(r.key) in on; sp += [cgo.COLOR, *rgb(NODE if act else "#BBBBBB"), cgo.SPHERE, *Y[r.bead], 3.0 if act else 1.5]
        if act: sat += sats(r.bead, 4.4)
    sat += sats(se_b, 6.6); cmd.load_cgo(sp, "anchors"); cmd.load_cgo(sat, "tfs")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.005, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.5, specular=0.1, direct=0.5, reflect=0.25, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k, v)
    w = a.size; cmd.viewport(w // 4, w // 4); cmd.reset()
    for i, q in enumerate(Y[np.r_[idx[::3], P.bead.values, se_b]]): cmd.pseudoatom("frame", pos=[float(x) for x in q], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=8, complete=1); cmd.delete("frame"); view = cmd.get_view(); cmd.png(str(O / "render.png"), width=w, height=w, dpi=600, ray=1); cmd.save(str(O / "render.pse")); R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); ii = np.r_[se_b, P.bead.values]; vc = (Y[ii] - org) @ R_ + cam; hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=ii, px=w / 2 * (1 + vc[:, 0] / hh), py=w / 2 * (1 - vc[:, 1] / hh))).to_csv(O / "projection.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon, PathPatch
    from matplotlib.path import Path as MP
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42}); Mt = pd.read_csv(R0 / "figure/conformation_metrics.tsv", sep="\t"); me = Mt[Mt.conf == a.conf].iloc[0]; cn = H[H.closed]; lc = int(((H.from_se == 1) & (H.on_loop == 1) & H.closed).sum())
    W, H_ = 12.95, 5.75; fig = plt.figure(figsize=(W, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.set_axis_off(); ax.text(0.08, H_ - 0.2, f"One typical structure: {len(cn)} of {len(H)} hub edges closed", fontsize=14, fontweight="bold", va="center")
    im = Image.open(O / "render.png").convert("RGBA"); PJ = pd.read_csv(O / "projection.tsv", sep="\t"); aa = np.asarray(im)[..., 3]; ys, xs = np.where(aa > 25); c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.01); x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); PJ["px"] -= x0; PJ["py"] -= y0; w = im.size[0]; IW = 5.25; IX, IY = 1.75, 0.05; ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="lanczos", zorder=0); ax.set_xlim(0, W); ax.set_ylim(0, H_); PJ["x"] = IX + PJ.px / w * IW; PJ["y"] = IY + IW - PJ.py / w * IW; Q = PJ.iloc[1:].reset_index(drop=True); Q["label"] = P.label; Q["key"] = P.key.astype(str); on = set(cn.a.astype(str)) | set(cn.b.astype(str)); Q = Q[Q.key.isin(on)]; midx = IX + IW / 2
    for side, sub in ((-1, Q[Q.x < midx]), (1, Q[Q.x >= midx])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 0.27)
        if yy and yy[-1] > H_ - 0.42: yy = [v - (yy[-1] - (H_ - 0.42)) for v in yy]
        for i in range(len(yy) - 2, -1, -1): yy[i] = min(yy[i], yy[i + 1] - 0.27)
        lx = IX - 0.05 if side < 0 else IX + IW + 0.05
        for r, ty in zip(sub.itertuples(), yy): ax.plot([r.x, lx], [r.y, ty], color="#888888", lw=0.5, zorder=3); ax.text(lx + (-0.04 if side < 0 else 0.04), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", color=INK)
    lx, y = 8.75, 5.1; ax.text(lx, y + 0.27, "Loop colour = what it connects", fontweight="bold", va="center")
    for t in ("promoter", "enhancer", "other"): ax.plot([lx, lx + 0.25], [y, y], color=TY[t], lw=4, solid_capstyle="butt"); ax.text(lx + 0.33, y, TN[t], va="center"); y -= 0.27
    ax.plot([lx, lx + 0.25], [y, y], color="#555555", lw=1.8); ax.text(lx + 0.33, y, "on a Hi-C loop", va="center"); y -= 0.27; ax.plot([lx, lx + 0.25], [y, y], color="#555555", lw=1.5, ls=(0, (2, 1.5))); ax.text(lx + 0.33, y, "Cicero only", va="center"); y -= 0.3; ax.plot(lx + 0.12, y, "o", ms=11, color=SEC, mec="none"); ax.text(lx + 0.33, y, "super-enhancer (SEdb, HEK293T)", va="center"); y -= 0.27; ax.plot(lx + 0.12, y, "o", ms=8, color=NODE, mec="none"); ax.text(lx + 0.33, y, "partner, with a closed edge here", va="center"); y -= 0.27; ax.plot(lx + 0.12, y, "o", ms=4.5, color="#BBBBBB", mec="none"); ax.text(lx + 0.33, y, "partner, no closed edge here", va="center"); y -= 0.32; ax.text(lx, y, "TF spheres:", va="center"); y -= 0.25
    for k, (c_, l_) in enumerate([(COL[t], t) for t in TFS] + [(OTHER, "other")]):
        xx, yy_ = lx + 0.1 + (k % 3) * 1.25, y - (k // 3) * 0.23; ax.add_patch(Circle((xx, yy_), 0.07, color=c_, lw=0)); ax.text(xx + 0.12, yy_, l_, va="center", fontstyle="italic" if l_ in TFS else "normal", fontsize=11)
    y = y - 4 * 0.23 - 0.0; ax.text(lx, y, f"Structure #{a.conf}: Rg {me.rg_pct:.0f}th, crumpling {me.crumple_pct:.0f}th percentile,\n{lc} of 18 SE loop-edges closed (median). Open edges\nnot drawn. Only SE loop-edges were forced on.", va="top", color="#444444", linespacing=1.08, fontsize=10.5)
    for e in ("svg", "pdf"): fig.savefig(O / f"structure_3d_typed.{e}", dpi=600, transparent=True)
    fig.savefig(O / "structure_3d_typed.png", dpi=600, facecolor="white"); fig.savefig(O / "structure_3d_typed_check.png", dpi=144, facecolor="white"); plt.close(fig)
    # network flower
    fig = plt.figure(figsize=(W, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.set_axis_off(); ax.text(0.08, H_ - 0.2, "All Cicero edges in the super-enhancer hub", fontsize=14, fontweight="bold", va="center"); cx, cy, Rg = 4.7, 2.72, 2.05; n = len(P); ang = {str(r.key): np.pi / 2 - 2 * np.pi * (i + 0.5) / n for i, r in P.iterrows()}; pt = lambda k: np.array([cx + Rg * np.cos(ang[k]), cy + Rg * np.sin(ang[k])])
    for r in H.sort_values(["on_loop", "from_se"]).itertuples():
        c_ = TY[r.type]; ls = "-" if r.on_loop == 1 else (0, (2.5, 2)); al = 0.95 if r.on_loop == 1 else 0.55; lw = 1.5 if r.from_se else 0.9 if r.on_loop == 1 else 0.6
        if r.from_se: k = str(r.b if r.a == "SE" else r.a); q = pt(k); ax.plot([cx, q[0]], [cy, q[1]], color=c_, lw=lw, ls=ls, alpha=al, zorder=2 + r.on_loop)
        else: q0, q1 = pt(str(r.a)), pt(str(r.b)); ctr = np.array([cx, cy]) + 0.35 * ((q0 + q1) / 2 - np.array([cx, cy])); ax.add_patch(PathPatch(MP([q0, ctr, q1], [MP.MOVETO, MP.CURVE3, MP.CURVE3]), fc="none", ec=c_, lw=lw, ls=ls, alpha=al, zorder=1 + r.on_loop))
    tips = []
    for r in P.itertuples():
        q = pt(str(r.key)); ax.plot(*q, "o", ms=7, color=NODE, mec="white", mew=0.5, zorder=6); tips.append((q[0], q[1], r.label))
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=0.34, fc="#FDF1E2", ec=SEC, lw=2, zorder=7)); ax.text(cx, cy, "SE", ha="center", va="center", fontweight="bold", color="#B85F00", zorder=8, fontsize=13)
    for side, sub in ((1, [t for t in tips if t[0] >= cx]), (-1, [t for t in tips if t[0] < cx])):
        sub = sorted(sub, key=lambda t: t[1]); m_ = len(sub); ys_ = np.linspace(cy - 2.45, cy + 2.45, m_); lx_ = cx + side * 2.6
        for t, ty in zip(sub, ys_): ax.plot([t[0], lx_], [t[1], ty], color="#BBBBBB", lw=0.4, zorder=0); ax.text(lx_ + side * 0.04, ty, t[2], ha="left" if side > 0 else "right", va="center", fontsize=10.5, fontstyle="italic", color=INK)
    lx, y = 9.0, 4.9; ax.text(lx, y + 0.3, "Loop colour = what it connects", fontweight="bold", va="center")
    for t in ("promoter", "enhancer", "other"):
        m = H.type == t; ax.plot([lx, lx + 0.28], [y, y], color=TY[t], lw=3); ax.text(lx + 0.38, y, f"{TN[t]}: {int((m & (H.on_loop == 1)).sum())} + {int((m & (H.on_loop == 0)).sum())}", va="center"); y -= 0.3
    y -= 0.08; ax.plot([lx, lx + 0.28], [y, y], color="#555555", lw=1.6); ax.text(lx + 0.38, y, f"on a Hi-C loop ({int((H.on_loop == 1).sum())})", va="center"); y -= 0.3; ax.plot([lx, lx + 0.28], [y, y], color="#555555", lw=1.2, ls=(0, (2.5, 2))); ax.text(lx + 0.38, y, f"Cicero only ({int((H.on_loop == 0).sum())})", va="center"); y -= 0.42; ax.text(lx, y, "Counts: on a Hi-C loop + Cicero only.\nSpokes: 31 edges from the super-enhancer.\nInner curves: 104 partner–partner edges.\nPartners run clockwise in genomic order\nfrom chr19:0.51 Mb at the top.", va="top", color="#444444", linespacing=1.15, fontsize=11)
    for e in ("svg", "pdf"): fig.savefig(O / f"flower_typed.{e}", dpi=600, transparent=True)
    fig.savefig(O / "flower_typed.png", dpi=600, facecolor="white"); fig.savefig(O / "flower_typed_check.png", dpi=144, facecolor="white")
