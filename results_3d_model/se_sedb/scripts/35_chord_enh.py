"""SE hub with edges coloured by what they connect (teal = promoter-promoter, amber = an enhancer end, the SE included; purple = the rest), solid = on a Hi-C loop, dashed = Cicero only; nodes neutral, SE orange. Closure of every hub edge across the ensemble against same-separation random pairs (partner-partner edges were not forced). 3D: structure --conf; closed edges only, loop-edges <= 400 kb as chromatin petals, others as chords. Flower: all edges, SE in the centre, partners on a ring in genomic order.
  render : python 10_typed_structure_flower.py --sim <traj dir> --conf N   (pymol-open-source)
  figure : python 10_typed_structure_flower.py --sim <traj dir> --conf N --figure"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--conf", type=int, required=True); ap.add_argument("--figure", action="store_true"); ap.add_argument("--maxp", type=int, default=150); ap.add_argument("--size", type=int, default=3000); a = ap.parse_args()
R0 = Path(__file__).resolve().parents[1]; O = R0 / "figure_chord_enh"; I = R0 / "model/inputs"; N = 910; L0 = 430000; MAXP = a.maxp; files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
TY = dict(promoter="#0072B2", enhancer="#D55E00", other="#BBBBBB"); TN = dict(promoter="promoter–promoter", enhancer="to an enhancer or the SE", other="other"); OTHER, INK, GHOST, NODE, SEC = "#BDBDBD", "#222222", "#D9D9D9", "#6A6A6A", "#262626"
H = pd.read_csv(R0 / "hub_edges_all.tsv", sep="\t"); B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"tfs": ""}); mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000; H["b1"] = [tb(mid(x)) for x in H.reg_a]; H["b2"] = [tb(mid(x)) for x in H.reg_b]; H["kb"] = (H.b1 - H.b2).abs() * 2; EC = dict(se_promoter="#0072B2", se_enhancer="#D55E00", se_other="#999999", pp_promoter="#56B4E9", pp_enhancer="#F0A35E", pp_other="#D0D0D0"); EN = dict(se_promoter="promoter", se_enhancer="enhancer", se_other="other", pp_promoter="promoter\u2013promoter", pp_enhancer="involving an enhancer", pp_other="other"); H["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(H.cls_a, H.cls_b, H.from_se, H.type)]; se_b = tb((1248039 + 1254347) // 2); H["label_a"] = H.label_a.str.replace("RNU6-2/TMEM259", "TMEM259"); H["label_b"] = H.label_b.str.replace("RNU6-2/TMEM259", "TMEM259"); tfset = lambda b: set(x for x in B.tfs[b].split(",") if x)
Pn = {}
for r in H.itertuples():
    for k, l, b in ((r.a, r.label_a, r.b1), (r.b, r.label_b, r.b2)):
        if str(k) != "SE": Pn[str(k)] = (l, b)
P = pd.DataFrame([dict(key=k, label=v[0], bead=v[1]) for k, v in Pn.items()]); P["keyn"] = P.key.astype(int); P = P.sort_values("keyn").reset_index(drop=True); Tx = pd.read_csv(R0 / "edges_exact_se.tsv", sep="\t"); Tx["bin"] = [mid(x) // 10000 for x in Tx.partner]; CL = dict(zip(Tx.bin, Tx.cls)); EM = pd.read_csv(O / "enhancer_map.tsv", sep="\t").set_index("bin_10kb").name.to_dict(); P["cls"] = [CL[k] for k in P.keyn]; P["label"] = [EM.get(k, l) for k, l in zip(P.keyn, P.label)]; t_, k_ = divmod(a.conf - 1, 50); X = np.load(files[t_])["X"][k_, :N].astype(float); H["closed"] = np.linalg.norm(X[H.b1] - X[H.b2], axis=1) < 2.0; H.to_csv(O / "hub_edges_in_structure.tsv", sep="\t", index=False)
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
    Y = (X - X[se_b]) * 10.0; cl = H[H.closed]; sl = cl[(cl.from_se == 1) & (cl.on_loop == 1)]; ends = sorted({int(r.b1 if str(r.a) != "SE" else r.b2) for r in sl.itertuples()}); ch = sorted(set(ends + [se_b])); idx = np.arange(ch[0], ch[-1] + 1); Z = gaussian_filter1d(Y, 6.0, axis=0, mode="nearest"); P0 = Z[idx].copy(); w_ = np.array([min(1.0, np.abs(idx[i] - np.array(ch)).min() / 8) for i in range(len(idx))])[:, None]; P0 = w_ * P0 + (1 - w_) * Y[idx]
    Vt = np.array(json.load(open(O / "flower_view.json"))["R"])
    Y = Y @ Vt.T; P0 = P0 @ Vt.T; cmd.reinitialize(); pe = []; pcl = {int(r.b1 if str(r.a) != "SE" else r.b2): (r.cls_a if str(r.a) != "SE" else r.cls_b) for r in sl.itertuples()}; g = rgb(GHOST); gl = [cgo.ALPHA, 0.3]; Keep = []
    for k_, (p_, q_) in enumerate(zip(ch[:-1], ch[1:])):
        seg = P0[p_ - idx[0]:q_ - idx[0] + 1]
        if len(seg) < 7: continue
        n_k = max(5, len(seg) // 8); kn = seg[np.linspace(0, len(seg) - 1, n_k).round().astype(int)]; tck, _ = splprep(kn.T, s=0, k=3); fine = np.array(splev(np.linspace(0, 1, len(seg) * 6), tck)).T
        if (q_ - p_) * 2 > MAXP:
            for u, v in zip(fine[:-1], fine[1:]): gl += [cgo.CYLINDER, *u, *v, 0.35, *g, *g]
            continue
        c = rgb(EC["se_" + pcl[p_ if q_ <= se_b else q_]]); Keep.append(fine)
        for u, v in zip(fine[:-1], fine[1:]): pe += [cgo.CYLINDER, *u, *v, 1.0, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 1.0]
    print("petals", len(Keep), "of", len(ch) - 1); P0 = np.vstack(Keep); cmd.load_cgo(gl, "long_gaps")
    cmd.load_cgo(pe, "petals"); g = rgb(GHOST); gh = [cgo.ALPHA, 0.18]; Zr = Z @ Vt.T
    for b in list(range(P.bead.min(), ch[0])) + list(range(ch[-1], P.bead.max())): gh += [cgo.CYLINDER, *Zr[b], *Zr[b + 1], 0.4, *g, *g]
    ln = []
    for r in cl[(cl.from_se == 0) | (cl.on_loop == 0)].sort_values("type", key=lambda v: v.map(dict(other=0, enhancer=1, promoter=2))).itertuples():
        q0, q1 = Y[r.b1], Y[r.b2]; L = np.linalg.norm(q1 - q0); c = rgb(EC[r.ek]); rad = 1.3 if r.type == "promoter" else 0.8
        if r.on_loop == 1: ln += [cgo.CYLINDER, *q0, *q1, rad, *c, *c]
        else:
            u_ = (q1 - q0) / max(L, 1e-6)
            for s in np.arange(0, L, 2.4): ln += [cgo.CYLINDER, *(q0 + u_ * s), *(q0 + u_ * min(s + 1.2, L)), 0.55, *c, *c]
    if ln: cmd.load_cgo(ln, "chords")
    on = {int(r.b1 if str(r.a) != "SE" else r.b2) for r in cl[cl.from_se == 1].itertuples()} | set(cl[cl.from_se == 0].b1) | set(cl[cl.from_se == 0].b2); sp = [cgo.COLOR, *rgb(SEC), cgo.SPHERE, *Y[se_b], 5.0]; sat = []; gold = np.pi * (3 - np.sqrt(5))
    def sats(b, rad):
        tf = tfset(b); tl = [x for x in TFS if x in tf] + ["other"] * min(len(tf - set(TFS)), 6); n_ = max(len(tl), 1); o = []
        for j, x in enumerate(tl):
            zz = 1 - 2 * (j + 0.5) / n_; rr = np.sqrt(1 - zz * zz); ph = j * gold; o += [cgo.COLOR, *rgb(COL.get(x, OTHER)), cgo.SPHERE, *(Y[b] + rad * np.array([rr * np.cos(ph), rr * np.sin(ph), zz])), 1.1 if x in COL else 0.6]
        return o
    for r in P.itertuples():
        act = r.bead in on
        if act: sp += [cgo.COLOR, *rgb(NODE), cgo.SPHERE, *Y[r.bead], 2.6]
        if r.bead in ends: sat += sats(r.bead, 4.0)
    sat += sats(se_b, 6.4); cmd.load_cgo(sp, "anchors"); cmd.load_cgo(sat, "tfs")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.005, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.5, specular=0.1, direct=0.5, reflect=0.25, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k, v)
    w = a.size; cmd.viewport(w // 4, w // 4); cmd.reset()
    rmax = np.linalg.norm(P0[:, :2], axis=1).max() * 1.1
    for i, q in enumerate(((rmax, rmax, 0), (-rmax, -rmax, 0), (rmax, -rmax, 0), (-rmax, rmax, 0))): cmd.pseudoatom("frame", pos=[float(x) for x in q], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=0, complete=1); cmd.delete("frame"); view = cmd.get_view(); cmd.png(str(O / "render.png"), width=w, height=w, dpi=600, ray=1); cmd.save(str(O / "render.pse")); R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); ii = np.r_[se_b, P.bead.values]; vc = (Y[ii] - org) @ R_ + cam; hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=ii, px=w / 2 * (1 + vc[:, 0] / hh), py=w / 2 * (1 - vc[:, 1] / hh))).to_csv(O / "projection.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon, PathPatch
    from matplotlib.path import Path as MP
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42}); Mt = pd.read_csv(R0 / "figure/conformation_metrics.tsv", sep="\t"); me = Mt[Mt.conf == a.conf].iloc[0]; cn = H[H.closed]; lc = int(((H.from_se == 1) & (H.on_loop == 1) & H.closed).sum())
    W, H_ = 12.95, 5.75; fig = plt.figure(figsize=(W, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.set_axis_off(); ax.text(0.08, H_ - 0.2, f"Most-closed structure (top {(Mt.n_loop_closed >= lc).mean() * 100:.2f}% of 10,000; typical = 7 of 18): {lc} of 18 SE loop-edges closed", fontsize=14, fontweight="bold", va="center")
    im = Image.open(O / "render.png").convert("RGBA"); PJ = pd.read_csv(O / "projection.tsv", sep="\t"); aa = np.asarray(im)[..., 3]; ys, xs = np.where(aa > 25); c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.01); x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); PJ["px"] -= x0; PJ["py"] -= y0; w = im.size[0]; IW = 5.25; IX, IY = 1.75, 0.05; ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="lanczos", zorder=0); ax.set_xlim(0, W); ax.set_ylim(0, H_); PJ["x"] = IX + PJ.px / w * IW; PJ["y"] = IY + IW - PJ.py / w * IW; Q = PJ.iloc[1:].reset_index(drop=True); Q["label"] = P.label; Q["key"] = P.key.astype(str); on = set(cn.a.astype(str)) | set(cn.b.astype(str)); Q = Q[Q.key.isin(on)]; midx = IX + IW / 2
    for side, sub in ((-1, Q[Q.x < midx]), (1, Q[Q.x >= midx])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 0.27)
        if yy and yy[-1] > H_ - 0.42: yy = [v - (yy[-1] - (H_ - 0.42)) for v in yy]
        for i in range(len(yy) - 2, -1, -1): yy[i] = min(yy[i], yy[i + 1] - 0.27)
        lx = IX - 0.05 if side < 0 else IX + IW + 0.05
        for r, ty in zip(sub.itertuples(), yy): ax.plot([r.x, lx], [r.y, ty], color="#888888", lw=0.5, zorder=3); ax.text(lx + (-0.04 if side < 0 else 0.04), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic", color=INK)
    lx, y = 8.75, 5.27
    for hd, keys, lw_ in (("From the super-enhancer:", ("se_promoter", "se_enhancer", "se_other"), 4), ("Between partners:", ("pp_promoter", "pp_enhancer", "pp_other"), 2.5)):
        ax.text(lx, y, hd, fontweight="bold", va="center"); y -= 0.25
        for k in keys: ax.plot([lx, lx + 0.25], [y, y], color=EC[k], lw=lw_, solid_capstyle="butt"); ax.text(lx + 0.33, y, EN[k], va="center"); y -= 0.25
    ax.plot([lx, lx + 0.25], [y, y], color="#555555", lw=1.8); ax.text(lx + 0.33, y, "on a Hi-C loop", va="center"); y -= 0.25; ax.plot([lx, lx + 0.25], [y, y], color="#555555", lw=1.5, ls=(0, (2, 1.5))); ax.text(lx + 0.33, y, "Cicero only", va="center"); y -= 0.26; ax.plot(lx + 0.12, y, "o", ms=11, color=SEC, mec="none"); ax.text(lx + 0.33, y, "super-enhancer (SEdb, HEK293T)", va="center"); y -= 0.27; ax.plot(lx + 0.12, y, "o", ms=8, color=NODE, mec="none"); ax.text(lx + 0.33, y, "partner, closed edge here", va="center"); y -= 0.27; y -= 0.05; ax.text(lx, y, "TF spheres:", va="center"); y -= 0.25
    for k, (c_, l_) in enumerate([(COL[t], t) for t in TFS] + [(OTHER, "other")]):
        xx, yy_ = lx + 0.1 + (k % 3) * 1.25, y - (k // 3) * 0.23; ax.add_patch(Circle((xx, yy_), 0.07, color=c_, lw=0)); ax.text(xx + 0.12, yy_, l_, va="center", fontstyle="italic" if l_ in TFS else "normal", fontsize=11)
    y = y - 4 * 0.23 - 0.0; ax.text(lx, y, f"Structure #{a.conf}: {len(cn)} of {len(H)} hub edges closed; Rg {me.rg_pct:.0f}th,\ncrumpling {me.crumple_pct:.0f}th percentile. A selected extreme, not typical.\nThick chain = closed SE loops, up to 150 kb between anchors,\ncoloured by the far partner; faint = longer stretches;\nstraight lines = other closed edges.\nCamera picked to spread the petals; coordinates unchanged.", va="top", color="#444444", linespacing=1.05, fontsize=10)
    for e in ("svg", "pdf"): fig.savefig(O / f"structure_3d_final.{e}", dpi=600, transparent=True)
    fig.savefig(O / "structure_3d_final.png", dpi=600, facecolor="white"); fig.savefig(O / "structure_3d_final_check.png", dpi=144, facecolor="white"); plt.close(fig)
    # network flower
    fig = plt.figure(figsize=(W, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.set_axis_off(); ax.text(0.08, H_ - 0.2, "All Cicero edges in the super-enhancer hub", fontsize=14, fontweight="bold", va="center"); cx, cy, Rg = 4.7, 2.72, 2.05; n = len(P); ang = {str(r.key): np.pi / 2 - 2 * np.pi * (i + 0.5) / n for i, r in P.iterrows()}; pt = lambda k: np.array([cx + Rg * np.cos(ang[k]), cy + Rg * np.sin(ang[k])])
    for r in H.assign(o_=H.type.map(dict(other=0, enhancer=1, promoter=2))).sort_values(["from_se", "on_loop", "o_"]).itertuples():
        c_ = EC[r.ek]; ls = "-" if r.on_loop == 1 else (0, (2.5, 2)); al = 1 if r.on_loop == 1 else 0.8 if r.from_se else 0.55; lw = 2.6 if (r.from_se and r.on_loop == 1) else 1.2 if r.on_loop == 1 else 1.1 if r.from_se else 0.5
        if r.from_se: k = str(r.b if r.a == "SE" else r.a); q = pt(k); ax.plot([cx, q[0]], [cy, q[1]], color=c_, lw=lw, ls=ls, alpha=al, zorder=3 + r.on_loop)
        else: q0, q1 = pt(str(r.a)), pt(str(r.b)); ctr = np.array([cx, cy]) + 0.35 * ((q0 + q1) / 2 - np.array([cx, cy])); ax.add_patch(PathPatch(MP([q0, ctr, q1], [MP.MOVETO, MP.CURVE3, MP.CURVE3]), fc="none", ec=c_, lw=lw, ls=ls, alpha=al, zorder=1 + r.on_loop + 0.1 * r.o_))
    tips = []
    for r in P.itertuples():
        q = pt(str(r.key)); ax.plot(*q, "o", ms=7.5, color=dict(promoter="#0072B2", enhancer="#D55E00").get(r.cls, "#8C8C8C"), mec="white", mew=0.5, zorder=6)
        if r.cls != "other": tips.append((q[0], q[1], r.label, r.cls))
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=0.34, fc="#FFC20A", ec="#5A4500", lw=1.6, zorder=7)); ax.text(cx, cy, "SE", ha="center", va="center", fontweight="bold", color="#2B2100", zorder=8, fontsize=13)
    for side, sub in ((1, [t for t in tips if t[0] >= cx]), (-1, [t for t in tips if t[0] < cx])):
        sub = sorted(sub, key=lambda t: t[1]); ys_ = [t[1] for t in sub]; lx_ = cx + side * 2.4
        for _ in range(100):
            for i_ in range(1, len(ys_)):
                if ys_[i_] < ys_[i_ - 1] + 0.3: d_ = (ys_[i_ - 1] + 0.3 - ys_[i_]) / 2; ys_[i_] += d_; ys_[i_ - 1] -= d_
        if side > 0: ys_[0] -= 0.25   # keeps the POLR2E leader clear of the unlabelled dot beside it
        for t, ty in zip(sub, ys_): ax.plot([t[0], lx_], [t[1], ty], color="#BBBBBB", lw=0.4, zorder=0); ax.text(lx_ + side * 0.04, ty, t[2], ha="left" if side > 0 else "right", va="center", fontsize=10.5, fontstyle="italic", color="#D55E00" if t[3] == "enhancer" else INK)
    lx, y = 9.0, 5.2; ax.text(lx + 3.0, y, "loop + Cicero", va="center", ha="right", fontsize=11, color="#444444")
    for hd, keys, lw_ in (("From the super-enhancer", ("se_promoter", "se_enhancer", "se_other"), 3.5), ("Between partners", ("pp_promoter", "pp_enhancer", "pp_other"), 2.2)):
        ax.text(lx, y, hd, fontweight="bold", va="center"); y -= 0.29
        for k in keys:
            m = H.ek == k; ax.plot([lx, lx + 0.28], [y, y], color=EC[k], lw=lw_, solid_capstyle="butt"); ax.text(lx + 0.38, y, EN[k], va="center"); ax.text(lx + 3.0, y, f"{int((m & (H.on_loop == 1)).sum())} + {int((m & (H.on_loop == 0)).sum())}", va="center", ha="right"); y -= 0.29
        y -= 0.08
    ax.plot([lx, lx + 0.28], [y, y], color="#555555", lw=1.6); ax.text(lx + 0.38, y, f"on a Hi-C loop ({int((H.on_loop == 1).sum())})", va="center"); y -= 0.29; ax.plot([lx, lx + 0.28], [y, y], color="#555555", lw=1.2, ls=(0, (2.5, 2))); ax.text(lx + 0.38, y, f"Cicero only ({int((H.on_loop == 0).sum())})", va="center"); y -= 0.33; ax.plot(lx + 0.14, y, "o", ms=8, color="#0072B2", mec="none"); ax.text(lx + 0.38, y, "promoter", va="center"); ax.plot(lx + 1.44, y, "o", ms=8, color="#D55E00", mec="none"); ax.text(lx + 1.68, y, "enhancer", va="center"); y -= 0.29; ax.plot(lx + 0.14, y, "o", ms=8, color="#8C8C8C", mec="none"); ax.text(lx + 0.38, y, "other site (not labelled)", va="center"); y -= 0.3; ax.text(lx, y, "Partners run clockwise in genomic order\nfrom chr19:0.51 Mb at the top.", va="top", color="#444444", linespacing=1.15, fontsize=11)
    for e in ("svg", "pdf"): fig.savefig(O / f"flower_final.{e}", dpi=600, transparent=True)
    fig.savefig(O / "flower_final.png", dpi=600, facecolor="white"); fig.savefig(O / "flower_final_check.png", dpi=144, facecolor="white")
