"""SE hub in the style of the chr16 simple rosette (tf_polymer_hicfirst/scripts/15_simple_figure.py): one Hi-C-constrained structure; shows only the largest set of hub elements linked within 2 sigma that contains the SE, and the chain between consecutive linked elements when <= 200 kb apart (splprep s = 0.08 n; tube radius 2, spheres as chr16). Tube colour = class of the element at the far end from the SE (from the SE: blue promoter, vermillion enhancer, grey other). Closed hub edges among linked elements as chords (lighter shades between partners, dashed = Cicero only). Structure: among those closing the maximum 14 of 18 SE loop-edges, the largest linked set.
  pick   : python 21_se_rosette.py --sim <traj dir> --pick
  render : python 21_se_rosette.py --sim <traj dir> --conf N   (pymol-open-source)
  figure : python 21_se_rosette.py --sim <traj dir> --conf N --figure"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--conf", type=int, default=0); ap.add_argument("--pick", action="store_true"); ap.add_argument("--figure", action="store_true"); ap.add_argument("--size", type=int, default=3000); a = ap.parse_args()
R0 = Path(__file__).resolve().parents[1]; O = R0 / "figure_rosette_enh"; N = 910; L0 = 430000; S = 10.0; files = sorted(glob.glob(a.sim + "/traj*.npz")); H = pd.read_csv(R0 / "hub_edges_all.tsv", sep="\t"); mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000; H["b1"] = [tb(mid(x)) for x in H.reg_a]; H["b2"] = [tb(mid(x)) for x in H.reg_b]; se_b = tb((1248039 + 1254347) // 2)
EC = dict(se_promoter="#0072B2", se_enhancer="#D55E00", se_other="#999999", pp_promoter="#56B4E9", pp_enhancer="#F0A35E", pp_other="#D0D0D0"); H["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(H.cls_a, H.cls_b, H.from_se, H.type)]; H["label_a"] = H.label_a.str.replace("RNU6-2/TMEM259", "TMEM259"); H["label_b"] = H.label_b.str.replace("RNU6-2/TMEM259", "TMEM259")
Kb = {"SE": se_b}; El = {se_b: ("super-enhancer", "se")}
for r in H.itertuples():
    for k_, b_, l_, c_ in ((str(r.a), r.b1, r.label_a, r.cls_a), (str(r.b), r.b2, r.label_b, r.cls_b)):
        if k_ not in Kb: Kb[k_] = int(b_); El[int(b_)] = (l_, c_)
H["k1"] = [Kb[str(x)] for x in H.a]; H["k2"] = [Kb[str(x)] for x in H.b]
EM = pd.read_csv(O / "enhancer_map.tsv", sep="\t").set_index("old_label").name.to_dict(); El = {b: (EM.get(l, l), c) for b, (l, c) in El.items()}
hb = np.array(sorted(El)); M = pd.read_csv(R0 / "figure/conformation_metrics.tsv", sep="\t")
def load(c):
    t, k = divmod(int(c) - 1, 50); return np.load(files[t])["X"][k, :N].astype(float)
def comp(P):
    pr = cKDTree(P[hb]).query_pairs(2.0, output_type="ndarray"); _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(hb), len(hb))), directed=False); return np.sort(hb[lab == lab[np.where(hb == se_b)[0][0]]])
if a.pick:
    rows = []
    for c in M[M.n_loop_closed == 14].conf:
        P = load(c); m_ = comp(P); gaps = [q - p for p, q in zip(m_[:-1], m_[1:])]; sh = sum(g * 2 <= 200 for g in gaps); e = M[M.conf == c].iloc[0]; rows.append(dict(conf=int(c), linked=len(m_), petals=int(sh), overlap=int(e.overlap), rg_pct=e.rg_pct, crumple_pct=e.crumple_pct))
    T = pd.DataFrame(rows).sort_values(["linked", "petals", "overlap"], ascending=[False, False, True]); T.to_csv(O / "rosette_candidates.tsv", sep="\t", index=False); print(T.round(1).to_string(index=False)); raise SystemExit
P = load(a.conf); mem = comp(P); loops = [(int(p), int(q)) for p, q in zip(mem[:-1], mem[1:]) if q - p <= 100]; show = np.zeros(N, bool)
for p, q in loops: show[p:q + 1] = True
show[mem] = True; H["closed"] = np.linalg.norm(P[H.b1] - P[H.b2], axis=1) < 2.0; ms = set(mem.tolist()); C = H[H.closed & H.k1.isin(ms) & H.k2.isin(ms)]
if not a.figure:
    cen = P[mem].mean(0); Y = (P - cen) * S; _, _, Vt = np.linalg.svd(Y[show] - Y[show].mean(0), full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    cmd.reinitialize()
    for k, (p, q) in enumerate(loops):
        seg = np.arange(p, q + 1); pts = Y[seg]; far = p if q <= se_b else q; cl = El[far][1]; c = rgb("#B2182B")
        if len(seg) >= 4: tck, _ = splprep(pts.T, s=len(seg) * 0.08, k=3); pts = np.array(splev(np.linspace(0, 1, len(seg) * 8), tck)).T
        o = []
        for u, v in zip(pts[:-1], pts[1:]): o += [cgo.CYLINDER, *u, *v, 1.4, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 1.4]
        cmd.load_cgo(o, f"loop{k}")
    ch = []
    for r in C.itertuples():
        c = rgb(EC[r.ek]); q0, q1 = Y[r.k1], Y[r.k2]; L = np.linalg.norm(q1 - q0)
        if r.on_loop: ch += [cgo.CYLINDER, *q0, *q1, 0.9, *c, *c]
        else:
            u_ = (q1 - q0) / max(L, 1e-6)
            for s in np.arange(0, L, 2.2): ch += [cgo.CYLINDER, *(q0 + u_ * s), *(q0 + u_ * min(s + 1.1, L)), 0.6, *c, *c]
    # edge chords are not drawn: the legend no longer has edge colours
    sp = []
    for b in mem:
        if b != se_b: sp += [cgo.COLOR, *rgb("#D55E00" if El[b][1] == "enhancer" else "#222222"), cgo.SPHERE, *Y[b], 4.6]
    cmd.load_cgo(sp, "nodes")
    for k_, v in dict(ray_trace_mode=1, ray_trace_gain=0.006, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.45, specular=0.1, direct=0.55, reflect=0.3, orthoscopic=1, antialias=2, cgo_lighting=1, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k_, v)
    w = a.size; cmd.viewport(w // 4, w // 4); cmd.reset()
    for i, q in enumerate(Y[show]): cmd.pseudoatom("frame", pos=[float(x) for x in q], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=3, complete=1); cmd.delete("frame"); view = cmd.get_view(); cmd.set_color("segold", rgb("#FFC20A")); cmd.pseudoatom("se_gold", pos=[float(x) for x in Y[se_b]], vdw=7.0, color="segold"); cmd.show_as("spheres", "se_gold"); cmd.set("sphere_scale", 1.0, "se_gold"); cmd.set("specular", 1.0, "se_gold"); cmd.set("spec_power", 200, "se_gold"); cmd.set("spec_reflect", 1.5, "se_gold"); cmd.set("spec_direct", 1.0, "se_gold"); cmd.set_view(view); cmd.png(str(O / "render.png"), width=w, height=w, dpi=600, ray=1); cmd.save(str(O / "render.pse")); R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); vc = (Y - org) @ R_ + cam; hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=mem, label=[El[b][0] for b in mem], cls=[El[b][1] for b in mem], px=(w / 2 * (1 + vc[:, 0] / hh))[mem], py=(w / 2 * (1 - vc[:, 1] / hh))[mem])).to_csv(O / "rosette_nodes.tsv", sep="\t", index=False); print("linked", len(mem), "segments", len(loops), "edges", len(C))
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42}); me = M[M.conf == a.conf].iloc[0]; lc = int(me.n_loop_closed); W, H_ = 12.95, 5.75; fig = plt.figure(figsize=(W, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.set_axis_off(); ax.text(0.08, H_ - 0.2, f"Most-closed structure (top {(M.n_loop_closed >= lc).mean() * 100:.2f}% of 10,000; typical = 7 of 18): {lc} of 18 SE loop-edges closed", fontsize=14, fontweight="bold", va="center")
    im = Image.open(O / "render.png").convert("RGBA"); T = pd.read_csv(O / "rosette_nodes.tsv", sep="\t"); aa = np.asarray(im)[..., 3]; ys, xs = np.where(aa > 5); c0, c1 = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; hf = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.02); x0, y0 = max(0, c0 - hf), max(0, c1 - hf); im = im.crop((x0, y0, x0 + 2 * hf, y0 + 2 * hf)); T["px"] -= x0; T["py"] -= y0; w = im.size[0]; IW = 5.3; IX, IY = 1.8, 0.02; ax.imshow(np.asarray(im), extent=(IX, IX + IW, IY, IY + IW), interpolation="lanczos", zorder=0); ax.set_xlim(0, W); ax.set_ylim(0, H_); T["x"] = IX + T.px / w * IW; T["y"] = IY + IW - T.py / w * IW; midx = IX + IW / 2; TC = dict(se="#8A6A00", promoter="#0072B2", enhancer="#D55E00", other="#666666")
    for side, sub in ((-1, T[T.x < midx]), (1, T[T.x >= midx])):
        sub = sub.sort_values("y"); yy = list(sub.y.values)
        for i in range(1, len(yy)): yy[i] = max(yy[i], yy[i - 1] + 0.3)
        if yy and yy[-1] > H_ - 0.45: yy = [v - (yy[-1] - (H_ - 0.45)) for v in yy]
        for i in range(len(yy) - 2, -1, -1): yy[i] = min(yy[i], yy[i + 1] - 0.3)
        lx = IX - 0.03 if side < 0 else IX + IW + 0.03
        for r, ty in zip(sub.itertuples(), yy): ax.plot([r.x, lx], [r.y, ty], color="#444444", lw=0.6, zorder=3); ax.text(lx + (-0.04 if side < 0 else 0.04), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="normal" if r.cls == "se" else "italic", fontweight="bold" if r.cls == "se" else "normal", color=TC[r.cls])
    lx, y = 9.2, 4.9
    ax.plot([lx, lx + 0.27], [y, y], color="#B2182B", lw=5, solid_capstyle="round"); ax.text(lx + 0.36, y, "chromatin", va="center"); y -= 0.32
    ax.plot(lx + 0.13, y, "o", ms=12, color="#FFC20A", mec="#5A4500", mew=0.8); ax.text(lx + 0.36, y, "super-enhancer (SEdb, HEK293T)", va="center"); y -= 0.32
    ax.plot(lx + 0.13, y, "o", ms=9, color="#222222", mec="none"); ax.text(lx + 0.36, y, "promoter (blue label)", va="center"); y -= 0.32
    ax.plot(lx + 0.13, y, "o", ms=9, color="#D55E00", mec="none"); ax.text(lx + 0.36, y, "enhancer", va="center"); y -= 0.32
    ax.plot(lx + 0.13, y, "o", ms=9, color="#222222", mec="none"); ax.text(lx + 0.36, y, "other site (grey label)", va="center"); y -= 0.42
    ax.text(lx, y, f"Structure #{a.conf}: {len(T) - 1} of 31 partners linked into one core;\nRg {me.rg_pct:.0f}th, crumpling {me.crumple_pct:.0f}th percentile. A selected extreme,\nnot typical. Tube = chromatin between linked\nelements up to 200 kb. Enhancers are numbered\nas on the next slide.", va="top", color="#444444", linespacing=1.08, fontsize=10.5)
    for e in ("svg", "pdf"): fig.savefig(O / f"se_rosette.{e}", dpi=600, transparent=True)
    fig.savefig(O / "se_rosette.png", dpi=600, facecolor="white"); fig.savefig(O / "se_rosette_check.png", dpi=144, facecolor="white")
