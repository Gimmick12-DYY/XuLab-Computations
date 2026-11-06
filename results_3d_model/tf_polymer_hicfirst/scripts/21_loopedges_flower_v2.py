"""v2 colours (figure_loopedges_flower_v2): CVD-optimised TF palette (worst pairwise CAM02-UCS distance 9.7 across normal,
deutan, protan and tritan simulation vs 3.6 before), single red hue for loop-edges (light fill + dark red outline), radial
rim labels, simplified legend.

Loop-edge flower: the petals are the hub's 18 loop-edges (Cicero shared clique edges on a Peakachu 0.7 loop, +-30 kb rule;
the red arcs of the browser figure). Hi-C-first model (no TF term), one conformation (--conf).
Closed = edge end beads < --closed sigma apart. Rim of the schematic core = the edge-end promoters (TF pies = TFs with a MACS2
peak in the 2 kb bead), ordered by a 2-opt search that keeps each edge's two ends close (start: genomic order).
Petal = teardrop from one end out and back to the other, length ~ log10(span), red family; solid = closed in this structure,
dashed/lighter = open. The 6 Cicero-only (blue) hub edges are listed in a note. 3D: each loop-edge's chromatin segment as a
red-family petal (beads coloured by the shortest edge covering them), edge ends dark + TF satellites, rest a faint ghost.
"""
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/tf_polymer_hicfirst")
ap.add_argument("--peaks", required=True); ap.add_argument("--sim", required=True)
ap.add_argument("--conf", type=int, default=1506); ap.add_argument("--closed", type=float, default=2.0)
ap.add_argument("--figure", action="store_true"); ap.add_argument("--size", type=int, default=2600)
ap.add_argument("--slide", action="store_true", help="no side notes / footnotes (text goes on the slide)")
ap.add_argument("--max-petal", type=int, default=400, help="3D: loop-edges up to this span (kb) are drawn as chromatin petals, longer ones as chords")
a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_loopedges_flower_v2"; O0 = Bd / "figure_loopedges_flower"; (O / "raster").mkdir(parents=True, exist_ok=True)
N, L0 = 910, 29730000
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]
COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, INK, GHOST, BLUE = "#BDBDBD", "#222222", "#D9D9D9", "#4A6FA5"
REDS = ["#C0392B"] * 6; PFILL = "#F3C9C4"
EL = pd.read_csv(Bd / "hicfirst/figure_hub_flow/hub_elements.tsv", sep="\t"); lab_of = dict(zip(EL.bead, EL.label))
HE = pd.read_csv(Bd / "hicfirst/figure_hub_flow/hub_edges_beads.tsv", sep="\t")
T = pd.read_csv(O0 / "loop_edges.tsv", sep="\t").sort_values(["b1", "b2"]).reset_index(drop=True)
BL = HE[HE["on_loop_0.7"] == 0].copy(); BL[["b1", "b2"]] = np.sort(BL[["b1", "b2"]].values, 1)
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); t, k = divmod(a.conf - 1, 50); X = np.load(files[t])["X"][k, :N].astype(float)
T["dist"] = np.linalg.norm(X[T.b1] - X[T.b2], axis=1); T["closed"] = T.dist < a.closed
BL["dist"] = np.linalg.norm(X[BL.b1] - X[BL.b2], axis=1); BL["closed"] = BL.dist < a.closed
D = pd.read_csv(O0 / "closed_per_conf.tsv", sep="\t"); colk = "closed2" if a.closed == 2.0 else "closed15"
kk = int(T.closed.sum()); pct = (D[colk] >= kk).mean() * 100
ends = sorted(set(T.b1) | set(T.b2))
bind = {b: set() for b in ends}
for f in sorted(glob.glob(os.path.join(a.peaks, "TF.*_peaks.q005.narrowPeak"))):
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.split("\t")
        if x[0] != "chr16": continue
        s, e = int(x[1]), int(x[2])
        for b in ends:
            if s < L0 + 2000 * b + 2000 and e > L0 + 2000 * b: bind[b].add(tf)
T.to_csv(O / "loop_edges_conf.tsv", sep="\t", index=False)
json.dump(dict(conf=a.conf, closed_sigma=a.closed, k=kk, n=len(T), pct_at_or_above=pct, mean_closed=float(D[colk].mean()),
               mean_closed_15=float(D.closed15.mean()), k_15=int((T.dist < 1.5).sum()), blue_closed=int(BL.closed.sum()), n_ends=len(ends)), open(O / "meta.json", "w"), indent=1)
print(f"conf {a.conf}: {kk}/18 loop-edges closed (<{a.closed} sigma; {int((T.dist < 1.5).sum())} <1.5); conformations with >= {kk}: {pct:.2f}%; mean {D[colk].mean():.1f}; blue closed {BL.closed.sum()}/6; ends {len(ends)}")

if not a.figure:
    owner = np.full(N, -1); span = (T.b2 - T.b1).values; SHORT = span * 2 <= a.max_petal
    for i in np.argsort(-span):
        if SHORT[i]: owner[T.b1[i]:T.b2[i] + 1] = i
    Y = (X - X[ends].mean(0)) * 10.0
    _, _, Vt = np.linalg.svd(Y[owner >= 0] - Y[owner >= 0].mean(0), full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    cmd.reinitialize()
    tck, _ = splprep(Y[:N].T, s=N * 0.08, k=3); tt = np.linspace(0, 1, N * 6); pts = np.array(splev(tt, tck)).T
    bo = np.minimum((tt * (N - 1)).round().astype(int), N - 1); gh, pe = [cgo.ALPHA, 0.15], []
    for u, v, b in zip(pts[:-1], pts[1:], bo[1:]):
        if owner[b] < 0: c = rgb(GHOST); gh += [cgo.CYLINDER, *u, *v, 0.8, *c, *c]
        else: c = rgb(REDS[owner[b] % len(REDS)]); pe += [cgo.CYLINDER, *u, *v, 1.3, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 1.3]
    cmd.load_cgo(gh, "ghost"); cmd.load_cgo(pe, "loop_edges")
    lc = []
    for i in np.where(~SHORT)[0]:
        c = rgb("#8E2A20"); lc += [cgo.CYLINDER, *Y[T.b1[i]], *Y[T.b2[i]], 0.55, *c, *c]
    if lc: cmd.load_cgo(lc, "long_edges")
    gold = np.pi * (3 - np.sqrt(5)); core, sat = [cgo.COLOR, *rgb(INK)], []
    for b in ends:
        core += [cgo.SPHERE, *Y[b], 4.0]
        tfl = [t_ for t_ in TFS if t_ in bind[b]] + ["other"] * min(len(bind[b] - set(TFS)), 12); n = len(tfl)
        for j, t_ in enumerate(tfl):
            zz = 1 - 2 * (j + 0.5) / n; rr = np.sqrt(1 - zz * zz); ph = j * gold
            sat += [cgo.COLOR, *rgb(COL.get(t_, OTHER)), cgo.SPHERE, *(Y[b] + 5.6 * np.array([rr * np.cos(ph), rr * np.sin(ph), zz])), 1.6 if t_ in COL else 1.0]
    cmd.load_cgo(core, "edge_ends"); cmd.load_cgo(sat, "tf_binding")
    for k_, v in dict(ray_trace_mode=1, ray_trace_gain=0.006, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15,
                      ambient=0.45, specular=0.1, direct=0.55, reflect=0.3, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3,
                      ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k_, v)
    w = h = a.size; cmd.viewport(w // 4, h // 4); cmd.reset()
    for i, p in enumerate(Y[owner >= 0]): cmd.pseudoatom("frame", pos=[float(x) for x in p], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=3, complete=1); cmd.delete("frame")
    view = cmd.get_view(); cmd.png(str(O / "loopedges_render.png"), width=w, height=h, dpi=600, ray=1); cmd.save(str(O / "loopedges.pse"))
    R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); vc = (Y - org) @ R_ + cam
    hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
    pd.DataFrame(dict(bead=ends, label=[lab_of[b] for b in ends], n_tfs=[len(bind[b]) for b in ends],
                      px=w / 2 * (1 + vc[ends, 0] / hh), py=h / 2 * (1 - vc[ends, 1] / hh))).to_csv(O / "anchor_projection.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
    MM = 1 / 25.4; PTS = 72 / 25.4
    FOOT = (f"Petals = the 18 Cicero shared edges supported by a Hi-C loop (0.7, \u00b130 kb). One example structure (conformation {a.conf}) from the Hi-C-constrained model;\n"
            f"all 18 close here (< {a.closed:g}\u03c3) because they were always-on springs ({pct:.1f}% of structures close all 18; mean {D[colk].mean():.1f}). "
            "TF colours = measured binding (MACS2 peaks); no TF input.")
    def legend(ax, x, y, dy=2.7, mode="schem"):
        ax.text(x - 1.5, y + 2.6, "TF with a MACS2 peak", fontsize=6, fontweight="bold", va="center")
        items = [(COL[t_], t_) for t_ in TFS] + [(OTHER, "other TFs")]
        for i, (c, l) in enumerate(items):
            yy = y - dy * i; ax.add_patch(Circle((x, yy), 0.9, color=c, lw=0)); ax.text(x + 2.2, yy, l, va="center", fontstyle="italic" if l in TFS else "normal")
        y2 = y - dy * len(items) - 2.0; ax.text(x - 1.5, y2, "Structure", fontsize=6, fontweight="bold", va="center")
        ax.add_patch(Circle((x, y2 - dy), 1.1, color=INK, lw=0)); ax.text(x + 2.2, y2 - dy, "hub promoter at a loop-edge end", va="center")
        if mode == "schem":
            ax.add_patch(plt.Rectangle((x - 1.5, y2 - 2 * dy - 0.9), 3.0, 1.8, fc=PFILL, ec=REDS[0], lw=1.0)); ax.text(x + 2.2, y2 - 2 * dy, "loop-edge (Hi-C-supported shared edge)", va="center")
        else:
            ax.plot([x - 1.5, x + 1.5], [y2 - 2 * dy] * 2, color=REDS[0], lw=3); ax.text(x + 2.2, y2 - 2 * dy, "chromatin of a loop-edge \u2264 200 kb", va="center")
            ax.plot([x - 1.5, x + 1.5], [y2 - 3 * dy] * 2, color="#8E2A20", lw=0.9); ax.text(x + 2.2, y2 - 3 * dy, "longer loop-edge (end to end)", va="center")
    # ---- 3D panel
    im = np.asarray(Image.open(O / "loopedges_render.png").convert("RGBA")); ys, xs = np.where(im[..., 3] > 30)
    cx_, cy_ = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; half = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.03)
    x0, y0 = max(0, cx_ - half), max(0, cy_ - half); cr = im[y0:cy_ + half, x0:cx_ + half]; chh, cww = cr.shape[:2]
    W, H = 200.0, 110.0; IMW = 86.0; IMH = IMW * chh / cww; IX = 28.0; IY = 13.0
    Image.fromarray(cr, "RGBA").save(O / "raster/loopedges_render.png", dpi=(round(cww / (IMW * MM)),) * 2)
    PJ = pd.read_csv(O / "anchor_projection.tsv", sep="\t"); PJ["x"] = IX + (PJ.px - x0) / cww * IMW; PJ["y"] = IY + IMH - (PJ.py - y0) / chh * IMH
    deg = pd.concat([T.b1, T.b2]).value_counts(); PJ["deg"] = [int(deg.get(b, 0)) for b in PJ.bead]
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
            ax.text(lx + (-0.5 if side < 0 else 0.5), ty, r.label, ha="right" if side < 0 else "left", va="center", fontstyle="italic")
    ax.text(W / 2, H - 2.5, "The hub's 18 loop-edges in one Hi-C-constrained structure", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(IX + IMW + 30, H - 11, f"10 most connected promoters labelled;\nsmall spheres around each = its TFs", va="top", fontsize=6)
    legend(ax, IX + IMW + 31, H - 30, mode="3d")
    if not a.slide: ax.text(W / 2, 4.6, FOOT, ha="center", va="center", fontsize=5.0, color="#444444")
    for e in ("svg", "pdf"): fig.savefig(O / f"loopedges_3d.{e}", transparent=True)
    fig.savefig(O / "loopedges_3d_preview.png", dpi=300, facecolor="white"); plt.close(fig)
    t_ = open(O / "loopedges_3d.svg").read(); m = next(re.finditer(r"<image\b.*?/>", t_, re.S))
    open(O / "loopedges_3d_linked.svg", "w").write(t_[:m.start()] + f'<image xlink:href="raster/loopedges_render.png" x="{IX * PTS:.3f}" y="{(H - IY - IMH) * PTS:.3f}" width="{IMW * PTS:.3f}" height="{IMH * PTS:.3f}" preserveAspectRatio="none"/>' + t_[m.end():])
    # ---- schematic flower
    n = len(ends); idx = {b: i for i, b in enumerate(ends)}; E_ = [(idx[r.b1], idx[r.b2]) for r in T.itertuples()]
    def cost(o):
        pos = {h: i for i, h in enumerate(o)}; return sum(min(abs(pos[x] - pos[y]), n - abs(pos[x] - pos[y])) for x, y in E_)
    order = list(range(n)); best = cost(order); improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]; c = cost(cand)
                if c < best: order, best, improved = cand, c, True
        for i in range(n):
            for j in range(n):
                if i == j: continue
                cand = order[:]; v = cand.pop(i); cand.insert(j, v); c = cost(cand)
                if c < best: order, best, improved = cand, c, True
    ang = {ends[h]: np.pi / 2 - 2 * np.pi * i / n for i, h in enumerate(order)}
    fig = plt.figure(figsize=(208 * MM, 125 * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-100, 108); ax.set_ylim(-64, 61); ax.set_aspect("equal"); ax.set_axis_off()
    cx0, R0 = -34.0, 19.0
    ax.add_patch(Circle((cx0, 0), R0, color="#F4F4F4", ec="#BBBBBB", lw=0.6, zorder=3))
    pol = lambda r_, a_: np.array([cx0 + r_ * np.cos(a_), r_ * np.sin(a_)])
    for i, r in T.iterrows():
        a1, a2 = ang[r.b1], ang[r.b2]; d_ = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi; a2 = a1 + d_
        Lp = 7 + 10 * np.log10(r.span_kb / 10 + 1); sg = 1.0 if a2 >= a1 else -1.0; w = 0.12
        P0, P3 = pol(R0, a1), pol(R0, a2); P1, P2 = pol(R0 + 1.35 * Lp, a1 - sg * w), pol(R0 + 1.35 * Lp, a2 + sg * w)
        tt = np.linspace(0, 1, 240)[:, None]; Bz = (1 - tt) ** 3 * P0 + 3 * (1 - tt) ** 2 * tt * P1 + 3 * (1 - tt) * tt ** 2 * P2 + tt ** 3 * P3
        ax.fill(Bz[:, 0], Bz[:, 1], color=PFILL, alpha=0.45, lw=0, zorder=1)
        ax.plot(Bz[:, 0], Bz[:, 1], color=REDS[0], lw=0.9, ls="-" if r.closed else (0, (2, 1.2)), zorder=2)
        dd = np.hypot(Bz[:, 0] - cx0, Bz[:, 1]); j = int(np.argmax(dd)); ux, uy = (Bz[j, 0] - cx0) / dd[j], Bz[j, 1] / dd[j]
        off_ = 1.0 + (0.9 if r.span_kb in (48,) else 0) ; ax.text(Bz[j, 0] + 4.6 * ux * off_ + (5.5 if r.span_kb == 48 else (-5.5 if r.span_kb == 80 else 0)), Bz[j, 1] + 3.3 * uy * off_, f"{r.span_kb} kb\n{r.gene1}–{r.gene2}", ha="center", va="center", fontsize=4.6, color="#333333", zorder=6)
    for b in ends:
        x, y = pol(R0, ang[b]); tf = [t_ for t_ in TFS if t_ in bind[b]]; no = len(bind[b] - set(TFS)); tot = max(1, len(tf) + no); st = 90.0
        for t_ in tf + (["other"] if no else []):
            w_ = 360.0 * (no if t_ == "other" else 1) / tot; ax.add_patch(Wedge((x, y), 1.9, st, st + w_, color=COL.get(t_, OTHER), lw=0, zorder=4)); st += w_
        ax.add_patch(Circle((x, y), 0.75, color=INK, lw=0, zorder=5))
        aa_ = ang[b] % (2 * np.pi); left = np.pi / 2 < aa_ < 3 * np.pi / 2; lx_, ly_ = pol(R0 - 2.8, ang[b])
        ax.text(lx_, ly_, lab_of[b], ha="left" if left else "right", va="center", fontsize=4.6, fontstyle="italic", zorder=6,
                rotation=np.degrees(aa_) - 180 if left else np.degrees(aa_), rotation_mode="anchor")
    ax.text(cx0, 58, "The hub's 18 loop-edges form a rosette", ha="center", va="center", fontsize=7.5, fontweight="bold")
    legend(ax, 33, 50)
    bl = "\n".join(f"  {lab_of[r.b1]}–{lab_of[r.b2]} ({(r.b2 - r.b1) * 2} kb; {'closed' if r.closed else 'open'} here)" for r in BL.sort_values("b1").itertuples())
    NOTES = [("Inputs", "WT HEK293T Hi-C (10 kb, ICE O/E \u2265 2; 20% of pixels held out); 165 Peakachu 0.7 loops;\n"
                         "the 18 loop-edges (Cicero shared edges on a 0.7 loop, \u00b130 kb). TF peaks used only for colour."),
             ("Model", "Hi-C-restrained polymer: 910 \u00d7 2 kb beads, Langevin dynamics (OpenMM). Hi-C contacts and loops\n"
                       "= soft springs (switched on in proportion to O/E); loop-edges always on. No TF attraction. 9,850 structures."),
             ("Output", f"Each structure closes {D[colk].mean():.0f}/18 loop-edges on average; this one closes all 18 (top {pct:.1f}%).\n"
                        "The 20 edge promoters gather into one core with the loop-edges (48\u2013844 kb) as petals.\n"
                        "Fit to held-out Hi-C: O/E Spearman 0.37; the 6 Cicero-only edges stay open (1.0\u00d7)."),
             ("Interpretation", "Hi-C and the loop-edges are consistent with one rosette-like hub where TF-co-bound\n"
                                "promoters meet. The rosette follows from the restraints (an input, not an independent test);\n"
                                "Cicero co-accessibility alone does not bring promoters together. TF colours are measured,\nnot modelled.")]
    if a.slide: NOTES = []
    yy = -2.0
    for h_, t_ in NOTES:
        ax.text(30, yy, h_, fontsize=6.2, fontweight="bold", va="top", color="#222222"); yy -= 3.0
        ax.text(30, yy, t_, fontsize=5.4, va="top", color="#333333", linespacing=1.25); yy -= 2.45 * (t_.count("\n") + 1) + 2.4
    if not a.slide: ax.text(0, -60, FOOT, ha="center", va="center", fontsize=4.9, color="#444444")
    for e in ("svg", "pdf"): fig.savefig(O / f"loopedges_schematic.{e}", transparent=True)
    fig.savefig(O / "loopedges_schematic_preview.png", dpi=300, facecolor="white"); plt.close(fig)
    print(T[["gene1", "gene2", "span_kb", "dist", "closed"]].round(2).to_string(index=False))
