"""v2 figure of the chr16p11.2 PASTIS model (no refit; reuses results_3d_model/out).

Stage 'render' (needs PyMOL): best model -> PCA-rotated, spline-smoothed CA trace (PDB), ray-traced tube coloured by genomic
position (viridis), promoter-hub cores as dark spheres, hub-component Cicero shared edges as CGO cylinders.
Camera: PCA of the hub-component nodes, viewing down the smallest axis (PC1 = horizontal, PC2 = vertical).
Writes pymol/model.pdb, model.pse, render.png and projected pixel coordinates of beads (projection.tsv).
Stage 'figure' (matplotlib): A split Hi-C / model contact map, B render + labels, C linear tracks, D quant + ensemble.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="results_3d_model folder")
ap.add_argument("--hub-edges", required=True, help="hub_chr16_edges.tsv (hub component of the browser figure)")
ap.add_argument("--genes", required=True, help="RefSeq genes table used by the browser figure")
ap.add_argument("--out", required=True)
ap.add_argument("--stage", choices=["render", "figure", "all"], default="all")
ap.add_argument("--cores", default="SRCAP,FBXL19,FBRS,PRR14,PHKG2,ZNF688,DCTPP1,TBX6,ZNF771,ALDOA")
ap.add_argument("--width", type=int, default=3000)
ap.add_argument("--height", type=int, default=2400)
ap.add_argument("--angstrom-per-bead", type=float, default=15.0)
a = ap.parse_args()
B = Path(a.base); O = B / "out"; M = O / "model_include"; V = Path(a.out); PM = V / "pymol"
PM.mkdir(parents=True, exist_ok=True)
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]

bins = pd.read_csv(O / "bins.tsv", sep="\t")
nb = len(bins); start0 = bins.start.iloc[0]; RES = int(bins.end.iloc[0] - bins.start.iloc[0])
keep = bins.keep.values.astype(bool); interp = ~keep
X = pd.read_csv(M / "best_model.tsv", sep="\t")[["x", "y", "z"]].values
qc = pd.read_csv(M / "model_qc.tsv", sep="\t", header=None, index_col=0)[1]
alpha = float(qc["alpha"])

def parse(r):
    c, s = r.split(":"); s, e = s.split("-"); return c, int(s), int(e)
def tobin(r):
    _, s, e = parse(r); return int(((s + e) // 2 - start0) // RES)

# hub-component edges with clique groups (edges linked through a shared TF clique id; numbered by size)
H = pd.read_csv(a.hub_edges, sep="\t")
T = pd.read_csv(B / "inputs" / "top_shared_clique_edges.tsv", sep="\t")
cl = {}
for r in T.itertuples():
    cl[(r.region1, r.region2)] = cl[(r.region2, r.region1)] = r.cliques.split(";")
H["cliques"] = [cl[(r.region1, r.region2)] for r in H.itertuples()]
par = list(range(len(H)))
def find(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
own = {}
for i, cs in enumerate(H.cliques):
    for c in cs:
        if c in own: par[find(i)] = find(own[c])
        else: own[c] = i
roots = [find(i) for i in range(len(H))]
sz = pd.Series(roots).value_counts()
H["bin1"] = [tobin(r) for r in H.region1]; H["bin2"] = [tobin(r) for r in H.region2]
order = sorted(sz.index, key=lambda r: (-sz[r], min(H.bin1[r], H.bin2[r])))
H["group"] = [order.index(r) for r in roots]
H["color"] = [OKABE[g % 4] for g in H.group]

# highlighted promoter-hub cores (one sphere per 10 kb bin; genes sharing a bin are joined with '/')
cores = pd.read_csv(B / "inputs" / "anchor_hub_cores_promoter_labels.tsv", sep="\t")
want = a.cores.split(",")
cores = cores[cores.promoter_gene.isin(want) & cores.core.str.startswith("chr16:")]
cores = cores[[29730000 <= parse(c)[1] <= 31550000 for c in cores.core]].copy()
cores["bin"] = [tobin(c) for c in cores.core]
cores["tss"] = [(parse(c)[1] + parse(c)[2]) // 2 for c in cores.core]
CB = cores.groupby("bin").agg(label=("promoter_gene", lambda s: "/".join(sorted(s, key=want.index))), n_tfs=("n_tfs", "max")).reset_index()

# PCA camera from hub-component nodes
nodes = sorted(set(H.bin1) | set(H.bin2))
P = X[nodes] - X[nodes].mean(0)
_, _, Vt = np.linalg.svd(P, full_matrices=False)
if np.linalg.det(Vt) < 0: Vt[2] *= -1
Xr = (X - X[nodes].mean(0)) @ Vt.T * a.angstrom_per_bead   # PC1 -> x, PC2 -> y, PC3 -> z (view axis)
np.save(V / "model_pca_angstrom.npy", Xr)

def render():
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    from scipy.interpolate import splprep, splev
    import matplotlib
    # spline through all beads (interpolating), resampled at ~3.8 A
    u = np.arange(nb, dtype=float)
    tck, _ = splprep(Xr.T, u=u, s=0, k=3)
    fine = np.linspace(0, nb - 1, nb * 40); S = np.array(splev(fine, tck)).T
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]
    ts = np.interp(np.arange(0, arc[-1], 3.8), arc, fine)
    pts = np.array(splev(ts, tck)).T
    gap = np.interp(ts, u, interp.astype(float)) > 0.5
    # chains: fitted segments A, B, ... ; gap atoms chain Z
    lines, chain, resi, prev = [], "A", 0, None
    for k, (p, t, g) in enumerate(zip(pts, ts, gap)):
        if prev is not None and g != prev and not g:
            chain = chr(ord(chain) + 1)
        ch = "Z" if g else chain
        resi += 1
        lines.append("ATOM  %5d  CA  GLY %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           C" % (k + 1, ch, resi % 10000, p[0], p[1], p[2], 1.0, t / (nb - 1)))
        prev = g
    (PM / "model.pdb").write_text("\n".join(lines) + "\nEND\n")
    cmd.load(str(PM / "model.pdb"), "chromatin")
    cmd.hide("everything")
    cmd.set("cartoon_trace_atoms", 1); cmd.set("cartoon_tube_radius", 3.2); cmd.set("cartoon_sampling", 14)
    cmd.cartoon("tube", "chromatin")
    cmd.show("cartoon", "chromatin and not chain Z")
    vir = matplotlib.colormaps["viridis"]
    nbin = 64
    for i in range(nbin):
        r, g, b, _ = vir(i / (nbin - 1)); cmd.set_color(f"vir{i}", [r, g, b])
        cmd.color(f"vir{i}", f"chromatin and b > {i/nbin - 1e-4:.5f} and b < {(i+1)/nbin + 1e-4:.5f}")
    cmd.set_color("gapgrey", [0.80, 0.80, 0.80])
    cmd.show("spheres", "chromatin and chain Z and (resi " + "+".join(str(int(x)) for x in np.unique([int(l[22:26]) for l in lines if l[21] == "Z"])[::2]) + ")")
    cmd.set("sphere_scale", 0.9, "chromatin and chain Z"); cmd.color("gapgrey", "chromatin and chain Z")
    # hub-core spheres
    cmd.set_color("coredark", [0.22, 0.22, 0.22])
    for r in CB.itertuples():
        x, y, z = Xr[r.bin]
        cmd.pseudoatom("cores", pos=[float(x), float(y), float(z)], name=f"C{r.bin}")
    cmd.show("spheres", "cores"); cmd.set("sphere_scale", 7.0, "cores"); cmd.color("coredark", "cores")
    # shared edges as cylinders
    obj = []
    for r in H.itertuples():
        c = [int(r.color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        p1, p2 = Xr[r.bin1], Xr[r.bin2]
        obj += [cgo.CYLINDER, *map(float, p1), *map(float, p2), 1.3, *c, *c]
    cmd.load_cgo(obj, "shared_edges")
    cmd.bg_color("white")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.004, ray_trace_color="black", ambient_occlusion_mode=1,
                     ambient_occlusion_scale=15, ray_shadows=1, ambient=0.35, specular=0.15, direct=0.6, reflect=0.4,
                     orthoscopic=0, depth_cue=0, antialias=2, field_of_view=20, cgo_lighting=1, ray_opaque_background=1).items():
        cmd.set(k, v)
    cmd.viewport(int(a.width / 4), int(a.height / 4))
    cmd.reset()                        # camera looks down -z: x = PC1 (right), y = PC2 (up)
    cmd.zoom("all", buffer=12, complete=1)
    view = cmd.get_view()
    cmd.png(str(PM / "render.png"), width=a.width, height=a.height, dpi=300, ray=1)
    cmd.save(str(PM / "model.pse"))
    json.dump(dict(view=list(view), width=a.width, height=a.height, fov=20.0), open(PM / "view.json", "w"))
    # project bead centres to render pixels (model -> camera with the view matrix, then perspective)
    R = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15])
    vc = (Xr - org) @ R + cam
    hh = -vc[:, 2] * np.tan(np.radians(20.0) / 2); hw = hh * a.width / a.height
    px = a.width / 2 * (1 + vc[:, 0] / hw); py = a.height / 2 * (1 - vc[:, 1] / hh)
    pd.DataFrame(dict(bin=bins.bin, px=px, py=py, depth=-vc[:, 2])).to_csv(PM / "projection.tsv", sep="\t", index=False)
    print("render done; view", np.round(view, 3).tolist())

def figure():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors, cm
    from matplotlib.patches import Rectangle, Arc
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42})
    W_IN, H_IN = 7.09, 6.6
    fig = plt.figure(figsize=(W_IN, H_IN))
    L0, L1 = 29.73, 31.55
    mb = lambda b: (start0 + (np.asarray(b) + 0.5) * RES) / 1e6
    def letter(x, y, s):
        fig.text(x, y, s, fontsize=9, weight="bold", va="top", ha="left")

    # ---------- A: split contact map ----------
    raw = np.load(O / "counts_raw.npz")["counts"]; w = bins.weight.values
    ice = raw * w[:, None] * w[None, :]
    top = np.load(M / "top10_models.npy")
    mc = np.mean([np.where(np.eye(nb, dtype=bool), np.nan, squareform(pdist(t)) + np.eye(nb)) ** alpha for t in top], axis=0)
    iu = np.triu_indices(nb, 1); okp = keep[iu[0]] & keep[iu[1]]
    rho = stats.spearmanr(ice[iu][okp], mc[iu][okp]).correlation
    def rankmat(A):
        R_ = np.full((nb, nb), np.nan); v = A[iu].copy(); v[~okp] = np.nan
        r = pd.Series(v).rank(pct=True).values
        R_[iu] = r; R_[(iu[1], iu[0])] = r; return R_
    ri, rm = rankmat(ice), rankmat(mc)
    S = np.where(np.triu(np.ones((nb, nb), bool), 1), ri, np.where(np.tril(np.ones((nb, nb), bool), -1), rm, np.nan))
    axA = fig.add_axes([0.08, 0.64, 0.265, 0.265 * W_IN / H_IN])
    cmapA = plt.get_cmap("Reds").copy(); cmapA.set_bad("#E6E6E6")
    ext = [L0, L1, L1, L0]
    imA = axA.imshow(S, cmap=cmapA, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    axA.plot([L0, L1], [L0, L1], color="white", lw=0.6)
    bb = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85)
    axA.text(0.97, 0.97, "WT Hi-C (ICE)", transform=axA.transAxes, ha="right", va="top", fontsize=6.3, bbox=bb)
    axA.text(0.03, 0.03, "model, top-10\nensemble ($d^{\\alpha}$)", transform=axA.transAxes, ha="left", va="bottom", fontsize=6.3, bbox=bb)
    axA.set_title(f"Spearman ρ = {rho:.2f} (fitted bin pairs)", fontsize=6.5, pad=3)
    axA.set_xticks([29.8, 30.2, 30.6, 31.0, 31.4]); axA.set_yticks([29.8, 30.2, 30.6, 31.0, 31.4])
    axA.set_xlabel("chr16 (Mb)"); axA.set_ylabel("chr16 (Mb)")
    caxA = fig.add_axes([0.08, 0.545, 0.11, 0.010])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="Reds"), cax=caxA, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.set_label("contact rank (percentile)", fontsize=6); cb.outline.set_linewidth(0.4)
    fig.text(0.205, 0.550, "grey: low-coverage\ngap (not fitted)", fontsize=5.8, color="#555555", va="center")
    letter(0.005, 0.985, "A")

    # ---------- B: PyMOL render with label overlay ----------
    img = plt.imread(PM / "render.png")
    pr = pd.read_csv(PM / "projection.tsv", sep="\t")
    rgb = img[..., :3] if img.ndim == 3 else np.dstack([img] * 3)
    nonw = np.where((rgb < 0.97).any(axis=2))
    pad = 40
    y0, y1 = max(nonw[0].min() - pad, 0), min(nonw[0].max() + pad, img.shape[0]); x0, x1 = max(nonw[1].min() - pad, 0), min(nonw[1].max() + pad, img.shape[1])
    crop = img[y0:y1, x0:x1]
    padx, pady = int(0.12 * crop.shape[1]), int(0.10 * crop.shape[0])   # white margin for labels
    canvas = np.ones((crop.shape[0] + 2 * pady, crop.shape[1] + 2 * padx, crop.shape[2]), dtype=crop.dtype)
    canvas[pady:pady + crop.shape[0], padx:padx + crop.shape[1]] = crop
    crop = canvas; x0 -= padx; y0 -= pady
    bx0, bw = 0.41, 0.575
    bh = bw * W_IN / H_IN * crop.shape[0] / crop.shape[1]
    bh = min(bh, 0.53);
    axB = fig.add_axes([bx0, 0.985 - bh, bw, bh]); axB.imshow(crop, interpolation="lanczos"); axB.set_axis_off()
    axB.set_xlim(0, crop.shape[1]); axB.set_ylim(crop.shape[0], 0)
    cx, cy = pr.px.values - x0, pr.py.values - y0
    pts = np.array([[cx[b], cy[b]] for b in CB.bin]); cen = pts.mean(0)
    span = max(crop.shape)
    lab = []
    for (b, l), p in zip(zip(CB.bin, CB.label), pts):
        v = p - cen; v = v / (np.linalg.norm(v) + 1e-9)
        lab.append([p, p + v * span * 0.11, l])
    ppi = crop.shape[1] / (bw * W_IN)                      # image pixels per inch in the final figure
    tw = lambda l: len(l) * 7 * 0.58 / 72 * ppi             # label text width (px)
    th = 7 * 1.5 / 72 * ppi
    def box(q, p, l):
        x0_ = q[0] if q[0] >= p[0] else q[0] - tw(l)
        return x0_, x0_ + tw(l), q[1] - th / 2, q[1] + th / 2
    for _ in range(400):   # push overlapping label boxes apart vertically, keep inside the image
        moved = False
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                a_, b_ = box(lab[i][1], lab[i][0], lab[i][2]), box(lab[j][1], lab[j][0], lab[j][2])
                if a_[0] < b_[1] and b_[0] < a_[1] and a_[2] < b_[3] and b_[2] < a_[3]:
                    s_ = 1 if lab[i][1][1] >= lab[j][1][1] else -1
                    lab[i][1][1] += s_ * th * 0.1; lab[j][1][1] -= s_ * th * 0.1; moved = True
        if not moved: break
    for p, q, l in lab:
        if q[0] >= p[0]: q[0] = min(q[0], crop.shape[1] - tw(l) - 10)
        else: q[0] = max(q[0], tw(l) + 10)
        q[1] = np.clip(q[1], th, crop.shape[0] - th)
        axB.plot([p[0], q[0]], [p[1], q[1]], color="#333333", lw=0.4, zorder=3)
        axB.text(q[0], q[1], l, fontsize=7, fontstyle="italic", ha="left" if q[0] >= p[0] else "right", va="center", zorder=4,
                 bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    letter(bx0 - 0.02, 0.985, "B")
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], marker="o", ls="", mfc="#383838", mec="black", ms=6, label="promoter-hub core"),
          Line2D([], [], marker="o", ls="", mfc="#CCCCCC", mec="none", ms=3, label="low-coverage gap (interpolated)")]
    ng = H.group.nunique()
    for g in range(min(ng, 4)):
        hd.append(Line2D([], [], color=OKABE[g], lw=1.6, label="shared edges, clique group " + ", ".join(str(k + 1) for k in range(g, ng, 4))))
    fig.legend(handles=hd, loc="upper left", bbox_to_anchor=(bx0 + 0.005, 0.985 - bh - 0.002), frameon=False, ncol=2,
               handlelength=1.8, columnspacing=1.2, borderaxespad=0, fontsize=6.3)
    fig.text(bx0 + 0.02, 0.975, "PASTIS PM2 best model, 10 kb beads; tube colour = genomic position (track C);\nview down PC3 of the hub nodes. Consensus model from population Hi-C; not a single-cell structure.",
             fontsize=5.8, color="#555555", ha="left", va="top")

    # ---------- C: linear tracks ----------
    cy_top = 0.985 - bh - 0.075
    tracks = [("pos", 0.016), ("genes", 0.10), ("tf", 0.06), ("arcs", 0.165)]
    tot = sum(h for _, h in tracks) + 0.008 * (len(tracks) - 1)
    yc = cy_top; axs = {}
    for name, h in tracks:
        yc -= h; axs[name] = fig.add_axes([bx0 + 0.075, yc, bw - 0.075, h]); yc -= 0.008
    for ax in axs.values():
        ax.set_xlim(L0, L1)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    ap_ = axs["pos"]
    ap_.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap="viridis", extent=[L0, L1, 0, 1])
    for b in np.where(interp)[0]:
        ap_.add_patch(Rectangle((mb(b) - RES / 2e6, 0), RES / 1e6, 1, color="#CCCCCC", lw=0))
    ap_.set_xlim(L0, L1); ap_.set_ylim(0, 1)
    ap_.set_axis_off()
    ap_.text(L0 - 0.01, 0.5, "position", ha="right", va="center", fontsize=6.5)
    # genes
    G = pd.read_csv(a.genes, sep="\t")
    G = G.sort_values("txStart").drop_duplicates("name")
    ag = axs["genes"]; rows_end = []; hlpos = []
    hl = set(sum([l.split("/") for l in CB.label], []))
    for r in G.itertuples():
        s, e = r.txStart / 1e6, r.txEnd / 1e6
        if e < L0 or s > L1: continue
        row = next((i for i, re_ in enumerate(rows_end) if s > re_ + 0.004), None)
        if row is None: rows_end.append(e); row = len(rows_end) - 1
        else: rows_end[row] = e
        y = -row
        col = "#222222" if r.name in hl else "#9A9A9A"
        ag.plot([s, e], [y, y], color=col, lw=0.5)
        for es, ee in zip(map(int, str(r.exonStarts).strip(",").split(",")), map(int, str(r.exonEnds).strip(",").split(","))):
            ag.add_patch(Rectangle((es / 1e6, y - 0.3), (ee - es) / 1e6, 0.6, color=col, lw=0))
        if r.name in hl:
            hlpos.append((r.txStart / 1e6 if r.strand == "+" else r.txEnd / 1e6, r.name, y))
    hlpos.sort()
    # label x positions: keep >= one label width apart within each of two staggered rows
    mb_per_in = (L1 - L0) / (ag.get_position().width * W_IN)
    wid = [len(nm) * 5.8 * 0.6 / 72 * mb_per_in + 0.02 for _, nm, _ in hlpos]
    xs = [t for t, _, _ in hlpos]
    for it in range(200):
        moved = False
        for i in range(len(xs) - 2):
            j = i + 2      # same row (rows alternate)
            need = (wid[i] + wid[j]) / 2
            if xs[j] - xs[i] < need:
                d = (need - (xs[j] - xs[i])) / 2; xs[i] -= d; xs[j] += d; moved = True
        if not moved: break
    for (t, nm, y), xl, k in zip(hlpos, xs, range(len(xs))):
        ytxt = 2.4 if k % 2 == 0 else 3.9
        ag.plot([t, t, xl], [y + 0.4, 1.4, ytxt - 0.25], color="#555555", lw=0.35)
        ag.text(xl, ytxt, nm, fontsize=5.8, fontstyle="italic", ha="center", va="bottom")
    ytxt = 3.9
    ag.set_ylim(-len(rows_end) + 0.3, ytxt + 1.4); ag.set_yticks([]); ag.spines["left"].set_visible(False)
    ag.set_xticks([]); ag.spines["bottom"].set_visible(False)
    ag.text(L0 - 0.01, -len(rows_end) / 2 + 0.5, "RefSeq", ha="right", va="center", fontsize=6.5)
    # TF counts
    at = axs["tf"]
    at.bar(mb(bins.bin), bins.n_tfs, width=RES / 1e6 * 0.9, color="#5E4FA2", lw=0)
    for b in CB.bin:
        at.bar(mb(b), bins.n_tfs[b], width=RES / 1e6 * 0.9, color="#222222", lw=0)
    at.set_ylabel("TFs\n(MACS2)", fontsize=6.3, rotation=0, ha="right", va="center", labelpad=4)
    at.set_xticks([]); at.spines["bottom"].set_visible(True); at.set_yticks([0, 20]); at.set_ylim(0, max(bins.n_tfs) * 1.05)
    # arcs: shared edges above, loops below
    aa = axs["arcs"]
    Lp = pd.read_csv(O / "loops.tsv", sep="\t")
    hmax = 0
    for r in H.itertuples():
        x1, x2 = sorted((mb(r.bin1), mb(r.bin2))); hgt = (x2 - x1)
        aa.add_patch(Arc(((x1 + x2) / 2, 0), x2 - x1, 2 * hgt, theta1=0, theta2=180, color=r.color, lw=0.6 + 0.35 * r.n_tfs))
        hmax = max(hmax, hgt)
    lmax = 0
    for r in Lp.itertuples():
        x1, x2 = mb(r.bin1), mb(r.bin2); hgt = (x2 - x1)
        aa.add_patch(Arc(((x1 + x2) / 2, 0), x2 - x1, hgt, theta1=180, theta2=360, color="#8C8C8C", lw=0.3))
        lmax = max(lmax, hgt / 2)
    aa.axhline(0, color="#BBBBBB", lw=0.4)
    aa.set_ylim(-lmax * 1.05, hmax * 1.05); aa.set_yticks([])
    aa.spines["left"].set_visible(False)
    aa.text(L0 - 0.01, hmax * 0.5, f"shared\nedges ({len(H)})", ha="right", va="center", fontsize=6.3)
    aa.text(L0 - 0.01, -lmax * 0.5, f"loops ≥0.7\n({len(Lp)})", ha="right", va="center", fontsize=6.3, color="#555555")
    aa.set_xlabel("chr16 (Mb, hg38)")
    aa.set_xticks([29.8, 30.0, 30.2, 30.4, 30.6, 30.8, 31.0, 31.2, 31.4])
    letter(bx0 - 0.02, cy_top + 0.02, "C")

    # ---------- D: quant + ensemble ----------
    qp = pd.read_csv(M / "quant_pairs_best.tsv", sep="\t"); Q = pd.read_csv(M / "quant_check.tsv", sep="\t")
    nulls = np.load(M / "quant_null_best.npz")
    order = [("shared_clique_edges", "shared\nedges"), ("cicero_step1_pairs", "Cicero\npairs"), ("peakachu_0.7_loops", "loops")]
    data = [np.log2(qp[qp.set == k].ratio.values) for k, _ in order]
    null = np.log2(np.random.RandomState(0).choice(np.concatenate([nulls[k] for k, _ in order]), 5000, replace=False))
    axD = fig.add_axes([0.085, 0.085, 0.19, 0.30])
    pos = [1, 2, 3, 4]
    vp = axD.violinplot(data + [null], positions=pos, widths=0.8, showextrema=False)
    for b_, c in zip(vp["bodies"], ["#5E4FA2", "#9E9AC8", "#7A7A7A", "#D9D9D9"]):
        b_.set_facecolor(c); b_.set_edgecolor("none"); b_.set_alpha(0.9)
    axD.boxplot(data + [null], positions=pos, widths=0.16, showfliers=False, medianprops=dict(color="black", lw=0.8),
                boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0))
    axD.axhline(0, color="#555555", lw=0.5, ls=":")
    qb = Q[Q.model_rank == 1].set_index("set")
    for x, (k, _) in zip(pos, order):
        r = qb.loc[k]; pv = "p<0.001" if r.p_one_sided <= 1.5 / 1001 else f"p={r.p_one_sided:.2f}"
        axD.text(x, 1.01, f"n={int(r.n_pairs)}\n{pv.replace('p<0.001', 'p<.001')}", transform=axD.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.4)
    axD.set_xticks(pos); axD.set_xticklabels([l for _, l in order] + ["random"], fontsize=6)
    axD.set_ylabel("log2 (3D distance / distance-\nmatched random median)")
    for s in ("top", "right"): axD.spines[s].set_visible(False)
    letter(0.005, 0.475, "D")
    axD.set_title("3D distance vs genomic-distance-\nmatched random bin pairs (best model)", fontsize=6.5, pad=20)
    # ensemble: top-10 Procrustes-aligned to the best, same rotation as B
    axE = fig.add_axes([0.28, 0.11, 0.115, 0.115 * W_IN / H_IN])
    ref = top[0][keep] - top[0][keep].mean(0)
    for i in range(len(top) - 1, -1, -1):
        t = top[i]; is_best = i == 0
        A_ = t - t[keep].mean(0)
        U, _, Vt2 = np.linalg.svd(A_[keep].T @ ref); A_ = A_ @ U @ Vt2
        Z = (A_ + top[0][keep].mean(0) - X[nodes].mean(0)) @ Vt.T
        Z[interp] = np.nan
        axE.plot(Z[:, 0], Z[:, 1], color="#222222" if is_best else "#5E4FA2", lw=0.6 if is_best else 0.5, alpha=1 if is_best else 0.25)
    axE.set_aspect("equal"); axE.set_axis_off()
    axE.set_title(f"top-10 ensemble\n(RMSD {float(qc['rmsd_top10_median']):.1f} bead steps)", fontsize=6, pad=2)
    for ext_ in ("png", "pdf"):
        fig.savefig(V / f"chr16p11.2_3d_model_v2.{ext_}", dpi=300)
    print("figure done; labels:", ", ".join(CB.label), "; groups:", H.group.nunique(), "; rho:", round(rho, 3))
    CB.assign(mb=mb(CB.bin)).to_csv(V / "highlighted_cores.tsv", sep="\t", index=False)
    H.drop(columns="cliques").to_csv(V / "hub_edges_with_groups.tsv", sep="\t", index=False)

if a.stage in ("render", "all"): render()
if a.stage in ("figure", "all"): figure()
