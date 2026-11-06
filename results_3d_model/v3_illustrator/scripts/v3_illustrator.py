"""Illustrator-ready exports of the chr16p11.2 3D-model figure (v2 content, no refit, no new render).

Every panel is drawn into a frame given in mm, so the full figure and the per-panel files share one layout.
Text stays editable (pdf/ps fonttype 42, svg fonttype none, Arial); only the Hi-C/model heatmap is rasterised (600 dpi).
Panel B embeds the existing 3000 x 2400 px PyMOL ray trace at native resolution; labels, leader lines, legend and the
position colour bar are vector objects placed from the projected bead coordinates (projection.tsv from the v2 render).

Usage: python v3_illustrator.py --base <copy of results_3d_model> --render <v2 pymol dir> --hub-edges ... --genes ... --out <dir>
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.patches import Rectangle, Arc
from matplotlib.lines import Line2D
from scipy import stats
from scipy.spatial.distance import pdist, squareform

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--render", required=True, help="folder with render.png and projection.tsv")
ap.add_argument("--hub-edges", required=True)
ap.add_argument("--genes", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cores", default="SRCAP,FBXL19,FBRS,PRR14,PHKG2,ZNF688,DCTPP1,TBX6,ZNF771,ALDOA")
a = ap.parse_args()
B = Path(a.base); O = B / "out"; M = O / "model_include"; PM = Path(a.render); OUT = Path(a.out)
(OUT / "panels").mkdir(parents=True, exist_ok=True)

FS, FS_SMALL, FS_LETTER = 7, 6, 8
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": FS,
                     "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "savefig.dpi": 600, "mathtext.default": "regular"})
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
MM = 1 / 25.4
L0, L1 = 29.73, 31.55

# ---------------- data (same definitions as v2) ----------------
bins = pd.read_csv(O / "bins.tsv", sep="\t")
nb = len(bins); start0 = bins.start.iloc[0]; RES = int(bins.end.iloc[0] - bins.start.iloc[0])
keep = bins.keep.values.astype(bool); interp = ~keep
X = pd.read_csv(M / "best_model.tsv", sep="\t")[["x", "y", "z"]].values
qc = pd.read_csv(M / "model_qc.tsv", sep="\t", header=None, index_col=0)[1]
alpha = float(qc["alpha"])
mb = lambda b: (start0 + (np.asarray(b) + 0.5) * RES) / 1e6
def parse(r):
    c, s = r.split(":"); s, e = s.split("-"); return c, int(s), int(e)
def tobin(r):
    _, s, e = parse(r); return int(((s + e) // 2 - start0) // RES)

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
NG = H.group.nunique()

cores = pd.read_csv(B / "inputs" / "anchor_hub_cores_promoter_labels.tsv", sep="\t")
want = a.cores.split(",")
cores = cores[cores.promoter_gene.isin(want) & cores.core.str.startswith("chr16:")]
cores = cores[[29730000 <= parse(c)[1] <= 31550000 for c in cores.core]].copy()
cores["bin"] = [tobin(c) for c in cores.core]
CB = cores.groupby("bin").agg(label=("promoter_gene", lambda s: "/".join(sorted(s, key=want.index))), n_tfs=("n_tfs", "max")).reset_index()
nodes = sorted(set(H.bin1) | set(H.bin2))
_, _, Vt = np.linalg.svd(X[nodes] - X[nodes].mean(0), full_matrices=False)
if np.linalg.det(Vt) < 0: Vt[2] *= -1
top = np.load(M / "top10_models.npy")

# render crop (white margin added for labels) and projected bead pixels
img = plt.imread(PM / "render.png")
pr = pd.read_csv(PM / "projection.tsv", sep="\t")
rgb = img[..., :3]
nonw = np.where((rgb < 0.97).any(axis=2))
pad = 40
y0, y1 = max(nonw[0].min() - pad, 0), min(nonw[0].max() + pad, img.shape[0])
x0, x1 = max(nonw[1].min() - pad, 0), min(nonw[1].max() + pad, img.shape[1])
crop = img[y0:y1, x0:x1]
padx, pady = int(0.12 * crop.shape[1]), int(0.10 * crop.shape[0])
canvas = np.ones((crop.shape[0] + 2 * pady, crop.shape[1] + 2 * padx, crop.shape[2]), dtype=crop.dtype)
canvas[pady:pady + crop.shape[0], padx:padx + crop.shape[1]] = crop
crop = canvas; x0 -= padx; y0 -= pady
CH, CW = crop.shape[:2]

# ---------------- layout helpers ----------------
def ax_mm(fig, frame, x, y, w, h, **kw):
    """axes at (x, y, w, h) mm inside frame (fx, fy) mm; fig size in mm known from fig."""
    FW, FH = fig.get_size_inches() / MM
    return fig.add_axes([(frame[0] + x) / FW, (frame[1] + y) / FH, w / FW, h / FH], **kw)
def text_mm(fig, frame, x, y, s, **kw):
    FW, FH = fig.get_size_inches() / MM
    return fig.text((frame[0] + x) / FW, (frame[1] + y) / FH, s, **kw)
def clean(ax, left=True, bottom=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)

# panel frame sizes (mm)
A_W, A_H = 62, 72
B_W = 108; B_IMG_H = B_W * CH / CW; B_H = B_IMG_H + 7 + 13
C_W, C_H = 108, 74
D_W, D_H = 66, 82

# ---------------- A ----------------
raw = np.load(O / "counts_raw.npz")["counts"]; w = bins.weight.values
ice = raw * w[:, None] * w[None, :]
with np.errstate(divide="ignore"):
    mc = np.mean([np.where(np.eye(nb, dtype=bool), np.nan, squareform(pdist(t)) + np.eye(nb)) ** alpha for t in top], axis=0)
iu = np.triu_indices(nb, 1); okp = keep[iu[0]] & keep[iu[1]]
RHO = stats.spearmanr(ice[iu][okp], mc[iu][okp]).correlation
def rankmat(A_):
    R_ = np.full((nb, nb), np.nan); v = A_[iu].copy(); v[~okp] = np.nan
    r = pd.Series(v).rank(pct=True).values
    R_[iu] = r; R_[(iu[1], iu[0])] = r; return R_
ri, rm = rankmat(ice), rankmat(mc)
SPLIT = np.where(np.triu(np.ones((nb, nb), bool), 1), ri, np.where(np.tril(np.ones((nb, nb), bool), -1), rm, np.nan))

def panel_A(fig, f):
    ax = ax_mm(fig, f, 11, 20, 47, 47)
    cmapA = plt.get_cmap("Reds").copy(); cmapA.set_bad("#E6E6E6")
    ax.imshow(SPLIT, cmap=cmapA, vmin=0, vmax=1, extent=[L0, L1, L1, L0], interpolation="nearest", rasterized=True)
    ax.plot([L0, L1], [L0, L1], color="white", lw=0.6)
    bb = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85)
    ax.text(0.97, 0.97, "WT Hi-C (ICE)", transform=ax.transAxes, ha="right", va="top", bbox=bb)
    ax.text(0.03, 0.03, "model, top-10\nensemble (d$^{α}$)", transform=ax.transAxes, ha="left", va="bottom", bbox=bb)
    ax.set_title(f"Spearman ρ = {RHO:.2f} (fitted bin pairs)", pad=3)
    t = [29.8, 30.2, 30.6, 31.0, 31.4]
    ax.set_xticks(t); ax.set_yticks(t); ax.set_xlabel("chr16 (Mb)", labelpad=2); ax.set_ylabel("chr16 (Mb)", labelpad=2)
    cax = ax_mm(fig, f, 11, 7.5, 18, 1.6)
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="Reds"), cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.set_label("contact rank (percentile)", labelpad=1.5); cb.outline.set_linewidth(0.4); cax.tick_params(length=1.5, pad=1.5)
    text_mm(fig, f, 32, 8.3, "grey: low-coverage\ngap (not fitted)", color="#555555", va="center")

# ---------------- B ----------------
def label_layout():
    cx, cy = pr.px.values - x0, pr.py.values - y0
    pts = np.array([[cx[b], cy[b]] for b in CB.bin]); cen = pts.mean(0)
    span = max(CH, CW)
    lab = [[p, p + (p - cen) / (np.linalg.norm(p - cen) + 1e-9) * span * 0.11, l] for p, l in zip(pts, CB.label)]
    ppi = CW / (B_W * MM)
    tw = lambda l: len(l) * FS * 0.66 / 72 * ppi
    th = FS * 1.5 / 72 * ppi
    def box(q, p, l):
        b0 = q[0] if q[0] >= p[0] else q[0] - tw(l)
        return b0, b0 + tw(l), q[1] - th / 2, q[1] + th / 2
    for _ in range(400):
        moved = False
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                p_, q_ = box(lab[i][1], lab[i][0], lab[i][2]), box(lab[j][1], lab[j][0], lab[j][2])
                if p_[0] < q_[1] and q_[0] < p_[1] and p_[2] < q_[3] and q_[2] < p_[3]:
                    s_ = 1 if lab[i][1][1] >= lab[j][1][1] else -1
                    lab[i][1][1] += s_ * th * 0.1; lab[j][1][1] -= s_ * th * 0.1; moved = True
        if not moved: break
    for p, q, l in lab:
        q[0] = min(q[0], CW - tw(l) - 0.02 * CW) if q[0] >= p[0] else max(q[0], tw(l) + 0.02 * CW)
        q[1] = np.clip(q[1], th, CH - th)
    return lab
LAB = label_layout()

def panel_B(fig, f, mode="full"):
    """mode: full (render + vectors), render (raster only), labels (vectors only, transparent)."""
    ax = ax_mm(fig, f, 0, 13, B_W, B_IMG_H)
    ax.set_xlim(0, CW); ax.set_ylim(CH, 0); ax.set_axis_off()
    if mode in ("full", "render"):
        ax.imshow(crop, extent=[0, CW, CH, 0], interpolation="none", aspect="auto")
    if mode == "render":
        return
    for p, q, l in LAB:
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#333333", lw=0.4)
        ax.text(q[0], q[1], l, fontstyle="italic", ha="left" if q[0] >= p[0] else "right", va="center",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    text_mm(fig, f, 2, B_H - 0.5, "PASTIS PM2 best model, 10 kb beads; view down PC3 of the hub nodes.\n"
            "Consensus model from population Hi-C; not a single-cell structure.", color="#555555", va="top")
    hd = [Line2D([], [], marker="o", ls="", mfc="#383838", mec="black", ms=5, label="promoter-hub core"),
          Line2D([], [], marker="o", ls="", mfc="#CCCCCC", mec="none", ms=2.5, label="low-coverage gap")]
    for g in range(min(NG, 4)):
        hd.append(Line2D([], [], color=OKABE[g], lw=1.4, label="clique groups " + ", ".join(str(k + 1) for k in range(g, NG, 4))))
    FW, FH = fig.get_size_inches() / MM
    fig.legend(handles=hd, loc="upper left", bbox_to_anchor=((f[0] + 1) / FW, (f[1] + 12.5) / FH), frameon=False, ncol=2,
               handlelength=1.6, columnspacing=1.0, borderaxespad=0, labelspacing=0.3, handletextpad=0.5)
    cax = ax_mm(fig, f, B_W - 28, 4.6, 25, 1.6)
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(L0, L1), cmap="viridis"), cax=cax, orientation="horizontal", ticks=[29.73, 31.55])
    cb.ax.set_xticklabels(["29.73", "31.55"]); cb.outline.set_linewidth(0.4); cax.tick_params(length=1.5, pad=1.5)
    text_mm(fig, f, B_W - 0.5, 7.4, "tube colour: genomic position (Mb)", ha="right", va="bottom")

# ---------------- C ----------------
def panel_C(fig, f):
    xl, xw = 17, C_W - 17
    axp = ax_mm(fig, f, xl, 70.5, xw, 2.5); axg = ax_mm(fig, f, xl, 54, xw, 15)
    axt = ax_mm(fig, f, xl, 41, xw, 10); axa = ax_mm(fig, f, xl, 9, xw, 29.5)
    for ax in (axp, axg, axt, axa): ax.set_xlim(L0, L1)
    axp.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap="viridis", extent=[L0, L1, 0, 1])
    for b in np.where(interp)[0]:
        axp.add_patch(Rectangle((mb(b) - RES / 2e6, 0), RES / 1e6, 1, color="#CCCCCC", lw=0))
    axp.set_xlim(L0, L1); axp.set_ylim(0, 1); axp.set_axis_off()
    text_mm(fig, f, xl - 1.5, 71.75, "position", ha="right", va="center")
    G = pd.read_csv(a.genes, sep="\t").sort_values("txStart").drop_duplicates("name")
    rows_end, hlpos = [], []
    hl = set(sum([l.split("/") for l in CB.label], []))
    for r in G.itertuples():
        s, e = r.txStart / 1e6, r.txEnd / 1e6
        if e < L0 or s > L1: continue
        row = next((i for i, re_ in enumerate(rows_end) if s > re_ + 0.004), None)
        if row is None: rows_end.append(e); row = len(rows_end) - 1
        else: rows_end[row] = e
        y = -row; col = "#222222" if r.name in hl else "#9A9A9A"
        axg.plot([s, e], [y, y], color=col, lw=0.5)
        for es, ee in zip(map(int, str(r.exonStarts).strip(",").split(",")), map(int, str(r.exonEnds).strip(",").split(","))):
            axg.add_patch(Rectangle((es / 1e6, y - 0.3), (ee - es) / 1e6, 0.6, color=col, lw=0))
        if r.name in hl:
            hlpos.append((r.txStart / 1e6 if r.strand == "+" else r.txEnd / 1e6, r.name, y))
    hlpos.sort()
    mb_per_in = (L1 - L0) / (xw * MM)
    wid = [len(nm) * FS_SMALL * 0.6 / 72 * mb_per_in + 0.02 for _, nm, _ in hlpos]
    xs = [t for t, _, _ in hlpos]
    for _ in range(300):
        moved = False
        for i in range(len(xs) - 2):
            j = i + 2; need = (wid[i] + wid[j]) / 2
            if xs[j] - xs[i] < need:
                d = (need - (xs[j] - xs[i])) / 2; xs[i] -= d; xs[j] += d; moved = True
        if not moved: break
    for k, ((t, nm, y), xl_) in enumerate(zip(hlpos, xs)):
        yt = 2.4 if k % 2 == 0 else 3.9
        axg.plot([t, t, xl_], [y + 0.4, 1.4, yt - 0.25], color="#555555", lw=0.35)
        axg.text(xl_, yt, nm, fontsize=FS_SMALL, fontstyle="italic", ha="center", va="bottom")
    axg.set_ylim(-len(rows_end) + 0.3, 5.3); axg.set_axis_off()
    text_mm(fig, f, xl - 1.5, 58, "RefSeq", ha="right", va="center")
    axt.bar(mb(bins.bin), bins.n_tfs, width=RES / 1e6 * 0.9, color="#5E4FA2", lw=0)
    axt.bar(mb(CB.bin), bins.n_tfs.values[CB.bin], width=RES / 1e6 * 0.9, color="#222222", lw=0)
    axt.set_ylim(0, max(bins.n_tfs) * 1.05); axt.set_yticks([0, 20]); axt.set_xticks([]); clean(axt)
    text_mm(fig, f, xl - 5, 46, "TFs\n(MACS2)", ha="right", va="center")
    Lp = pd.read_csv(O / "loops.tsv", sep="\t")
    hmax = lmax = 0
    for r in H.itertuples():
        x1_, x2_ = sorted((mb(r.bin1), mb(r.bin2))); h = x2_ - x1_
        axa.add_patch(Arc(((x1_ + x2_) / 2, 0), h, 2 * h, theta1=0, theta2=180, color=r.color, lw=0.5 + 0.3 * r.n_tfs))
        hmax = max(hmax, h)
    for r in Lp.itertuples():
        x1_, x2_ = mb(r.bin1), mb(r.bin2); h = x2_ - x1_
        axa.add_patch(Arc(((x1_ + x2_) / 2, 0), h, h, theta1=180, theta2=360, color="#8C8C8C", lw=0.25))
        lmax = max(lmax, h / 2)
    axa.axhline(0, color="#BBBBBB", lw=0.4)
    axa.set_ylim(-lmax * 1.05, hmax * 1.05); axa.set_yticks([]); clean(axa, left=False)
    axa.set_xticks([29.8, 30.0, 30.2, 30.4, 30.6, 30.8, 31.0, 31.2, 31.4]); axa.set_xlabel("chr16 (Mb, hg38)", labelpad=2)
    fr = hmax / (hmax + lmax)
    text_mm(fig, f, xl - 1.5, 9 + 29.5 * (1 - fr / 2), f"shared\nedges ({len(H)})", ha="right", va="center")
    text_mm(fig, f, xl - 1.5, 9 + 29.5 * (1 - fr) / 2, f"loops ≥0.7\n({len(Lp)})", ha="right", va="center", color="#555555")

# ---------------- D ----------------
def panel_D(fig, f):
    qp = pd.read_csv(M / "quant_pairs_best.tsv", sep="\t"); Q = pd.read_csv(M / "quant_check.tsv", sep="\t")
    nulls = np.load(M / "quant_null_best.npz")
    sets = [("shared_clique_edges", "shared\nedges"), ("cicero_step1_pairs", "Cicero\npairs"), ("peakachu_0.7_loops", "loops")]
    data = [np.log2(qp[qp.set == k].ratio.values) for k, _ in sets]
    null = np.log2(np.random.RandomState(0).choice(np.concatenate([nulls[k] for k, _ in sets]), 5000, replace=False))
    ax = ax_mm(fig, f, 13, 13, 35, 50)
    pos = [1, 2, 3, 4]
    vp = ax.violinplot(data + [null], positions=pos, widths=0.8, showextrema=False)
    for b_, c in zip(vp["bodies"], ["#5E4FA2", "#9E9AC8", "#7A7A7A", "#D9D9D9"]):
        b_.set_facecolor(c); b_.set_edgecolor("none"); b_.set_alpha(0.9)
    ax.boxplot(data + [null], positions=pos, widths=0.16, showfliers=False, medianprops=dict(color="black", lw=0.8),
               boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0))
    ax.axhline(0, color="#555555", lw=0.5, ls=":")
    qb = Q[Q.model_rank == 1].set_index("set")
    for x, (k, _) in zip(pos, sets):
        r = qb.loc[k]; pv = "p<.001" if r.p_one_sided <= 1.5 / 1001 else f"p={r.p_one_sided:.2f}"
        ax.text(x, 1.01, f"n={int(r.n_pairs)}\n{pv}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS_SMALL)
    ax.set_xticks(pos); ax.set_xticklabels([l for _, l in sets] + ["random"])
    ax.set_ylabel("log2 (3D distance / distance-\nmatched random median)", labelpad=2); clean(ax)
    text_mm(fig, f, 30.5, D_H - 1, "3D distance vs genomic-distance-\nmatched random bin pairs (best model)", ha="center", va="top")
    # ensemble inset
    axe = ax_mm(fig, f, 48.5, 16, 17.5, 11)
    ref = top[0][keep] - top[0][keep].mean(0)
    for i in range(len(top) - 1, -1, -1):
        A_ = top[i] - top[i][keep].mean(0)
        U, _, Vt2 = np.linalg.svd(A_[keep].T @ ref); A_ = A_ @ U @ Vt2
        Z = (A_ + top[0][keep].mean(0) - X[nodes].mean(0)) @ Vt.T; Z[interp] = np.nan
        axe.plot(Z[:, 0], Z[:, 1], color="#222222" if i == 0 else "#5E4FA2", lw=0.5 if i == 0 else 0.4, alpha=1 if i == 0 else 0.25)
    axe.set_aspect("equal", adjustable="datalim"); axe.set_axis_off()
    text_mm(fig, f, 57.25, 28, f"top-10 ensemble\n(RMSD {float(qc['rmsd_top10_median']):.1f} beads)", ha="center", va="bottom", fontsize=FS_SMALL)

# ---------------- outputs ----------------
def save(fig, stem, transparent=False, png_dpi=None):
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=600, transparent=transparent)
    if png_dpi:
        fig.savefig(f"{stem}.png", dpi=png_dpi, transparent=transparent)
    plt.close(fig)

FULL_W = 180; GAP = 4
left_h = A_H + D_H; right_h = B_H + C_H
FULL_H = max(left_h, right_h) + 2
fig = plt.figure(figsize=(FULL_W * MM, FULL_H * MM))
fA = (0, FULL_H - A_H - 1); fD = (0, fA[1] - D_H)
fB = (FULL_W - B_W, FULL_H - B_H - 1); fC = (FULL_W - C_W, fB[1] - C_H)
panel_A(fig, fA); panel_B(fig, fB); panel_C(fig, fC); panel_D(fig, fD)
for (fx, fy), (fw, fh), L in ((fA, (A_W, A_H), "A"), (fB, (B_W, B_H), "B"), (fC, (C_W, C_H), "C"), (fD, (D_W, D_H), "D")):
    text_mm(fig, (fx, fy), 0.3 if L in "AD" else -3.5, fh + 0.5, L, fontsize=FS_LETTER, weight="bold", va="top")
save(fig, OUT / "chr16p11.2_3d_model_full", png_dpi=150)

sizes = {"A_split_matrix": (A_W, A_H, panel_A), "C_tracks": (C_W, C_H, panel_C), "D_violin": (D_W, D_H, panel_D)}
for name, (wmm, hmm, fn) in sizes.items():
    fig = plt.figure(figsize=(wmm * MM, hmm * MM)); fn(fig, (0, 0)); save(fig, OUT / "panels" / name)
for mode, name, tr in (("full", "B_3d_structure", False), ("render", "B_3d_structure_render_only", False), ("labels", "B_3d_structure_labels_only", True)):
    fig = plt.figure(figsize=(B_W * MM, B_H * MM)); panel_B(fig, (0, 0), mode); save(fig, OUT / "panels" / name, transparent=tr)

# approximate transparent copy of the existing render (near-white pixels -> transparent; no re-ray-trace)
rgba = img.copy()
if rgba.shape[2] == 3: rgba = np.dstack([rgba, np.ones(rgba.shape[:2])])
mn = rgba[..., :3].min(axis=2)
rgba[..., 3] = np.clip((1 - mn) / 0.06, 0, 1)
plt.imsave(OUT / "panels" / "B_pymol_render_3000px_transparent_approx.png", rgba)

pd.DataFrame([("full figure", FULL_W, round(FULL_H, 1)), ("A_split_matrix", A_W, A_H), ("B_3d_structure (all 3 versions)", B_W, round(B_H, 1)),
              ("C_tracks", C_W, C_H), ("D_violin", D_W, D_H)], columns=["file", "width_mm", "height_mm"]).to_csv(OUT / "sizes_mm.tsv", sep="\t", index=False)
print(f"full {FULL_W} x {FULL_H:.1f} mm; B {B_W} x {B_H:.1f} mm (render {CW}x{CH} px -> {CW / (B_W * MM):.0f} dpi); rho {RHO:.3f}")

# ---------------- separate raster PNGs + SVGs that link to them ----------------
import re
from PIL import Image
RD = OUT / "raster"; RD.mkdir(exist_ok=True)
PT = 72 / 25.4
# A: heatmap pixels only (no axes, no diagonal line), exactly the 47 x 47 mm matrix box at 600 dpi
npx = int(round(47 / 25.4 * 600))
cmapA = plt.get_cmap("Reds").copy(); cmapA.set_bad("#E6E6E6")
rgbaA = (cmapA(np.ma.masked_invalid(SPLIT)) * 255).astype(np.uint8)
Image.fromarray(rgbaA, "RGBA").convert("RGB").resize((npx, npx), Image.NEAREST).save(RD / "A_hic_model_heatmap_600dpi.png", dpi=(600, 600))
# B: the render exactly as placed in panel B (model crop + white label margin)
b_dpi = CW / (B_W * MM)
Image.fromarray((np.clip(crop[..., :3], 0, 1) * 255).astype(np.uint8), "RGB").save(RD / "B_pymol_render.png", dpi=(round(b_dpi), round(b_dpi)))

def relink(src, dst, pick, href, x, y, w, h):
    """replace the pick-th embedded <image> of src by a linked, unflipped image at x, y, w, h (pt, from the top-left)."""
    t = open(src).read()
    ims = list(re.finditer(r"<image\b.*?/>", t, re.S))
    m = ims[pick]
    new = f'<image xlink:href="{href}" x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}" preserveAspectRatio="none"/>'
    open(dst, "w").write(t[:m.start()] + new + t[m.end():])
P = OUT / "panels"
relink(P / "A_split_matrix.svg", P / "A_split_matrix_linked.svg", 0, "../raster/A_hic_model_heatmap_600dpi.png",
       11 * PT, (A_H - 20 - 47) * PT, 47 * PT, 47 * PT)
for stem in ("B_3d_structure", "B_3d_structure_render_only"):
    relink(P / f"{stem}.svg", P / f"{stem}_linked.svg", 0, "../raster/B_pymol_render.png",
           0, (B_H - 13 - B_IMG_H) * PT, B_W * PT, B_IMG_H * PT)
print(f"raster: A heatmap {npx} px = 47 mm @ 600 dpi; B render {CW}x{CH} px = {B_W} x {B_IMG_H:.1f} mm @ {b_dpi:.0f} dpi")
