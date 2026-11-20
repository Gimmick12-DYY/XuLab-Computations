#!/usr/bin/env python
"""Round 6 peak heatmaps, re-rendered (replaces the r13_heatmaps.py figures; matrices from r13 heatmap_matrices.npz).
Same regions and the same row order in every column: each panel TF's D-B differential regions (top 1,000, shared regions
removed), centred on the region midpoint, stacked one block per TF; within a block rows sorted by that TF's ChIP CPM
within +-500 bp, descending. All columns on one +-2 kb axis with 50 bp bins:
  ChIP = bamCoverage CPM (dups ignored, reads extended to 200 bp);
  sc raw = aggregated 5' ends, CPM, 3-bin (150 bp) running mean;
  sc imputed = fraction of cells positive in the 1 kb genomic bin covering each 50 bp position (nearest-neighbour, so
  steps are 1 kb wide; imputation exists only at 1 kb and inside the ATAC union).
Colour per column: 0 to the 98th percentile of that column's nonzero values. Profiles: mean over all rows (L2) or the
block (L1, L3), y from 0, shared y-limits per data type.
Outputs OH: L2_byTF (6 blocks x 18 columns, 183 mm), L3_matched (each block shows its own TF; 3 columns, 89 mm),
L1_<TF> (own block: ChIP, raw, imputed, 79-TF pool). .png/.pdf/.svg."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from fig2_common import INP, OUT, ROOT, TFS, bin_index, load_imp7

O6 = OUT / "round6_diffpeaks"
OH = O6 / "heatmaps"
W, STEP, NB = 2000, 50, 80
MM = 1 / 25.4
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}

R = pd.read_csv(O6 / "rows.tsv", sep="\t")
S = np.load(O6 / "diff_sets.npz")
sets = {t: S[f"D-B_1000_{t}"] for t in TFS}
u, cnt = np.unique(np.concatenate(list(sets.values())), return_counts=True)
shared = set(u[cnt > 1])
sets = {t: np.array([i for i in v if i not in shared]) for t, v in sets.items()}
blk = np.concatenate([[t] * len(sets[t]) for t in TFS])
ix = np.concatenate([sets[t] for t in TFS])
chrom = R.chrom.values[ix]
centre = ((R.start.values[ix] + R.end.values[ix]) // 2).astype(int)
n = len(ix)

H = np.load(OH / "heatmap_matrices.npz", allow_pickle=True)
assert (H["blocks"] == blk).all()
bulk = dict(zip(TFS, H["bulk"]))
pool = H["pool"]
k3 = np.ones(3) / 3
raw = {t: np.apply_along_axis(lambda r: np.convolve(r, k3, mode="same"), 1, M) for t, M in zip(TFS, H["raw"])}
# imputed on the same 50 bp axis: genomic 1 kb bin covering each position
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]
ncell = fr["ncells_imp79"]
imp = load_imp7()
xs = np.arange(-W, W, STEP) + STEP // 2
base = np.array([off[c][0] for c in chrom])
gb = base[:, None] + (centre[:, None] + xs[None, :]) // 1000
gb = np.clip(gb, 0, imp.shape[1] - 1)
impm = {t: imp[i][gb] / ncell[tf79.index(t)] for i, t in enumerate(TFS)}

ctr = slice(NB // 2 - 10, NB // 2 + 10)  # +-500 bp
order = np.concatenate([np.flatnonzero(blk == t)[np.argsort(-bulk[t][blk == t][:, ctr].sum(1), kind="stable")] for t in TFS])
sizes = [len(sets[t]) for t in TFS]
bounds = np.cumsum(sizes)[:-1]
print("rows per block", dict(zip(TFS, sizes)), flush=True)

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
plt.rcParams.update({"font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 6, "ytick.labelsize": 6})
CM = P.seq_cmap()
TYPES = {"ChIP": bulk, "raw": raw, "imputed": impm}
UNIT = {"ChIP": "CPM", "raw": "CPM", "imputed": "frac. cells", "pool": "CPM"}


def vmax_of(M):
    nz = M[M > 0]
    return float(np.percentile(nz, 98)) if nz.size else 1.0


def draw(fig, gcell, M, rows, bnds, ylim, title, unit, show_y=False, ylabels=None, prof_rows=None):
    g = gcell.subgridspec(4, 1, height_ratios=[1.1, 6, 0.32, 0.16], hspace=0.06)
    axp = fig.add_subplot(g[0])
    pr = M[rows if prof_rows is None else prof_rows].mean(0)
    axp.plot(xs, pr, color=P.neutral.ink, lw=0.6)
    axp.set_xlim(-W, W); axp.set_ylim(0, ylim); axp.set_xticks([])
    axp.spines[["top", "right"]].set_visible(False)
    axp.set_yticks([0, float(f"{ylim / 1.08:.1g}")] if f"{ylim / 1.08:.1g}" != "0" else [0])
    axp.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    if not show_y:
        axp.tick_params(labelleft=False)
    axp.set_title(title, fontsize=7, pad=2)
    ax = fig.add_subplot(g[1])
    vm = vmax_of(M[rows])
    im = ax.imshow(M[rows], aspect="auto", cmap=CM, vmin=0, vmax=vm, interpolation="nearest",
                   extent=[-W, W, len(rows), 0], rasterized=True)
    for y in bnds:
        ax.axhline(y, color="white", lw=1.0)
    ax.set_xticks([-W, 0, W], ["−2", "0", "+2"])
    ax.tick_params(axis="x", pad=1)
    for lab, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
        lab.set_ha(ha)
    for s in ax.spines.values():
        s.set_visible(False)
    if ylabels is not None:
        mids = np.r_[0, bnds] + np.diff(np.r_[0, list(bnds), len(rows)]) / 2
        ax.set_yticks(mids, ylabels); ax.tick_params(axis="y", length=0)
    else:
        ax.set_yticks([])
    cax = fig.add_subplot(g[3])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    for lab, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        lab.set_ha(ha)
    return ax


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OH / f"{name}.{ext}", dpi=600)
    plt.close(fig)


blabels = [f"{t}\n{s:,}" for t, s in zip(TFS, sizes)]
# ---- L2: 6 blocks x 18 columns, grouped by TF ----
ylims = {ty: 1.08 * max(D[t][order].mean(0).max() for t in TFS) for ty, D in TYPES.items()}
fig = plt.figure(figsize=(183 * MM, 150 * MM))
gs = fig.add_gridspec(1, 6, wspace=0.14, left=0.085, right=0.995, top=0.90, bottom=0.07)
for k, t in enumerate(TFS):
    sg = gs[k].subgridspec(1, 3, wspace=0.12)
    for j, ty in enumerate(TYPES):
        draw(fig, sg[j], TYPES[ty][t], order, bounds, ylims[ty], {"ChIP": "ChIP", "raw": "raw", "imputed": "imp."}[ty],
             UNIT[ty], show_y=(k == 0 and j == 0), ylabels=blabels if (k == 0 and j == 0) else None)
    x0 = gs[k].get_position(fig)
    fig.text((x0.x0 + x0.x1) / 2, 0.955, f"{t} ({CELL[t]} ChIP)", ha="center", fontsize=7, fontweight="bold")
fig.text(0.535, 0.012, "Distance from region centre (kb)", ha="center", fontsize=7)
save(fig, "L2_byTF")

# ---- L3: matched (each block shows its own TF) ----
mat = {ty: np.zeros((n, NB)) for ty in TYPES}
for t in TFS:
    m = blk == t
    for ty in TYPES:
        mat[ty][m] = TYPES[ty][t][m]
fig = plt.figure(figsize=(89 * MM, 120 * MM))
gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.17, right=0.98, top=0.93, bottom=0.08)
for j, (ty, ttl) in enumerate((("ChIP", "Bulk ChIP"), ("raw", "sc raw"), ("imputed", "sc imputed"))):
    draw(fig, gs[j], mat[ty], order, bounds, 1.08 * mat[ty][order].mean(0).max(), ttl, UNIT[ty],
         show_y=True, ylabels=blabels if j == 0 else None)
fig.text(0.575, 0.012, "Distance from region centre (kb)", ha="center", fontsize=7)
save(fig, "L3_matched")

# ---- L1: own block, 4 columns ----
for t in TFS:
    rows = order[blk[order] == t]
    fig = plt.figure(figsize=(89 * MM, 80 * MM))
    gs = fig.add_gridspec(1, 4, wspace=0.5, left=0.13, right=0.98, top=0.88, bottom=0.11)
    cols = [(bulk[t], f"ChIP\n({CELL[t]})", "ChIP"), (raw[t], "sc raw", "raw"), (impm[t], "sc imputed", "imputed"),
            (pool, "79-TF pool\nraw", "pool")]
    for j, (M, ttl, ty) in enumerate(cols):
        draw(fig, gs[j], M, rows, [], 1.08 * M[rows].mean(0).max(), ttl, UNIT[ty], show_y=True,
             ylabels=[f"{t}\n{len(rows):,}"] if j == 0 else None)
    fig.text(0.555, 0.015, "Distance from region centre (kb)", ha="center", fontsize=7)
    save(fig, f"L1_{t}")

# own-block vs other-block numbers on the displayed (+-250 bp) centre, all on the same rows
c5 = slice(NB // 2 - 5, NB // 2 + 5)
rec = []
for t in TFS:
    for ty in TYPES:
        v = TYPES[ty][t][:, c5].mean(1)
        per = {u: v[blk == u].mean() for u in TFS}
        rec.append(dict(track=f"{t} {ty}", own=per[t], others=np.mean([per[u] for u in TFS if u != t]),
                        own_over_others=per[t] / max(np.mean([per[u] for u in TFS if u != t]), 1e-12),
                        own_rank_of6=1 + sum(per[u] > per[t] for u in TFS), **{f"block_{u}": per[u] for u in TFS}))
E = pd.DataFrame(rec)
E.to_csv(OH / "block_enrichment.tsv", sep="\t", index=False, float_format="%.5g")
E.to_excel(OH / "block_enrichment.xlsx", index=False)
pd.set_option("display.width", 250)
print(E.round(4).to_string())
print("done")
