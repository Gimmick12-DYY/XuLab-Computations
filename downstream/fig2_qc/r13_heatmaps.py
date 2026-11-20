#!/usr/bin/env python
"""Round 6 peak heatmaps (deepTools-style), same regions and row order in every column.
Regions: each panel TF's D-B differential regions (top 1,000 by p; r12b_diff_stats.py), centred on the region midpoint
(D-B regions are merged candidates without a summit). Regions in more than one TF's set are dropped (count reported).
Blocks stacked in panel order; within a block rows sorted by that TF's bulk ChIP signal (sum over +-2 kb), descending.
Tracks: bulk = ChIP CPM bigWig (bamCoverage: duplicates ignored, reads extended to 200 bp, 50 bp bins); raw = aggregated
single-cell 5' ends, CPM of that TF's reads, full depth, 50 bp bins; window +-2 kb. Imputed = fraction of cells positive per
1 kb bin (imputation is only at 1 kb and inside the ATAC union), shown over +-5 kb at 1 kb. Pool = all 79 TFs' raw reads, CPM.
Colour: per column 0 to that column's 99th percentile; second version z-scored per column against its own background
(outer bins: |d| >= 1.5 kb for 50 bp tracks, |d| >= 4 kb for imputed), shown 0 to the 99th percentile of z.
Outputs OUT/round6_diffpeaks/heatmaps/: L1_<TF>_{pct99,bgz}.*, L2_{bytype,byTF}_{pct99,bgz}.*, block_enrichment.tsv/.xlsx."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from matplotlib import font_manager

from fig2_common import INP, OUT, ROOT, TFS, bin_index, load_imp7

O6 = OUT / "round6_diffpeaks"
OH = O6 / "heatmaps"
OH.mkdir(parents=True, exist_ok=True)
W, STEP, NB = 2000, 50, 80
IMPW = 5
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
R = pd.read_csv(O6 / "rows.tsv", sep="\t")
S = np.load(O6 / "diff_sets.npz")
sets = {t: S[f"D-B_1000_{t}"] for t in TFS}
allidx = np.concatenate(list(sets.values()))
u, cnt = np.unique(allidx, return_counts=True)
shared = set(u[cnt > 1])
sets = {t: np.array([i for i in v if i not in shared]) for t, v in sets.items()}
print("regions per TF after dropping", len(shared), "shared:", {t: len(v) for t, v in sets.items()}, flush=True)
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]
ncell = fr["ncells_imp79"]
imp = load_imp7()
csize = dict(zip(bi.chrom, bi["size"]))

blocks = []
for t in TFS:
    ix = sets[t]
    blocks.append(pd.DataFrame({"block": t, "row": ix, "chrom": R.chrom.values[ix],
                                "centre": ((R.start.values[ix] + R.end.values[ix]) // 2).astype(int)}))
B = pd.concat(blocks, ignore_index=True)
n = len(B)


def bw_matrix(path):
    bw = pyBigWig.open(str(path))
    M = np.zeros((n, NB))
    for r, (c, x) in enumerate(zip(B.chrom.values, B.centre.values)):
        a, b = x - W, x + W
        if a < 0 or b > csize[c]:
            continue
        v = bw.stats(c, int(a), int(b), nBins=NB, type="mean")
        M[r] = np.nan_to_num(np.array(v, dtype=float))
    return M


bulk = {k: bw_matrix(OH / "bw" / f"{k}.bw") for k in TFS}
edges = np.arange(-W, W + 1, STEP)
five = {}


def raw_matrix(t):
    z = np.load(INP / "five" / f"{t}.npz")
    tot = sum(len(z[c]) for c in z.files)
    M = np.zeros((n, NB))
    for c in np.unique(B.chrom.values):
        rr = np.flatnonzero(B.chrom.values == c)
        p = z[c]
        lo = np.searchsorted(p, B.centre.values[rr] - W); hi = np.searchsorted(p, B.centre.values[rr] + W)
        for r, l, h in zip(rr, lo, hi):
            M[r] = np.histogram(p[l:h] - B.centre.values[r], edges)[0]
    return M, tot


raw, pool, ptot = {}, np.zeros((n, NB)), 0.0
for t in tf79:
    M, tot = raw_matrix(t)
    pool += M; ptot += tot
    if t in TFS:
        raw[t] = M / tot * 1e6
pool = pool / ptot * 1e6
impm = {}
cb = np.array([off[c][0] for c in B.chrom.values]) + B.centre.values // 1000
for i, t in enumerate(TFS):
    M = np.zeros((n, 2 * IMPW + 1))
    for d in range(-IMPW, IMPW + 1):
        M[:, d + IMPW] = imp[i, np.clip(cb + d, 0, imp.shape[1] - 1)] / ncell[tf79.index(t)]
    impm[t] = M
# order rows: within block by own bulk sum
order = []
for t in TFS:
    rr = np.flatnonzero(B.block.values == t)
    order.append(rr[np.argsort(-bulk[t][rr].sum(1), kind="stable")])
order = np.concatenate(order)
bounds = np.cumsum([len(sets[t]) for t in TFS])[:-1]
np.savez(OH / "heatmap_matrices.npz", order=order, blocks=B.block.values, raw=np.array([raw[t] for t in TFS]),
         bulk=np.array([bulk[t] for t in TFS]), imp=np.array([impm[t] for t in TFS]), pool=pool)

# block enrichment table
ctr50 = slice(NB // 2 - 5, NB // 2 + 5)    # +-250 bp
rows_e = []
for k, t in enumerate(TFS):
    for typ, M, cs in (("ChIP", bulk[t], ctr50), ("raw", raw[t], ctr50), ("imputed", impm[t], slice(IMPW, IMPW + 1))):
        v = M[:, cs].mean(1)
        own = v[B.block.values == t].mean(); oth = v[B.block.values != t].mean()
        per = {f"block_{u}": v[B.block.values == u].mean() for u in TFS}
        rows_e.append(dict(track=f"{t} {typ}", own_block_mean=own, other_blocks_mean=oth, own_over_other=own / max(oth, 1e-12),
                           own_block_rank_of6=int(1 + sum(per[f'block_{u}'] > own for u in TFS)), **per))
pv = pool[:, ctr50].mean(1)
rows_e.append(dict(track="79-TF pool raw", **{f"block_{u}": pv[B.block.values == u].mean() for u in TFS}))
E = pd.DataFrame(rows_e)
E.to_csv(OH / "block_enrichment.tsv", sep="\t", index=False, float_format="%.5g")
try:
    E.to_excel(OH / "block_enrichment.xlsx", index=False)
except Exception as e:
    print("xlsx", e)
pd.set_option("display.width", 250)
print(E.round(4).to_string(), flush=True)

# ---- plotting ----
for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
CM = P.seq_cmap()


def bgz(M, imp_track):
    bg = np.r_[M[:, :2], M[:, -2:]].ravel() if imp_track else np.r_[M[:, :10], M[:, -10:]].ravel()
    return (M - bg.mean()) / (bg.std() + 1e-12)


def panel(fig, gs_col, M, title, imp_track, scale, ylabel=False):
    g = gs_col.subgridspec(2, 1, height_ratios=[1, 5], hspace=0.05)
    X = bgz(M, imp_track) if scale == "bgz" else M
    Xo = X[order_rows]
    vmax = np.percentile(Xo, 99) or 1
    vmin = 0
    axp = fig.add_subplot(g[0])
    xs = np.arange(-IMPW, IMPW + 1) * 1000 if imp_track else (edges[:-1] + edges[1:]) / 2
    axp.plot(xs, Xo.mean(0), color=P.neutral.ink, lw=0.7)
    axp.set_xticks([]); axp.tick_params(labelsize=5)
    axp.set_title(title, fontsize=5.5)
    ax = fig.add_subplot(g[1])
    ext = [-(IMPW + .5) * 1000, (IMPW + .5) * 1000] if imp_track else [-W, W]
    im = ax.imshow(np.clip(Xo, vmin, vmax), aspect="auto", cmap=CM, vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=[ext[0], ext[1], len(Xo), 0], rasterized=True)
    for y in bnd_rows:
        ax.axhline(y, color=P.neutral.mid, lw=0.4)
    ax.set_xticks([ext[0] + (500 if imp_track else 0), 0, ext[1] - (500 if imp_track else 0)],
                  ["-5 kb" if imp_track else "-2 kb", "0", "+5 kb" if imp_track else "+2 kb"], fontsize=5)
    if ylabel:
        mids = np.r_[0, bnd_rows] + np.diff(np.r_[0, bnd_rows, len(Xo)]) / 2
        ax.set_yticks(mids, [f"{t}\n(n={len(sets[t])})" for t in blk_names], fontsize=5.5)
    else:
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.06)
    cbar.ax.tick_params(labelsize=4.5, length=1.5)
    cbar.set_label(("z vs own background" if scale == "bgz" else "CPM" if not imp_track else "fraction of cells") + ", 0–99th pct", fontsize=4.5)


LABEL = {"ChIP": lambda t: f"{t} ChIP ({CELL[t]})", "raw": lambda t: f"{t} sc raw", "imputed": lambda t: f"{t} sc imputed\n(1 kb bins)"}
NOTE = ("Rows: sc differential regions (D-B, top 1,000; shared regions removed), centred on region midpoint, sorted by own ChIP. "
        "ChIP/raw/pool: CPM, 50 bp bins, ±2 kb. Imputed: fraction of cells, 1 kb bins, ±5 kb (ATAC-masked).")
for scale in ("pct99", "bgz"):
    # L1 per TF
    for t in TFS:
        order_rows = np.flatnonzero(B.block.values[order] == t)
        order_rows = order[order_rows]
        bnd_rows, blk_names = [], [t]
        fig = plt.figure(figsize=(5.0, 4.6))
        gs = fig.add_gridspec(1, 4, wspace=0.35, left=0.12, right=0.98, top=0.88, bottom=0.12)
        panel(fig, gs[0], bulk[t], LABEL["ChIP"](t), False, scale, ylabel=True)
        panel(fig, gs[1], raw[t], LABEL["raw"](t), False, scale)
        panel(fig, gs[2], impm[t], LABEL["imputed"](t), True, scale)
        panel(fig, gs[3], pool, "79-TF pool raw\n(open chromatin ref.)", False, scale)
        fig.suptitle(f"{t} differential regions ({'colour 0–99th percentile per column' if scale == 'pct99' else 'z-scored per column vs own background'})", fontsize=6)
        fig.text(0.01, 0.01, NOTE, fontsize=4.5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OH / f"L1_{t}_{scale}.{ext}", dpi=600)
        plt.close(fig)
    # L2 combined
    order_rows = order
    bnd_rows = list(bounds); blk_names = TFS
    for lay in ("bytype", "byTF"):
        cols = ([("ChIP", t) for t in TFS] + [("raw", t) for t in TFS] + [("imputed", t) for t in TFS]) if lay == "bytype" else \
               [(ty, t) for t in TFS for ty in ("ChIP", "raw", "imputed")]
        fig = plt.figure(figsize=(14.0, 7.5))
        gs = fig.add_gridspec(1, 18, wspace=0.25, left=0.06, right=0.995, top=0.9, bottom=0.1)
        for c, (ty, t) in enumerate(cols):
            M = bulk[t] if ty == "ChIP" else raw[t] if ty == "raw" else impm[t]
            panel(fig, gs[c], M, LABEL[ty](t), ty == "imputed", scale, ylabel=(c == 0))
        fig.suptitle(f"All six blocks ({'colour 0–99th percentile per column' if scale == 'pct99' else 'z-scored per column vs own background'}); columns grouped {'by data type' if lay == 'bytype' else 'by TF'}", fontsize=7)
        fig.text(0.01, 0.01, NOTE, fontsize=5)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OH / f"L2_{lay}_{scale}.{ext}", dpi=600)
        plt.close(fig)
print("done")
