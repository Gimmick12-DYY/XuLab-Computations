#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# plot_correlation_complexity.py
#
# Clustered TF x TF correlation heatmap with an aligned per-TF cell-count bar
# panel, so the effect of complexity (number of cells / depth) on the correlation
# structure is visible at a glance. Layout: row dendrogram | heatmap | TF labels |
# horizontal bar (number of cells, log x), all sharing the clustered row order.
#
# Inputs:
#   --sim-tsv    a tf_similarity_{genome,A,B}.tsv (TF x TF, from tf_complexes.py)
#   --cell-meta  data/TF1000cells.meta.csv  (per-cell table with a TF column; counted
#                per TF) or a two-column TF,count table
#   --out        output PNG
#
#   python tf_complex/scripts/plot_correlation_complexity.py \
#     --sim-tsv tf_complex/results/tf_similarity_A.tsv \
#     --cell-meta data/TF1000cells.meta.csv --out tf_complex/results/corr_complexity_A.png
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


def load_sim(path):
    tfs, rows = [], []
    for i, ln in enumerate(open(path)):
        t = ln.rstrip("\n").split("\t")
        if i == 0:
            continue
        tfs.append(t[0]); rows.append([float(x) for x in t[1:]])
    return tfs, np.array(rows)


def load_cells(path):
    """Accept either a TF,count table or the per-cell TF1000cells.meta.csv."""
    with open(path) as fh:
        rd = csv.reader(fh)
        rows = list(rd)
    if not rows:
        return {}
    hdr = [h.strip() for h in rows[0]]
    # per-cell table: DNA_id,cell,TF,barcode,subset -> count rows per TF
    if "TF" in hdr:
        i = hdr.index("TF")
        cc = {}
        for r in rows[1:]:
            if len(r) <= i:
                continue
            tf = r[i].strip().lower()
            if tf:
                cc[tf] = cc.get(tf, 0) + 1
        return cc
    cc = {}
    for r in rows:
        if len(r) < 2 or not r[1].strip().isdigit():
            continue
        cc[r[0].strip().lower()] = int(r[1])
    return cc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-tsv", type=Path, required=True)
    ap.add_argument("--cell-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--n-clusters", type=int, default=8, help="color bars by this many clusters")
    ap.add_argument("--cell-in", type=float, default=0.38,
                    help="inches per TF; heatmap is square at n * cell-in")
    ap.add_argument("--font-size", type=float, default=16,
                    help="TF name font size (pt); title/axes scale from this")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tfs, R = load_sim(args.sim_tsv)
    n = len(tfs)
    cells = load_cells(args.cell_meta)
    counts = np.array([cells.get(t.lower(), 0) for t in tfs], dtype=float)
    missing = [t for t in tfs if t.lower() not in cells]
    if missing:
        print(f"[warn] no cell count for {len(missing)}: {','.join(missing[:8])}", flush=True)

    # hierarchical clustering on 1 - correlation
    D = 1.0 - R; D = (D + D.T) / 2.0; np.fill_diagonal(D, 0.0); D[D < 0] = 0.0
    Z = linkage(squareform(D, checks=False), method="average")
    clust = fcluster(Z, t=args.n_clusters, criterion="maxclust")

    # Square heatmap cells: height = n * cell-in; width = dendrogram + heatmap + labels + bars.
    # The old square figsize + aspect='auto' stretched the heatmap into tall rectangles.
    fs = args.font_size
    heat_in = n * args.cell_in
    dend_in, gap_in, bar_in = 1.8, 2.2, 4.4
    left_in, right_in, bot_in, top_in = 0.35, 0.5, 2.8, 1.0
    fig_w = left_in + dend_in + heat_in + gap_in + bar_in + right_in
    fig_h = bot_in + heat_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def box(x, w):
        return [x / fig_w, bot_in / fig_h, w / fig_w, heat_in / fig_h]

    x = left_in
    axd = fig.add_axes(box(x, dend_in)); x += dend_in
    axh = fig.add_axes(box(x, heat_in)); x += heat_in + gap_in
    axb = fig.add_axes(box(x, bar_in))
    cax = fig.add_axes([left_in / fig_w, 0.35 / fig_h, 2.2 / fig_w, 0.38 / fig_h])

    # dendrogram (left); leaves[0] is at the BOTTOM -> origin='lower' on the heatmap
    dn = dendrogram(Z, orientation="left", ax=axd, no_labels=True,
                    link_color_func=lambda k: "#555555")
    order = dn["leaves"]
    axd.set_xticks([]); axd.set_yticks([])
    # scipy left-dendrogram y is 5, 15, ..., 10n-5; keep that scale so leaves line up
    axd.set_ylim(0, 10 * n)
    for s in axd.spines.values():
        s.set_visible(False)

    Ro = R[np.ix_(order, order)]
    off = Ro[~np.eye(n, dtype=bool)]
    vmax = max(0.05, np.nanpercentile(np.abs(off), 99))
    im = axh.imshow(Ro, aspect="equal", origin="lower", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    axh.set_xlim(-0.5, n - 0.5); axh.set_ylim(-0.5, n - 0.5)
    axh.set_xticks(range(n))
    axh.set_xticklabels([tfs[i].upper() for i in order], rotation=90, fontsize=fs)
    axh.set_yticks([])
    axh.tick_params(length=0, pad=3)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Pearson r", fontsize=fs + 1)
    cb.ax.tick_params(labelsize=fs)
    cax.xaxis.set_ticks_position("bottom")

    # cell-count bars (right), same bottom-to-top order, log x, colored by cluster
    cmap = plt.get_cmap("tab10")
    y = np.arange(n)
    bar_counts = counts[order]
    bar_colors = [cmap(clust[order][i] % 10) for i in range(n)]
    axb.barh(y, np.maximum(bar_counts, 1), color=bar_colors, height=0.8)
    axb.set_xscale("log")
    axb.set_ylim(-0.5, n - 0.5)
    axb.set_yticks(range(n))
    axb.set_yticklabels([tfs[i].upper() for i in order], fontsize=fs)
    axb.tick_params(axis="y", length=0, pad=4)
    axb.tick_params(axis="x", labelsize=fs)
    axb.set_xlabel("Number of cells", fontsize=fs + 1)
    axb.grid(axis="x", ls=":", alpha=0.4)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    title = args.title or f"TF-TF correlation vs cell count ({args.sim_tsv.stem})"
    fig.suptitle(title, y=1.0 - 0.18 / fig_h, fontsize=fs + 5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[done] wrote {args.out}  (vmax={vmax:.2f}, {n} TFs, {fig_w:.1f}x{fig_h:.1f} in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
