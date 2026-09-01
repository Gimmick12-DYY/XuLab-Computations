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
#   --cell-meta  data/TF1000cells.meta.csv  ("Row Labels,Count of cell"): per-TF cells
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
    cc = {}
    with open(path) as fh:
        rd = csv.reader(fh)
        for r in rd:
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
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

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

    fig = plt.figure(figsize=(max(9, n * 0.28), max(8, n * 0.28)))
    gs = GridSpec(1, 3, width_ratios=[0.13, 0.58, 0.29], wspace=0.09)
    axd = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1])
    axb = fig.add_subplot(gs[2])

    # dendrogram (left); leaves[0] is at the BOTTOM -> use origin='lower' to align
    dn = dendrogram(Z, orientation="left", ax=axd, no_labels=True,
                    link_color_func=lambda k: "#555555")
    order = dn["leaves"]
    axd.set_xticks([]); axd.set_yticks([])
    for s in axd.spines.values():
        s.set_visible(False)

    Ro = R[np.ix_(order, order)]
    off = Ro[~np.eye(n, dtype=bool)]
    vmax = max(0.05, np.nanpercentile(np.abs(off), 99))
    im = axh.imshow(Ro, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axh.set_xticks(range(n)); axh.set_xticklabels([tfs[i].upper() for i in order], rotation=90, fontsize=6)
    axh.set_yticks([]); axh.tick_params(length=0)
    cax = fig.add_axes([0.055, 0.12, 0.013, 0.16])
    fig.colorbar(im, cax=cax, label="Pearson r")

    # cell-count bars (right), same bottom-to-top order, log x, colored by cluster.
    # TF labels sit on the LEFT of the bar panel = in the gap between heatmap and bars.
    cmap = plt.get_cmap("tab10")
    y = np.arange(n)
    bar_counts = counts[order]
    bar_colors = [cmap(clust[order][i] % 10) for i in range(n)]
    axb.barh(y, np.maximum(bar_counts, 1), color=bar_colors, height=0.8)
    axb.set_xscale("log")
    axb.set_ylim(-0.5, n - 0.5)
    axb.set_yticks(range(n))
    axb.set_yticklabels([tfs[i].upper() for i in order], fontsize=6)
    axb.tick_params(axis="y", length=0)
    axb.set_xlabel("Number of cells")
    axb.grid(axis="x", ls=":", alpha=0.4)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    title = args.title or f"TF-TF correlation vs cell count ({args.sim_tsv.stem})"
    fig.suptitle(title, y=0.995, fontsize=11)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {args.out}  (vmax={vmax:.2f}, {n} TFs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
