#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# chromhmm_correlation.py
#
# "TF chromHMM18 rpkm ratio pearson cor": Pearson-correlate TFs over their RPKM
# profile across the 18 ChromHMM states (build_chromhmm_matrix.py), cluster the TFs,
# and render the 3-panel figure:
#   (1) clustered TF x TF correlation heatmap
#   (2) per-TF cell-count bars (complexity)
#   (3) ChromHMM state composition per TF cluster (mean-RPKM proportion, stacked)
#
# Output (--out-dir): tf_similarity.tsv, tf_clusters.tsv, state_composition.tsv,
#   chromhmm_rpkm_figure.png
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal

# standard Roadmap 18-state colours (keyed on the core state name)
STATE_COLORS = {
    "TssA": "#FF0000", "TssFlnk": "#FF4500", "TssFlnkU": "#FF4500", "TssFlnkD": "#FF4500",
    "Tx": "#008000", "TxWk": "#006400", "EnhG1": "#C2E105", "EnhG2": "#C2E105",
    "EnhA1": "#FFC34D", "EnhA2": "#FFC34D", "EnhWk": "#FFFF00", "ZNF/Rpts": "#66CDAA",
    "Het": "#8A91D0", "TssBiv": "#CD5C5C", "EnhBiv": "#BDB76B", "ReprPC": "#808080",
    "ReprPCWk": "#C0C0C0", "Quies": "#DCDCDC",
}
STATE_ORDER = ["TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "Tx", "TxWk", "EnhG1", "EnhG2",
               "EnhA1", "EnhA2", "EnhWk", "ZNF/Rpts", "Het", "TssBiv", "EnhBiv",
               "ReprPC", "ReprPCWk", "Quies"]


def core(state):
    return state.split("_", 1)[1] if "_" in state and state.split("_")[0].isdigit() else state


def load_cells(path):
    """Accept either a TF,count table or the per-cell TF1000cells.meta.csv."""
    if not path or not Path(path).is_file():
        return {}
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return {}
    hdr = [h.strip() for h in rows[0]]
    if "TF" in hdr:                       # per-cell metadata -> count cells per TF
        i = hdr.index("TF")
        cc = {}
        for r in rows[1:]:
            if len(r) > i and r[i].strip():
                tf = r[i].strip().lower()
                cc[tf] = cc.get(tf, 0) + 1
        return cc
    cc = {}
    for r in rows:                        # already a TF,count table
        if len(r) >= 2 and r[1].strip().isdigit():
            cc[r[0].strip().lower()] = int(r[1])
    return cc


def state_stars(M, clust):
    """Kruskal-Wallis across TF clusters on each state's per-TF RPKM proportion."""
    tot = M.sum(axis=0, keepdims=True)
    tot[tot == 0] = 1.0
    P = M / tot
    groups = [np.flatnonzero(clust == c) for c in sorted(set(clust))]
    stars, pvals = [], []
    for k in range(P.shape[0]):
        vals = [P[k, g] for g in groups if g.size > 1]
        try:
            p = float(kruskal(*vals)[1]) if len(vals) >= 2 else 1.0
        except ValueError:                # identical values in every group
            p = 1.0
        pvals.append(p)
        stars.append("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "")
    return stars, pvals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cell-meta", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--transform", choices=["rpkm", "log", "ratio"], default="rpkm",
                    help="rpkm (default) | log = log1p(rpkm) | ratio = rpkm / mean-across-states")
    args = ap.parse_args()

    M = np.asarray(sp.load_npz(args.matrix_dir / "chromhmm_matrix.npz").todense())  # states x TFs
    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    states = [ln.split("\t")[0] for ln in open(args.matrix_dir / "states.tsv")]
    n_tf = len(tfs)
    X = M.copy()
    if args.transform == "log":
        X = np.log1p(X)
    elif args.transform == "ratio":
        mu = X.mean(axis=1, keepdims=True); mu[mu == 0] = 1.0
        X = X / mu
    R = np.nan_to_num(np.corrcoef(X, rowvar=False))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "tf_similarity.tsv").open("w") as f:
        f.write("TF\t" + "\t".join(tfs) + "\n")
        for i, t in enumerate(tfs):
            f.write(t + "\t" + "\t".join(f"{R[i, j]:.4f}" for j in range(n_tf)) + "\n")

    # cluster TFs
    D = 1 - R; D = (D + D.T) / 2; np.fill_diagonal(D, 0); D[D < 0] = 0
    Z = linkage(squareform(D, checks=False), method="average")
    clust = fcluster(Z, t=args.n_clusters, criterion="maxclust")
    with (args.out_dir / "tf_clusters.tsv").open("w") as f:
        f.write("tf\tcluster\n")
        for t, c in zip(tfs, clust):
            f.write(f"{t}\t{c}\n")

    # per-cluster ChromHMM composition (mean RPKM across cluster's TFs, normalized)
    stars, pvals = state_stars(M, clust)
    comp = {}
    with (args.out_dir / "state_composition.tsv").open("w") as f:
        f.write("cluster\t" + "\t".join(states) + "\n")
        for c in sorted(set(clust)):
            m = M[:, clust == c].mean(axis=1)
            prop = m / (m.sum() if m.sum() else 1.0)
            comp[c] = prop
            f.write(f"Cluster{c}\t" + "\t".join(f"{v:.5f}" for v in prop) + "\n")
        f.write("p_kruskal\t" + "\t".join(f"{p:.3g}" for p in pvals) + "\n")

    cells = load_cells(args.cell_meta)
    print(f"[cells] {len(cells)} TFs in {args.cell_meta}; "
          f"{sum(1 for t in tfs if t.lower() in cells)}/{n_tf} matched")
    _plot(R, tfs, states, Z, clust, comp, cells, args.out_dir, stars=stars)
    print(f"[done] {n_tf} TFs, {len(states)} states, {args.n_clusters} clusters -> {args.out_dir}")
    return 0


def _plot(R, tfs, states, Z, clust, comp, cells, out_dir, stars=None, vmin=0.65):
    """Reference layout: dendrogram | TFxTF heatmap (RdYlBu_r, 0.65-1) | cell-count bars |
    stacked ChromHMM state composition per TF cluster."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    n = len(tfs)
    stars = stars or [""] * len(states)
    dn = dendrogram(Z, no_plot=True); order = dn["leaves"]
    fig = plt.figure(figsize=(20, max(9, n * 0.135)))
    # Explicit geometry: the dendrogram must butt against the heatmap, while the
    # heatmap->bars gap has to hold two columns of TF labels.
    axd = fig.add_axes([0.130, 0.10, 0.085, 0.84])
    axh = fig.add_axes([0.215, 0.10, 0.345, 0.84])
    axb = fig.add_axes([0.620, 0.10, 0.130, 0.84])
    axc = fig.add_axes([0.805, 0.10, 0.115, 0.84])
    hsv = plt.get_cmap("hsv")
    # Cell bars: pink at top -> red at bottom (reference rainbow, no wrap).
    # ChromHMM states: salmon TssA -> rose Quies, one hue per state.
    def hsv_span(n, h0, h1):
        if n <= 1:
            return [hsv(h0)]
        return [hsv(h0 + (h1 - h0) * i / (n - 1)) for i in range(n)]
    bar_cols = hsv_span(n, 0.02, 0.92)   # red at bottom (y=0) -> pink at top

    fig.text(0.015, 0.80, "TF\nchromHMM18\nrpkm ratio\npearson cor", fontsize=15, va="top")

    dendrogram(Z, orientation="left", ax=axd, no_labels=True, link_color_func=lambda k: "#555")
    axd.set_xticks([]); axd.set_yticks([]); [s.set_visible(False) for s in axd.spines.values()]
    axd.set_ylim(0, 10 * n)

    Ro = R[np.ix_(order, order)]
    im = axh.imshow(Ro, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=vmin, vmax=1.0)
    axh.set_xticks(range(n)); axh.set_xticklabels([tfs[i].upper() for i in order], rotation=90, fontsize=5)
    axh.yaxis.tick_right(); axh.set_yticks(range(n))
    axh.set_yticklabels([tfs[i].upper() for i in order], fontsize=5)
    axh.set_xticks(np.arange(-0.5, n, 1), minor=True)
    axh.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axh.grid(which="minor", color="white", linewidth=0.4)
    axh.tick_params(which="minor", length=0); axh.tick_params(length=0)
    cax = fig.add_axes([0.045, 0.30, 0.009, 0.22])      # colorbar, far left (matches reference)
    fig.colorbar(im, cax=cax, ticks=[0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

    y = np.arange(n)
    cnt = np.array([cells.get(tfs[i].lower(), 1) for i in order], float)
    axb.barh(y, np.maximum(cnt, 1), color=bar_cols, height=0.8)
    axb.set_xscale("log"); axb.set_ylim(-0.5, n - 0.5)
    axb.set_yticks(range(n)); axb.set_yticklabels([tfs[i].upper() for i in order], fontsize=5)
    axb.tick_params(length=0); axb.set_xlabel("Number of cells")
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    # stacked ChromHMM composition per cluster, canonical state order (TssA on top)
    idx = sorted(range(len(states)), key=lambda k: STATE_ORDER.index(core(states[k]))
                 if core(states[k]) in STATE_ORDER else 99)
    state_cols = hsv_span(len(idx), 0.02, 0.90)
    scol = {k: state_cols[p] for p, k in enumerate(idx)}
    clusters = sorted(comp)
    for ci, c in enumerate(clusters):
        bottom = 0.0
        for k in reversed(idx):
            axc.bar(ci, comp[c][k], bottom=bottom, width=0.55, color=scol[k])
            bottom += comp[c][k]
    axc.set_xticks(range(len(clusters)))
    axc.set_xticklabels([f"Cluster{c}" for c in clusters], fontsize=8)
    axc.set_xlim(-0.6, len(clusters) - 0.4)
    axc.set_xlabel("TF Cluster", fontsize=9); axc.set_ylim(0, 1)
    axc.set_yticks([0, .25, .5, .75, 1.0]); axc.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axc.set_ylabel("Proportion of States (Based on Mean RPKM)", fontsize=9)
    axc.set_title("ChromHMM State Composition\nAcross TF Clusters", fontsize=10)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    handles = [Patch(color=scol[k], label=f"{core(states[k])}{stars[k]}") for k in idx]
    axc.legend(handles=handles, fontsize=6, bbox_to_anchor=(1.06, 0.96), loc="upper left",
               title="ChromHMM State\n(* p<0.05)", title_fontsize=7, frameon=False)

    fig.savefig(out_dir / "chromhmm_rpkm_figure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_dir/'chromhmm_rpkm_figure.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
