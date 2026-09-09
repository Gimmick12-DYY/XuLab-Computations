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
    cc = {}
    if path and Path(path).is_file():
        for r in csv.reader(open(path)):
            if len(r) >= 2 and r[1].strip().isdigit():
                cc[r[0].strip().lower()] = int(r[1])
    return cc


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
    comp = {}
    with (args.out_dir / "state_composition.tsv").open("w") as f:
        f.write("cluster\t" + "\t".join(states) + "\n")
        for c in sorted(set(clust)):
            m = M[:, clust == c].mean(axis=1)
            prop = m / (m.sum() if m.sum() else 1.0)
            comp[c] = prop
            f.write(f"Cluster{c}\t" + "\t".join(f"{v:.5f}" for v in prop) + "\n")

    cells = load_cells(args.cell_meta)
    _plot(R, tfs, states, Z, clust, comp, cells, args.out_dir)
    print(f"[done] {n_tf} TFs, {len(states)} states, {args.n_clusters} clusters -> {args.out_dir}")
    return 0


def _plot(R, tfs, states, Z, clust, comp, cells, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    n = len(tfs)
    dn = dendrogram(Z, no_plot=True); order = dn["leaves"]
    fig = plt.figure(figsize=(20, max(10, n * 0.22)))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.12, 0.52, 0.16, 0.20], wspace=0.04)
    axd = fig.add_subplot(gs[0]); axh = fig.add_subplot(gs[1])
    axb = fig.add_subplot(gs[2]); axc = fig.add_subplot(gs[3])

    dendrogram(Z, orientation="left", ax=axd, no_labels=True, link_color_func=lambda k: "#555")
    axd.set_xticks([]); axd.set_yticks([]); [s.set_visible(False) for s in axd.spines.values()]
    axd.set_ylim(0, 10 * n)

    Ro = R[np.ix_(order, order)]
    im = axh.imshow(Ro, aspect="auto", origin="lower", cmap="RdYlBu_r",
                    vmin=np.nanpercentile(Ro, 2), vmax=1.0)
    axh.set_xticks(range(n)); axh.set_xticklabels([tfs[i].upper() for i in order], rotation=90, fontsize=5)
    axh.set_yticks([])
    cax = fig.add_axes([0.05, 0.12, 0.012, 0.16])       # dedicated colorbar (bottom-left, no overlap)
    fig.colorbar(im, cax=cax, label="Pearson r")

    y = np.arange(n)
    cnt = np.array([cells.get(tfs[i].lower(), 1) for i in order], float)
    cmap = plt.get_cmap("tab10")
    axb.barh(y, np.maximum(cnt, 1), color=[cmap(clust[i] % 10) for i in order], height=0.8)
    axb.set_xscale("log"); axb.set_ylim(-0.5, n - 0.5)
    axb.set_yticks(range(n)); axb.set_yticklabels([tfs[i].upper() for i in order], fontsize=5)
    axb.tick_params(length=0); axb.set_xlabel("Number of cells")
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    # per-cluster ChromHMM composition (stacked)
    clusters = sorted(comp)
    scored = sorted(range(len(states)), key=lambda k: STATE_ORDER.index(core(states[k]))
                    if core(states[k]) in STATE_ORDER else 99)
    for ci, c in enumerate(clusters):
        bottom = 0.0
        for k in scored:
            v = comp[c][k]
            axc.bar(ci, v, bottom=bottom, width=0.7,
                    color=STATE_COLORS.get(core(states[k]), "#999999"))
            bottom += v
    axc.set_xticks(range(len(clusters))); axc.set_xticklabels([f"Cluster{c}" for c in clusters])
    axc.set_ylabel("Proportion of states (mean RPKM)"); axc.set_ylim(0, 1)
    axc.set_title("ChromHMM state composition across TF clusters", fontsize=9)
    handles = [Patch(color=STATE_COLORS.get(s, "#999"), label=s) for s in STATE_ORDER
               if any(core(st) == s for st in states)]
    axc.legend(handles=handles, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left", title="ChromHMM State")

    fig.suptitle("TF chromHMM18 RPKM  Pearson correlation", fontsize=13)
    fig.savefig(out_dir / "chromhmm_rpkm_figure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_dir/'chromhmm_rpkm_figure.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
