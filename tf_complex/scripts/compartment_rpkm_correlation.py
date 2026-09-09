#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# compartment_rpkm_correlation.py
#
# "TF compartment rpkm pearson cor": the ChromHMM-18 RPKM workflow
# (build_chromhmm_matrix.py + chromhmm_correlation.py), with the unit swapped from
# a chromatin state to an A/B compartment domain. Pearson-correlate TFs over their
# RPKM profile across compartment units (build_compartment_matrix.py --transform
# rpkm), cluster the TFs, and render:
#   (1) clustered TF x TF correlation heatmap
#   (2) per-TF cell-count bars (complexity)
# There is no state-composition panel -- compartments carry only an A/B label, and
# that split is expressed instead by running the whole thing per SCOPE.
#
# Scopes: genome (all units), A (A-only units = A-A co-binding), B (B-only = B-B).
#
# Output (--out-dir), one set per scope:
#   tf_similarity_<scope>.tsv, tf_clusters_<scope>.tsv, compartment_rpkm_<scope>.png
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


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


def write_matrix(path, M, labels):
    with open(path, "w") as f:
        f.write("TF\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels))) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True,
                    help="build_compartment_matrix.py --transform rpkm output")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cell-meta", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--transform", choices=["rpkm", "log"], default="rpkm",
                    help="rpkm (default, matrix used as built) | log = log1p(rpkm)")
    ap.add_argument("--min-reads", type=float, default=0.0,
                    help="drop units whose summed signal across TFs is <= this")
    ap.add_argument("--vmin", type=float, default=None,
                    help="heatmap colour floor; default = 2nd percentile of the scope")
    args = ap.parse_args()

    M = np.asarray(sp.load_npz(args.matrix_dir / "compartment_matrix.npz").todense())  # units x TFs
    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    rows = [ln.rstrip("\n").split("\t") for ln in open(args.matrix_dir / "domains.tsv")]
    labs = np.array([r[3].strip() for r in rows], dtype="<U1")
    n_tf, nD = len(tfs), M.shape[0]
    print(f"[matrix] {nD:,} compartment units x {n_tf} TFs (A={int((labs=='A').sum()):,} "
          f"B={int((labs=='B').sum()):,})", flush=True)

    X = np.log1p(M) if args.transform == "log" else M
    keep = np.asarray(M.sum(axis=1)).ravel() > args.min_reads
    cells = load_cells(args.cell_meta)
    print(f"[cells] {len(cells)} TFs in {args.cell_meta}; "
          f"{sum(1 for t in tfs if t.lower() in cells)}/{n_tf} matched", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for scope, mask in (("genome", np.ones(nD, bool)), ("A", labs == "A"), ("B", labs == "B")):
        idx = np.flatnonzero(mask & keep)
        if idx.size < 10:
            print(f"[{scope}] only {idx.size} units; skipping", flush=True); continue
        R = np.nan_to_num(np.corrcoef(X[idx], rowvar=False))
        write_matrix(args.out_dir / f"tf_similarity_{scope}.tsv", R, tfs)
        off = R[~np.eye(n_tf, dtype=bool)]
        print(f"[{scope}] {idx.size:,} units  off-diag median={np.median(off):+.3f} "
              f"p95={np.percentile(off, 95):+.3f} max={off.max():+.3f}", flush=True)

        D = 1 - R; D = (D + D.T) / 2; np.fill_diagonal(D, 0); D[D < 0] = 0
        Z = linkage(squareform(D, checks=False), method="average")
        clust = fcluster(Z, t=args.n_clusters, criterion="maxclust")
        with (args.out_dir / f"tf_clusters_{scope}.tsv").open("w") as f:
            f.write("tf\tcluster\n")
            for t, c in zip(tfs, clust):
                f.write(f"{t}\t{c}\n")
        _plot(R, tfs, Z, cells, args.out_dir / f"compartment_rpkm_{scope}.png", scope,
              vmin=args.vmin if args.vmin is not None else float(np.percentile(R, 2)))
    print(f"[done] -> {args.out_dir}")
    return 0


def _plot(R, tfs, Z, cells, out_png, scope, vmin=0.65):
    """dendrogram | TF x TF heatmap (RdYlBu_r) | cell-count bars -- ChromHMM layout
    without the state-composition panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    n = len(tfs)
    dn = dendrogram(Z, no_plot=True); order = dn["leaves"]
    fig = plt.figure(figsize=(15, max(9, n * 0.135)))
    # Dendrogram butts against the heatmap; the heatmap->bars gap holds two label columns.
    axd = fig.add_axes([0.175, 0.10, 0.110, 0.84])
    axh = fig.add_axes([0.285, 0.10, 0.450, 0.84])
    axb = fig.add_axes([0.815, 0.10, 0.165, 0.84])
    hsv = plt.get_cmap("hsv")
    bar_cols = [hsv(0.02 + 0.90 * i / max(n - 1, 1)) for i in range(n)]  # red bottom -> pink top

    fig.text(0.020, 0.80, f"TF compartment\nRPKM\npearson cor\n({scope})", fontsize=15, va="top")

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
    cax = fig.add_axes([0.060, 0.30, 0.012, 0.22])
    fig.colorbar(im, cax=cax)

    y = np.arange(n)
    cnt = np.array([cells.get(tfs[i].lower(), 1) for i in order], float)
    axb.barh(y, np.maximum(cnt, 1), color=bar_cols, height=0.8)
    axb.set_xscale("log"); axb.set_ylim(-0.5, n - 0.5)
    axb.set_yticks(range(n)); axb.set_yticklabels([tfs[i].upper() for i in order], fontsize=5)
    axb.tick_params(length=0); axb.set_xlabel("Number of cells")
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


if __name__ == "__main__":
    raise SystemExit(main())
