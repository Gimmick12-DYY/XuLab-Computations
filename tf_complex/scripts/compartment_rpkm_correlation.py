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
# Two unit layouts are accepted, auto-detected from --matrix-dir:
#   * CLASS units (chromhmm_matrix.npz + states.tsv, from build_compartment_classes.py
#     -> build_chromhmm_matrix.py): 2n eigenvector-strength classes pooled genome-wide.
#     This is the layout that behaves -- see build_compartment_classes.py for why.
#     Scope: genome only (the A/B split is already an axis of the units).
#   * DOMAIN/BIN units (compartment_matrix.npz + domains.tsv, from
#     build_compartment_matrix.py): one unit per compartment domain or 25 kb bin.
#     Scopes: genome, A (A-A co-binding), B (B-B). Depth-confounded at this
#     granularity; kept for comparison, not for interpretation.
#
# Output (--out-dir), one set per scope:
#   tf_similarity_<scope>.tsv   raw Pearson r (input TF order) — use this for numbers
#   tf_clusters_<scope>.tsv     average-linkage clusters on d = 1-r (raw Pearson)
#   compartment_rpkm_<scope>.png
# Heatmap colours default to a global empirical percentile of the unique
# off-diagonal r's (visualization only). Clustering always uses raw 1-r.
# Plotting lives in plot_correlation_matrix.py.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster

from plot_correlation_matrix import (
    linkage_from_pearson,
    load_similarity_tsv,
    plot_correlation_matrix,
)


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
    ap.add_argument("--matrix-dir", type=Path, default=None,
                    help="build_compartment_matrix.py --transform rpkm output")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cell-meta", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--transform", choices=["rpkm", "log"], default="rpkm",
                    help="rpkm (default, matrix used as built) | log = log1p(rpkm)")
    ap.add_argument("--min-reads", type=float, default=0.0,
                    help="drop units whose summed signal across TFs is <= this")
    ap.add_argument("--min-bp", type=int, default=0,
                    help="drop domains shorter than this (bp). 0 = keep all. "
                         "Only applies to domain/bin matrices.")
    ap.add_argument("--vmin", type=float, default=None,
                    help="heatmap colour floor for --color r|clip; "
                         "default = 2nd percentile (r) or 0.92 (clip)")
    ap.add_argument("--color", choices=["percentile", "r", "clip"], default="percentile",
                    help="heatmap colours: global off-diagonal percentile of r "
                         "(default, visualization only); raw r; or raw r clipped "
                         "to [--vmin, 1]. Clustering always uses raw 1-r.")
    ap.add_argument("--replot-dir", type=Path, default=None,
                    help="skip matrix build: load tf_similarity_<scope>.tsv from this dir and replot")
    args = ap.parse_args()
    if args.replot_dir is not None:
        return _replot_from_similarity(args)
    if args.matrix_dir is None:
        raise SystemExit("--matrix-dir is required unless --replot-dir is set")

    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    npz_cls = args.matrix_dir / "chromhmm_matrix.npz"
    if npz_cls.is_file():                     # eigenvector-strength classes, genome scope only
        M = np.asarray(sp.load_npz(npz_cls).todense())
        names = [ln.split("\t")[0].strip() for ln in
                 open(args.matrix_dir / "states.tsv") if ln.strip()]
        labs = np.array([n[0] for n in names], dtype="<U1")
        unit_kind, scopes = "class", ("genome",)
        size_ok = np.ones(M.shape[0], bool)
    else:                                     # one unit per domain / 25 kb bin
        M = np.asarray(sp.load_npz(args.matrix_dir / "compartment_matrix.npz").todense())
        rows = [ln.rstrip("\n").split("\t") for ln in open(args.matrix_dir / "domains.tsv")]
        labs = np.array([r[3].strip() for r in rows], dtype="<U1")
        bp = np.array([int(r[4]) if len(r) > 4 else 0 for r in rows], dtype=np.int64)
        unit_kind, scopes = "domain", ("genome", "A", "B")
        if args.min_bp > 0:
            size_ok = bp >= args.min_bp
            print(f"[min-bp] keep {int(size_ok.sum()):,}/{len(bp):,} domains ≥ {args.min_bp:,} bp "
                  f"({bp[size_ok].sum()/1e6:.0f} Mb)", flush=True)
        else:
            size_ok = np.ones(len(bp), bool)
    n_tf, nD = len(tfs), M.shape[0]
    print(f"[matrix] {nD:,} compartment {unit_kind} units x {n_tf} TFs "
          f"(A={int((labs=='A').sum()):,} B={int((labs=='B').sum()):,})", flush=True)

    X = np.log1p(M) if args.transform == "log" else M
    keep = (np.asarray(M.sum(axis=1)).ravel() > args.min_reads) & size_ok
    cells = load_cells(args.cell_meta)
    print(f"[cells] {len(cells)} TFs in {args.cell_meta}; "
          f"{sum(1 for t in tfs if t.lower() in cells)}/{n_tf} matched", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    masks = {"genome": np.ones(nD, bool), "A": labs == "A", "B": labs == "B"}
    for scope in scopes:
        idx = np.flatnonzero(masks[scope] & keep)
        if idx.size < 3:
            print(f"[{scope}] only {idx.size} units; skipping", flush=True); continue
        R = np.nan_to_num(np.corrcoef(X[idx], rowvar=False))
        write_matrix(args.out_dir / f"tf_similarity_{scope}.tsv", R, tfs)
        off = R[~np.eye(n_tf, dtype=bool)]
        print(f"[{scope}] {idx.size:,} units  off-diag median={np.median(off):+.3f} "
              f"p95={np.percentile(off, 95):+.3f} max={off.max():+.3f}", flush=True)

        Z = linkage_from_pearson(R)
        clust = fcluster(Z, t=args.n_clusters, criterion="maxclust")
        with (args.out_dir / f"tf_clusters_{scope}.tsv").open("w") as f:
            f.write("tf\tcluster\n")
            for t, c in zip(tfs, clust):
                f.write(f"{t}\t{c}\n")
        _draw(R, tfs, cells, args, scope, Z)
    print(f"[done] -> {args.out_dir}")
    return 0


def _title(color: str, scope: str) -> str:
    if color == "percentile":
        return f"TF compartment\nRPKM\nPearson r\npercentile\n({scope})"
    return f"TF compartment\nRPKM\npearson cor\n({scope})"


def _draw(R, tfs, cells, args, scope, Z=None):
    if Z is None:
        Z = linkage_from_pearson(R)
    plot_correlation_matrix(
        R, tfs, args.out_dir / f"compartment_rpkm_{scope}.png",
        cells=cells, color=args.color, vmin=args.vmin, Z=Z,
        title=_title(args.color, scope), write_tsv=True,
    )
    plot_correlation_matrix(
        R, tfs, args.out_dir / f"compartment_rpkm_{scope}_r.png",
        cells=cells, color="r", vmin=args.vmin, Z=Z,
        title=_title("r", scope), write_tsv=True,
    )
    plot_correlation_matrix(
        R, tfs, args.out_dir / f"tf_similarity_{scope}.png",
        cells=cells, color="r", vmin=args.vmin, Z=Z,
        title=_title("r", scope), write_tsv=False,
    )


def _replot_from_similarity(args) -> int:
    src = args.replot_dir
    cells = load_cells(args.cell_meta)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    found = False
    for scope in ("genome", "A", "B"):
        p = src / f"tf_similarity_{scope}.tsv"
        if not p.is_file():
            continue
        found = True
        tfs, R = load_similarity_tsv(p)
        _draw(R, tfs, cells, args, scope)
        print(f"[{scope}] replotted {len(tfs)} TFs from {p}", flush=True)
    if not found:
        raise SystemExit(f"no tf_similarity_*.tsv in {src}")
    print(f"[done] -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
