#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# tf_correlation.py
#
# TF x TF co-occupancy correlation from the raw binding matrix (build_tf_matrix.py),
# stratified by A/B compartment, then hierarchical clustering -> candidate protein
# complexes. Part of the parallel A/B-compartment -> correlation -> motif -> complex
# pipeline (independent of the Cicero co-binding analysis).
#
# For each scope (genome-wide feature bins, A-compartment only, B only) it ranks
# each TF's per-bin signal and correlates TF columns (Spearman; or --metric
# pearson/jaccard). TFs are clustered on the genome-wide correlation (1-corr
# distance, average linkage); clusters >= 2 TFs are candidate complexes.
#
# Also reports each TF's A/B preference (size-normalized: obs vs expected by the
# number of A vs B feature bins).
#
# Outputs (--out-dir):
#   corr_genome.tsv / corr_A.tsv / corr_B.tsv   TF x TF correlation matrices
#   tf_compartment.tsv                          per-TF A/B enrichment
#   candidate_complexes.tsv                     clustered TF groups
#   correlation_heatmap.png (--plot)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

_REGION = re.compile(r"^([^:]+):(\d+)-(\d+)$")


def load_compartment(ab_bed: Path, regions: list[str]) -> np.ndarray:
    """Per-region compartment label ('A'/'B'/'.') via the 25 kb bin it falls in."""
    lab: dict[tuple, str] = {}
    widths: dict[int, int] = {}
    with open(ab_bed) as f:
        for ln in f:
            p = ln.split("\t")
            if len(p) < 4:
                continue
            w = int(p[2]) - int(p[1]); widths[w] = widths.get(w, 0) + 1
    res = max(widths, key=widths.get)
    with open(ab_bed) as f:
        for ln in f:
            p = ln.split("\t")
            if len(p) < 4:
                continue
            lab[(p[0], int(p[1]) // res)] = p[3].strip()
    out = np.empty(len(regions), dtype="<U1")
    for i, r in enumerate(regions):
        m = _REGION.match(r)
        out[i] = lab.get((m.group(1), int(m.group(2)) // res), ".") if m else "."
    return out


def corr_tf_by_tf(sub: np.ndarray, metric: str) -> np.ndarray:
    """sub: bins x TFs (dense). Return TF x TF correlation."""
    if metric == "jaccard":
        B = (sub > 0)
        n = B.shape[1]
        R = np.eye(n)
        colsum = B.sum(axis=0)
        for i in range(n):
            inter = (B[:, i:i + 1] & B).sum(axis=0)
            union = colsum[i] + colsum - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                R[i] = np.where(union > 0, inter / union, 0.0)
        return R
    X = sub.astype(np.float64)
    if metric == "spearman":
        X = np.apply_along_axis(rankdata, 0, X)
    # drop zero-variance TFs to nan rows handled by corrcoef
    C = np.corrcoef(X, rowvar=False)
    return C


def write_matrix(path: Path, M: np.ndarray, labels: list[str]) -> None:
    with path.open("w") as f:
        f.write("TF\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels))) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True, help="dir with tf_matrix.npz + tfs.txt + regions.tsv")
    ap.add_argument("--ab-bed", type=Path, required=True, help="classify_compartments.py *.AB.bed")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["spearman", "pearson", "jaccard"], default="spearman")
    ap.add_argument("--min-tfs-per-bin", type=int, default=2,
                    help="feature bins = bound (signal>0) in at least this many TFs (drops "
                         "singleton/empty bins that inflate co-absence). Default 2.")
    ap.add_argument("--corr-threshold", type=float, default=0.5,
                    help="cluster TFs whose linkage merges below 1-this correlation (default 0.5)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    M = sp.load_npz(args.matrix_dir / "tf_matrix.npz").tocsc()
    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    regions = [r.strip() for r in (args.matrix_dir / "regions.tsv").read_text().splitlines() if r.strip()]
    n_bins, n_tf = M.shape
    print(f"[matrix] {n_bins:,} bins x {n_tf} TFs", flush=True)

    comp = load_compartment(args.ab_bed, regions)

    # feature bins: bound in >= min-tfs-per-bin TFs
    nnz_per_bin = np.asarray((M > 0).sum(axis=1)).ravel()
    feat = nnz_per_bin >= args.min_tfs_per_bin
    print(f"[features] bins bound in >= {args.min_tfs_per_bin} TFs: {int(feat.sum()):,}", flush=True)
    if feat.sum() < 10:
        raise SystemExit("too few feature bins; lower --min-tfs-per-bin")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scopes = {"genome": feat, "A": feat & (comp == "A"), "B": feat & (comp == "B")}
    corr = {}
    for name, mask in scopes.items():
        idx = np.flatnonzero(mask)
        if idx.size < 10:
            print(f"[corr] {name}: only {idx.size} bins; skipping", flush=True)
            continue
        sub = M[idx].toarray()
        R = corr_tf_by_tf(sub, args.metric)
        corr[name] = R
        write_matrix(args.out_dir / f"corr_{name}.tsv", np.nan_to_num(R), tfs)
        print(f"[corr] {name}: {idx.size:,} bins -> corr_{name}.tsv", flush=True)

    # per-TF A/B enrichment (size-normalized by number of A vs B feature bins)
    nA = int((scopes["A"]).sum()); nB = int((scopes["B"]).sum())
    exp_a = nA / (nA + nB) if (nA + nB) else float("nan")
    sig = M.tocsr()
    with (args.out_dir / "tf_compartment.tsv").open("w") as f:
        f.write("tf\tnonzero_bins\ttotal_signal\tsignal_fracA\texp_fracA\tlog2_enrichA\n")
        Acol = np.flatnonzero(scopes["A"]); Bcol = np.flatnonzero(scopes["B"])
        for j, tf in enumerate(tfs):
            col = M[:, j]
            sA = float(col[Acol].sum()); sB = float(col[Bcol].sum())
            tot = sA + sB
            fa = sA / tot if tot else float("nan")
            l2 = math.log2(fa / exp_a) if tot and fa > 0 and exp_a > 0 else float("nan")
            f.write(f"{tf}\t{int((col.toarray() > 0).sum())}\t{float(col.sum()):.0f}\t"
                    f"{fa:.4f}\t{exp_a:.4f}\t{l2:.4f}\n")

    # cluster TFs on genome-wide correlation -> candidate complexes
    if "genome" in corr:
        R = np.nan_to_num(corr["genome"], nan=0.0)
        np.fill_diagonal(R, 1.0)
        D = 1.0 - R
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        clusters = fcluster(Z, t=1.0 - args.corr_threshold, criterion="distance")
        order = leaves_list(Z)
        groups: dict[int, list[str]] = {}
        for tf, c in zip(tfs, clusters):
            groups.setdefault(int(c), []).append(tf)
        with (args.out_dir / "candidate_complexes.tsv").open("w") as f:
            f.write("complex_id\tn_tfs\tmean_intra_corr\tmembers\n")
            cid = 0
            for c, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                if len(mem) < 2:
                    continue
                cid += 1
                mi = [tfs.index(m) for m in mem]
                sub = R[np.ix_(mi, mi)]
                mic = (sub.sum() - len(mem)) / (len(mem) * (len(mem) - 1)) if len(mem) > 1 else 1.0
                f.write(f"CX{cid:03d}\t{len(mem)}\t{mic:.3f}\t{','.join(sorted(mem))}\n")
        n_cx = sum(1 for m in groups.values() if len(m) >= 2)
        print(f"[complexes] {n_cx} candidate complexes (>=2 TFs, corr>={args.corr_threshold})", flush=True)
        if args.plot:
            _plot(R, [tfs[i] for i in order], order, args.out_dir)
    print(f"[done] outputs under {args.out_dir}")
    return 0


def _plot(R, ordered_labels, order, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    Ro = R[np.ix_(order, order)]
    n = len(order)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.25),) * 2)
    im = ax.imshow(Ro, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n), ordered_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n), ordered_labels, fontsize=5)
    fig.colorbar(im, ax=ax, label="TF co-occupancy correlation")
    ax.set_title("TF x TF co-occupancy (clustered)")
    fig.tight_layout(); fig.savefig(out_dir / "correlation_heatmap.png", dpi=150); plt.close(fig)
    print(f"[plot] wrote {out_dir/'correlation_heatmap.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
