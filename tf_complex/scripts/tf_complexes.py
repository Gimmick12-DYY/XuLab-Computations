#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# tf_complexes.py
#
# TF co-occupancy -> candidate protein complexes, from the peak occupancy matrix
# (build_peak_matrix.py). Follows the field-standard method (Partridge et al. 2020,
# Nature, "Occupancy maps of 208 chromatin-associated proteins"): build a
# loci x TF binding matrix, remove high-occupancy artifacts, reduce with PCA, and
# cluster the TF correlation -> complexes (they recovered cohesin, NuRD, POL2...).
#
# Two corrections the literature demands (see tf_complex/README.md):
#   1. Remove HOT loci (bound by > --hot-frac of TFs) + optional ENCODE blacklist.
#      HOT / high-occupancy-target regions are largely ChIP-seq artifacts
#      (GC/CpG-rich, motif-free) that make every TF pair look co-bound.
#   2. Correlate binding PROFILES (default via PCA), not raw pairwise overlap.
#
# A/B compartments are used as an ANNOTATION of the resulting complexes/TFs, NOT as
# the axis that defines co-binding (every TF is A-leaning, so A/B can't discriminate).
#
# Outputs (--out-dir):
#   tf_similarity_{genome,A,B}.tsv   TF x TF similarity (chosen metric)
#   candidate_complexes.tsv          clustered TF groups
#   tf_compartment.tsv               per-TF A/B enrichment (size-normalized)
#   complex_heatmap.png (--plot)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def load_blacklist(path):
    iv = defaultdict(list)
    for ln in open(path):
        if not ln.strip() or ln.startswith(("#", "track", "browser")):
            continue
        p = ln.split("\t"); iv[p[0]].append((int(p[1]), int(p[2])))
    for c in iv:
        iv[c].sort()
    return iv


def loci_hit_blacklist(loci_bed, bl):
    hit = []
    for ln in open(loci_bed):
        p = ln.split("\t"); c, s, e = p[0], int(p[1]), int(p[2])
        ivs = bl.get(c, [])
        h = any(s < be and bs < e for bs, be in ivs)   # small lists per locus; fine
        hit.append(h)
    return np.array(hit, dtype=bool)


def pca_similarity(Bkeep, n_pc):
    """Partridge-style: PCA of the loci x TF matrix, then Pearson correlation between
    TFs in PC space. Centers per-locus, SVD, take TF loadings on top PCs, correlate."""
    X = Bkeep.astype(np.float64).toarray()                # loci x TF
    X -= X.mean(axis=1, keepdims=True)                    # center each locus (row)
    k = min(n_pc, min(X.shape) - 1)
    # right singular vectors give per-TF coordinates
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    load = (Vt[:k].T * S[:k])                             # TF x k (loadings)
    return np.corrcoef(load)                              # TF x TF Pearson in PC space


def jaccard_similarity(Bkeep):
    B = Bkeep.astype(bool)
    n = B.shape[1]
    cs = np.asarray(B.sum(axis=0)).ravel()
    Bi = B.astype(np.int32)
    inter = np.asarray((Bi.T @ Bi).todense())
    union = cs[:, None] + cs[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(J, 1.0)
    return J


def pearson_similarity(Bkeep):
    return np.corrcoef(Bkeep.astype(np.float64).toarray().T)   # TF x TF phi


def write_matrix(path, M, labels):
    with open(path, "w") as f:
        f.write("TF\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels))) + "\n")


def similarity(Bkeep, metric, n_pc):
    if metric == "pca":
        return pca_similarity(Bkeep, n_pc)
    if metric == "jaccard":
        return jaccard_similarity(Bkeep)
    return pearson_similarity(Bkeep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True, help="build_peak_matrix.py output")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["pca", "pearson", "jaccard"], default="pca")
    ap.add_argument("--n-pc", type=int, default=25, help="PCs for --metric pca")
    ap.add_argument("--hot-frac", type=float, default=0.5,
                    help="drop loci bound by more than this FRACTION of TFs (HOT / high-occupancy "
                         "artifacts). 0.5 = drop loci bound by >half the panel.")
    ap.add_argument("--min-occ", type=int, default=2,
                    help="keep loci bound by >= this many TFs (drop singletons; they carry no "
                         "co-occupancy signal).")
    ap.add_argument("--blacklist", type=Path, default=None, help="ENCODE blacklist BED to remove")
    ap.add_argument("--cluster", choices=["threshold", "hierarchical"], default="hierarchical")
    ap.add_argument("--cluster-threshold", type=float, default=0.5,
                    help="threshold mode: edge if similarity >= this. hierarchical mode: distance "
                         "cut = 1 - this. Set from the printed p90/p95.")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    B = sp.load_npz(args.matrix_dir / "peak_matrix.npz").tocsc()
    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    n_tf = len(tfs)
    occ = np.asarray(B.sum(axis=1)).ravel()
    print(f"[matrix] {B.shape[0]:,} loci x {n_tf} TFs (nnz={B.nnz:,})", flush=True)

    # --- remove HOT loci + blacklist + singletons ---
    hot_cut = args.hot_frac * n_tf
    keep = (occ >= args.min_occ) & (occ <= hot_cut)
    n_hot = int((occ > hot_cut).sum())
    print(f"[filter] drop {n_hot:,} HOT loci (>{hot_cut:.0f}/{n_tf} TFs), "
          f"{int((occ < args.min_occ).sum()):,} below min-occ; keep {int(keep.sum()):,}", flush=True)
    if args.blacklist and args.blacklist.is_file():
        bl = loci_hit_blacklist(args.matrix_dir / "loci.bed", load_blacklist(args.blacklist))
        print(f"[filter] drop {int((keep & bl).sum()):,} blacklist-overlapping loci", flush=True)
        keep &= ~bl

    # --- A/B per-locus labels (annotation) ---
    ab_path = args.matrix_dir / "loci_ab.tsv"
    labs = None
    if ab_path.is_file():
        labs = np.array([ln.split("\t")[1].strip() for ln in open(ab_path)], dtype="<U1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scopes = {"genome": keep}
    if labs is not None:
        scopes["A"] = keep & (labs == "A")
        scopes["B"] = keep & (labs == "B")

    prim = None
    for name, m in scopes.items():
        idx = np.flatnonzero(m)
        if idx.size < 50:
            print(f"[{name}] only {idx.size} loci; skipping", flush=True)
            continue
        Bk = B[idx]
        # drop TFs with no peaks in this scope would break corr; keep all, corr handles via nan->0
        R = np.nan_to_num(similarity(Bk, args.metric, args.n_pc))
        write_matrix(args.out_dir / f"tf_similarity_{name}.tsv", R, tfs)
        off = R[~np.eye(n_tf, dtype=bool)]
        print(f"[{name}] {idx.size:,} loci  {args.metric} off-diag: median={np.median(off):.3f} "
              f"p90={np.percentile(off,90):.3f} p95={np.percentile(off,95):.3f} "
              f"max={np.nanmax(off):.3f}", flush=True)
        if name == "genome":
            prim = R

    # --- clustering -> candidate complexes ---
    if prim is not None:
        if args.cluster == "hierarchical":
            D = 1.0 - prim
            D = (D + D.T) / 2.0
            np.fill_diagonal(D, 0.0)
            D[D < 0] = 0.0
            Z = linkage(squareform(D, checks=False), method="average")
            labels_c = fcluster(Z, t=1.0 - args.cluster_threshold, criterion="distance")
            groups = defaultdict(list)
            for i, c in enumerate(labels_c):
                groups[c].append(i)
            members_list = list(groups.values())
        else:  # threshold connected components
            A = prim >= args.cluster_threshold
            np.fill_diagonal(A, False)
            parent = list(range(n_tf))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for i in range(n_tf):
                for j in range(i + 1, n_tf):
                    if A[i, j]:
                        a, b = find(i), find(j)
                        if a != b:
                            parent[a] = b
            groups = defaultdict(list)
            for i in range(n_tf):
                groups[find(i)].append(i)
            members_list = list(groups.values())

        with (args.out_dir / "candidate_complexes.tsv").open("w") as f:
            f.write("complex_id\tn_tfs\tmean_intra_sim\tmembers\n")
            cid = 0
            for members in sorted(members_list, key=len, reverse=True):
                if len(members) < 2:
                    continue
                cid += 1
                s = prim[np.ix_(members, members)]
                mic = (s.sum() - np.trace(s)) / (len(members) * (len(members) - 1))
                f.write(f"CX{cid:03d}\t{len(members)}\t{mic:.3f}\t"
                        f"{','.join(sorted(tfs[i] for i in members))}\n")
        print(f"[complexes] {cid} candidate complexes (>=2 TFs)", flush=True)

    # --- per-TF A/B enrichment (size-normalized), annotation ---
    if labs is not None:
        isA = labs == "A"; isB = labs == "B"
        nA, nB = int(isA.sum()), int(isB.sum())
        exp_a = nA / (nA + nB) if (nA + nB) else float("nan")
        with (args.out_dir / "tf_compartment.tsv").open("w") as f:
            f.write("tf\tpeaks\tpeaks_in_A\tfracA\texp_fracA\tlog2_enrichA\n")
            for j, tf in enumerate(tfs):
                col = np.asarray(B[:, j].todense()).ravel().astype(bool)
                pA = int((col & isA).sum()); pB = int((col & isB).sum()); tot = pA + pB
                fa = pA / tot if tot else float("nan")
                l2 = np.log2(fa / exp_a) if tot and fa > 0 and exp_a > 0 else float("nan")
                f.write(f"{tf}\t{int(col.sum())}\t{pA}\t{fa:.4f}\t{exp_a:.4f}\t{l2:.4f}\n")

    if args.plot and prim is not None:
        _plot(prim, tfs, args.out_dir, args.metric)
    print(f"[done] -> {args.out_dir}")
    return 0


def _plot(R, tfs, out_dir, metric):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    D = 1.0 - R; D = (D + D.T) / 2.0; np.fill_diagonal(D, 0.0); D[D < 0] = 0.0
    order = leaves_list(linkage(squareform(D, checks=False), method="average"))
    Ro = R[np.ix_(order, order)]; n = len(order)
    off = Ro[~np.eye(n, dtype=bool)]
    vmax = np.nanpercentile(np.abs(off), 99) or 1.0
    fig, ax = plt.subplots(figsize=(max(6, n * 0.25),) * 2)
    im = ax.imshow(Ro, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(n), [tfs[i] for i in order], rotation=90, fontsize=5)
    ax.set_yticks(range(n), [tfs[i] for i in order], fontsize=5)
    fig.colorbar(im, ax=ax, label=f"TF co-occupancy ({metric})")
    ax.set_title(f"TF x TF co-occupancy ({metric}, HOT-removed, clustered)")
    fig.tight_layout(); fig.savefig(out_dir / "complex_heatmap.png", dpi=150); plt.close(fig)
    print(f"[plot] wrote {out_dir/'complex_heatmap.png'} (vmax={vmax:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
