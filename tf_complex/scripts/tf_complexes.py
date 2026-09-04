#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# tf_complexes.py
#
# TF co-occupancy -> candidate protein complexes, from the peak occupancy matrix
# (build_peak_matrix.py). Follows the field-standard method (Partridge et al. 2020,
# Nature, "Occupancy maps of 208 chromatin-associated proteins"): build a
# loci x TF binding matrix, remove high-occupancy artifacts, correlate TFs, and
# cluster -> complexes (they recovered cohesin, NuRD, POL2...).
#
# Correction the literature demands (see tf_complex/README.md):
#   Remove HOT loci (bound by > --hot-frac of TFs) + optional ENCODE blacklist.
#   HOT / high-occupancy-target regions are largely ChIP-seq artifacts (GC/CpG-rich,
#   motif-free) that make every TF pair look co-bound.
#
# Metric = pearson (phi of binary peak vectors, default) or jaccard. Both recover
# complexes across the panel's ~1000x peak-count spread WITHOUT a coverage block,
# once HOT loci are removed. (A PCA/SVD step was tried and REMOVED: row-centering the
# loci made sparse low-peak TFs collinear -> a spurious 36-TF "complex" that just
# tracked peak count.)
#
# A/B compartments are used as an ANNOTATION of the resulting complexes/TFs, NOT as
# the axis that defines co-binding (every TF is A-leaning, so A/B can't discriminate).
#
# Outputs (--out-dir):
#   tf_similarity_{genome,A,B}.tsv   TF x TF similarity (chosen metric)
#   candidate_complexes.tsv          clustered TF groups
#   tf_compartment.tsv               per-TF A/B enrichment (size-normalized)
#   corr_complexity_{genome,A,B}.png (--plot)  dendrogram + heatmap + per-TF cell-count bars
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import subprocess
import sys
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


def similarity(Bkeep, metric):
    if metric == "jaccard":
        return jaccard_similarity(Bkeep)
    return pearson_similarity(Bkeep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True, help="build_peak_matrix.py output")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["pearson", "jaccard"], default="pearson",
                    help="TF-TF similarity over HOT-removed loci. pearson (phi of binary peak "
                         "vectors, default) and jaccard both recover complexes without a "
                         "coverage block; peak-count spread is handled by HOT removal, not a "
                         "dimensionality reduction.")
    ap.add_argument("--hot-frac", type=float, default=0.5,
                    help="drop loci bound by more than this FRACTION of TFs (HOT / high-occupancy "
                         "artifacts). 0.5 = drop loci bound by >half the panel.")
    ap.add_argument("--min-occ", type=int, default=2,
                    help="keep loci bound by >= this many TFs (drop singletons; they carry no "
                         "co-occupancy signal).")
    ap.add_argument("--blacklist", type=Path, default=None, help="ENCODE blacklist BED to remove")
    ap.add_argument("--cluster", choices=["threshold", "hierarchical"], default="hierarchical")
    ap.add_argument("--cluster-threshold", type=float, default=0.2,
                    help="threshold mode: edge if similarity >= this. hierarchical mode: distance "
                         "cut = 1 - this. pearson co-occupancy is small in abs terms (~0.1-0.4); "
                         "set near the printed p95.")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--cell-meta", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "TF1000cells.meta.csv",
                    help="per-TF cell counts for the complexity-format heatmap (dendrogram + "
                         "heatmap + cell-count bars). Every scope's matrix is plotted this way.")
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

    # bottom-quartile TFs by peak count -- used to flag any residual coverage block
    tf_peaks = np.asarray((B > 0).sum(axis=0)).ravel()
    low_q = np.flatnonzero(tf_peaks <= np.percentile(tf_peaks, 25))

    Rs = {}
    for name, m in scopes.items():
        idx = np.flatnonzero(m)
        if idx.size < 50:
            print(f"[{name}] only {idx.size} loci; skipping", flush=True)
            continue
        Bk = B[idx]
        # keep all TFs; corr of a zero-peak-in-scope TF -> nan -> 0
        R = np.nan_to_num(similarity(Bk, args.metric))
        write_matrix(args.out_dir / f"tf_similarity_{name}.tsv", R, tfs)
        off = R[~np.eye(n_tf, dtype=bool)]
        # coverage-artifact check: similarity among low-peak TFs, and corr(peak_count, mean_sim)
        low_sub = R[np.ix_(low_q, low_q)]
        low_off = low_sub[~np.eye(len(low_q), dtype=bool)] if len(low_q) > 1 else np.array([0.0])
        msim = (R.sum(axis=1) - np.diag(R)) / (n_tf - 1)
        cov_r = np.corrcoef(tf_peaks, msim)[0, 1] if n_tf > 2 else float("nan")
        print(f"[{name}] {idx.size:,} loci  {args.metric} off-diag: median={np.median(off):.3f} "
              f"p90={np.percentile(off,90):.3f} p95={np.percentile(off,95):.3f} "
              f"max={np.nanmax(off):.3f}", flush=True)
        # low-peak-TF block median ~0 = no depth artifact; a large POSITIVE value means
        # low-peak TFs cluster trivially (the PCA failure mode). cov_r can be nonzero
        # legitimately if a real complex happens to be depth-correlated, so read it with
        # the block median, not alone.
        print(f"[{name}] coverage check: low-peak-TF block median={np.median(low_off):+.3f} "
              f"(near 0 = clean), corr(peak_count, mean_sim)={cov_r:+.3f}", flush=True)
        Rs[name] = R

    # --- per-scope clustering -> candidate complexes (genome + A-A + B-B) ---
    for name, R in Rs.items():
        members_list = cluster_members(R, n_tf, args)
        out_name = "candidate_complexes.tsv" if name == "genome" else f"candidate_complexes_{name}.tsv"
        ncx = write_complexes(args.out_dir / out_name, members_list, R, tfs)
        print(f"[complexes:{name}] {ncx} candidate complexes (>=2 TFs) -> {out_name}", flush=True)
        if args.plot:
            plot_complexity(args.out_dir / f"tf_similarity_{name}.tsv", args.cell_meta,
                            args.out_dir / f"corr_complexity_{name}.png",
                            f"TF-TF {args.metric} co-occupancy ({name}) vs cell count",
                            R, tfs, args.metric, name)

    # --- A vs B differential: which pairs co-bind MORE in A vs in B ---
    if "A" in Rs and "B" in Rs:
        dAB = Rs["A"] - Rs["B"]
        write_matrix(args.out_dir / "tf_ab_differential.tsv", dAB, tfs)
        iu = np.triu_indices(n_tf, 1)
        d = dAB[iu]; order = np.argsort(d)
        with (args.out_dir / "tf_ab_top_pairs.tsv").open("w") as f:
            f.write("pair\tsim_A\tsim_B\tdelta_A_minus_B\tcompartment_pref\n")
            for k in list(order[::-1][:25]) + list(order[:25]):   # top A-specific, then top B-specific
                i, j = int(iu[0][k]), int(iu[1][k])
                f.write(f"{tfs[i]}-{tfs[j]}\t{Rs['A'][i,j]:.4f}\t{Rs['B'][i,j]:.4f}\t"
                        f"{d[k]:+.4f}\t{'A' if d[k] > 0 else 'B'}\n")
        print(f"[A/B diff] wrote tf_ab_differential.tsv + tf_ab_top_pairs.tsv "
              f"(A-specific pairs: delta>0, B-specific: delta<0)", flush=True)

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

    print(f"[done] -> {args.out_dir}")
    return 0


def cluster_members(R, n_tf, args):
    """Cluster the TF x TF similarity R into groups (list of member-index lists)."""
    if args.cluster == "hierarchical":
        D = 1.0 - R; D = (D + D.T) / 2.0; np.fill_diagonal(D, 0.0); D[D < 0] = 0.0
        Z = linkage(squareform(D, checks=False), method="average")
        labels_c = fcluster(Z, t=1.0 - args.cluster_threshold, criterion="distance")
        groups = defaultdict(list)
        for i, c in enumerate(labels_c):
            groups[c].append(i)
        return list(groups.values())
    A = R >= args.cluster_threshold
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
    return list(groups.values())


def write_complexes(path, members_list, R, tfs):
    cid = 0
    with open(path, "w") as f:
        f.write("complex_id\tn_tfs\tmean_intra_sim\tmembers\n")
        for members in sorted(members_list, key=len, reverse=True):
            if len(members) < 2:
                continue
            cid += 1
            s = R[np.ix_(members, members)]
            mic = (s.sum() - np.trace(s)) / (len(members) * (len(members) - 1))
            f.write(f"CX{cid:03d}\t{len(members)}\t{mic:.3f}\t"
                    f"{','.join(sorted(tfs[i] for i in members))}\n")
    return cid


def plot_complexity(sim_tsv, cell_meta, out_png, title, R, tfs, metric, name):
    """Complexity-format heatmap (dendrogram + heatmap + per-TF cell-count bars) via
    plot_correlation_complexity.py -- applied to EVERY scope (genome/A/B). Falls back to
    the plain heatmap only if the cell-count metadata is unavailable."""
    script = Path(__file__).resolve().parent / "plot_correlation_complexity.py"
    if cell_meta and Path(cell_meta).is_file() and script.is_file() and Path(sim_tsv).is_file():
        rc = subprocess.run([sys.executable, str(script),
                             "--sim-tsv", str(sim_tsv), "--cell-meta", str(cell_meta),
                             "--out", str(out_png), "--title", title]).returncode
        if rc == 0:
            return
        print(f"[plot] complexity plot failed (rc={rc}); plain-heatmap fallback", flush=True)
    else:
        print(f"[plot] cell-meta {cell_meta} not found; plain-heatmap fallback", flush=True)
    _plot(R, tfs, out_png.parent, metric, suffix=("" if name == "genome" else f"_{name}"))


def _plot(R, tfs, out_dir, metric, suffix=""):
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
    scope = suffix.lstrip("_") or "genome"
    ax.set_title(f"TF x TF co-occupancy ({metric}, {scope}, HOT-removed, clustered)")
    out = out_dir / f"complex_heatmap{suffix}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[plot] wrote {out} (vmax={vmax:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
