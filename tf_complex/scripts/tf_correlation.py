#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# tf_correlation.py
#
# TF x TF co-occupancy from the raw binding matrix (build_tf_matrix.py), A/B
# stratified, then clustering -> candidate protein complexes.
#
# Raw scCUT&Tag is sparse and each TF is profiled in DIFFERENT cells, so at 1 kb
# two co-binding TFs rarely hit the same bin -> Spearman ~ 0 (flat heatmap). So:
#   * --agg-bp : aggregate fine bins into coarser windows (default 10 kb) so
#     co-localizing TFs share a bin.
#   * --metric cooccur (default): log2(observed/expected) overlap of bound bins --
#     robust to sparsity, measures "bind together above chance". Also spearman/
#     pearson/jaccard. A cooccur matrix is always written for interpretation.
#
# Prints sparsity + correlation-magnitude diagnostics so a flat result is explained.
#
# Outputs (--out-dir):
#   cooccur_genome.tsv / _A.tsv / _B.tsv     log2 obs/exp overlap (enrichment)
#   corr_genome.tsv (metric matrix, if metric != cooccur)
#   tf_compartment.tsv                       per-TF A/B enrichment
#   candidate_complexes.tsv                  clustered TF groups
#   correlation_heatmap.png (--plot)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

_REGION = re.compile(r"^([^:]+):(\d+)-(\d+)$")


def parse_regions(regions):
    chrom = np.empty(len(regions), dtype=object)
    start = np.zeros(len(regions), dtype=np.int64)
    for i, r in enumerate(regions):
        m = _REGION.match(r)
        chrom[i], start[i] = m.group(1), int(m.group(2))
    return chrom, start


def aggregate(M, chrom, start, agg_bp):
    """Sum fine bins into agg_bp windows. Returns (M_coarse csc, coarse_chrom, coarse_start)."""
    keys = {}
    coarse_idx = np.empty(M.shape[0], dtype=np.int64)
    cchrom, cstart = [], []
    for i in range(M.shape[0]):
        k = (chrom[i], start[i] // agg_bp)
        j = keys.get(k)
        if j is None:
            j = len(keys); keys[k] = j
            cchrom.append(k[0]); cstart.append(k[1] * agg_bp)
        coarse_idx[i] = j
    Agg = sp.csr_matrix((np.ones(M.shape[0]), (coarse_idx, np.arange(M.shape[0]))),
                        shape=(len(keys), M.shape[0]))
    return (Agg @ M).tocsc(), np.array(cchrom, dtype=object), np.array(cstart)


def compartment_of(ab_bed, cchrom, cstart):
    lab, widths = {}, {}
    for ln in open(ab_bed):
        p = ln.split("\t")
        if len(p) < 4:
            continue
        w = int(p[2]) - int(p[1]); widths[w] = widths.get(w, 0) + 1
    res = max(widths, key=widths.get)
    for ln in open(ab_bed):
        p = ln.split("\t")
        if len(p) < 4:
            continue
        lab[(p[0], int(p[1]) // res)] = p[3].strip()
    return np.array([lab.get((cchrom[i], cstart[i] // res), ".") for i in range(len(cchrom))],
                    dtype="<U1")


def cooccur_enrich(Bscope, N):
    """log2(observed/expected) overlap of bound bins over a scope of N universe bins.

    Bscope: scope_bins x TFs (sparse bool). obs = |Ai n Aj| = Bscope.T @ Bscope;
    expected under independence = |Ai||Aj|/N.

    N is the size of the null universe. Use the BINDABLE universe (bins bound by >=1
    TF), NOT the whole genome: TFs draw their bins from a tiny accessible fraction of
    the genome, so a genome-wide null makes *every* pair look enriched (universal
    positive gradient, no discrete complexes). Conditioning on the bindable universe
    puts accessibility-only pairs at ~0 and leaves only specific co-binding positive.
    """
    Bi = Bscope.astype(np.int32)                     # bool matmul OR's; int sums the overlap count
    obs = np.asarray((Bi.T @ Bi).todense()).astype(np.float64)
    n1 = np.asarray(Bscope.sum(axis=0)).ravel().astype(np.float64)
    exp = np.outer(n1, n1) / N if N else np.zeros_like(obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((obs > 0) & (exp > 0), np.log2(obs / exp), 0.0)


def corr_matrix(sub, metric):
    if metric == "cooccur":
        return cooccur(sub > 0)
    if metric == "jaccard":
        B = (sub > 0); n = B.shape[1]; cs = B.sum(0); R = np.eye(n)
        for i in range(n):
            inter = (B[:, i:i + 1] & B).sum(0)
            union = cs[i] + cs - inter
            R[i] = np.where(union > 0, inter / union, 0.0)
        return R
    X = sub.astype(np.float64)
    if metric == "spearman":
        X = np.apply_along_axis(rankdata, 0, X)
    return np.corrcoef(X, rowvar=False)


def write_matrix(path, M, labels):
    with open(path, "w") as f:
        f.write("TF\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels))) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True)
    ap.add_argument("--ab-bed", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--agg-bp", type=int, default=10000,
                    help="aggregate fine bins into this window (bp) before correlating (default 10000)")
    ap.add_argument("--top-n", type=int, default=20000,
                    help="define 'bound' as each TF's top-N bins by signal (coverage-equalized, "
                         "removes background). 0 = any signal>0 (not recommended: broad/deep TFs "
                         "then overlap trivially). Default 20000.")
    ap.add_argument("--metric", choices=["cooccur", "spearman", "pearson", "jaccard"], default="cooccur")
    ap.add_argument("--universe", choices=["bound", "genome"], default="bound",
                    help="null universe for cooccur expected: 'bound' = bins bound by >=1 TF "
                         "(default; removes shared-accessibility inflation), 'genome' = all bins "
                         "in scope (every pair looks enriched).")
    ap.add_argument("--min-tfs-per-bin", type=int, default=2)
    ap.add_argument("--cluster-threshold", type=float, default=1.0,
                    help="TFs join a complex when their genome-scope score >= this, in the "
                         "metric's units (cooccur: log2 fold, so 1.0 = 2x co-binding; "
                         "spearman/jaccard: ~0.3-0.5). Connected components >= 2 = complexes.")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    M = sp.load_npz(args.matrix_dir / "tf_matrix.npz").tocsc()
    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    regions = [r.strip() for r in (args.matrix_dir / "regions.tsv").read_text().splitlines() if r.strip()]
    print(f"[matrix] {M.shape[0]:,} bins x {M.shape[1]} TFs (nnz={M.nnz:,})", flush=True)

    chrom, start = parse_regions(regions)
    if args.agg_bp and args.agg_bp > 0:
        M, chrom, start = aggregate(M, chrom, start, args.agg_bp)
        print(f"[aggregate] -> {M.shape[0]:,} x {args.agg_bp}bp bins", flush=True)
    comp = compartment_of(args.ab_bed, chrom, start)

    # "bound" = each TF's top-N strongest bins (coverage-equalized, drops background).
    if args.top_n and args.top_n > 0:
        Mc = M.tocsc()
        ri, ci = [], []
        for j in range(M.shape[1]):
            st, en = Mc.indptr[j], Mc.indptr[j + 1]
            data = Mc.data[st:en]; ind = Mc.indices[st:en]
            if data.size > args.top_n:
                keep = ind[np.argpartition(data, -args.top_n)[-args.top_n:]]  # exactly top-N
            else:
                keep = ind[data > 0]
            ri.append(keep); ci.append(np.full(keep.size, j))
        ri = np.concatenate(ri) if ri else np.array([], int)
        ci = np.concatenate(ci) if ci else np.array([], int)
        Bb = sp.csc_matrix((np.ones(ri.size, dtype=bool), (ri, ci)), shape=M.shape)
    else:
        Bb = (M > 0).tocsc()

    nz = np.asarray(Bb.sum(axis=0)).ravel()
    print(f"[bound] top_n={args.top_n}: bound bins per TF median={int(np.median(nz))} "
          f"min={int(nz.min())} max={int(nz.max())}  (of {M.shape[0]:,} bins)", flush=True)

    feat = np.asarray(Bb.sum(axis=1)).ravel() >= args.min_tfs_per_bin   # for spearman/pearson only
    Bb = Bb.tocsr()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # scope = the FULL bin universe restricted to genome / A / B (the enrichment null).
    bin_scopes = {"genome": np.ones(M.shape[0], dtype=bool), "A": comp == "A", "B": comp == "B"}
    prim = None
    for name, m in bin_scopes.items():
        N = int(m.sum())
        if N < 10:
            print(f"[{name}] only {N} bins; skipping", flush=True)
            continue
        Bscope = Bb[np.flatnonzero(m)]
        if args.universe == "bound":
            Nuniv = int((np.asarray(Bscope.sum(axis=1)).ravel() > 0).sum())  # bins bound by >=1 TF
        else:
            Nuniv = N
        co = cooccur_enrich(Bscope, Nuniv)
        write_matrix(args.out_dir / f"cooccur_{name}.tsv", np.nan_to_num(co), tfs)
        if args.metric == "cooccur":
            R = co
        elif args.metric == "jaccard":
            R = corr_matrix(Bscope.toarray().astype(np.float64), "jaccard")
        else:  # spearman/pearson over feature bins in the scope
            fidx = np.flatnonzero(m & feat)
            R = corr_matrix(M[fidx].toarray(), args.metric)
            write_matrix(args.out_dir / f"corr_{name}.tsv", np.nan_to_num(R), tfs)
        off = R[~np.eye(len(tfs), dtype=bool)]
        print(f"[{name}] scope={N:,} bins, null_universe={Nuniv:,} ({args.universe})  "
              f"{args.metric} off-diag: median={np.median(off):.3f} "
              f"p90={np.percentile(off,90):.3f} max={np.nanmax(off):.3f}", flush=True)
        if name == "genome":
            prim = np.nan_to_num(R)

    # per-TF A/B enrichment (size-normalized by number of A vs B feature bins)
    isA = comp == "A"; isB = comp == "B"
    nA = int(isA.sum()); nB = int(isB.sum())
    exp_a = nA / (nA + nB) if (nA + nB) else float("nan")
    with (args.out_dir / "tf_compartment.tsv").open("w") as f:
        f.write("tf\tbound_bins\ttotal_signal\tsignal_fracA\texp_fracA\tlog2_enrichA\n")
        Ac = np.flatnonzero(isA); Bc = np.flatnonzero(isB)
        for j, tf in enumerate(tfs):
            col = M[:, j]
            sA = float(col[Ac].sum()); sB = float(col[Bc].sum()); tot = sA + sB
            fa = sA / tot if tot else float("nan")
            l2 = math.log2(fa / exp_a) if tot and fa > 0 and exp_a > 0 else float("nan")
            f.write(f"{tf}\t{int((col.toarray()>0).sum())}\t{float(col.sum()):.0f}\t{fa:.4f}\t{exp_a:.4f}\t{l2:.4f}\n")

    # candidate complexes = connected components where genome score >= threshold
    if prim is not None:
        n = len(tfs)
        A = prim >= args.cluster_threshold
        np.fill_diagonal(A, False)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        comp = defaultdict(list)
        for i in range(n):
            comp[find(i)].append(i)
        with (args.out_dir / "candidate_complexes.tsv").open("w") as f:
            f.write("complex_id\tn_tfs\tmean_intra_score\tmembers\n")
            cid = 0
            for members in sorted(comp.values(), key=len, reverse=True):
                if len(members) < 2:
                    continue
                cid += 1
                s = prim[np.ix_(members, members)]
                mic = (s.sum() - np.trace(s)) / (len(members) * (len(members) - 1))
                f.write(f"CX{cid:03d}\t{len(members)}\t{mic:.3f}\t"
                        f"{','.join(sorted(tfs[i] for i in members))}\n")
        print(f"[complexes] {cid} candidate complexes (>=2 TFs, {args.metric}>="
              f"{args.cluster_threshold})", flush=True)
        if args.plot:
            R = np.nan_to_num(prim); sim = (R - R.min()) / (R.max() - R.min() + 1e-12)
            D = 1.0 - sim; D = (D + D.T) / 2.0; np.fill_diagonal(D, 0.0)
            _plot(prim, tfs, leaves_list(linkage(squareform(D, checks=False), method="average")),
                  args.out_dir, args.metric)
    print(f"[done] outputs under {args.out_dir}")
    return 0


def _plot(R, tfs, order, out_dir, metric):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped: {e}"); return
    Ro = R[np.ix_(order, order)]; n = len(order)
    vmax = np.nanpercentile(np.abs(Ro[~np.eye(n, dtype=bool)]), 99) or 1.0
    fig, ax = plt.subplots(figsize=(max(6, n * 0.25),) * 2)
    im = ax.imshow(Ro, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(n), [tfs[i] for i in order], rotation=90, fontsize=5)
    ax.set_yticks(range(n), [tfs[i] for i in order], fontsize=5)
    fig.colorbar(im, ax=ax, label=f"TF co-occupancy ({metric})")
    ax.set_title(f"TF x TF co-occupancy ({metric}, clustered)")
    fig.tight_layout(); fig.savefig(out_dir / "correlation_heatmap.png", dpi=150); plt.close(fig)
    print(f"[plot] wrote {out_dir/'correlation_heatmap.png'} (vmax={vmax:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
