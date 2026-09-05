#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# compartment_correlation.py
#
# TF x TF correlation across A/B compartment DOMAINS (build_compartment_matrix.py),
# with per-TF CPM-normalized read FRACTIONS as the input (not raw counts).
#
#   input  : (domains x TFs) read counts
#   CPM    : per TF, counts / total x 1e6  (fraction of the TF's reads); log1p optional
#   corr   : Pearson between TFs, over ALL domains (genome), A-only (A-A), B-only (B-B)
#
# Output (--out-dir):
#   tf_similarity_{genome,A,B}.tsv     TF x TF Pearson
#   corr_complexity_{genome,A,B}.png   complexity-format heatmap (--plot)
#   candidate_complexes.tsv            (optional clustering)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def write_matrix(path, M, labels):
    with open(path, "w") as f:
        f.write("TF\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels))) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True, help="build_compartment_matrix.py output")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--input", choices=["tf_fraction", "domain_fraction", "count"], default="tf_fraction",
                    help="tf_fraction (default) = per-TF fraction: each TF's reads in a compartment / "
                         "that TF's TOTAL reads (its distribution across compartments, sums to 1). "
                         "domain_fraction = each compartment's reads / that compartment's total across "
                         "all TFs (the domain's TF composition). count = raw reads.")
    ap.add_argument("--size-normalize", action="store_true", default=False,
                    help="divide each domain's reads by its size (reads/bp) before correlating. "
                         "Off by default = 'just the fraction'. NOTE per-TF fraction is Pearson "
                         "scale-invariant, so tf_fraction alone == raw counts (size-dominated, "
                         "all-high); turn this on (or use --input domain_fraction) to de-confound.")
    ap.add_argument("--log1p", action="store_true", default=False,
                    help="log1p-transform the input before correlating. OFF by default: 'just the "
                         "fraction' = raw per-domain fraction, Pearson (simpler and a stronger "
                         "complex signal in tests). Turn on for count/cpm-heavy tails.")
    ap.add_argument("--min-reads", type=float, default=0.0,
                    help="drop domains with total reads (across TFs) <= this (sparse units add noise)")
    ap.add_argument("--cluster-threshold", type=float, default=0.0,
                    help=">0 = also cluster (hierarchical, distance cut 1-thr) -> candidate_complexes")
    ap.add_argument("--cell-meta", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    M = sp.load_npz(args.matrix_dir / "compartment_matrix.npz").tocsc().astype(np.float64)
    tfs = [t.strip() for t in (args.matrix_dir / "tfs.txt").read_text().split() if t.strip()]
    n_tf = len(tfs)
    dom_rows = [ln.split("\t") for ln in open(args.matrix_dir / "domains.tsv")]
    labs = np.array([r[3].strip() for r in dom_rows], dtype="<U1")
    bp = np.array([float(r[4]) for r in dom_rows], dtype=np.float64)   # domain size (bp)
    nD = M.shape[0]
    print(f"[matrix] {nD:,} domains x {n_tf} TFs", flush=True)

    X = np.asarray(M.todense())                          # domains x TFs (counts)
    tf_total = X.sum(axis=0)                              # per-TF library size (for confound diag)
    if args.size_normalize:                              # reads/bp -> remove domain-size domination
        X = X / np.where(bp > 0, bp, 1.0)[:, None]
        print("[size] normalized reads by domain size (reads/bp)", flush=True)
    if args.input == "tf_fraction":
        tot = X.sum(axis=0, keepdims=True); tot[tot == 0] = 1.0
        X = X / tot                                       # per-TF fraction (TF's reads / TF total)
        print(f"[input] per-TF FRACTION{'+log1p' if args.log1p else ''} "
              "(each TF's reads in compartment / its total)", flush=True)
    elif args.input == "domain_fraction":
        dt = X.sum(axis=1, keepdims=True); dt[dt == 0] = 1.0
        X = X / dt                                        # per-domain fraction (domain's TF composition)
        print(f"[input] per-domain FRACTION{'+log1p' if args.log1p else ''}", flush=True)
    else:
        print("[input] raw counts", flush=True)
    if args.log1p:
        X = np.log1p(X * (1e4 if args.input != "count" else 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scopes = {"genome": np.ones(nD, dtype=bool), "A": labs == "A", "B": labs == "B"}
    Rs = {}
    for name, m in scopes.items():
        if args.min_reads > 0:
            m = m & (np.asarray(M.sum(1)).ravel() > args.min_reads)
        idx = np.flatnonzero(m)
        if idx.size < 10:
            print(f"[{name}] only {idx.size} domains; skipping", flush=True); continue
        R = np.nan_to_num(np.corrcoef(X[idx], rowvar=False))
        write_matrix(args.out_dir / f"tf_similarity_{name}.tsv", R, tfs)
        off = R[~np.eye(n_tf, dtype=bool)]
        ms = (R.sum(1) - np.diag(R)) / (n_tf - 1)
        conf = spearmanr(tf_total, ms)[0]
        print(f"[{name}] {idx.size:,} domains  off-diag median={np.median(off):+.3f} "
              f"p95={np.percentile(off,95):+.3f} max={off.max():+.3f} | "
              f"confound(TF_total_reads, mean_corr)={conf:+.3f}", flush=True)
        Rs[name] = R

    # A vs B differential
    if "A" in Rs and "B" in Rs:
        dAB = Rs["A"] - Rs["B"]; write_matrix(args.out_dir / "tf_ab_differential.tsv", dAB, tfs)
        iu = np.triu_indices(n_tf, 1); d = dAB[iu]; order = np.argsort(d)
        with (args.out_dir / "tf_ab_top_pairs.tsv").open("w") as f:
            f.write("pair\tsim_A\tsim_B\tdelta_A_minus_B\tpref\n")
            for k in list(order[::-1][:25]) + list(order[:25]):
                i, j = int(iu[0][k]), int(iu[1][k])
                f.write(f"{tfs[i]}-{tfs[j]}\t{Rs['A'][i,j]:.4f}\t{Rs['B'][i,j]:.4f}\t{d[k]:+.4f}\t{'A' if d[k]>0 else 'B'}\n")

    # optional clustering
    if args.cluster_threshold > 0 and "genome" in Rs:
        R = Rs["genome"]; D = 1 - R; D = (D + D.T) / 2; np.fill_diagonal(D, 0); D[D < 0] = 0
        lab_c = fcluster(linkage(squareform(D, checks=False), "average"),
                         t=1 - args.cluster_threshold, criterion="distance")
        groups = defaultdict(list)
        for i, c in enumerate(lab_c):
            groups[c].append(i)
        with (args.out_dir / "candidate_complexes.tsv").open("w") as f:
            f.write("complex_id\tn_tfs\tmean_intra\tmembers\n"); cid = 0
            for mem in sorted(groups.values(), key=len, reverse=True):
                if len(mem) < 2:
                    continue
                cid += 1; s = R[np.ix_(mem, mem)]
                mic = (s.sum() - np.trace(s)) / (len(mem) * (len(mem) - 1))
                f.write(f"CX{cid:03d}\t{len(mem)}\t{mic:.3f}\t{','.join(sorted(tfs[i] for i in mem))}\n")
        print(f"[complexes] {cid} groups", flush=True)

    if args.plot:
        script = Path(__file__).resolve().parent / "plot_correlation_complexity.py"
        for name in Rs:
            sim = args.out_dir / f"tf_similarity_{name}.tsv"
            if script.is_file() and args.cell_meta.is_file():
                subprocess.run([sys.executable, str(script), "--sim-tsv", str(sim),
                                "--cell-meta", str(args.cell_meta),
                                "--out", str(args.out_dir / f"corr_complexity_{name}.png"),
                                "--title", f"TF-TF {args.input} corr over {name} compartments vs cell count"])
    print(f"[done] -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
