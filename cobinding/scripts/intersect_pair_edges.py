#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# intersect_pair_edges.py   (two-stage Cicero: reproducibility filter)
#
# Intersect two peak_edges.tsv pair sets (round 1 and round 2, each optionally
# q-filtered) as UNORDERED pairs. Pairs significant in BOTH rounds are the
# reproducible coordinated-binding calls -- more likely real (motivated by the
# low overlap of single-round Cicero pairs with Hi-C loops).
#
# Output keeps the round-1 row + appends round-2 coaccess/qval, and prints
# counts + Jaccard. Does not touch the input files.
#
#   python intersect_pair_edges.py --a <tf>.peak_edges.tsv --b stage2/<tf>.peak_edges.tsv \
#       --qval 0.05 --out <tf>.reproducible_pairs.tsv
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path


def load(path: Path, qmax: float):
    """unordered-pair key -> (full_row_list, coaccess, qval); q-filtered."""
    rows: dict[frozenset, tuple] = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(hdr)}
        i1, i2 = col.get("peak1", 0), col.get("peak2", 1)
        ica = col.get("coaccess", 2)
        iq = col.get("qval", col.get("fdr"))
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) <= max(i1, i2):
                continue
            q = float("nan")
            if iq is not None and iq < len(p):
                try:
                    q = float(p[iq])
                except ValueError:
                    q = float("nan")
            if q == q and q > qmax:
                continue
            ca = float(p[ica]) if ica < len(p) and p[ica] not in ("", "nan") else float("nan")
            rows[frozenset((p[i1], p[i2]))] = (p, ca, q)
    return hdr, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="round-1 peak_edges.tsv")
    ap.add_argument("--b", type=Path, required=True, help="round-2 peak_edges.tsv")
    ap.add_argument("--qval", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hdr_a, A = load(args.a, args.qval)
    _, B = load(args.b, args.qval)
    inter = set(A) & set(B)
    union = set(A) | set(B)
    jac = len(inter) / len(union) if union else 0.0
    print(f"[intersect] round1={len(A):,} round2={len(B):,} reproducible={len(inter):,} "
          f"(jaccard={jac:.4f}; {100*len(inter)/max(len(A),1):.1f}% of round1)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("\t".join(hdr_a) + "\tcoaccess_r2\tqval_r2\n")
        for key in inter:
            row, _, _ = A[key]
            _, ca2, q2 = B[key]
            f.write("\t".join(row) + f"\t{ca2:.6g}\t{q2:.6g}\n")
    print(f"[intersect] wrote {len(inter):,} reproducible pairs -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
