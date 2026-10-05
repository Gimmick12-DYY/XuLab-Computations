#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# extract_pairs_peaks.py   (two-stage Cicero, stage-2 input)
#
# Unique peak regions (colon form chr:start-end) from a cicero_conns_to_edges
# peak_edges.tsv, optionally restricted to significant pairs (qval <= --qval).
# Feed as --regions to export_pmat_tf_mm.R to re-run Cicero on just the
# called-coordinated-binding-pair peaks.
#
#   python extract_pairs_peaks.py --edges <tf>.peak_edges.tsv --qval 0.05 --out peaks1.txt
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qval", type=float, default=1.0,
                    help="keep peaks from pairs with qval <= this (1 = all pairs)")
    args = ap.parse_args()

    peaks: list[str] = []
    seen: set[str] = set()
    n_pair = 0
    with open(args.edges) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(hdr)}
        i1 = col.get("peak1", 0); i2 = col.get("peak2", 1)
        iq = col.get("qval", col.get("fdr"))
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) <= max(i1, i2):
                continue
            if iq is not None and iq < len(p):
                try:
                    if float(p[iq]) > args.qval:
                        continue
                except ValueError:
                    pass
            n_pair += 1
            for pk in (p[i1], p[i2]):
                if pk and pk not in seen:
                    seen.add(pk); peaks.append(pk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(peaks) + "\n")
    print(f"[peaks] {n_pair} pairs (qval<={args.qval}) -> {len(peaks)} unique peaks -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
