#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_compartment_classes.py
#
# Turn the A/B calls into a small set of genome-wide COMPARTMENT CLASSES, so the
# validated ChromHMM-18 RPKM workflow can be applied verbatim (the emitted BED is
# fed straight to build_chromhmm_matrix.py, which aggregates by the 4th column).
#
# Why classes and not one unit per domain: a ChromHMM state is an *aggregate* of all
# its segments genome-wide, so even a 1,060-cell TF puts ~10^5 reads in each of the
# 18 states and its profile is precisely estimated. Split the genome into 22,215
# compartment domains instead and a shallow TF gets ~3 reads/domain -- Poisson noise
# dominates, and noise attenuates Pearson r in proportion to depth (mean r tracked
# cell count at rho=+0.98 that way, vs +0.22 for ChromHMM-18). Plain A vs B is the
# opposite failure: 2 units make Pearson degenerate (always +/-1).
#
# So: bin the compartment eigenvector (E1) into quantile classes within each arm,
#   B<n> ... B1 | A1 ... A<n>   (B<n> = strongest B, A<n> = strongest A),
# giving 2*n units that each pool ~1/2n of the genome. --per-arm 9 -> 18 classes,
# matching ChromHMM-18's dimensionality and reads per unit.
#
# Input : hic/work/compartments/compartments_25000.AB.bed  (chrom start end A|B E1)
# Output: <out>.bed  (chrom start end CLASS)  +  <out>.classes.tsv  (class bp n_bins)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
from collections import OrderedDict
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ab-bed", type=Path, required=True,
                    help="compartments_25000.AB.bed (chrom start end A|B E1)")
    ap.add_argument("--out", type=Path, required=True, help="output BED path")
    ap.add_argument("--per-arm", type=int, default=9,
                    help="quantile classes per arm; total units = 2*per-arm (default 9 -> 18)")
    args = ap.parse_args()

    opener = gzip.open if str(args.ab_bed).endswith(".gz") else open
    rec = []
    for ln in opener(args.ab_bed, "rt"):
        if not ln.strip() or ln.startswith(("#", "track", "browser", "chrom")):
            continue
        p = ln.rstrip("\n").split("\t")
        if len(p) < 5 or not p[4].strip():
            continue
        try:
            e1 = float(p[4])
        except ValueError:
            continue
        rec.append((p[0], int(p[1]), int(p[2]), p[3].strip(), e1))
    if not rec:
        raise SystemExit(f"no usable rows (need chrom start end A|B E1) in {args.ab_bed}")
    e1 = np.array([r[4] for r in rec])
    print(f"[in] {len(rec):,} bins with E1 "
          f"(A={sum(r[3]=='A' for r in rec):,} B={sum(r[3]=='B' for r in rec):,})", flush=True)

    # Quantile-bin |E1| within each arm so every class holds ~the same number of bins.
    k = args.per_arm
    lab = np.empty(len(rec), dtype=object)
    for arm, sign in (("A", 1), ("B", -1)):
        m = np.array([r[3] == arm for r in rec])
        if not m.any():
            continue
        v = np.abs(e1[m])
        # rank-based split -> equal-occupancy classes even with a skewed E1 distribution
        q = np.argsort(np.argsort(v)) * k // max(m.sum(), 1)      # 0..k-1, weakest -> strongest
        lab[m] = [f"{arm}{int(j)+1}" for j in q]

    order = [f"B{i}" for i in range(k, 0, -1)] + [f"A{i}" for i in range(1, k + 1)]
    bp, nb = OrderedDict((c, 0) for c in order), OrderedDict((c, 0) for c in order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for (c, s, e, _, _), L in zip(rec, lab):
            f.write(f"{c}\t{s}\t{e}\t{L}\n")
            bp[L] += e - s
            nb[L] += 1
    tsv = args.out.with_suffix(".classes.tsv")
    with tsv.open("w") as f:
        f.write("class\tbp\tn_bins\n")
        for c in order:
            f.write(f"{c}\t{bp[c]}\t{nb[c]}\n")
    print("[classes] " + "  ".join(f"{c}={nb[c]:,}bins/{bp[c]/1e6:.1f}Mb" for c in order), flush=True)
    print(f"[done] {len(order)} classes -> {args.out} (+ {tsv.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
