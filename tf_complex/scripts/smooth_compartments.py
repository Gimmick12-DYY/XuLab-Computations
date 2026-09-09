#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# smooth_compartments.py
#
# Absorb spurious short compartment flips: a short B run sandwiched between two A
# runs becomes A (and vice versa), then adjacent same-label runs are merged. Repeat
# until nothing changes.
#
# Motivation: in the 25 kb HEK293T calls, 48.5% of the 22,215 domains are a SINGLE
# bin and 10,425 of those single-bin domains sit between two opposite-label
# neighbours. Those are eigenvector flicker near E1 ~ 0, not 25 kb compartments.
# They also wreck the RPKM correlation: they inflate the unit count, so each unit
# gets few reads and shallow TFs go Poisson-dominated (see README step 5).
#
# Runs are built from consecutive same-label bins on a chromosome; a gap larger than
# --max-gap (unmappable / no-E1 stretches) breaks a run and blocks a flip across it.
#
# Input : compartments_25000.AB.bed   (chrom start end A|B E1)
# Output: <out>.bed          smoothed per-bin BED (chrom start end A|B E1) -- drop-in
#                            replacement for AB.bed (UNIT=class / UNIT=bin)
#         <out>.domains.bed  smoothed domains     (chrom start end A|B n_bins bp)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np


def read_bins(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    by_c = {}
    for ln in opener(path, "rt"):
        if not ln.strip() or ln.startswith(("#", "track", "browser", "chrom")):
            continue
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4 or p[3].strip() not in ("A", "B"):
            continue
        e1 = p[4].strip() if len(p) > 4 else ""
        by_c.setdefault(p[0], []).append([int(p[1]), int(p[2]), p[3].strip(), e1])
    for c in by_c:
        by_c[c].sort()
    return by_c


def runs_of(bins, max_gap):
    """[start, end, label, n_bins] for consecutive same-label bins (gap <= max_gap)."""
    out = []
    for s, e, lab, _ in bins:
        if out and out[-1][2] == lab and s - out[-1][1] <= max_gap:
            out[-1][1] = e; out[-1][3] += 1
        else:
            out.append([s, e, lab, 1])
    return out


def smooth(runs, min_bins, max_gap):
    """Flip short sandwiched runs, re-merge, repeat. Returns (runs, n_flipped)."""
    flipped = 0
    while True:
        hit = 0
        for i in range(1, len(runs) - 1):
            a, b, c = runs[i - 1], runs[i], runs[i + 1]
            if b[3] > min_bins or a[2] == b[2] or c[2] == b[2] or a[2] != c[2]:
                continue
            if b[0] - a[1] > max_gap or c[0] - b[1] > max_gap:   # don't flip across a gap
                continue
            b[2] = a[2]; hit += 1
        if not hit:
            return runs, flipped
        flipped += hit
        merged = []                                              # re-merge adjacent same-label
        for r in runs:
            if merged and merged[-1][2] == r[2] and r[0] - merged[-1][1] <= max_gap:
                merged[-1][1] = r[1]; merged[-1][3] += r[3]
            else:
                merged.append(r)
        runs = merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ab-bed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output per-bin BED path")
    ap.add_argument("--min-bins", type=int, default=4,
                    help="absorb sandwiched runs of <= this many bins (default 4 = 100 kb)")
    ap.add_argument("--max-gap", type=int, default=100_000,
                    help="largest gap still treated as contiguous (default 100 kb)")
    args = ap.parse_args()

    by_c = read_bins(args.ab_bed)
    n_in = sum(len(v) for v in by_c.values())
    pre = {c: runs_of(v, args.max_gap) for c, v in by_c.items()}
    n_pre = sum(len(v) for v in pre.values())

    post, n_flip = {}, 0
    for c, r in pre.items():
        post[c], f = smooth([x[:] for x in r], args.min_bins, args.max_gap)
        n_flip += f
    n_post = sum(len(v) for v in post.values())

    # relabel every bin by the smoothed run covering it
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dom_path = args.out.with_suffix(".domains.bed")
    n_changed = 0
    with args.out.open("w") as fb, dom_path.open("w") as fd:
        for c in sorted(by_c):
            rs = post[c]
            starts = np.array([r[0] for r in rs])
            for r in rs:
                fd.write(f"{c}\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[1]-r[0]}\n")
            for s, e, lab, e1 in by_c[c]:
                k = int(np.searchsorted(starts, s, side="right")) - 1
                new = rs[k][2] if k >= 0 and s < rs[k][1] else lab
                n_changed += (new != lab)
                fb.write(f"{c}\t{s}\t{e}\t{new}\t{e1}\n")

    sz = np.array([r[1] - r[0] for c in post for r in post[c]], float)
    one = sum(1 for c in post for r in post[c] if r[3] == 1)
    print(f"[in]  {n_in:,} bins -> {n_pre:,} domains")
    print(f"[out] {n_post:,} domains ({100*(1-n_post/n_pre):.1f}% fewer), "
          f"{n_flip:,} run flips, {n_changed:,} bins relabelled ({100*n_changed/n_in:.1f}%)")
    print(f"[size] median {int(np.median(sz)):,} bp (was 50,000), "
          f"mean {int(sz.mean()):,} bp, single-bin domains {one:,} ({100*one/n_post:.1f}%)")
    print(f"[done] {args.out}  +  {dom_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
