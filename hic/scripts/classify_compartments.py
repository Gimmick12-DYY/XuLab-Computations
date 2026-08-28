#!/usr/bin/env python3
"""Classify 25 kb bins into A/B compartments and compute compartment sizes.

From the GC-phased first eigenvector (cooltools eigs-cis; E1 > 0 = A, E1 < 0 = B),
label each bin A or B, merge contiguous bins into compartment domains, and report
the total genomic size of A vs B -- the normalization factor used by
assign_regions_to_compartments.py for size-corrected enrichment.

Inputs:
  --e1-bedgraph   compartments_<res>.E1.bedGraph  (chrom start end E1)

Outputs (<out-prefix>):
  .AB.bed       chrom start end compartment E1            (per bin)
  .domains.bed  chrom start end compartment n_bins bp     (merged runs)
  .sizes.tsv    compartment sizes genome-wide + per chrom (bp, bins, fraction)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def read_e1(path: Path):
    out = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 4:
                continue
            try:
                v = float(p[3])
            except ValueError:
                continue
            if v != v:  # NaN
                continue
            out.append((p[0], int(p[1]), int(p[2]), v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1-bedgraph", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    bins = read_e1(args.e1_bedgraph)
    if not bins:
        raise SystemExit(f"no usable E1 rows in {args.e1_bedgraph}")
    bins.sort(key=lambda r: (r[0], r[1]))
    labeled = [(c, s, e, "A" if v > 0 else "B", v) for c, s, e, v in bins if v != 0]

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ab_path = Path(str(args.out_prefix) + ".AB.bed")
    with ab_path.open("w") as o:
        for c, s, e, lab, v in labeled:
            o.write(f"{c}\t{s}\t{e}\t{lab}\t{v:.5g}\n")

    # merge contiguous same-compartment bins into domains
    domains = []
    cur = None
    for c, s, e, lab, _ in labeled:
        if cur and cur[0] == c and cur[3] == lab and s <= cur[2]:
            cur[2] = e
            cur[4] += 1
        else:
            if cur:
                domains.append(cur)
            cur = [c, s, e, lab, 1]
    if cur:
        domains.append(cur)
    dom_path = Path(str(args.out_prefix) + ".domains.bed")
    with dom_path.open("w") as o:
        for c, s, e, lab, n in domains:
            o.write(f"{c}\t{s}\t{e}\t{lab}\t{n}\t{e - s}\n")

    # sizes: genome-wide + per chrom
    bp = defaultdict(lambda: defaultdict(int))     # chrom -> {A: bp, B: bp}
    nb = defaultdict(lambda: defaultdict(int))     # chrom -> {A: bins, B: bins}
    for c, s, e, lab, _ in labeled:
        bp[c][lab] += e - s
        nb[c][lab] += 1
    tot_a = sum(bp[c]["A"] for c in bp)
    tot_b = sum(bp[c]["B"] for c in bp)
    tot = tot_a + tot_b
    n_dom_a = sum(1 for d in domains if d[3] == "A")
    n_dom_b = sum(1 for d in domains if d[3] == "B")

    sizes_path = Path(str(args.out_prefix) + ".sizes.tsv")
    with sizes_path.open("w") as o:
        o.write("scope\tcompartment\tn_bins\tn_domains\tbp\tgenome_fraction\n")
        o.write(f"genome\tA\t{sum(nb[c]['A'] for c in nb)}\t{n_dom_a}\t{tot_a}\t{tot_a/tot:.5f}\n")
        o.write(f"genome\tB\t{sum(nb[c]['B'] for c in nb)}\t{n_dom_b}\t{tot_b}\t{tot_b/tot:.5f}\n")
        for c in sorted(bp):
            ct = bp[c]["A"] + bp[c]["B"]
            for lab in ("A", "B"):
                frac = bp[c][lab] / ct if ct else 0.0
                nd = sum(1 for d in domains if d[0] == c and d[3] == lab)
                o.write(f"{c}\t{lab}\t{nb[c][lab]}\t{nd}\t{bp[c][lab]}\t{frac:.5f}\n")

    print(f"[classify] {len(labeled):,} bins  A={tot_a/1e6:.1f}Mb ({100*tot_a/tot:.1f}%, {n_dom_a} domains)  "
          f"B={tot_b/1e6:.1f}Mb ({100*tot_b/tot:.1f}%, {n_dom_b} domains)")
    print(f"[classify] wrote {ab_path}\n[classify] wrote {dom_path}\n[classify] wrote {sizes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
