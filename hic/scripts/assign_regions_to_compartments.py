#!/usr/bin/env python3
"""Assign regions to A/B compartments and score size-normalized enrichment.

General tool: works on any region set -- co-binding modules/cliques/peaks (chr:s-e
tokens) or a BED of TF peaks. Each region is assigned to the compartment covering
the majority of its length (against classify_compartments.py's <prefix>.AB.bed),
then enrichment is normalized to compartment SIZE:

    obs_A = fraction of regions in A ;  exp_A = fraction of the genome that is A
    log2 enrichment = log2(obs_A / exp_A)   (+ binomial test vs exp_A)

so a bigger compartment doesn't trivially win.

Inputs:
  --regions   BED (chrom start end [name]) OR any file containing chr:start-end
              tokens (auto-detected; e.g. cobinding nodes.tsv / modules.tsv).
  --ab-bed    <prefix>.AB.bed from classify_compartments.py

Outputs:
  --out       per-region assignment TSV (chrom start end name compartment frac_A)
  stdout      size-normalized enrichment summary (also --summary <tsv>)
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

_COORD = re.compile(r"(chr[0-9A-Za-z]+)[:_](\d+)[-_](\d+)")


def load_ab(path: Path):
    """Return (per-chrom {bin_index: 'A'/'B'}, resolution, A_bp, B_bp)."""
    by_chrom: dict[str, dict[int, str]] = defaultdict(dict)
    widths: dict[int, int] = defaultdict(int)
    abp = bbp = 0
    with open(path) as f:
        for ln in f:
            p = ln.split("\t")
            if len(p) < 4:
                continue
            c, s, e, lab = p[0], int(p[1]), int(p[2]), p[3].strip()
            widths[e - s] += 1
            if lab == "A":
                abp += e - s
            elif lab == "B":
                bbp += e - s
    res = max(widths, key=widths.get)
    with open(path) as f:
        for ln in f:
            p = ln.split("\t")
            if len(p) < 4:
                continue
            c, s, e, lab = p[0], int(p[1]), int(p[2]), p[3].strip()
            by_chrom[c][s // res] = lab
    return by_chrom, res, abp, bbp


def iter_regions(path: Path, bounding_box: bool = False):
    """Yield (chrom, start, end, name) from a BED or any chr:s-e-bearing file.

    bounding_box: collapse every chr:s-e token on a line into one span (same
    chromosome) named by the first column — for cobinding modules.tsv / cliques.tsv.
    """
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            cols = ln.split("\t")
            # BED: chrom \t int \t int  (chrom must be a bare chrN, not chr:s-e)
            if (len(cols) >= 3 and re.fullmatch(r"chr[0-9A-Za-z]+", cols[0])
                    and cols[1].isdigit() and cols[2].isdigit()):
                name = cols[3] if len(cols) > 3 else f"{cols[0]}:{cols[1]}-{cols[2]}"
                yield cols[0], int(cols[1]), int(cols[2]), name
                continue
            name = cols[0]
            hits = list(_COORD.finditer(ln))
            if not hits:
                continue
            if bounding_box:
                chroms = {m.group(1) for m in hits}
                if len(chroms) != 1:
                    continue  # skip trans-chrom modules
                c = chroms.pop()
                s = min(int(m.group(2)) for m in hits)
                e = max(int(m.group(3)) for m in hits)
                yield c, s, e, name
            else:
                for m in hits:
                    yield m.group(1), int(m.group(2)), int(m.group(3)), name


def assign(chrom, s, e, by_chrom, res):
    """Majority-overlap compartment + fraction of length in A."""
    ov = {"A": 0, "B": 0}
    labels = by_chrom.get(chrom)
    if labels:
        for b in range(s // res, (e - 1) // res + 1):
            lab = labels.get(b)
            if lab is None:
                continue
            bp = min(e, (b + 1) * res) - max(s, b * res)
            ov[lab] += bp
    tot = ov["A"] + ov["B"]
    if tot == 0:
        return ".", float("nan")
    comp = "A" if ov["A"] >= ov["B"] else "B"
    return comp, ov["A"] / tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--ab-bed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--label", default="regions", help="label for the summary row")
    ap.add_argument("--bounding-box", action="store_true",
                    help="one region per input line = bounding box of all chr:s-e tokens "
                         "(cobinding modules.tsv / cliques.tsv)")
    args = ap.parse_args()

    by_chrom, res, abp, bbp = load_ab(args.ab_bed)
    exp_a = abp / (abp + bbp)
    print(f"[compartments] resolution={res} A={abp/1e6:.1f}Mb B={bbp/1e6:.1f}Mb "
          f"(expected A fraction={exp_a:.3f})", flush=True)

    n_a = n_b = n_un = 0
    bp_a = bp_b = 0
    seen = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as o:
        o.write("chrom\tstart\tend\tname\tcompartment\tfrac_A\n")
        for c, s, e, name in iter_regions(args.regions, bounding_box=args.bounding_box):
            key = (c, s, e)
            comp, fa = assign(c, s, e, by_chrom, res)
            o.write(f"{c}\t{s}\t{e}\t{name}\t{comp}\t{fa:.3f}\n" if fa == fa
                    else f"{c}\t{s}\t{e}\t{name}\t{comp}\tNA\n")
            if key in seen:      # count each unique region once for the stats
                continue
            seen.add(key)
            if comp == "A":
                n_a += 1; bp_a += fa * (e - s); bp_b += (1 - fa) * (e - s)
            elif comp == "B":
                n_b += 1; bp_a += fa * (e - s); bp_b += (1 - fa) * (e - s)
            else:
                n_un += 1

    n = n_a + n_b
    obs_a = n_a / n if n else float("nan")
    log2e = math.log2(obs_a / exp_a) if n and obs_a > 0 and exp_a > 0 else float("nan")
    # binomial test vs the size expectation
    try:
        from scipy.stats import binomtest
        pval = binomtest(n_a, n, exp_a).pvalue if n else float("nan")
    except Exception:  # noqa: BLE001
        pval = float("nan")
    bp_frac_a = bp_a / (bp_a + bp_b) if (bp_a + bp_b) else float("nan")

    hdr = ("label\tn_regions\tn_A\tn_B\tn_unassigned\tobs_fracA\texp_fracA\t"
           "log2_enrichA\tbinom_p\tbp_fracA\tbp_log2_enrichA")
    bp_log2 = math.log2(bp_frac_a / exp_a) if bp_frac_a == bp_frac_a and bp_frac_a > 0 else float("nan")
    row = (f"{args.label}\t{n}\t{n_a}\t{n_b}\t{n_un}\t{obs_a:.4f}\t{exp_a:.4f}\t"
           f"{log2e:.4f}\t{pval:.3g}\t{bp_frac_a:.4f}\t{bp_log2:.4f}")
    print("\n" + hdr.replace("\t", "  "))
    print(row.replace("\t", "  "))
    print(f"\n=> {args.label}: {n_a}/{n} regions in A ({100*obs_a:.1f}%) vs {100*exp_a:.1f}% expected "
          f"by size  =>  log2 enrichment(A) = {log2e:+.3f}  (p={pval:.2g})")
    if args.summary:
        with args.summary.open("w") as o:
            o.write(hdr + "\n" + row + "\n")
    print(f"[done] per-region -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
