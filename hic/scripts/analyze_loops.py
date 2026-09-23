#!/usr/bin/env python3
"""Annotate Peakachu loops with A/B compartments and CTCF peaks.

Reads a pooled Peakachu BEDPE (chrom1 start1 end1 chrom2 start2 end2 score
[count]) and writes:

  <prefix>.annotated.tsv   per-loop A/B + CTCF overlap
  <prefix>.anchors.bed     both anchors (for IGV)
  <prefix>.summary.tsv     counts / distances / enrichment

Compartment assignment uses the 25 kb A/B bins from classify_compartments.py
(majority overlap). CTCF overlap is any intersection of an anchor with the
peak BED (default: CTCF majority ≥2-of-3 consensus).
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_ab(path: Path):
    by_chrom: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    abp = bbp = 0
    with open(path) as f:
        for ln in f:
            p = ln.split("\t")
            if len(p) < 4:
                continue
            c, s, e, lab = p[0], int(p[1]), int(p[2]), p[3].strip()
            by_chrom[c].append((s, e, lab))
            if lab == "A":
                abp += e - s
            elif lab == "B":
                bbp += e - s
    for c in by_chrom:
        by_chrom[c].sort()
    return by_chrom, abp, bbp


def assign_ab(by_chrom, chrom: str, start: int, end: int) -> str:
    bins = by_chrom.get(chrom)
    if not bins:
        return "NA"
    # binary search for first bin that could overlap
    lo, hi = 0, len(bins)
    while lo < hi:
        mid = (lo + hi) // 2
        if bins[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    a = b = 0
    for i in range(lo, len(bins)):
        s, e, lab = bins[i]
        if s >= end:
            break
        ov = min(end, e) - max(start, s)
        if ov <= 0:
            continue
        if lab == "A":
            a += ov
        elif lab == "B":
            b += ov
    if a == 0 and b == 0:
        return "NA"
    return "A" if a >= b else "B"


def load_bed(path: Path):
    by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    n = 0
    with open(path) as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#") or ln.startswith("track"):
                continue
            p = ln.split("\t")
            if len(p) < 3:
                continue
            by_chrom[p[0]].append((int(p[1]), int(p[2])))
            n += 1
    for c in by_chrom:
        by_chrom[c].sort()
    return by_chrom, n


def overlaps(by_chrom, chrom: str, start: int, end: int) -> int:
    ivs = by_chrom.get(chrom)
    if not ivs:
        return 0
    lo, hi = 0, len(ivs)
    while lo < hi:
        mid = (lo + hi) // 2
        if ivs[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    hits = 0
    for i in range(lo, len(ivs)):
        s, e = ivs[i]
        if s >= end:
            break
        if min(end, e) > max(start, s):
            hits += 1
    return hits


def parse_bedpe(path: Path):
    loops = []
    with open(path) as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#") or ln.startswith("track"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            try:
                c1, s1, e1 = p[0], int(p[1]), int(p[2])
                c2, s2, e2 = p[3], int(p[4]), int(p[5])
            except ValueError:
                continue
            score = p[6] if len(p) > 6 else ""
            count = p[-1] if len(p) > 7 else ""
            loops.append((c1, s1, e1, c2, s2, e2, score, count))
    return loops


def log2fc(obs: float, exp: float) -> float:
    if obs <= 0 or exp <= 0:
        return float("nan")
    return math.log2(obs / exp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loops", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True, type=Path)
    ap.add_argument("--ab-bed", type=Path, default=None)
    ap.add_argument("--ctcf", type=Path, default=None)
    args = ap.parse_args()

    loops = parse_bedpe(args.loops)
    ab_map, abp, bbp = (load_ab(args.ab_bed) if args.ab_bed else ({}, 0, 0))
    ctcf_map, n_ctcf = (load_bed(args.ctcf) if args.ctcf else ({}, 0))
    genome_ab = abp + bbp
    exp_a = (abp / genome_ab) if genome_ab else float("nan")

    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ann = prefix.with_name(prefix.name + ".annotated.tsv")
    anchors = prefix.with_name(prefix.name + ".anchors.bed")
    summary = prefix.with_name(prefix.name + ".summary.tsv")

    pair_ab = Counter()
    ctcf_class = Counter()
    dists: list[int] = []
    n_cis = n_trans = 0
    ctcf_at_anchor = 0  # unique-ish: count of CTCF overlaps across anchors

    with open(ann, "w") as oa, open(anchors, "w") as ob:
        oa.write(
            "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tscore\tcount\t"
            "span_bp\tcompartment1\tcompartment2\tpair_AB\t"
            "n_ctcf1\tn_ctcf2\tctcf_class\n"
        )
        for i, (c1, s1, e1, c2, s2, e2, score, count) in enumerate(loops, 1):
            span = abs(((s2 + e2) // 2) - ((s1 + e1) // 2)) if c1 == c2 else -1
            if c1 == c2:
                n_cis += 1
                dists.append(span)
            else:
                n_trans += 1
            a1 = assign_ab(ab_map, c1, s1, e1) if ab_map else "NA"
            a2 = assign_ab(ab_map, c2, s2, e2) if ab_map else "NA"
            labels = sorted([x for x in (a1, a2) if x in ("A", "B")])
            pair = "".join(labels) if len(labels) == 2 else "NA"
            if pair == "BA":
                pair = "AB"
            pair_ab[pair] += 1
            n1 = overlaps(ctcf_map, c1, s1, e1) if ctcf_map else 0
            n2 = overlaps(ctcf_map, c2, s2, e2) if ctcf_map else 0
            n_hit = (1 if n1 else 0) + (1 if n2 else 0)
            klass = {0: "none", 1: "one", 2: "both"}[n_hit]
            ctcf_class[klass] += 1
            if n1:
                ctcf_at_anchor += 1
            if n2:
                ctcf_at_anchor += 1
            oa.write(
                f"{c1}\t{s1}\t{e1}\t{c2}\t{s2}\t{e2}\t{score}\t{count}\t"
                f"{span}\t{a1}\t{a2}\t{pair}\t{n1}\t{n2}\t{klass}\n"
            )
            name = f"loop{i}"
            ob.write(f"{c1}\t{s1}\t{e1}\t{name}_L\t{score or '0'}\t+\n")
            ob.write(f"{c2}\t{s2}\t{e2}\t{name}_R\t{score or '0'}\t+\n")

    n = len(loops)
    n_aa, n_ab, n_bb = pair_ab["AA"], pair_ab["AB"], pair_ab["BB"]
    n_known = n_aa + n_ab + n_bb
    # expected pair fractions under independent A/B assignment of two anchors
    if 0 < exp_a < 1:
        exp_aa = exp_a * exp_a
        exp_bb = (1 - exp_a) * (1 - exp_a)
        exp_ab = 2 * exp_a * (1 - exp_a)
    else:
        exp_aa = exp_bb = exp_ab = float("nan")

    rows = [
        ("n_loops", n),
        ("n_cis", n_cis),
        ("n_trans", n_trans),
        ("median_span_bp", int(statistics.median(dists)) if dists else ""),
        ("mean_span_bp", int(statistics.mean(dists)) if dists else ""),
        ("p10_span_bp", int(sorted(dists)[max(0, int(0.1 * len(dists)) - 1)]) if dists else ""),
        ("p90_span_bp", int(sorted(dists)[min(len(dists) - 1, int(0.9 * len(dists)))]) if dists else ""),
        ("n_AA", n_aa),
        ("n_AB", n_ab),
        ("n_BB", n_bb),
        ("n_AB_NA", pair_ab["NA"]),
        ("frac_AA", f"{n_aa / n_known:.4f}" if n_known else ""),
        ("frac_AB", f"{n_ab / n_known:.4f}" if n_known else ""),
        ("frac_BB", f"{n_bb / n_known:.4f}" if n_known else ""),
        ("exp_frac_A_genome", f"{exp_a:.4f}" if exp_a == exp_a else ""),
        ("log2_AA_vs_indep", f"{log2fc(n_aa / n_known, exp_aa):.4f}" if n_known and exp_aa == exp_aa else ""),
        ("log2_AB_vs_indep", f"{log2fc(n_ab / n_known, exp_ab):.4f}" if n_known and exp_ab == exp_ab else ""),
        ("log2_BB_vs_indep", f"{log2fc(n_bb / n_known, exp_bb):.4f}" if n_known and exp_bb == exp_bb else ""),
        ("n_ctcf_peaks", n_ctcf),
        ("n_loops_ctcf_both", ctcf_class["both"]),
        ("n_loops_ctcf_one", ctcf_class["one"]),
        ("n_loops_ctcf_none", ctcf_class["none"]),
        ("frac_loops_ctcf_both", f"{ctcf_class['both'] / n:.4f}" if n else ""),
        ("frac_loops_ctcf_any", f"{(ctcf_class['both'] + ctcf_class['one']) / n:.4f}" if n else ""),
        ("n_anchors_with_ctcf", ctcf_at_anchor),
        ("frac_anchors_with_ctcf", f"{ctcf_at_anchor / (2 * n):.4f}" if n else ""),
    ]
    with open(summary, "w") as o:
        o.write("metric\tvalue\n")
        for k, v in rows:
            o.write(f"{k}\t{v}\n")

    print(f"n_loops={n}  cis={n_cis}  median_span={rows[3][1]}")
    print(f"AB pairs  AA={n_aa} AB={n_ab} BB={n_bb}")
    print(f"CTCF anchors  both={ctcf_class['both']} one={ctcf_class['one']} none={ctcf_class['none']}")
    print(f"wrote {ann}")
    print(f"wrote {anchors}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
