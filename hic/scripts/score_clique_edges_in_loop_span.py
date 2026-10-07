#!/usr/bin/env python3
"""Score cobinding pairs / clique edges as Peakachu loops by SPAN containment.

A pair/edge matches a loop if BOTH peaks lie fully inside that loop's genomic
span [min(anchor starts), max(anchor ends)] (±slop). Prefer 5 kb when both
resolutions hit. Does not overwrite the both-anchors tables.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import compare_union_pairs_and_clique_edges as C
import union_5k_10k_loops_vs_pairs as U


def build_span_index(loops: list[dict]):
    """chrom -> list of (left, right, idx) sorted by left."""
    by: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for i, lp in enumerate(loops):
        a, b = lp["a"], lp["b"]
        if a[0] != b[0]:
            continue
        left = min(a[1], b[1])
        right = max(a[2], b[2])
        by[a[0]].append((left, right, i))
    for c in by:
        by[c].sort()
    return by


def assign_span(pr, fine, ix5, coarse, ix10, slop: int):
    """Return preferred loop (5 kb first) if both peaks sit inside its span."""
    c, p1, p2 = pr["p1"][0], pr["p1"], pr["p2"]
    lo = min(p1[1], p2[1])
    hi = max(p1[2], p2[2])

    def hit(loops, ix):
        best = None
        for left, right, i in ix.get(c, ()):
            # sorted by left: once left-slop exceeds the left peak edge, stop
            if left - slop > lo:
                break
            if right + slop >= hi:
                if best is None or loops[i]["score"] > loops[best]["score"]:
                    best = i
        return best

    i5 = hit(fine, ix5)
    if i5 is not None:
        return fine[i5]
    i10 = hit(coarse, ix10)
    if i10 is not None:
        return coarse[i10]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine", type=Path, required=True)
    ap.add_argument("--coarse", type=Path, required=True)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--cliques", type=Path, required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--slop", type=int, default=10_000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fine = U.load_bedpe(args.fine, "5000")
    coarse = U.load_bedpe(args.coarse, "10000")
    ix5, ix10 = build_span_index(fine), build_span_index(coarse)
    slop = args.slop
    pad = f"pad{slop // 1000}kb"

    pairs = U.load_edges(args.edges)
    cl_edges = C.load_clique_edges(args.cliques)

    pair_header = ("peak1\tpeak2\tchrom1\tstart1\tend1\tchrom2\tstart2\tend2\t"
                   "resolution\tloop_score\n")
    cl_header = ("clique_id\tn_regions\tpeak1\tpeak2\tchrom1\tstart1\tend1\t"
                 "chrom2\tstart2\tend2\tresolution\tloop_score\n")

    n_both = n_5 = n_10 = 0
    pair_path = args.out_dir / f"RBBP4_pairs_in_loop_span.{args.cutoff}.{pad}.tsv"
    with pair_path.open("w") as o:
        o.write(pair_header)
        for pr in pairs:
            lp = assign_span(pr, fine, ix5, coarse, ix10, slop)
            if lp is None:
                continue
            n_both += 1
            if lp["res"] == "5000":
                n_5 += 1
            else:
                n_10 += 1
            o.write(f"{C.pk(pr['p1'])}\t{C.pk(pr['p2'])}\t{C.loop_cols(lp)}\n")

    n_cl = len(cl_edges)
    n_cl_loop = n_cl_5 = n_cl_10 = 0
    cliques_any: set[str] = set()
    cliques_all: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cl_path = args.out_dir / f"RBBP4_clique_edges_in_loop_span.{args.cutoff}.{pad}.tsv"
    cl_all_path = args.out_dir / f"RBBP4_clique_edges_in_loop_span_all.{args.cutoff}.{pad}.tsv"
    with cl_path.open("w") as hit, cl_all_path.open("w") as allf:
        hit.write(cl_header)
        allf.write(cl_header)
        for ed in cl_edges:
            cliques_all[ed["clique_id"]][0] += 1
            lp = assign_span(ed, fine, ix5, coarse, ix10, slop)
            if lp is None:
                allf.write(f"{ed['clique_id']}\t{ed['n']}\t{C.pk(ed['p1'])}\t{C.pk(ed['p2'])}\t"
                           f".\t.\t.\t.\t.\t.\t.\t.\n")
                continue
            n_cl_loop += 1
            cliques_any.add(ed["clique_id"])
            cliques_all[ed["clique_id"]][1] += 1
            if lp["res"] == "5000":
                n_cl_5 += 1
            else:
                n_cl_10 += 1
            row = (f"{ed['clique_id']}\t{ed['n']}\t{C.pk(ed['p1'])}\t{C.pk(ed['p2'])}\t"
                   f"{C.loop_cols(lp)}\n")
            hit.write(row)
            allf.write(row)

    n_cliques = len(cliques_all)
    n_all_loop = sum(1 for n_e, n_l in cliques_all.values() if n_e and n_l == n_e)
    n = len(pairs)
    lines = [
        f"mode=both_peaks_inside_loop_span  cutoff={args.cutoff}  "
        f"5 kb loops={len(fine):,}  10 kb loops={len(coarse):,}  pad=±{slop}",
        "prefer 5 kb when a pair/edge hits both resolutions",
        "",
        "all cobinding pairs",
        f"  n={n:,}",
        f"  both peaks inside same loop span: {n_both:,} ({100 * n_both / n:.1f}%)  "
        f"5 kb={n_5:,}  10 kb={n_10:,}",
        "",
        "clique edges (every pair of regions inside a clique)",
        f"  n_cliques={n_cliques:,}  n_edges={n_cl:,}  (edges counted per clique)",
        f"  edges inside a loop span: {n_cl_loop:,}/{n_cl:,} ({100 * n_cl_loop / n_cl:.1f}%)  "
        f"5 kb={n_cl_5:,}  10 kb={n_cl_10:,}",
        f"  cliques with ≥1 such edge: {len(cliques_any):,}/{n_cliques:,} "
        f"({100 * len(cliques_any) / n_cliques:.1f}%)",
        f"  cliques where every edge is inside a loop span: {n_all_loop:,}/{n_cliques:,}",
        "",
        "contrast (locked both-anchors @ 0.7 ±10 kb): 25/1,098 = 2.3%",
        "",
        f"wrote {pair_path}",
        f"wrote {cl_path}",
        f"wrote {cl_all_path}",
    ]
    text = "\n".join(lines)
    print(text)
    (args.out_dir / f"summary_in_loop_span.{args.cutoff}.{pad}.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
