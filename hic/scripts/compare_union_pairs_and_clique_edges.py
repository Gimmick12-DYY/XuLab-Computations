#!/usr/bin/env python3
"""Score cobinding pairs and clique edges as Peakachu loops (5 kb ∪ 10 kb).

A pair/edge sits as a loop if the two regions each overlap a different
anchor of the same loop (±slop). If both resolutions hit, keep 5 kb.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import union_5k_10k_loops_vs_pairs as U


def load_clique_edges(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            cid = p[idx["clique_id"]]
            peaks = []
            for tok in p[idx["regions"]].split(";"):
                pk = U.parse_peak(tok)
                if pk:
                    peaks.append(pk)
            peaks.sort()
            for a, b in combinations(peaks, 2):
                if a[0] != b[0]:
                    continue
                if (a[0], a[1]) > (b[0], b[1]):
                    a, b = b, a
                rows.append({"clique_id": cid, "n": len(peaks), "p1": a, "p2": b})
    return rows


def assign_loop(pr, fine, ix5, coarse, ix10, slop: int):
    _, both5, i5 = U.pair_hits(pr, fine, ix5, slop, 5000)
    if both5:
        return fine[i5]
    _, both10, i10 = U.pair_hits(pr, coarse, ix10, slop, 10000)
    if both10:
        return coarse[i10]
    return None


def hit_one(pr, fine, ix5, coarse, ix10, slop: int) -> bool:
    one5, _, _ = U.pair_hits(pr, fine, ix5, slop, 5000)
    one10, _, _ = U.pair_hits(pr, coarse, ix10, slop, 10000)
    return one5 or one10


def pk(p) -> str:
    return f"{p[0]}:{p[1]}-{p[2]}"


def loop_cols(lp) -> str:
    a, b = lp["a"], lp["b"]
    return (f"{a[0]}\t{a[1]}\t{a[2]}\t{b[0]}\t{b[1]}\t{b[2]}\t"
            f"{lp['res']}\t{lp['score']:.6g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine", type=Path, required=True)
    ap.add_argument("--coarse", type=Path, required=True)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--cliques", type=Path, required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--slop", type=int, default=U.SLOP)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fine = U.load_bedpe(args.fine, "5000")
    coarse = U.load_bedpe(args.coarse, "10000")
    ix5, ix10 = U.index_anchors(fine, 5000), U.index_anchors(coarse, 10000)
    slop = args.slop

    pairs = U.load_edges(args.edges)
    cl_edges = load_clique_edges(args.cliques)

    pair_header = ("peak1\tpeak2\tchrom1\tstart1\tend1\tchrom2\tstart2\tend2\t"
                   "resolution\tloop_score\n")
    cl_header = ("clique_id\tn_regions\tpeak1\tpeak2\tchrom1\tstart1\tend1\t"
                 "chrom2\tstart2\tend2\tresolution\tloop_score\n")

    n_both = n_one = n_5 = n_10 = 0
    pair_path = args.out_dir / f"RBBP4_pairs_as_loops.{args.cutoff}.tsv"
    with pair_path.open("w") as o:
        o.write(pair_header)
        for pr in pairs:
            if hit_one(pr, fine, ix5, coarse, ix10, slop):
                n_one += 1
            lp = assign_loop(pr, fine, ix5, coarse, ix10, slop)
            if lp is None:
                continue
            n_both += 1
            if lp["res"] == "5000":
                n_5 += 1
            else:
                n_10 += 1
            o.write(f"{pk(pr['p1'])}\t{pk(pr['p2'])}\t{loop_cols(lp)}\n")

    n_cl = len(cl_edges)
    n_cl_loop = n_cl_5 = n_cl_10 = 0
    cliques_any = set()
    cliques_all = defaultdict(lambda: [0, 0])  # n_edges, n_loop
    cl_path = args.out_dir / f"RBBP4_clique_edges_as_loops.{args.cutoff}.tsv"
    cl_all_path = args.out_dir / f"RBBP4_clique_edges_all.{args.cutoff}.tsv"
    with cl_path.open("w") as hit, cl_all_path.open("w") as allf:
        hit.write(cl_header)
        allf.write(cl_header)
        for ed in cl_edges:
            cliques_all[ed["clique_id"]][0] += 1
            lp = assign_loop(ed, fine, ix5, coarse, ix10, slop)
            if lp is None:
                allf.write(f"{ed['clique_id']}\t{ed['n']}\t{pk(ed['p1'])}\t{pk(ed['p2'])}\t"
                           f".\t.\t.\t.\t.\t.\t.\t.\n")
                continue
            n_cl_loop += 1
            cliques_any.add(ed["clique_id"])
            cliques_all[ed["clique_id"]][1] += 1
            if lp["res"] == "5000":
                n_cl_5 += 1
            else:
                n_cl_10 += 1
            row = (f"{ed['clique_id']}\t{ed['n']}\t{pk(ed['p1'])}\t{pk(ed['p2'])}\t"
                   f"{loop_cols(lp)}\n")
            hit.write(row)
            allf.write(row)

    n_cliques = len(cliques_all)
    n_all_loop = sum(1 for n_e, n_l in cliques_all.values() if n_e and n_l == n_e)
    n = len(pairs)
    lines = [
        f"cutoff={args.cutoff}  5 kb loops={len(fine):,}  10 kb loops={len(coarse):,}  pad=±{slop}",
        f"prefer 5 kb when a pair/edge hits both resolutions",
        "",
        "all cobinding pairs",
        f"  n={n:,}",
        f"  both anchors of same loop: {n_both:,} ({100*n_both/n:.1f}%)  5 kb={n_5:,}  10 kb={n_10:,}",
        f"  at least 1 anchor: {n_one:,} ({100*n_one/n:.1f}%)",
        "",
        "clique edges (every pair of regions inside a clique)",
        f"  n_cliques={n_cliques:,}  n_edges={n_cl:,}  (edges counted per clique)",
        f"  edges that sit as a loop: {n_cl_loop:,}/{n_cl:,} ({100*n_cl_loop/n_cl:.1f}%)  "
        f"5 kb={n_cl_5:,}  10 kb={n_cl_10:,}",
        f"  cliques with ≥1 loop edge: {len(cliques_any):,}/{n_cliques:,} ({100*len(cliques_any)/n_cliques:.1f}%)",
        f"  cliques where every edge is a loop: {n_all_loop:,}/{n_cliques:,}",
        "",
        f"wrote {pair_path}",
        f"wrote {cl_path}",
        f"wrote {cl_all_path}",
    ]
    text = "\n".join(lines)
    print(text)
    (args.out_dir / f"summary_pairs_and_clique_edges.{args.cutoff}.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
