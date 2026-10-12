#!/usr/bin/env python3
"""Slide-table numbers at the locked ±30 kb / score≥0.5 loop definition.

For Dr. Wang's pairs and the k=50 new run:

* recompute after dropping pairs whose midpoint span is ≤25 kb
* same-chromosome permutation null for the unfiltered rates
  (median, standard deviation, one-sided p)

Pairs are shifted independently. A clique is shifted as one block so its
internal spacing stays fixed.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

XULAB = Path("/work/users/d/y/dyy12/XuLab")
sys.path.insert(0, str(XULAB / "hic" / "scripts"))

import compare_union_pairs_and_clique_edges as C  # noqa: E402
import union_5k_10k_loops_vs_pairs as U  # noqa: E402

FINE = XULAB / "hic/work/loops/peakachu_5000.loops.0.5.bedpe"
COARSE = XULAB / "hic/work/loops/peakachu_10000.loops.0.5.bedpe"
SLOP = 30_000
MIN_SPAN = 25_000

HG38 = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895,
}

SETS = {
    "wang": {
        "edges": XULAB / "cobinding/work/RBBP4.peak_edges.tsv",
        "cliques": XULAB / "cobinding/results/RBBP4/cliques.tsv",
    },
    "new": {
        "edges": XULAB / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv",
        "cliques": XULAB / "cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv",
    },
}


def span_bp(pr: dict) -> int:
    a, b = pr["p1"], pr["p2"]
    return abs(((a[1] + a[2]) // 2) - ((b[1] + b[2]) // 2))


def shift_pair(pr: dict, chrom_len: dict, rng: np.random.Generator) -> dict:
    left = min(pr["p1"][1], pr["p2"][1])
    span = max(pr["p1"][2], pr["p2"][2]) - left
    c = pr["p1"][0]
    mx = chrom_len.get(c, 0) - span
    if mx <= 0:
        return pr
    ns = int(rng.integers(0, mx + 1))
    shift = ns - left
    return {
        "p1": (c, pr["p1"][1] + shift, pr["p1"][2] + shift),
        "p2": (c, pr["p2"][1] + shift, pr["p2"][2] + shift),
    }


def permute_pairs(pairs, chrom_len, rng) -> list[dict]:
    return [shift_pair(pr, chrom_len, rng) for pr in pairs]


def permute_clique_edges(edges, chrom_len, rng) -> list[dict]:
    by = defaultdict(list)
    order = []
    for e in edges:
        if e["clique_id"] not in by:
            order.append(e["clique_id"])
        by[e["clique_id"]].append(e)
    out = []
    for cid in order:
        group = by[cid]
        peaks = [e["p1"] for e in group] + [e["p2"] for e in group]
        lo = min(p[1] for p in peaks)
        hi = max(p[2] for p in peaks)
        c = group[0]["p1"][0]
        mx = chrom_len.get(c, 0) - (hi - lo)
        if mx <= 0:
            out.extend(group)
            continue
        ns = int(rng.integers(0, mx + 1))
        shift = ns - lo
        for e in group:
            out.append({
                "clique_id": cid,
                "p1": (c, e["p1"][1] + shift, e["p1"][2] + shift),
                "p2": (c, e["p2"][1] + shift, e["p2"][2] + shift),
            })
    return out


def score_pairs(pairs, fine, coarse, ix5, ix10) -> tuple[int, int, int]:
    n_both = n5 = n10 = 0
    for pr in pairs:
        lp = C.assign_loop(pr, fine, ix5, coarse, ix10, SLOP)
        if lp is None:
            continue
        n_both += 1
        if lp["res"] == "5000":
            n5 += 1
        else:
            n10 += 1
    return n_both, n5, n10


def score_cliques(edges, fine, coarse, ix5, ix10) -> dict:
    by = defaultdict(list)
    for e in edges:
        by[e["clique_id"]].append(e)
    n_loop_edges = n5 = n10 = 0
    n_ge1 = n_ge2 = n_all = 0
    hist = {}
    for group in by.values():
        k = 0
        for e in group:
            lp = C.assign_loop(e, fine, ix5, coarse, ix10, SLOP)
            if lp is None:
                continue
            k += 1
            n_loop_edges += 1
            if lp["res"] == "5000":
                n5 += 1
            else:
                n10 += 1
        hist[k] = hist.get(k, 0) + 1
        if k >= 1:
            n_ge1 += 1
        if k >= 2:
            n_ge2 += 1
        if k == len(group) and k > 0:
            n_all += 1
    return {
        "n_cliques": len(by),
        "n_edges": len(edges),
        "n_loop_edges": n_loop_edges,
        "n5": n5,
        "n10": n10,
        "n_ge1": n_ge1,
        "n_ge2": n_ge2,
        "n_all": n_all,
        "hist": hist,
    }


def rates(pairs, edges, fine, coarse, ix5, ix10) -> dict:
    nb, n5, n10 = score_pairs(pairs, fine, coarse, ix5, ix10)
    cl = score_cliques(edges, fine, coarse, ix5, ix10)
    return {
        "pair_rate": nb / len(pairs) if pairs else float("nan"),
        "pair_n": nb,
        "pair_d": len(pairs),
        "pair_5": n5,
        "pair_10": n10,
        "edge_rate": cl["n_loop_edges"] / cl["n_edges"] if cl["n_edges"] else float("nan"),
        "edge_n": cl["n_loop_edges"],
        "edge_d": cl["n_edges"],
        "edge_5": cl["n5"],
        "edge_10": cl["n10"],
        "ge1_rate": cl["n_ge1"] / cl["n_cliques"] if cl["n_cliques"] else float("nan"),
        "ge1_n": cl["n_ge1"],
        "ge1_d": cl["n_cliques"],
        "ge2_rate": cl["n_ge2"] / cl["n_cliques"] if cl["n_cliques"] else float("nan"),
        "ge2_n": cl["n_ge2"],
        "ge2_d": cl["n_cliques"],
        "all_rate": cl["n_all"] / cl["n_cliques"] if cl["n_cliques"] else float("nan"),
        "all_n": cl["n_all"],
        "all_d": cl["n_cliques"],
        "hist": cl["hist"],
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt_count(n: int, d: int, rate: float) -> str:
    return f"{n:,} / {d:,} ({pct(rate)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", action="append", default=None,
                    help="Repeat with --edges and --cliques to score a set other than the RBBP4 defaults")
    ap.add_argument("--edges", type=Path, action="append", default=None)
    ap.add_argument("--cliques", type=Path, action="append", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.label or args.edges or args.cliques:
        if not (args.label and args.edges and args.cliques):
            raise SystemExit("--label, --edges, and --cliques must be given together")
        if not (len(args.label) == len(args.edges) == len(args.cliques)):
            raise SystemExit("--label, --edges, and --cliques must be repeated the same number of times")
        sets = {
            lab: {"edges": ed, "cliques": cl}
            for lab, ed, cl in zip(args.label, args.edges, args.cliques)
        }
    else:
        sets = SETS

    out_dir = args.out_dir or (XULAB / "hic/work/loops/union_5k_10k_0.5/slide_table_pad30kb")
    out_dir.mkdir(parents=True, exist_ok=True)
    fine = U.load_bedpe(FINE, "5000")
    coarse = U.load_bedpe(COARSE, "10000")
    ix5 = U.index_anchors(fine, 5000)
    ix10 = U.index_anchors(coarse, 10000)
    print(f"loops 5kb={len(fine):,} 10kb={len(coarse):,}", flush=True)

    loaded = {}
    span_lines = []
    for name, spec in sets.items():
        pairs = U.load_edges(spec["edges"])
        edges = C.load_clique_edges(spec["cliques"])
        loaded[name] = (pairs, edges)
        long_pairs = [p for p in pairs if span_bp(p) > MIN_SPAN]
        long_edges = [e for e in edges if span_bp(e) > MIN_SPAN]
        base = rates(pairs, edges, fine, coarse, ix5, ix10)
        long = rates(long_pairs, long_edges, fine, coarse, ix5, ix10)
        print(f"\n== {name} unfiltered ==", flush=True)
        print(f"pairs {fmt_count(base['pair_n'], base['pair_d'], base['pair_rate'])} "
              f"5kb={base['pair_5']:,} 10kb={base['pair_10']:,}", flush=True)
        print(f"clique edges {base['edge_d']:,}", flush=True)
        print(f"clique-edge loops {fmt_count(base['edge_n'], base['edge_d'], base['edge_rate'])} "
              f"5kb={base['edge_5']:,} 10kb={base['edge_10']:,}", flush=True)
        print(f">=1 {fmt_count(base['ge1_n'], base['ge1_d'], base['ge1_rate'])}", flush=True)
        print(f">=2 {fmt_count(base['ge2_n'], base['ge2_d'], base['ge2_rate'])}", flush=True)
        print(f"all {fmt_count(base['all_n'], base['all_d'], base['all_rate'])} hist={base['hist']}", flush=True)
        print(f"\n== {name} span>{MIN_SPAN} ==", flush=True)
        print(f"pairs {fmt_count(long['pair_n'], long['pair_d'], long['pair_rate'])} "
              f"5kb={long['pair_5']:,} 10kb={long['pair_10']:,}", flush=True)
        print(f"clique edges {long['edge_d']:,}", flush=True)
        print(f"clique-edge loops {fmt_count(long['edge_n'], long['edge_d'], long['edge_rate'])} "
              f"5kb={long['edge_5']:,} 10kb={long['edge_10']:,}", flush=True)
        print(f">=1 {fmt_count(long['ge1_n'], long['ge1_d'], long['ge1_rate'])}", flush=True)
        print(f">=2 {fmt_count(long['ge2_n'], long['ge2_d'], long['ge2_rate'])}", flush=True)
        print(f"all {fmt_count(long['all_n'], long['all_d'], long['all_rate'])} hist={long['hist']}", flush=True)
        span_lines.append(
            "\t".join([
                name,
                fmt_count(long["pair_n"], long["pair_d"], long["pair_rate"]),
                f"{long['pair_5']:,} at 5 kb; {long['pair_10']:,} at 10 kb",
                f"{long['edge_d']:,}",
                fmt_count(long["edge_n"], long["edge_d"], long["edge_rate"]),
                f"{long['edge_5']:,} at 5 kb; {long['edge_10']:,} at 10 kb",
                fmt_count(long["ge1_n"], long["ge1_d"], long["ge1_rate"]),
                fmt_count(long["ge2_n"], long["ge2_d"], long["ge2_rate"]),
                fmt_count(long["all_n"], long["all_d"], long["all_rate"]),
            ])
        )
    (out_dir / "span_gt_25kb.tsv").write_text("\n".join(span_lines) + "\n")

    if args.n_perm <= 0:
        return 0

    keys = ["pair_rate", "edge_rate", "ge1_rate", "ge2_rate", "all_rate"]
    report = []
    for name, (pairs, edges) in loaded.items():
        rng = np.random.default_rng(args.seed)
        obs = rates(pairs, edges, fine, coarse, ix5, ix10)
        nulls = {k: [] for k in keys}
        for i in range(args.n_perm):
            shuf_pairs = permute_pairs(pairs, HG38, rng)
            shuf_edges = permute_clique_edges(edges, HG38, rng)
            r = rates(shuf_pairs, shuf_edges, fine, coarse, ix5, ix10)
            for k in keys:
                nulls[k].append(r[k])
            if (i + 1) % 25 == 0:
                print(f"  {name} perm {i + 1}/{args.n_perm}", flush=True)
        print(f"\n== {name} permutation n={args.n_perm} seed={args.seed} ==", flush=True)
        for k in keys:
            arr = np.asarray(nulls[k], dtype=float)
            n_ge = int(np.sum(arr >= obs[k] - 1e-15))
            p = (1 + n_ge) / (1 + args.n_perm)
            med = float(np.median(arr))
            sd = float(np.std(arr, ddof=1))
            line = (f"{name}\t{k}\tobs={obs[k]:.6f}\tnull_median={med:.6f}\t"
                    f"null_std={sd:.6f}\tp={p:.6g}\tn_ge={n_ge}")
            print(line, flush=True)
            report.append(line)
    (out_dir / "permutation.tsv").write_text("\n".join(report) + "\n")
    print("wrote permutation.tsv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
