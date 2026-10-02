#!/usr/bin/env python3
"""Observed ±20 kb loop overlap and same-chromosome permutation p-values.

Null: keep each pair's (or clique's) peak widths and relative spacing, place
the constellation at a random start on the same chromosome. One-sided p =
(1 + n_perm with rate >= observed) / (1 + n_perm).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import compare_union_pairs_and_clique_edges as C
import union_5k_10k_loops_vs_pairs as U

MAIN = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)


def load_chrom_sizes(path: Path) -> dict[str, int]:
    out = {}
    with path.open() as f:
        for ln in f:
            chrom, n = ln.split()[:2]
            if chrom in MAIN:
                out[chrom] = int(n)
    return out


def load_cliques(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            peaks = []
            for tok in p[idx["regions"]].split(";"):
                pk = U.parse_peak(tok)
                if pk:
                    peaks.append(pk)
            peaks.sort()
            rows.append({"id": p[idx["clique_id"]], "n": int(p[idx["n_regions"]]),
                         "peaks": peaks, "chrom": peaks[0][0] if peaks else ""})
    return rows


def live_191(edges_path: Path, cliques: list[dict]) -> set[str]:
    df = pd.read_csv(edges_path, sep="\t")
    df = df[(df["qval"] <= 0.05) & (df["coaccess"] >= 0.0) & (df["n_links"] >= 1)]
    G = nx.Graph()
    for r in df.itertuples(index=False):
        G.add_edge(r.peak1, r.peak2)
    keep = set().union(*(c for c in nx.connected_components(G) if len(c) >= 10))
    live = set()
    for cl in cliques:
        if any(C.pk(p) in keep for p in cl["peaks"]):
            live.add(cl["id"])
    return live


def score_pair(pr, fine, ix5, coarse, ix10, slop: int) -> tuple[bool, bool, str | None, dict | None]:
    one5, both5, i5 = U.pair_hits(pr, fine, ix5, slop, 5000)
    one10, both10, i10 = U.pair_hits(pr, coarse, ix10, slop, 10000)
    one = one5 or one10
    if both5:
        return one, True, "5000", fine[i5]
    if both10:
        return one, True, "10000", coarse[i10]
    return one, False, None, None


def clique_edges(cliques, ids=None) -> list[dict]:
    from itertools import combinations
    rows = []
    for cl in cliques:
        if ids is not None and cl["id"] not in ids:
            continue
        for a, b in combinations(cl["peaks"], 2):
            if a[0] != b[0]:
                continue
            if (a[0], a[1]) > (b[0], b[1]):
                a, b = b, a
            rows.append({"clique_id": cl["id"], "n": cl["n"], "p1": a, "p2": b})
    return rows


def place_pair(p1, p2, new_s1: int):
    d = p2[1] - p1[1]
    w1 = p1[2] - p1[1]
    w2 = p2[2] - p2[1]
    n1 = (p1[0], new_s1, new_s1 + w1)
    n2 = (p2[0], new_s1 + d, new_s1 + d + w2)
    return n1, n2


def pair_max_start(p1, p2, chrom_len: int) -> int:
    span = max(p1[2], p2[2]) - min(p1[1], p2[1])
    return chrom_len - span


def pick_locus(span: int, chrom_len: dict[str, int], rng) -> tuple[str, int]:
    """Uniform random start among all genome positions where a block of `span` fits."""
    chroms, weights = [], []
    for c, L in chrom_len.items():
        w = L - span
        if w > 0:
            chroms.append(c)
            weights.append(w)
    i = int(rng.choice(len(chroms), p=np.array(weights, dtype=float) / sum(weights)))
    c = chroms[i]
    ns = int(rng.integers(0, (chrom_len[c] - span) + 1))
    return c, ns


def permute_pairs(pairs, chrom_len, rng, genome_wide: bool = True):
    out = []
    for pr in pairs:
        left = min(pr["p1"][1], pr["p2"][1])
        span = max(pr["p1"][2], pr["p2"][2]) - left
        if genome_wide:
            c, ns = pick_locus(span, chrom_len, rng)
        else:
            c = pr["p1"][0]
            mx = chrom_len[c] - span
            if mx <= 0:
                out.append(pr)
                continue
            ns = int(rng.integers(0, mx + 1))
        shift = ns - left
        p1 = (c, pr["p1"][1] + shift, pr["p1"][2] + shift)
        p2 = (c, pr["p2"][1] + shift, pr["p2"][2] + shift)
        out.append({"p1": p1, "p2": p2})
    return out


def permute_cliques(cliques, chrom_len, rng, ids=None, genome_wide: bool = True):
    out = []
    for cl in cliques:
        if ids is not None and cl["id"] not in ids:
            continue
        lo = min(p[1] for p in cl["peaks"])
        hi = max(p[2] for p in cl["peaks"])
        span = hi - lo
        if genome_wide:
            c, ns = pick_locus(span, chrom_len, rng)
        else:
            c = cl["chrom"]
            mx = chrom_len.get(c, 0) - span
            if mx <= 0:
                out.append({"id": cl["id"], "n": cl["n"], "peaks": cl["peaks"], "chrom": c})
                continue
            ns = int(rng.integers(0, mx + 1))
        shift = ns - lo
        peaks = [(c, p[1] + shift, p[2] + shift) for p in cl["peaks"]]
        out.append({"id": cl["id"], "n": cl["n"], "peaks": peaks, "chrom": c})
    return out


def tally_pairs(pairs, fine, ix5, coarse, ix10, slop):
    n_one = n_both = n5 = n10 = 0
    hits = []
    for pr in pairs:
        one, both, res, lp = score_pair(pr, fine, ix5, coarse, ix10, slop)
        if one:
            n_one += 1
        if both:
            n_both += 1
            if res == "5000":
                n5 += 1
            else:
                n10 += 1
            hits.append((pr, lp))
    n = len(pairs)
    return {"n": n, "one": n_one, "both": n_both, "n5": n5, "n10": n10,
            "frac_one": n_one / n, "frac_both": n_both / n, "hits": hits}


def tally_clique_edges(edges, fine, ix5, coarse, ix10, slop):
    n_both = n5 = n10 = 0
    any_cl = set()
    hits = []
    for ed in edges:
        _, both, res, lp = score_pair(ed, fine, ix5, coarse, ix10, slop)
        if both:
            n_both += 1
            any_cl.add(ed["clique_id"])
            if res == "5000":
                n5 += 1
            else:
                n10 += 1
            hits.append((ed, lp))
    n = len(edges)
    n_cliq = len({ed["clique_id"] for ed in edges})
    return {"n": n, "n_cliq": n_cliq, "both": n_both, "n5": n5, "n10": n10,
            "frac_both": n_both / n if n else 0,
            "n_cliq_hit": len(any_cl), "hits": hits}


def pval(obs, nulls):
    n = len(nulls)
    ge = sum(1 for x in nulls if x >= obs - 1e-15)
    return (1 + ge) / (1 + n), float(np.mean(nulls)), float(np.percentile(nulls, 95))


def write_pair_hits(path, hits):
    with path.open("w") as o:
        o.write("peak1\tpeak2\tchrom1\tstart1\tend1\tchrom2\tstart2\tend2\t"
                "resolution\tloop_score\n")
        for pr, lp in hits:
            o.write(f"{C.pk(pr['p1'])}\t{C.pk(pr['p2'])}\t{C.loop_cols(lp)}\n")


def write_clique_hits(path, hits):
    with path.open("w") as o:
        o.write("clique_id\tn_regions\tpeak1\tpeak2\tchrom1\tstart1\tend1\t"
                "chrom2\tstart2\tend2\tresolution\tloop_score\n")
        for ed, lp in hits:
            o.write(f"{ed['clique_id']}\t{ed['n']}\t{C.pk(ed['p1'])}\t{C.pk(ed['p2'])}\t"
                    f"{C.loop_cols(lp)}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine", type=Path, required=True)
    ap.add_argument("--coarse", type=Path, required=True)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--cliques", type=Path, required=True)
    ap.add_argument("--chrom-sizes", type=Path, required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--slop", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--genome-wide", action="store_true", default=True,
                    help="place blocks uniformly across chr1–22,X (default)")
    ap.add_argument("--same-chrom", action="store_true",
                    help="restrict relocation to the original chromosome")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chrom_len = load_chrom_sizes(args.chrom_sizes)
    fine = U.load_bedpe(args.fine, "5000")
    coarse = U.load_bedpe(args.coarse, "10000")
    ix5, ix10 = U.index_anchors(fine, 5000), U.index_anchors(coarse, 10000)
    slop = args.slop

    pairs = [pr for pr in U.load_edges(args.edges) if pr["p1"][0] in chrom_len]
    cliques = [cl for cl in load_cliques(args.cliques) if cl["chrom"] in chrom_len]
    live_ids = live_191(args.edges, cliques)
    edges_all = clique_edges(cliques)
    edges_191 = clique_edges(cliques, live_ids)

    obs_pairs = tally_pairs(pairs, fine, ix5, coarse, ix10, slop)
    obs_355 = tally_clique_edges(edges_all, fine, ix5, coarse, ix10, slop)
    obs_191 = tally_clique_edges(edges_191, fine, ix5, coarse, ix10, slop)

    pad = f"pad{slop // 1000}kb"
    write_pair_hits(args.out_dir / f"RBBP4_pairs_as_loops.{args.cutoff}.{pad}.tsv",
                    obs_pairs["hits"])
    write_clique_hits(args.out_dir / f"RBBP4_clique_edges_as_loops.{args.cutoff}.{pad}.tsv",
                      obs_355["hits"])
    write_clique_hits(args.out_dir / f"RBBP4_191cliques_edges_as_loops.{args.cutoff}.{pad}.tsv",
                      obs_191["hits"])

    gw = not args.same_chrom
    tag = "genomewide" if gw else "samechrom"

    rng = np.random.default_rng(args.seed)
    null_both, null_one, null_355, null_191 = [], [], [], []
    for i in range(args.n_perm):
        pp = permute_pairs(pairs, chrom_len, rng, genome_wide=gw)
        tp = tally_pairs(pp, fine, ix5, coarse, ix10, slop)
        null_both.append(tp["frac_both"])
        null_one.append(tp["frac_one"])
        c355 = permute_cliques(cliques, chrom_len, rng, genome_wide=gw)
        t355 = tally_clique_edges(clique_edges(c355), fine, ix5, coarse, ix10, slop)
        null_355.append(t355["frac_both"])
        c191 = permute_cliques(cliques, chrom_len, rng, live_ids, genome_wide=gw)
        t191 = tally_clique_edges(clique_edges(c191, live_ids), fine, ix5, coarse, ix10, slop)
        null_191.append(t191["frac_both"])
        if (i + 1) % 50 == 0:
            print(f"[perm] {i+1}/{args.n_perm}", flush=True)

    rows = [
        ("all_pairs_both_anchors", obs_pairs["frac_both"], obs_pairs["both"], obs_pairs["n"],
         *pval(obs_pairs["frac_both"], null_both), obs_pairs["n5"], obs_pairs["n10"]),
        ("all_pairs_at_least_1_anchor", obs_pairs["frac_one"], obs_pairs["one"], obs_pairs["n"],
         *pval(obs_pairs["frac_one"], null_one), -1, -1),
        ("clique_edges_355_as_loop", obs_355["frac_both"], obs_355["both"], obs_355["n"],
         *pval(obs_355["frac_both"], null_355), obs_355["n5"], obs_355["n10"]),
        ("clique_edges_191_as_loop", obs_191["frac_both"], obs_191["both"], obs_191["n"],
         *pval(obs_191["frac_both"], null_191), obs_191["n5"], obs_191["n10"]),
    ]

    print(f"cutoff={args.cutoff}  pad=±{slop}  n_perm={args.n_perm}  seed={args.seed}  null={tag}")
    print("null: random genomic start (chr1–22,X, length-weighted); keep widths and spacing")
    print(f"{'test':<32}{'obs':>8}{'n':>8}{'N':>8}{'%':>8}{'null_mean':>10}{'null_95':>10}{'p':>10}{'5kb':>8}{'10kb':>8}")
    lines = [
        f"cutoff={args.cutoff} pad=±{slop} n_perm={args.n_perm} seed={args.seed} null={tag}",
        "null: uniform random start on chr1–22,X (weighted by usable chrom length); "
        "keep peak widths and relative spacing; chromosome can change",
        "p = (1 + count(null >= obs)) / (1 + n_perm)",
        "",
        f"{'test':<32}{'obs':>8}{'n':>8}{'N':>8}{'pct':>8}{'null_mean':>10}{'null_95':>10}{'p':>10}{'5kb':>8}{'10kb':>8}",
    ]
    for name, frac, k, n, p, mu, p95, n5, n10 in rows:
        extra = f"{n5:>8}{n10:>8}" if n5 >= 0 else f"{'':>8}{'':>8}"
        row = (f"{name:<32}{k:>8}{n:>8}{100*frac:>7.1f}%{100*mu:>9.1f}%{100*p95:>9.1f}%"
               f"{p:>10.4g}{extra}")
        print(row)
        lines.append(row)
    lines.append("")
    lines.append(f"191 cliques with ≥1 loop edge: {obs_191['n_cliq_hit']}/{obs_191['n_cliq']}")
    lines.append(f"355 cliques with ≥1 loop edge: {obs_355['n_cliq_hit']}/{obs_355['n_cliq']}")
    text = "\n".join(lines)
    (args.out_dir / f"overlap_{pad}_perm.{args.cutoff}.{tag}.txt").write_text(text + "\n")
    tsv = args.out_dir / f"overlap_{pad}_perm.{args.cutoff}.{tag}.tsv"
    with tsv.open("w") as o:
        o.write("test\tn_hit\tn_total\tpercent\tobs_frac\tnull_mean\tnull_95\tp_value\tn_5kb\tn_10kb\n")
        for name, frac, k, n, p, mu, p95, n5, n10 in rows:
            o.write(f"{name}\t{k}\t{n}\t{100*frac:.4f}\t{frac:.6g}\t{mu:.6g}\t{p95:.6g}\t{p:.6g}\t"
                    f"{'' if n5<0 else n5}\t{'' if n10<0 else n10}\n")
    print(f"wrote {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
