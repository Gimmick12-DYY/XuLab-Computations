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


def _same_loop(h1: dict, h2: dict) -> tuple[bool, int | None]:
    best = None
    for idx, e1 in h1.items():
        e2 = h2.get(idx)
        if not e2:
            continue
        if ("a" in e1 and "b" in e2) or ("b" in e1 and "a" in e2):
            if best is None or idx < best:
                best = idx
    return best is not None, best


def classify(pr, fine, ix5, coarse, ix10, slop: int):
    """one anchor, both anchors of one loop, both peaks on some anchor, loop."""
    c, p1, p2 = pr["p1"][0], pr["p1"], pr["p2"]
    h1_5 = U.loops_on_peak(ix5, fine, c, p1[1], p1[2], slop, 5000)
    h2_5 = U.loops_on_peak(ix5, fine, c, p2[1], p2[2], slop, 5000)
    h1_10 = U.loops_on_peak(ix10, coarse, c, p1[1], p1[2], slop, 10000)
    h2_10 = U.loops_on_peak(ix10, coarse, c, p2[1], p2[2], slop, 10000)
    one = bool(h1_5 or h2_5 or h1_10 or h2_10)
    hub = bool(h1_5 or h1_10) and bool(h2_5 or h2_10)
    both5, i5 = _same_loop(h1_5, h2_5)
    both10, i10 = _same_loop(h1_10, h2_10)
    if both5:
        return one, True, hub, "5000", fine[i5]
    if both10:
        return one, True, hub, "10000", coarse[i10]
    return one, False, hub, None, None


def build_span_index(fine, coarse):
    tmp: dict = defaultdict(list)
    for lp in list(fine) + list(coarse):
        a, b = lp["a"], lp["b"]
        if a[0] != b[0]:
            continue
        tmp[a[0]].append((min(a[1], b[1]), max(a[2], b[2])))
    index = {}
    for chrom, rows in tmp.items():
        rows.sort()
        starts = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        ends = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
        index[chrom] = (starts, ends)
    return index


def loops_inside(index, chrom: str, lo: int, hi: int) -> int:
    rec = index.get(chrom)
    if rec is None:
        return 0
    starts, ends = rec
    i = int(np.searchsorted(starts, lo, side="left"))
    j = int(np.searchsorted(starts, hi, side="right"))
    if j <= i:
        return 0
    return int(np.count_nonzero(ends[i:j] <= hi))


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


def tally_pairs(pairs, fine, ix5, coarse, ix10, slop, store_hits: bool = True):
    n_one = n_both = n_hub = n5 = n10 = 0
    hits = []
    for pr in pairs:
        one, both, hub, res, lp = classify(pr, fine, ix5, coarse, ix10, slop)
        if one:
            n_one += 1
        if hub:
            n_hub += 1
        if both:
            n_both += 1
            if res == "5000":
                n5 += 1
            else:
                n10 += 1
            if store_hits:
                hits.append((pr, lp))
    n = len(pairs)
    return {"n": n, "one": n_one, "both": n_both, "hub": n_hub, "n5": n5, "n10": n10,
            "frac_one": n_one / n, "frac_both": n_both / n, "frac_hub": n_hub / n,
            "hits": hits}


def tally_clique_edges(edges, fine, ix5, coarse, ix10, slop, store_hits: bool = True):
    n_both = n5 = n10 = 0
    any_cl = set()
    hits = []
    for ed in edges:
        _, both, _, res, lp = classify(ed, fine, ix5, coarse, ix10, slop)
        if both:
            n_both += 1
            any_cl.add(ed["clique_id"])
            if res == "5000":
                n5 += 1
            else:
                n10 += 1
            if store_hits:
                hits.append((ed, lp))
    n = len(edges)
    n_cliq = len({ed["clique_id"] for ed in edges})
    return {"n": n, "n_cliq": n_cliq, "both": n_both, "n5": n5, "n10": n10,
            "frac_both": n_both / n if n else 0,
            "n_cliq_hit": len(any_cl), "hits": hits}


def tally_span(cliques, index, slop: int):
    n0 = g1 = g2 = g3 = 0
    for cl in cliques:
        lo = min(p[1] for p in cl["peaks"]) - slop
        hi = max(p[2] for p in cl["peaks"]) + slop
        k = loops_inside(index, cl["chrom"], lo, hi)
        n0 += k == 0
        g1 += k >= 1
        g2 += k >= 2
        g3 += k >= 3
    n = len(cliques)
    return {"n": n, "n0": n0, "g1": g1, "g2": g2, "g3": g3,
            "f0": n0 / n, "f1": g1 / n, "f2": g2 / n, "f3": g3 / n}


def pval(obs, nulls, lower: bool = False):
    nulls = np.asarray(nulls, dtype=float)
    n = int(nulls.size)
    if lower:
        k = int(np.count_nonzero(nulls <= obs + 1e-15))
    else:
        k = int(np.count_nonzero(nulls >= obs - 1e-15))
    return (1 + k) / (1 + n), float(np.mean(nulls)), float(np.percentile(nulls, 95))


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


def write_report(out_dir, pad, cutoff, seed, tag, null, meta) -> int:
    null = np.asarray(null, dtype=float)
    n_perm = int(null.shape[0])
    names = [str(x) for x in meta["names"]]
    tails = [str(x) for x in meta["tails"]]
    print(f"cutoff={cutoff}  pad={pad}  n_perm={n_perm}  seed={seed}  null={tag}")
    lines = [
        f"cutoff={cutoff} pad={pad} n_perm={n_perm} seed={seed} null={tag}",
        "null: uniform random start on chr1–22,X (weighted by usable chrom length); "
        "keep peak widths and relative spacing; chromosome can change",
        "upper tail: p = (1 + count(null >= obs)) / (1 + n_perm)",
        "lower tail: p = (1 + count(null <= obs)) / (1 + n_perm)   [clique_span_0_loops]",
        "",
        f"{'test':<36}{'obs':>8}{'n':>8}{'pct':>8}{'null_mean':>10}{'null_95':>10}"
        f"{'p':>12}{'tail':>8}{'5kb':>8}{'10kb':>8}",
    ]
    records = []
    for i, name in enumerate(names):
        k = int(meta["obs_k"][i])
        n = int(meta["obs_n"][i])
        frac = k / n
        lower = tails[i] == "lower"
        p, mu, p95 = pval(frac, null[:, i], lower=lower)
        n5, n10 = int(meta["n5"][i]), int(meta["n10"][i])
        extra = f"{n5:>8}{n10:>8}" if n5 >= 0 else f"{'':>8}{'':>8}"
        lines.append(f"{name:<36}{k:>8}{n:>8}{100*frac:>7.1f}%{100*mu:>9.1f}%"
                     f"{100*p95:>9.1f}%{p:>12.6g}{tails[i]:>8}{extra}")
        records.append((name, k, n, frac, mu, p95, p, tails[i], n5, n10))
    note355 = meta["note_355"]
    note191 = meta["note_191"]
    lines.append("")
    lines.append(f"191 cliques with ≥1 loop edge: {int(note191[0])}/{int(note191[1])}")
    lines.append(f"355 cliques with ≥1 loop edge: {int(note355[0])}/{int(note355[1])}")
    text = "\n".join(lines)
    print(text)
    (out_dir / f"overlap_{pad}_perm.{cutoff}.{tag}.txt").write_text(text + "\n")
    tsv = out_dir / f"overlap_{pad}_perm.{cutoff}.{tag}.tsv"
    with tsv.open("w") as o:
        o.write("test\tn_hit\tn_total\tpercent\tobs_frac\tnull_mean\tnull_95\t"
                "p_value\ttail\tn_5kb\tn_10kb\n")
        for name, k, n, frac, mu, p95, p, tail, n5, n10 in records:
            o.write(f"{name}\t{k}\t{n}\t{100*frac:.4f}\t{frac:.6g}\t{mu:.6g}\t{p95:.6g}\t"
                    f"{p:.6g}\t{tail}\t{'' if n5<0 else n5}\t{'' if n10<0 else n10}\n")
    print(f"wrote {tsv}")
    return 0


def merge_shards(args) -> int:
    slop = args.slop
    pad = f"pad{slop // 1000}kb"
    tag = "samechrom" if args.same_chrom else "genomewide"
    paths = sorted(args.out_dir.glob(f"null_{pad}_perm.{args.cutoff}.{tag}.shard*.npz"))
    if len(paths) != args.n_shards:
        raise SystemExit(f"expected {args.n_shards} shard files, found {len(paths)} in {args.out_dir}")
    parts = [np.load(p) for p in paths]
    shards = sorted(int(p["shard"]) for p in parts)
    if shards != list(range(args.n_shards)):
        raise SystemExit(f"shard ids {shards} do not cover 0..{args.n_shards - 1}")
    parts.sort(key=lambda p: int(p["shard"]))
    null = np.concatenate([p["null"] for p in parts], axis=0)
    meta = {k: parts[0][k] for k in
            ("names", "tails", "obs_k", "obs_n", "n5", "n10", "note_191", "note_355")}
    return write_report(args.out_dir, pad, args.cutoff, args.seed, tag, null, meta)


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
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard of the permutation stream (0-based)")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="combine shard npz files already in --out-dir into the report")
    ap.add_argument("--genome-wide", action="store_true", default=True,
                    help="place blocks uniformly across chr1–22,X (default)")
    ap.add_argument("--same-chrom", action="store_true",
                    help="restrict relocation to the original chromosome")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.merge:
        return merge_shards(args)
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

    span_ix = build_span_index(fine, coarse)
    obs_span = tally_span(cliques, span_ix, slop)

    pad = f"pad{slop // 1000}kb"
    gw = not args.same_chrom
    tag = "genomewide" if gw else "samechrom"
    if args.n_shards <= 1 or args.shard == 0:
        write_pair_hits(args.out_dir / f"RBBP4_pairs_as_loops.{args.cutoff}.{pad}.tsv",
                        obs_pairs["hits"])
        write_clique_hits(args.out_dir / f"RBBP4_clique_edges_as_loops.{args.cutoff}.{pad}.tsv",
                          obs_355["hits"])
        write_clique_hits(args.out_dir / f"RBBP4_191cliques_edges_as_loops.{args.cutoff}.{pad}.tsv",
                          obs_191["hits"])

    # Independent stream per shard so array tasks do not repeat shuffles.
    rng = np.random.default_rng(args.seed + args.shard * 1_000_003)
    n_perm = args.n_perm
    null = np.empty((n_perm, 9), dtype=np.float64)
    step = 500 if n_perm >= 2000 else 50
    for i in range(n_perm):
        pp = permute_pairs(pairs, chrom_len, rng, genome_wide=gw)
        tp = tally_pairs(pp, fine, ix5, coarse, ix10, slop, store_hits=False)
        c355 = permute_cliques(cliques, chrom_len, rng, genome_wide=gw)
        t355 = tally_clique_edges(clique_edges(c355), fine, ix5, coarse, ix10, slop,
                                  store_hits=False)
        c191 = permute_cliques(cliques, chrom_len, rng, live_ids, genome_wide=gw)
        t191 = tally_clique_edges(clique_edges(c191, live_ids), fine, ix5, coarse, ix10, slop,
                                  store_hits=False)
        ts = tally_span(c355, span_ix, slop)
        null[i] = (tp["frac_both"], tp["frac_one"], tp["frac_hub"],
                   t355["frac_both"], t191["frac_both"],
                   ts["f0"], ts["f1"], ts["f2"], ts["f3"])
        if (i + 1) % step == 0:
            print(f"[perm] shard {args.shard}/{args.n_shards}  {i+1}/{n_perm}", flush=True)

    meta = {
        "names": np.array([
            "all_pairs_both_anchors",
            "all_pairs_at_least_1_anchor",
            "all_pairs_both_peaks_any_anchor",
            "clique_edges_355_as_loop",
            "clique_edges_191_as_loop",
            "clique_span_0_loops",
            "clique_span_ge1",
            "clique_span_ge2",
            "clique_span_ge3",
        ]),
        "tails": np.array(["upper", "upper", "upper", "upper", "upper",
                           "lower", "upper", "upper", "upper"]),
        "obs_k": np.array([
            obs_pairs["both"], obs_pairs["one"], obs_pairs["hub"],
            obs_355["both"], obs_191["both"],
            obs_span["n0"], obs_span["g1"], obs_span["g2"], obs_span["g3"],
        ], dtype=np.int64),
        "obs_n": np.array([
            obs_pairs["n"], obs_pairs["n"], obs_pairs["n"],
            obs_355["n"], obs_191["n"],
            obs_span["n"], obs_span["n"], obs_span["n"], obs_span["n"],
        ], dtype=np.int64),
        "n5": np.array([obs_pairs["n5"], -1, -1, obs_355["n5"], obs_191["n5"],
                        -1, -1, -1, -1], dtype=np.int64),
        "n10": np.array([obs_pairs["n10"], -1, -1, obs_355["n10"], obs_191["n10"],
                         -1, -1, -1, -1], dtype=np.int64),
        "note_191": np.array([obs_191["n_cliq_hit"], obs_191["n_cliq"]], dtype=np.int64),
        "note_355": np.array([obs_355["n_cliq_hit"], obs_355["n_cliq"]], dtype=np.int64),
    }
    if args.n_shards > 1:
        shard_path = args.out_dir / (
            f"null_{pad}_perm.{args.cutoff}.{tag}.shard{args.shard:02d}.npz")
        np.savez(shard_path, null=null, **meta,
                 seed=args.seed, shard=args.shard, n_shards=args.n_shards)
        print(f"wrote {shard_path}")
        return 0
    return write_report(args.out_dir, pad, args.cutoff, args.seed, tag, null, meta)


if __name__ == "__main__":
    raise SystemExit(main())
