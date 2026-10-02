#!/usr/bin/env python3
"""Compare a nested Peakachu 10 kb loop set to gold RBBP4 Cicero cobinding pairs.

Does not modify cobinding/results/RBBP4 or anything under /vast/som/xujie_lab.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

BIN = 10_000
LAYER_RANK = {"0.95": 0, "0.9": 1, "0.75": 2}


def parse_peak(tok: str) -> tuple[str, int, int] | None:
    if ":" not in tok or "-" not in tok:
        return None
    chrom, se = tok.split(":", 1)
    s, e = se.split("-", 1)
    try:
        return chrom, int(s), int(e)
    except ValueError:
        return None


def load_loops(path: Path, layer: str) -> list[dict]:
    out = []
    with path.open() as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#") or ln.startswith("track"):
                continue
            x = ln.split()
            c1, s1, e1 = x[0], int(x[1]), int(x[2])
            c2, s2, e2 = x[3], int(x[4]), int(x[5])
            score = float(x[6]) if len(x) > 6 else 0.0
            a = (c1, s1, e1)
            b = (c2, s2, e2)
            if (c1, s1) > (c2, s2):
                a, b = b, a
            span = abs(((b[1] + b[2]) // 2) - ((a[1] + a[2]) // 2)) if a[0] == b[0] else None
            out.append({"key": frozenset((a, b)), "a": a, "b": b, "score": score,
                        "span": span, "layer": layer})
    return out


def nested_union(core, mid, loose) -> list[dict]:
    seen: set[frozenset] = set()
    stacked = []
    for layer, rows in (("0.95", core), ("0.9", mid), ("0.75", loose)):
        for r in rows:
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            stacked.append({**r, "layer": layer})
    return stacked


def load_edges(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            p1 = parse_peak(p[idx["peak1"]])
            p2 = parse_peak(p[idx["peak2"]])
            if not p1 or not p2:
                continue
            q = float(p[idx["qval"]]) if "qval" in idx and p[idx["qval"]] else 1.0
            ca = float(p[idx["coaccess"]]) if "coaccess" in idx else 0.0
            if p1[0] != p2[0]:
                continue
            if (p1[0], p1[1]) > (p2[0], p2[1]):
                p1, p2 = p2, p1
            mid1 = (p1[1] + p1[2]) // 2
            mid2 = (p2[1] + p2[2]) // 2
            rows.append({
                "p1": p1, "p2": p2, "q": q, "ca": ca,
                "span": abs(mid2 - mid1),
            })
    return rows


def peak_bins(s: int, e: int, slop: int = 0) -> range:
    lo = max(0, s - slop)
    hi = max(lo + 1, e + slop)
    return range(lo // BIN, (hi - 1) // BIN + 1)


def build_indexes(loops: list[dict]):
    """Anchor lookup: (chrom, bin) -> list of (loop_idx, end 'a'|'b').
    Containment: chrom -> (lefts sorted, rights aligned, idxs aligned).
    """
    anchors: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    lefts: dict[str, list[int]] = defaultdict(list)
    rights: dict[str, list[int]] = defaultdict(list)
    idxs: dict[str, list[int]] = defaultdict(list)
    for i, lp in enumerate(loops):
        a, b = lp["a"], lp["b"]
        if a[0] != b[0]:
            continue
        for pos in range(a[1] // BIN, max(a[1] // BIN + 1, (a[2] - 1) // BIN + 1)):
            anchors[(a[0], pos)].append((i, "a"))
        for pos in range(b[1] // BIN, max(b[1] // BIN + 1, (b[2] - 1) // BIN + 1)):
            anchors[(b[0], pos)].append((i, "b"))
        lefts[a[0]].append(min(a[1], b[1]))
        rights[a[0]].append(max(a[2], b[2]))
        idxs[a[0]].append(i)
    # sort containment arrays by left coordinate
    contain = {}
    for chrom in lefts:
        order = sorted(range(len(lefts[chrom])), key=lambda k: lefts[chrom][k])
        contain[chrom] = (
            [lefts[chrom][k] for k in order],
            [rights[chrom][k] for k in order],
            [idxs[chrom][k] for k in order],
        )
    return anchors, contain


def loops_on_peak(anchors, chrom: str, s: int, e: int, slop: int = 0) -> dict[int, set[str]]:
    hit: dict[int, set[str]] = defaultdict(set)
    for b in peak_bins(s, e, slop):
        for idx, end in anchors.get((chrom, b), ()):
            hit[idx].add(end)
    return hit


def classify_pair(pair, loops, anchors, contain, slop: int = 0) -> tuple[str, int | None]:
    """anchor: each peak overlaps a different loop anchor.
    inside: both peaks sit in the interval between the two anchors.
    Prefer 0.95 then 0.9 then 0.75 (loop list is already nested in that order).
    """
    c, p1, p2 = pair["p1"][0], pair["p1"], pair["p2"]
    h1 = loops_on_peak(anchors, c, p1[1], p1[2], slop)
    h2 = loops_on_peak(anchors, c, p2[1], p2[2], slop)
    best_anchor = None
    for idx, ends1 in h1.items():
        ends2 = h2.get(idx)
        if not ends2:
            continue
        if ("a" in ends1 and "b" in ends2) or ("b" in ends1 and "a" in ends2):
            r = LAYER_RANK.get(loops[idx]["layer"], 99)
            if best_anchor is None:
                best_anchor = idx
            else:
                r0 = LAYER_RANK.get(loops[best_anchor]["layer"], 99)
                if r < r0 or (r == r0 and idx < best_anchor):
                    best_anchor = idx
    if best_anchor is not None:
        return "anchor", best_anchor

    rec = contain.get(c)
    if rec is None:
        return "none", None
    lefts, rights, idxs = rec
    pair_lo = min(p1[1], p2[1])
    pair_hi = max(p1[2], p2[2])
    # pad the loop span by slop on both sides
    n = bisect_right(lefts, pair_lo + slop)  # lefts[k] - slop <= pair_lo
    best_inside = None
    best_span = None
    for k in range(n):
        if rights[k] + slop < pair_hi:
            continue
        idx = idxs[k]
        span = loops[idx]["span"] or (rights[k] - lefts[k])
        if best_inside is None or span < best_span:
            best_inside = idx
            best_span = span
    if best_inside is not None:
        return "inside", best_inside
    return "none", None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--mid", type=Path, required=True)
    ap.add_argument("--loose", type=Path, required=True)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--qval", type=float, default=0.05)
    ap.add_argument("--slop", type=int, default=BIN,
                    help="bp padding on both sides of each peak vs each loop anchor "
                         "and of the loop span (default: 10000)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loops = nested_union(load_loops(args.core, "0.95"),
                         load_loops(args.mid, "0.9"),
                         load_loops(args.loose, "0.75"))
    anchors, contain = build_indexes(loops)
    edges = load_edges(args.edges)
    gold = [e for e in edges if e["q"] <= args.qval]

    union_bedpe = args.out_dir / "loops_10kb_nested_0.95_0.9_0.75.bedpe"
    with union_bedpe.open("w") as o:
        for lp in loops:
            a, b = lp["a"], lp["b"]
            o.write(f"{a[0]}\t{a[1]}\t{a[2]}\t{b[0]}\t{b[1]}\t{b[2]}\t"
                    f"{lp['score']:.6g}\t{lp['layer']}\n")

    def tally(pairs, tag, slop: int):
        n_anchor = n_inside = n_none = 0
        loop_hit = set()
        layer_anchor = defaultdict(int)
        layer_inside = defaultdict(int)
        hits = []
        for pr in pairs:
            klass, idx = classify_pair(pr, loops, anchors, contain, slop=slop)
            if klass == "anchor":
                n_anchor += 1
                loop_hit.add(idx)
                layer_anchor[loops[idx]["layer"]] += 1
            elif klass == "inside":
                n_inside += 1
                loop_hit.add(idx)
                layer_inside[loops[idx]["layer"]] += 1
            else:
                n_none += 1
            hits.append((pr, klass, idx))
        n = len(pairs)
        return {
            "tag": tag, "n": n, "slop": slop,
            "n_anchor": n_anchor, "n_inside": n_inside, "n_none": n_none,
            "frac_anchor": n_anchor / n if n else 0,
            "frac_inside_only": n_inside / n if n else 0,
            "frac_any": (n_anchor + n_inside) / n if n else 0,
            "n_loops_hit": len(loop_hit),
            "layer_anchor": dict(layer_anchor),
            "layer_inside": dict(layer_inside),
            "hits": hits,
            "loop_hit_anchor": {i for _pr, k, i in hits if k == "anchor"},
        }

    slop = args.slop
    stats = [
        tally(edges, f"all_gold_edges  ±{slop/1000:.0f} kb pad", slop),
        tally(gold, f"gold_q<={args.qval}  ±{slop/1000:.0f} kb pad", slop),
        tally(gold, f"gold_q<={args.qval}  exact (no pad)", 0),
    ]
    stats_pad = stats[1]
    stats_exact = stats[2]

    layer_n = defaultdict(int)
    for lp in loops:
        layer_n[lp["layer"]] += 1

    # span bins of gold pairs vs match class (exact-anchor)
    def span_bin(sp):
        if sp < 50_000:
            return "<50kb"
        if sp < 200_000:
            return "50-200kb"
        if sp < 500_000:
            return "200-500kb"
        return ">=500kb"

    span_tot = defaultdict(int)
    span_anchor = defaultdict(int)
    span_inside = defaultdict(int)
    unmatched_span = defaultdict(int)
    for pr, klass, _idx in stats_pad["hits"]:
        b = span_bin(pr["span"])
        span_tot[b] += 1
        if klass == "anchor":
            span_anchor[b] += 1
        elif klass == "inside":
            span_inside[b] += 1
        else:
            unmatched_span[b] += 1

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"nested 10 kb Peakachu loops: n={len(loops)}  "
         f"core_0.95={layer_n['0.95']}  added_0.9={layer_n['0.9']}  added_0.75={layer_n['0.75']}")
    emit(f"gold RBBP4 Cicero edges: {len(edges):,}   q<={args.qval}: {len(gold):,}")
    emit("source: cobinding/work/RBBP4.peak_edges.tsv (read-only)")
    emit("")
    emit("Match rules (primary = ±10 kb padding on both sides):")
    emit("  each Cicero peak is expanded ±slop; each loop span is expanded ±slop")
    emit("  anchor = each padded peak overlaps a different 10 kb loop anchor (swap OK)")
    emit("  inside = both peaks fall in the padded interval between the two anchors")
    emit("  when several loops match, keep the nested-highest (0.95 > 0.9 > 0.75);")
    emit("  for inside-only, keep the shortest spanning loop")
    emit("")
    for st in stats:
        emit(f"=== {st['tag']} ===")
        emit(f"  pairs: {st['n']:,}")
        emit(f"  anchor-anchor: {st['n_anchor']:,}  ({100*st['frac_anchor']:.1f}%)")
        emit(f"  inside loop, not on both anchors: {st['n_inside']:,}  ({100*st['frac_inside_only']:.1f}%)")
        emit(f"  any (anchor or inside): {st['n_anchor']+st['n_inside']:,}  ({100*st['frac_any']:.1f}%)")
        emit(f"  no loop: {st['n_none']:,}  ({100*st['n_none']/st['n']:.1f}%)")
        emit(f"  distinct loops hit: {st['n_loops_hit']:,} / {len(loops):,}")
        emit(f"  anchor hits by loop layer: {st['layer_anchor']}")
        emit(f"  inside hits by loop layer: {st['layer_inside']}")
        emit("")

    emit("=== loops that capture a q<=0.05 cobinding pair (±10 kb pad) ===")
    loops_with_pair = stats_pad["loop_hit_anchor"]
    emit(f"  loops with anchor-anchor pair: {len(loops_with_pair):,} / {len(loops):,} "
         f"({100*len(loops_with_pair)/len(loops):.1f}%)")
    by_layer_hit = defaultdict(int)
    for i in loops_with_pair:
        by_layer_hit[loops[i]["layer"]] += 1
    emit("  fraction of each nested layer with ≥1 anchor-anchor cobinding pair:")
    for ly in ("0.95", "0.9", "0.75"):
        n = layer_n[ly]
        h = by_layer_hit[ly]
        emit(f"    {ly}: {h}/{n}  ({100*h/n if n else 0:.1f}%)")
    emit("")
    emit("=== q<=0.05 pair genomic span vs match (±10 kb pad) ===")
    emit(f"  {'span':<12} {'n_pairs':>8} {'anchor':>8} {'inside':>8} {'none':>8}")
    for b in ("<50kb", "50-200kb", "200-500kb", ">=500kb"):
        n = span_tot[b]
        a = span_anchor[b]
        inn = span_inside[b]
        none = unmatched_span[b]
        emit(f"  {b:<12} {n:8d} {a:8d} {inn:8d} {none:8d}")
    emit("")
    n_either = n_one = n_both_any = 0
    peaks = set()
    peaks_on_anchor = 0
    for pr in gold:
        c, p1, p2 = pr["p1"][0], pr["p1"], pr["p2"]
        h1 = loops_on_peak(anchors, c, p1[1], p1[2], slop)
        h2 = loops_on_peak(anchors, c, p2[1], p2[2], slop)
        e1, e2 = bool(h1), bool(h2)
        if e1 or e2:
            n_either += 1
        if e1 ^ e2:
            n_one += 1
        if e1 and e2:
            n_both_any += 1
        peaks.add(p1)
        peaks.add(p2)
    for p in peaks:
        if loops_on_peak(anchors, p[0], p[1], p[2], slop):
            peaks_on_anchor += 1
    emit("")
    emit("=== one-end overlap (q<=0.05, ±10 kb pad) ===")
    emit(f"  either peak within 10 kb of a nested-loop anchor: {n_either:,}  ({100*n_either/len(gold):.1f}%)")
    emit(f"  exactly one peak: {n_one:,}")
    emit(f"  both peaks on some anchor (not necessarily the same loop): {n_both_any:,}")
    emit(f"  unique peaks within 10 kb of an anchor: {peaks_on_anchor:,} / {len(peaks):,} "
         f"({100*peaks_on_anchor/len(peaks):.1f}%)")
    emit("")
    emit(f"exact (no pad) q<=0.05 anchor-anchor was {stats_exact['n_anchor']:,}; "
         f"±{slop/1000:.0f} kb pad is {stats_pad['n_anchor']:,}.")

    summary = args.out_dir / "compare_nested_loops_vs_RBBP4_cicero.txt"
    summary.write_text("\n".join(lines) + "\n")

    hits = args.out_dir / "RBBP4_q05_pairs_overlapping_loops.tsv"
    with hits.open("w") as o:
        o.write("peak1\tpeak2\tcoaccess\tqval\tpair_span\tmatch\tloop_layer\tloop_span\t"
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\n")
        for pr, klass, idx in stats_pad["hits"]:
            if idx is None:
                continue
            lp = loops[idx]
            a, b = lp["a"], lp["b"]
            o.write(
                f"{pr['p1'][0]}:{pr['p1'][1]}-{pr['p1'][2]}\t"
                f"{pr['p2'][0]}:{pr['p2'][1]}-{pr['p2'][2]}\t"
                f"{pr['ca']:.6g}\t{pr['q']:.4g}\t{pr['span']}\t"
                f"{klass}\t{lp['layer']}\t{lp['span'] or ''}\t"
                f"{a[0]}\t{a[1]}\t{a[2]}\t{b[0]}\t{b[1]}\t{b[2]}\n"
            )
    print(f"wrote {union_bedpe}")
    print(f"wrote {summary}")
    print(f"wrote {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
