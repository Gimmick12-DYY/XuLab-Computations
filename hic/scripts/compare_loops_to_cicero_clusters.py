#!/usr/bin/env python3
"""Compare nested Peakachu 10 kb loops to gold RBBP4 Cicero cliques/modules.

Does not modify cobinding/results/RBBP4.
"""
from __future__ import annotations

import argparse
import importlib.util
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "loopcmp",
    Path(__file__).with_name("compare_loops_to_cicero_pairs.py"),
)
loopcmp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loopcmp)

LAYER_RANK = loopcmp.LAYER_RANK


def parse_peak(tok: str):
    return loopcmp.parse_peak(tok)


def load_clusters(path: Path, id_col: str) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            peaks = []
            for tok in p[idx["regions"]].split(";"):
                pk = parse_peak(tok)
                if pk:
                    peaks.append(pk)
            if len(peaks) < 2:
                continue
            chroms = {pk[0] for pk in peaks}
            chrom = p[idx["chromosome"]] if "chromosome" in idx else next(iter(chroms))
            span = int(float(p[idx["span_bp"]])) if "span_bp" in idx and p[idx["span_bp"]] else 0
            extra = {}
            if "n_source_cliques" in idx:
                extra["n_source"] = int(p[idx["n_source_cliques"]])
            rows.append({
                "id": p[idx[id_col]],
                "n": int(p[idx["n_regions"]]),
                "chrom": chrom,
                "span": span,
                "peaks": peaks,
                **extra,
            })
    return rows


def classify_cluster(cl, loops, anchors, contain, slop: int):
    """two_anchors: ≥1 member near each end of the same loop.
    all_inside: every member in some (padded) loop span, else not two_anchors.
    any_anchor: ≥1 member near some loop anchor.
    """
    peaks = cl["peaks"]
    chrom = cl["chrom"]
    per_peak = [loopcmp.loops_on_peak(anchors, pk[0], pk[1], pk[2], slop) for pk in peaks]
    loop_ends: dict[int, dict[str, int]] = defaultdict(lambda: {"a": 0, "b": 0})
    n_on_anchor = 0
    for hit in per_peak:
        if hit:
            n_on_anchor += 1
        seen_idx = set()
        for idx, ends in hit.items():
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            if "a" in ends:
                loop_ends[idx]["a"] += 1
            if "b" in ends:
                loop_ends[idx]["b"] += 1

    best_two = None
    for idx, ab in loop_ends.items():
        if ab["a"] >= 1 and ab["b"] >= 1:
            r = LAYER_RANK.get(loops[idx]["layer"], 99)
            if best_two is None:
                best_two = idx
            else:
                r0 = LAYER_RANK.get(loops[best_two]["layer"], 99)
                if r < r0 or (r == r0 and idx < best_two):
                    best_two = idx

    pair_lo = min(pk[1] for pk in peaks)
    pair_hi = max(pk[2] for pk in peaks)
    rec = contain.get(chrom)
    best_inside = None
    best_span = None
    n_inside_best = 0
    if rec is not None:
        lefts, rights, idxs = rec
        n = bisect_right(lefts, pair_lo + slop)
        for k in range(n):
            if rights[k] + slop < pair_hi:
                continue
            idx = idxs[k]
            span = loops[idx]["span"] or (rights[k] - lefts[k])
            n_in = sum(1 for pk in peaks if pk[1] < rights[k] + slop and pk[2] > lefts[k] - slop)
            if best_inside is None or span < best_span:
                best_inside = idx
                best_span = span
                n_inside_best = n_in

    if best_two is not None:
        lp = loops[best_two]
        ab = loop_ends[best_two]
        return {
            "match": "two_anchors",
            "idx": best_two,
            "n_on_A": ab["a"],
            "n_on_B": ab["b"],
            "n_on_anchor": n_on_anchor,
            "n_inside": n_inside_best,
            "layer": lp["layer"],
            "loop_span": lp["span"],
        }
    if best_inside is not None and n_inside_best == len(peaks):
        lp = loops[best_inside]
        return {
            "match": "all_inside",
            "idx": best_inside,
            "n_on_A": 0,
            "n_on_B": 0,
            "n_on_anchor": n_on_anchor,
            "n_inside": n_inside_best,
            "layer": lp["layer"],
            "loop_span": lp["span"],
        }
    if n_on_anchor:
        return {
            "match": "anchor_partial",
            "idx": best_inside,
            "n_on_A": 0,
            "n_on_B": 0,
            "n_on_anchor": n_on_anchor,
            "n_inside": n_inside_best,
            "layer": loops[best_inside]["layer"] if best_inside is not None else "",
            "loop_span": loops[best_inside]["span"] if best_inside is not None else "",
        }
    if best_inside is not None:
        return {
            "match": "partial_inside",
            "idx": best_inside,
            "n_on_A": 0,
            "n_on_B": 0,
            "n_on_anchor": 0,
            "n_inside": n_inside_best,
            "layer": loops[best_inside]["layer"],
            "loop_span": loops[best_inside]["span"],
        }
    return {
        "match": "none",
        "idx": None,
        "n_on_A": 0,
        "n_on_B": 0,
        "n_on_anchor": 0,
        "n_inside": 0,
        "layer": "",
        "loop_span": "",
    }


def summarize(clusters, loops, anchors, contain, slop: int, tag: str):
    counts = Counter()
    by_size = defaultdict(Counter)
    by_layer = Counter()
    n_on_hist = Counter()
    loops_two = set()
    hits = []
    for cl in clusters:
        rec = classify_cluster(cl, loops, anchors, contain, slop)
        counts[rec["match"]] += 1
        by_size[cl["n"]][rec["match"]] += 1
        n_on_hist[rec["n_on_anchor"]] += 1
        if rec["match"] == "two_anchors":
            by_layer[rec["layer"]] += 1
            loops_two.add(rec["idx"])
        hits.append((cl, rec))
    n = len(clusters)
    return {
        "tag": tag, "n": n, "counts": counts, "by_size": by_size,
        "by_layer": by_layer, "n_on_hist": n_on_hist,
        "n_loops_two": len(loops_two), "hits": hits,
    }


def emit_block(st, loops_n, emit):
    n = st["n"]
    c = st["counts"]
    two = c["two_anchors"]
    inside = c["all_inside"]
    part_a = c["anchor_partial"]
    part_i = c["partial_inside"]
    none = c["none"]
    emit(f"=== {st['tag']}  n={n} ===")
    emit(f"  both loop anchors occupied (≥1 member on each end of same loop): {two}  ({100*two/n:.1f}%)")
    emit(f"  all members inside a loop span, not two-anchor: {inside}  ({100*inside/n:.1f}%)")
    emit(f"  ≥1 member near an anchor, not two-anchor: {part_a}  ({100*part_a/n:.1f}%)")
    emit(f"  partial span overlap only: {part_i}  ({100*part_i/n:.1f}%)")
    emit(f"  none: {none}  ({100*none/n:.1f}%)")
    emit(f"  distinct loops with both anchors hit by a cluster: {st['n_loops_two']} / {loops_n}")
    emit(f"  two-anchor hits by loop layer: {dict(st['by_layer'])}")
    emit("  members near some loop anchor (histogram n_members): "
         + ", ".join(f"{k}:{st['n_on_hist'][k]}" for k in sorted(st["n_on_hist"])))
    if st["by_size"]:
        emit("  by cluster size:")
        for sz in sorted(st["by_size"]):
            cc = st["by_size"][sz]
            tot = sum(cc.values())
            emit(f"    k={sz} n={tot}  two_anchors={cc['two_anchors']}  "
                 f"all_inside={cc['all_inside']}  anchor_partial={cc['anchor_partial']}  "
                 f"none={cc['none']}")
    emit("")


def write_hits(path: Path, hits, loops):
    with path.open("w") as o:
        o.write("id\tn_regions\tchromosome\tspan_bp\tmatch\tn_on_anchor\tn_on_A\tn_on_B\t"
                "n_inside\tloop_layer\tloop_span\tchrom1\tstart1\tend1\tchrom2\tstart2\tend2\tregions\n")
        for cl, rec in hits:
            if rec["idx"] is None:
                loop_bed = "\t".join(["."] * 6)
            else:
                lp = loops[rec["idx"]]
                a, b = lp["a"], lp["b"]
                loop_bed = f"{a[0]}\t{a[1]}\t{a[2]}\t{b[0]}\t{b[1]}\t{b[2]}"
            regs = ";".join(f"{p[0]}:{p[1]}-{p[2]}" for p in cl["peaks"])
            o.write(
                f"{cl['id']}\t{cl['n']}\t{cl['chrom']}\t{cl['span']}\t{rec['match']}\t"
                f"{rec['n_on_anchor']}\t{rec['n_on_A']}\t{rec['n_on_B']}\t{rec['n_inside']}\t"
                f"{rec['layer']}\t{rec['loop_span'] or ''}\t{loop_bed}\t{regs}\n"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--mid", type=Path, required=True)
    ap.add_argument("--loose", type=Path, required=True)
    ap.add_argument("--cliques", type=Path, required=True)
    ap.add_argument("--modules", type=Path, required=True)
    ap.add_argument("--slop", type=int, default=loopcmp.BIN)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loops = loopcmp.nested_union(
        loopcmp.load_loops(args.core, "0.95"),
        loopcmp.load_loops(args.mid, "0.9"),
        loopcmp.load_loops(args.loose, "0.75"),
    )
    anchors, contain = loopcmp.build_indexes(loops)
    cliques = load_clusters(args.cliques, "clique_id")
    modules = load_clusters(args.modules, "module_id")
    multi = [m for m in modules if m.get("n_source", 1) > 1]

    slop = args.slop
    pad = f"±{slop/1000:.0f} kb"
    stats = [
        summarize(cliques, loops, anchors, contain, slop, f"cliques (>=3) {pad}"),
        summarize(modules, loops, anchors, contain, slop, f"overlap modules {pad}"),
        summarize(multi, loops, anchors, contain, slop, f"multi-clique modules {pad}"),
        summarize(cliques, loops, anchors, contain, 0, "cliques exact (no pad)"),
        summarize(cliques, loops, anchors, contain, 20_000, "cliques ±20 kb"),
    ]

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"nested 10 kb Peakachu loops: n={len(loops)}")
    emit(f"gold RBBP4 cliques: {len(cliques)}  modules: {len(modules)}  "
         f"multi-clique modules: {len(multi)}")
    emit("source: cobinding/results/RBBP4/cliques.tsv and modules.tsv (read-only)")
    emit("")
    emit("Match rules:")
    emit("  two_anchors = ≥1 clique/module member near each of the two anchors of the same loop")
    emit("  all_inside  = every member falls in that loop's padded span, but not two_anchors")
    emit("  anchor_partial = ≥1 member near some loop anchor, not two_anchors")
    emit(f"  padding: {pad} on peaks and loop spans (primary)")
    emit("")
    for st in stats:
        emit_block(st, len(loops), emit)

    summary = args.out_dir / "compare_nested_loops_vs_RBBP4_clusters.txt"
    summary.write_text("\n".join(lines) + "\n")
    write_hits(args.out_dir / "RBBP4_cliques_vs_loops.tsv", stats[0]["hits"], loops)
    write_hits(args.out_dir / "RBBP4_modules_vs_loops.tsv", stats[1]["hits"], loops)
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
