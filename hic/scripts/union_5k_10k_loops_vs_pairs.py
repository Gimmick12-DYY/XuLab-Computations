#!/usr/bin/env python3
"""Union Peakachu 5 kb and 10 kb loops at cutoff 0.7 (prefer 5 kb).

All 5 kb loops are kept. 10 kb loops are kept as-is in the BEDPE union
(coordinates never match exactly across resolutions). When a cobinding
pair overlaps loops at both resolutions, the 5 kb loop is the match.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

SLOP = 10_000


def parse_peak(tok: str):
    chrom, se = tok.split(":", 1)
    s, e = se.split("-", 1)
    return chrom, int(s), int(e)


def load_bedpe(path: Path, res: str) -> list[dict]:
    out = []
    with path.open() as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            x = ln.split()
            a = (x[0], int(x[1]), int(x[2]))
            b = (x[3], int(x[4]), int(x[5]))
            if (a[0], a[1]) > (b[0], b[1]):
                a, b = b, a
            score = float(x[6]) if len(x) > 6 else 0.0
            span = abs(((b[1] + b[2]) // 2) - ((a[1] + a[2]) // 2)) if a[0] == b[0] else None
            out.append({"a": a, "b": b, "score": score, "span": span, "res": res})
    return out


def load_edges(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            p1, p2 = parse_peak(p[idx["peak1"]]), parse_peak(p[idx["peak2"]])
            if p1[0] != p2[0]:
                continue
            if (p1[0], p1[1]) > (p2[0], p2[1]):
                p1, p2 = p2, p1
            q = float(p[idx["qval"]]) if p[idx["qval"]] else 1.0
            rows.append({"p1": p1, "p2": p2, "q": q})
    return rows


def iv_overlap(chrom_iv, s, e, slop: int) -> bool:
    c, a, b = chrom_iv
    return c == chrom_iv[0] and min(b, e + slop) > max(a, s - slop)


def index_anchors(loops: list[dict], bin_size: int):
    """(chrom, bin) -> list of (loop_idx, 'a'|'b')."""
    ix = defaultdict(list)
    for i, lp in enumerate(loops):
        for end, iv in (("a", lp["a"]), ("b", lp["b"])):
            for b in range(iv[1] // bin_size, (iv[2] - 1) // bin_size + 1):
                ix[(iv[0], b)].append((i, end))
    return ix


def loops_on_peak(ix, loops, chrom, s, e, slop: int, bin_size: int) -> dict[int, set[str]]:
    hit: dict[int, set[str]] = defaultdict(set)
    lo, hi = max(0, s - slop), e + slop
    for b in range(lo // bin_size, max(lo // bin_size, (hi - 1) // bin_size) + 1):
        for idx, end in ix.get((chrom, b), ()):
            iv = loops[idx][end]
            if min(iv[2], hi) > max(iv[1], lo):
                hit[idx].add(end)
    return hit


def pair_hits(pr, loops, ix, slop: int, bin_size: int) -> tuple[bool, bool, int | None]:
    """Return (hit_one_anchor, hit_both_anchors_same_loop, loop_idx)."""
    c, p1, p2 = pr["p1"][0], pr["p1"], pr["p2"]
    h1 = loops_on_peak(ix, loops, c, p1[1], p1[2], slop, bin_size)
    h2 = loops_on_peak(ix, loops, c, p2[1], p2[2], slop, bin_size)
    one = bool(h1) or bool(h2)
    best = None
    for idx, e1 in h1.items():
        e2 = h2.get(idx)
        if not e2:
            continue
        if ("a" in e1 and "b" in e2) or ("b" in e1 and "a" in e2):
            if best is None or idx < best:
                best = idx
    return one, best is not None, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine", type=Path, required=True, help="5 kb 0.7 bedpe")
    ap.add_argument("--coarse", type=Path, required=True, help="10 kb 0.7 bedpe")
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--slop", type=int, default=SLOP)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fine = load_bedpe(args.fine, "5000")
    coarse = load_bedpe(args.coarse, "10000")
    union = fine + coarse
    union_path = args.out_dir / "peakachu_union_5k_10k.loops.0.7.bedpe"
    with union_path.open("w") as o:
        for lp in union:
            a, b = lp["a"], lp["b"]
            o.write(f"{a[0]}\t{a[1]}\t{a[2]}\t{b[0]}\t{b[1]}\t{b[2]}\t"
                    f"{lp['score']:.6g}\t{lp['res']}\n")

    pairs = load_edges(args.edges)
    slop = args.slop
    ix5, ix10 = index_anchors(fine, 5000), index_anchors(coarse, 10000)

    n_both = n_one = 0
    n_both_5 = n_both_10_only = n_both_bothres = 0
    n_one_5 = n_one_10_only = 0
    for pr in pairs:
        one5, both5, _ = pair_hits(pr, fine, ix5, slop, 5000)
        one10, both10, _ = pair_hits(pr, coarse, ix10, slop, 10000)
        one = one5 or one10
        both = both5 or both10
        if both:
            n_both += 1
            if both5 and both10:
                n_both_bothres += 1
                n_both_5 += 1
            elif both5:
                n_both_5 += 1
            else:
                n_both_10_only += 1
        if one:
            n_one += 1
            if one5 and not one10:
                n_one_5 += 1
            elif one10 and not one5:
                n_one_10_only += 1
            # if both resolutions have ≥1 anchor, still "at least 1"; counted in n_one

    n = len(pairs)
    lines = [
        f"loop union @ 0.7: 5 kb={len(fine):,}  10 kb={len(coarse):,}  cat={len(union):,}",
        "preference: if a cobinding pair hits loops at both resolutions, keep 5 kb",
        f"padding: ±{slop} bp on peaks vs anchors",
        f"gold RBBP4 pairs (all): {n:,}",
        "",
        f"hit both anchors of the same loop: {n_both:,}  ({100*n_both/n:.1f}%)",
        f"  assigned 5 kb (incl. pairs that also hit 10 kb): {n_both_5:,}",
        f"    of those, hit both resolutions: {n_both_bothres:,}  (kept 5 kb)",
        f"  10 kb only: {n_both_10_only:,}",
        f"hit at least 1 loop anchor: {n_one:,}  ({100*n_one/n:.1f}%)",
        f"  5 kb only (no 10 kb anchor): {n_one_5:,}",
        f"  10 kb only (no 5 kb anchor): {n_one_10_only:,}",
        f"  either resolution: {n_one:,}",
        f"neither anchor: {n-n_one:,}  ({100*(n-n_one)/n:.1f}%)",
        f"wrote {union_path}",
    ]
    text = "\n".join(lines)
    print(text)
    (args.out_dir / "compare_union_0.7_vs_RBBP4_pairs.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
