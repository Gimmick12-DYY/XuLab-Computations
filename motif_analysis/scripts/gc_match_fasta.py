#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# gc_match_fasta.py
#
# Build GC-matched foreground + control FASTAs for AME --control.
#
# Why: OCR background is compositionally poorer than imputed peaks (e.g. CTCF
# fg GC~0.57 vs OCR bg GC~0.47). Without matching, any GC-rich PWM (SP2, MAZ,
# NRF1, E2F3, CTCFL) looks enriched at absurd p-values under either --scoring.
#
# Method: bin sequences by GC, then in each bin take n = min(|fg|,|bg|) without
# replacement from both sides. Never upsample -- duplicates in the control made
# AME treat the same sequence as independent observations and re-inflated p.
# When |OCR bg| << |peaks| the matched sets shrink to the OCR size; that is the
# honest comparison (and what HOMER's -bg already does by construction).
#
#   python gc_match_fasta.py --fg peaks.fa --bg background.fa \
#       --out-fg peaks.gc.fa --out-bg background.gc.fa
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_fa(path):
    out, name, buf = [], None, []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith(">"):
                if name is not None:
                    out.append((name, "".join(buf)))
                name, buf = ln[1:].strip(), []
            else:
                buf.append(ln.strip())
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def gc(seq: str) -> float:
    s = seq.upper()
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def write_fa(path: Path, recs, idxs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i in idxs:
            n, s = recs[i]
            fh.write(f">{n}\n")
            for j in range(0, len(s), 80):
                fh.write(s[j:j + 80] + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fg", type=Path, required=True)
    ap.add_argument("--bg", type=Path, required=True)
    ap.add_argument("--out-bg", type=Path, required=True)
    ap.add_argument("--out-fg", type=Path, default=None,
                    help="optional GC-matched fg subsample (same n as out-bg)")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    fg, bg = read_fa(args.fg), read_fa(args.bg)
    if not fg or not bg:
        raise SystemExit(f"empty FASTA: fg={len(fg)} bg={len(bg)}")

    def bucket(seq):
        return min(int(gc(seq) * args.bins), args.bins - 1)

    fg_b = [[] for _ in range(args.bins)]
    bg_b = [[] for _ in range(args.bins)]
    for i, (_, s) in enumerate(fg):
        fg_b[bucket(s)].append(i)
    for i, (_, s) in enumerate(bg):
        bg_b[bucket(s)].append(i)

    fg_pick, bg_pick = [], []
    for b in range(args.bins):
        n = min(len(fg_b[b]), len(bg_b[b]))
        if n == 0:
            continue
        fg_pick.extend(rng.sample(fg_b[b], n))
        bg_pick.extend(rng.sample(bg_b[b], n))

    write_fa(args.out_bg, bg, bg_pick)
    if args.out_fg is not None:
        write_fa(args.out_fg, fg, fg_pick)

    def mean_gc(recs, idxs):
        xs = [recs[i][1] for i in idxs]
        return sum(gc(s) for s in xs) / max(len(xs), 1)

    print(f"[gc-match] fg n={len(fg):,} mean_GC={sum(gc(s) for _,s in fg)/len(fg):.3f}  "
          f"bg n={len(bg):,} mean_GC={sum(gc(s) for _,s in bg)/len(bg):.3f}  "
          f"-> matched n={len(bg_pick):,}  "
          f"fg_GC={mean_gc(fg, fg_pick):.3f} bg_GC={mean_gc(bg, bg_pick):.3f}  "
          f"-> {args.out_bg}"
          + (f" + {args.out_fg}" if args.out_fg else ""),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
