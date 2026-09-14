#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# split_chromhmm_states.py
#
# Dissect each ChromHMM-18 state into N smaller states (default 5) by ranking
# that state's segments on the Hi-C eigenvector (E1) and cutting into equal-bp
# quantile classes:
#     {state}.q1  lowest E1 (most B-like)
#     ...
#     {state}.qN  highest E1 (most A-like)
# Each class still pools genome-wide, so a shallow TF keeps enough reads per
# unit (18 -> 90 units, not 22k domains).
#
#   python split_chromhmm_states.py \
#     --chromhmm-bed data/HEK293T_chromHMM18.bed.gz \
#     --ab-bed hic/work/compartments/compartments_25000.AB.bed \
#     --n-split 5 --out tf_complex/work/chromhmm_x5/chromhmm18_x5.bed
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_e1(path):
    by_c = {}
    for ln in _open(path):
        if not ln.strip() or ln.startswith(("#", "track", "browser", "chrom")):
            continue
        p = ln.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        try:
            e1 = float(p[4])
        except ValueError:
            continue
        by_c.setdefault(p[0], []).append((int(p[1]), int(p[2]), e1))
    idx = {}
    for c, rows in by_c.items():
        rows.sort()
        idx[c] = (np.array([r[0] for r in rows], dtype=np.int64),
                  np.array([r[1] for r in rows], dtype=np.int64),
                  np.array([r[2] for r in rows], dtype=float))
    return idx


def segment_e1(starts, ends, e1, s, e):
    """Length-weighted mean E1 of compartment bins overlapping [s, e)."""
    k0 = int(np.searchsorted(ends, s, side="right"))
    k1 = int(np.searchsorted(starts, e, side="left"))
    if k1 <= k0:
        return np.nan
    w = np.minimum(ends[k0:k1], e) - np.maximum(starts[k0:k1], s)
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        return np.nan
    return float(np.average(e1[k0:k1], weights=w))


def assign_quantiles(e1, bp, n_split):
    """Equal-bp E1 quantiles within one state. q=1 is lowest E1."""
    n = len(e1)
    lab = np.ones(n, dtype=int)
    ok = np.isfinite(e1)
    if ok.sum() == 0:
        return lab
    order = np.argsort(np.where(ok, e1, np.inf))
    target = bp[ok].sum() / float(n_split)
    cum, q = 0.0, 1
    for i in order:
        if not ok[i]:
            lab[i] = (n_split + 1) // 2
            continue
        if cum >= q * target and q < n_split:
            q += 1
        lab[i] = q
        cum += bp[i]
    return lab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chromhmm-bed", type=Path, required=True)
    ap.add_argument("--ab-bed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-split", type=int, default=5)
    args = ap.parse_args()

    e1_idx = load_e1(args.ab_bed)
    segs = []  # chrom, start, end, state, e1, bp
    for ln in _open(args.chromhmm_bed):
        if not ln.strip() or ln.startswith(("#", "track", "browser")):
            continue
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        c, s, e, st = p[0], int(p[1]), int(p[2]), p[3].strip()
        if c in e1_idx:
            starts, ends, e1 = e1_idx[c]
            ev = segment_e1(starts, ends, e1, s, e)
        else:
            ev = np.nan
        segs.append((c, s, e, st, ev, e - s))
    print(f"[in] {len(segs):,} ChromHMM segments; "
          f"E1 for {sum(np.isfinite(s[4]) for s in segs):,}", flush=True)

    by_st = defaultdict(list)
    for i, rec in enumerate(segs):
        by_st[rec[3]].append(i)

    label = [""] * len(segs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tsv = args.out.with_suffix(".classes.tsv")
    with tsv.open("w") as ft:
        ft.write("class\tparent\tq\tbp\tn_segs\tmean_e1\n")
        print("[classes]", flush=True)
        for st in by_st:
            ix = np.array(by_st[st], dtype=int)
            e1 = np.array([segs[i][4] for i in ix], dtype=float)
            bp = np.array([segs[i][5] for i in ix], dtype=float)
            q = assign_quantiles(e1, bp, args.n_split)
            for i, qi in zip(ix, q):
                label[i] = f"{st}.q{int(qi)}"
            for qi in range(1, args.n_split + 1):
                m = q == qi
                ev = e1[m]
                ev = ev[np.isfinite(ev)]
                mu = float(ev.mean()) if ev.size else float("nan")
                ft.write(f"{st}.q{qi}\t{st}\t{qi}\t{int(bp[m].sum())}\t{int(m.sum())}\t{mu:.4f}\n")
            print(f"  {st}: " + "  ".join(
                f"q{qi}={int((q==qi).sum())}segs/{bp[q==qi].sum()/1e6:.1f}Mb"
                for qi in range(1, args.n_split + 1)), flush=True)

    with args.out.open("w") as f:
        for rec, lab in zip(segs, label):
            f.write(f"{rec[0]}\t{rec[1]}\t{rec[2]}\t{lab}\n")
    print(f"[done] {args.n_split} x {len(by_st)} states -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
