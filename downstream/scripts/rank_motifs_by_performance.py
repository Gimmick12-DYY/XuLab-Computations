#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# rank_motifs_by_performance.py   (Novel Motif Finding, Step 3)
#
# "Rank motifs by PERFORMANCE, not information content." For each candidate motif
# (de-novo HOMER/STREME, or any MEME-format PWM), score every positive and
# negative sequence by its best PWM match (max log-odds over all offsets + both
# strands), then rank motifs by how well that score SEPARATES pos from neg:
#   AUROC + AUPRC (held-out pos vs neg bins).
# Also reports each motif's information content, so you can SEE that low-info
# ("flat") motifs often out-predict high-IC ones -- the whole point of Step 3.
#
# Inputs are FASTA (pos/neg sequences, e.g. bedtools getfasta on held-out bins)
# and a MEME motif file, so this is tool-agnostic (no FIMO threshold tuning).
#
#   python rank_motifs_by_performance.py --motifs cand.meme \
#       --pos-fa pos.fa --neg-fa neg.fa --out ranked_motifs.tsv
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

_B = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMP = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T, C<->G column swap for reverse complement


def read_fasta(path: Path) -> list[np.ndarray]:
    """FASTA -> list of int8 sequences over {0..3}; non-ACGT positions = -1."""
    seqs, cur = [], []

    def flush():
        if cur:
            seqs.append(np.frombuffer("".join(cur).encode(), dtype=np.uint8))

    with open(path) as fh:
        for ln in fh:
            if ln.startswith(">"):
                flush(); cur.clear()
            else:
                cur.append(ln.strip().upper())
    flush()
    out = []
    lut = np.full(256, -1, dtype=np.int8)
    for b, i in _B.items():
        lut[ord(b)] = i
    for s in seqs:
        out.append(lut[s])
    return out


def read_meme(path: Path):
    """Yield (motif_id, name, pwm[L,4] probabilities)."""
    motifs = []
    mid = name = None
    rows: list[list[float]] = []
    in_mat = False
    with open(path) as fh:
        for ln in fh:
            t = ln.strip()
            if t.startswith("MOTIF"):
                if mid is not None and rows:
                    motifs.append((mid, name, np.asarray(rows, float)))
                parts = t.split()
                mid = parts[1] if len(parts) > 1 else "motif"
                name = parts[2] if len(parts) > 2 else mid
                rows, in_mat = [], False
            elif t.startswith("letter-probability matrix"):
                in_mat = True
            elif in_mat:
                vals = t.split()
                if len(vals) >= 4:
                    try:
                        rows.append([float(x) for x in vals[:4]])
                    except ValueError:
                        in_mat = False
                else:
                    in_mat = False
        if mid is not None and rows:
            motifs.append((mid, name, np.asarray(rows, float)))
    return motifs


def info_content(pwm: np.ndarray) -> float:
    """Total information content (bits) vs uniform background."""
    p = np.clip(pwm, 1e-9, 1.0)
    return float(np.sum(p * np.log2(p / 0.25)))


def logodds(pwm: np.ndarray, bg=0.25) -> np.ndarray:
    return np.log2(np.clip(pwm, 1e-9, 1.0) / bg)


def revcomp_lo(lo: np.ndarray) -> np.ndarray:
    rc = lo[::-1].copy()
    return rc[:, [_COMP[0], _COMP[1], _COMP[2], _COMP[3]]]


def best_scores(seqs: list[np.ndarray], lo: np.ndarray) -> np.ndarray:
    """Per-sequence best (max) log-odds over all offsets and both strands."""
    L = lo.shape[0]
    lo_rc = revcomp_lo(lo)
    out = np.full(len(seqs), -np.inf)
    for i, s in enumerate(seqs):
        n = s.shape[0]
        if n < L:
            continue
        # windows x L int codes; mask any window containing a non-ACGT (-1)
        idx = np.arange(n - L + 1)[:, None] + np.arange(L)[None, :]
        w = s[idx]                                   # (W, L)
        valid = (w >= 0).all(axis=1)
        if not valid.any():
            continue
        wv = w[valid]
        pos = np.arange(L)
        fwd = lo[pos, wv].sum(axis=1)
        rev = lo_rc[pos, wv].sum(axis=1)
        out[i] = max(fwd.max(), rev.max())
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC (rank-based), NaN-safe."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auprc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    recall = tp / pos.size
    precision = tp / np.maximum(tp + fp, 1)
    # step integration over recall
    rec = np.concatenate([[0.0], recall])
    prec = np.concatenate([[1.0], precision])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motifs", type=Path, required=True, help="candidate motifs, MEME format")
    ap.add_argument("--pos-fa", type=Path, required=True, help="positive (bound) sequences FASTA")
    ap.add_argument("--neg-fa", type=Path, required=True, help="negative (background) sequences FASTA")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pos = read_fasta(args.pos_fa)
    neg = read_fasta(args.neg_fa)
    motifs = read_meme(args.motifs)
    if not motifs:
        raise SystemExit(f"no motifs parsed from {args.motifs}")
    print(f"[rank] {len(motifs)} motifs; pos={len(pos)} neg={len(neg)} seqs", flush=True)

    rows = []
    for mid, name, pwm in motifs:
        lo = logodds(pwm)
        sp = best_scores(pos, lo)
        sn = best_scores(neg, lo)
        rows.append((mid, name, pwm.shape[0], info_content(pwm),
                     auroc(sp, sn), auprc(sp, sn)))
    rows.sort(key=lambda r: (-(r[4] if not math.isnan(r[4]) else -1)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("motif_id\tname\twidth\tinfo_bits\tauroc\tauprc\n")
        for mid, name, w, ic, ar, ap_ in rows:
            f.write(f"{mid}\t{name}\t{w}\t{ic:.3f}\t{ar:.4f}\t{ap_:.4f}\n")
    top = rows[0]
    print(f"[rank] best: {top[0]} ({top[1]}) AUROC={top[4]:.3f} AUPRC={top[5]:.3f} "
          f"IC={top[3]:.2f} bits -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
