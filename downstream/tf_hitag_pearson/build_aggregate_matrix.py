#!/usr/bin/env python3
"""Pearson correlation at 10 kb for one TF: raw aggregate, imputed aggregate,
Chromnitron (SA / former pred_2, if present), and bulk CPM (if present).

Usage:
  python build_aggregate_matrix.py CTCF
  python build_aggregate_matrix.py --list   # print TF order for SLURM array
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
from scipy import sparse

ROOT = Path("/work/users/d/y/dyy12/XuLab")
OUT = ROOT / "downstream" / "tf_hitag_pearson" / "all_tfs_aggregate"
BINS = ROOT / "downstream" / "ctcf_hitag_pearson" / "bins_10kb.bed"
TAG = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig")
SA = Path("/users/x/u/xujie/HEK293T_SA")
BULK_DIR = ROOT / "downstream" / "tracks" / "bulk_chipseq"
SMOOTH_BINS = 5


def discover_tfs() -> list[str]:
    tfs = []
    for p in sorted((ROOT / "unified" / "work").glob("*/impute/matrix_csr.npz")):
        tfs.append(p.parent.parent.name)
    return tfs


def load_bins(path: Path):
    chroms, starts, ends = [], [], []
    with path.open() as fh:
        for line in fh:
            p = line.split()
            chroms.append(p[0])
            starts.append(int(p[1]))
            ends.append(int(p[2]))
    return np.array(chroms), np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def score_bw(path: Path, chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    bw = pyBigWig.open(str(path))
    if bw is None:
        raise FileNotFoundError(path)
    out = np.full(chroms.size, np.nan)
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[str(c)].append(i)
    for c, idx in by.items():
        if c not in bw.chroms():
            continue
        for i in idx:
            v = bw.stats(c, int(starts[i]), int(ends[i]), type="mean")
            if v is not None and v[0] is not None:
                out[i] = float(v[0])
    bw.close()
    return out


def parse_region(tok: str) -> tuple[str, int, int]:
    chrom, se = tok.split(":", 1)
    s, e = se.split("-", 1)
    return chrom, int(s), int(e)


def smooth_1kb(chroms: list[str], starts: np.ndarray, scores: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(scores)
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    k = np.ones(w, dtype=float) / w
    for c, idx in by.items():
        idx = np.asarray(idx, dtype=np.int64)
        order = np.argsort(starts[idx])
        idx = idx[order]
        st = starts[idx]
        if len(st) == 0:
            continue
        lo, hi = int(st[0]), int(st[-1])
        nslot = (hi - lo) // 1000 + 1
        grid = np.zeros(nslot, dtype=float)
        grid[(st - lo) // 1000] = scores[idx]
        sm = np.convolve(grid, k, mode="same")
        out[idx] = sm[(st - lo) // 1000]
    return out


def reduce_on_10kb(chroms, starts, ends, scores, q_chrom, q_start, q_end, how: str):
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    grids = {}
    for c, idx in by.items():
        idx = np.asarray(idx, dtype=np.int64)
        st = starts[idx]
        lo = int(st.min())
        hi = int(ends[idx].max())
        nslot = max(1, (hi - lo) // 1000)
        grid = np.zeros(nslot, dtype=float)
        rel = np.clip((st - lo) // 1000, 0, nslot - 1)
        grid[rel] = scores[idx]
        grids[c] = (lo, grid)
    out = np.full(len(q_chrom), np.nan)
    for i, c in enumerate(q_chrom):
        rec = grids.get(str(c))
        if rec is None:
            continue
        lo, grid = rec
        a = max(0, (int(q_start[i]) - lo) // 1000)
        b = max(a + 1, (int(q_end[i]) - lo + 999) // 1000)
        b = min(len(grid), b)
        if a >= len(grid):
            continue
        sl = grid[a:b]
        out[i] = float(sl.sum() if how == "sum" else sl.mean())
    return out


def imputed_aggregate_log1p(tf: str, q_chrom, q_start, q_end) -> np.ndarray:
    imp_dir = ROOT / "unified" / "work" / tf.lower() / "impute"
    sm = sparse.load_npz(imp_dir / "matrix_csr.npz")
    scores_1kb = np.asarray(sm.sum(axis=1)).ravel().astype(float)
    regs = [parse_region(ln) for ln in (imp_dir / "regions.tsv").read_text().splitlines() if ln.strip()]
    chroms = [r[0] for r in regs]
    starts = np.array([r[1] for r in regs], dtype=np.int64)
    ends = np.array([r[2] for r in regs], dtype=np.int64)
    smooth = smooth_1kb(chroms, starts, scores_1kb, SMOOTH_BINS)
    agg10 = reduce_on_10kb(chroms, starts, ends, smooth, q_chrom, q_start, q_end, "mean")
    return np.log1p(np.clip(agg10, 0, None))


def write_tsv(path: Path, labels: list[str], R: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("track\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{R[i, j]:.4f}" for j in range(len(labels))) + "\n")
    print(f"wrote {path}", flush=True)


def plot_matrix(tsv: Path, png: Path, title: str) -> None:
    lines = tsv.read_text().splitlines()
    labels = lines[0].split("\t")[1:]
    R = np.array([[float(x) for x in ln.split("\t")[1:]] for ln in lines[1:]], dtype=float)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.05 * n + 2.2, 1.05 * n + 1.4))
    im = ax.imshow(R, cmap="YlOrRd", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=("white" if v >= 0.65 else "black"), fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print(f"wrote {png}", flush=True)


def resolve_tracks(tf_up: str) -> list[tuple[str, Path]]:
    """Required: HiTAG_aggregate. Optional: Chromnitron, Bulk."""
    rows: list[tuple[str, Path]] = []
    agg = TAG / "mtx2bw" / f"{tf_up}_HiTAG_unweighted_smooth.bw"
    if not agg.is_file():
        raise FileNotFoundError(f"missing HiTAG aggregate: {agg}")
    rows.append(("HiTAG_aggregate", agg))

    chrom = SA / tf_up / "processed" / "data.bigwig"
    if chrom.is_file():
        rows.append(("Chromnitron", chrom))
    else:
        print(f"  [skip] Chromnitron missing: {chrom}", flush=True)

    bulk = BULK_DIR / f"{tf_up}_bulk_CPM.bw"
    if bulk.is_file():
        rows.append(("Bulk", bulk))
    else:
        print(f"  [skip] Bulk missing: {bulk}", flush=True)
    return rows


def run_tf(tf: str) -> None:
    tf_lo = tf.lower()
    tf_up = tf.upper()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== {tf_up} ===", flush=True)
    chroms, starts, ends = load_bins(BINS)
    labels: list[str] = []
    cols: list[np.ndarray] = []

    for lab, path in resolve_tracks(tf_up):
        print(f"  score {lab} {path}", flush=True)
        v = score_bw(path, chroms, starts, ends)
        print(f"    finite={np.isfinite(v).sum():,} max={np.nanmax(v):.4g}", flush=True)
        labels.append(lab)
        cols.append(v)

    print("  imputed aggregate log1p", flush=True)
    imp = imputed_aggregate_log1p(tf_lo, chroms, starts, ends)
    print(f"    finite={np.isfinite(imp).sum():,} max={np.nanmax(imp):.4g}", flush=True)
    labels.append("Imputed_aggregate")
    cols.append(imp)

    X = np.column_stack(cols)
    keep = np.isfinite(X).all(axis=1)
    Xk = X[keep]
    print(f"  complete bins {int(keep.sum()):,} / {len(keep):,}", flush=True)
    if Xk.shape[0] < 1000:
        raise RuntimeError(f"too few complete bins: {Xk.shape[0]}")

    np.savez_compressed(
        OUT / f"{tf_up}_signals_10kb.npz",
        chrom=chroms, start=starts, end=ends, **dict(zip(labels, cols)),
    )
    R = np.corrcoef(Xk, rowvar=False)
    stem = f"{tf_up}_pearson_10kb"
    tsv = OUT / f"{stem}.tsv"
    write_tsv(tsv, labels, R)
    plot_matrix(tsv, OUT / f"{stem}.png", f"{tf_up}  Pearson r, 10 kb bins")
    # key pairwise print
    for a in labels:
        for b in labels:
            if a >= b:
                continue
            print(f"  {a} vs {b}: {R[labels.index(a), labels.index(b)]:.4f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tf", nargs="?", help="TF name (case-insensitive)")
    ap.add_argument("--list", action="store_true", help="print discovered TF order")
    ap.add_argument("--index", type=int, help="SLURM array index into --list order")
    args = ap.parse_args()
    tfs = discover_tfs()
    if args.list:
        for i, t in enumerate(tfs):
            print(f"{i}\t{t}")
        return 0
    if args.index is not None:
        if args.index < 0 or args.index >= len(tfs):
            raise SystemExit(f"index {args.index} out of range 0-{len(tfs)-1}")
        run_tf(tfs[args.index])
        return 0
    if not args.tf:
        raise SystemExit("pass a TF name, --index N, or --list")
    run_tf(args.tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
