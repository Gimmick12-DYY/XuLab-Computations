#!/usr/bin/env python3
"""Add a cell-sum + 5-bin-smooth imputed aggregate column and rebuild matrices."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

OUT = Path("/work/users/d/y/dyy12/XuLab/downstream/ctcf_hitag_pearson")
IMPUTE = Path("/work/users/d/y/dyy12/XuLab/unified/work/ctcf/impute")
SMOOTH_BINS = 5  # 1 kb bins; matches TAGATG unweighted_smooth (~5 kb box)


def parse_region(tok: str) -> tuple[str, int, int]:
    chrom, se = tok.split(":", 1)
    s, e = se.split("-", 1)
    return chrom, int(s), int(e)


def zscore(a: np.ndarray) -> np.ndarray:
    m = np.isfinite(a)
    o = np.full_like(a, np.nan, dtype=float)
    x = a[m]
    sd = x.std()
    o[m] = (x - x.mean()) / (sd if sd > 0 else 1.0)
    return o


def smooth_1kb(chroms: list[str], starts: np.ndarray, scores: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(scores)
    by = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    k = np.ones(w, dtype=float) / w
    for c, idx in by.items():
        idx = np.asarray(idx, dtype=np.int64)
        order = np.argsort(starts[idx])
        idx = idx[order]
        st = starts[idx]
        # fill gaps with 0 so the kernel is in genomic 1 kb steps
        if len(st) == 0:
            continue
        lo, hi = int(st[0]), int(st[-1])
        nslot = (hi - lo) // 1000 + 1
        grid = np.zeros(nslot, dtype=float)
        grid[(st - lo) // 1000] = scores[idx]
        sm = np.convolve(grid, k, mode="same")
        out[idx] = sm[(st - lo) // 1000]
    return out


def mean_on_10kb(chroms, starts, ends, scores, q_chrom, q_start, q_end):
    by = defaultdict(list)
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
        rel = (st - lo) // 1000
        rel = np.clip(rel, 0, nslot - 1)
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
        out[i] = float(grid[a:b].mean())
    return out


def corr_matrix(X: np.ndarray, method: str) -> np.ndarray:
    n = X.shape[1]
    R = np.eye(n)
    if method == "pearson":
        C = np.corrcoef(X, rowvar=False)
        return C
    ranks = np.column_stack([rankdata(X[:, j], method="average") for j in range(n)])
    return np.corrcoef(ranks, rowvar=False)


def write_tsv(path: Path, labels: list[str], R: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("track\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{R[i, j]:.4f}" for j in range(len(labels))) + "\n")
    print(f"wrote {path}")


def main() -> None:
    print("loading CSR cell-sum (unweighted aggregate at 1 kb)...")
    sm = sparse.load_npz(IMPUTE / "matrix_csr.npz")
    scores_1kb = np.asarray(sm.sum(axis=1)).ravel().astype(float)
    regs = [parse_region(ln) for ln in (IMPUTE / "regions.tsv").read_text().splitlines() if ln.strip()]
    chroms = [r[0] for r in regs]
    starts = np.array([r[1] for r in regs], dtype=np.int64)
    ends = np.array([r[2] for r in regs], dtype=np.int64)
    print(f"  {len(scores_1kb):,} 1 kb bins  nnz={(scores_1kb > 0).sum():,}")
    print(f"smoothing {SMOOTH_BINS} x 1 kb box (same as TAGATG unweighted_smooth)...")
    smooth = smooth_1kb(chroms, starts, scores_1kb, SMOOTH_BINS)

    z = np.load(OUT / "signals_partial.npz", allow_pickle=True)
    q_chrom, q_start, q_end = z["chrom"].astype(str), z["start"], z["end"]
    agg10 = mean_on_10kb(chroms, starts, ends, smooth, q_chrom, q_start, q_end)
    print(f"  10 kb aggregate  finite={np.isfinite(agg10).sum():,}  max={np.nanmax(agg10):.4g}")

    index = {(q_chrom[i], int(q_start[i]), int(q_end[i])): i for i in range(q_chrom.size)}
    bulk3 = np.full((q_chrom.size, 3), np.nan)
    with open(OUT / "consensus_bulk.tab") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            key = (p[0].strip("'"), int(p[1]), int(p[2]))
            i = index[key]
            for j in range(3):
                v = p[3 + j]
                bulk3[i, j] = np.nan if v in ("nan", "NaN", "") else float(v)
    bulk = np.nanmean(np.vstack([zscore(bulk3[:, j]) for j in range(3)]), axis=0)

    labels = [
        "Chromnitron",
        "Consensus_bulk",
        "HiTAG_weight",
        "HiTAG_macs2",
        "HiTAG_aggregate",
        "Imputed",
        "Imputed_aggregate",
    ]
    cols = {
        "Chromnitron": z["Chromnitron_pred2"].astype(float),
        "Consensus_bulk": bulk,
        "HiTAG_weight": z["HiTAG_weight"].astype(float),
        "HiTAG_macs2": z["HiTAG_macs2"].astype(float),
        "HiTAG_aggregate": z["HiTAG_aggregate"].astype(float),
        "Imputed": z["imputed"].astype(float),
        "Imputed_aggregate": agg10,
    }
    X = np.column_stack([cols[k] for k in labels])
    keep = np.isfinite(X).all(axis=1)
    Xk = X[keep]
    print(f"complete bins {keep.sum():,} / {len(keep):,}")
    for method, name in (("pearson", "CTCF_pearson_10kb"), ("spearman", "CTCF_spearman_10kb")):
        R = corr_matrix(Xk, method)
        write_tsv(OUT / f"{name}.tsv", labels, R)
        # print imputed rows
        ii = labels.index("Imputed")
        ia = labels.index("Imputed_aggregate")
        hm = labels.index("HiTAG_macs2")
        ha = labels.index("HiTAG_aggregate")
        print(f"  {method} Imputed vs MACS2 {R[ii, hm]:.3f}  vs HiTAG_agg {R[ii, ha]:.3f}")
        print(f"  {method} Imputed_aggregate vs MACS2 {R[ia, hm]:.3f}  vs HiTAG_agg {R[ia, ha]:.3f}")

    np.savez_compressed(OUT / "imputed_aggregate_10kb.npz", chrom=q_chrom, start=q_start, end=q_end, score=agg10)
    print("wrote imputed_aggregate_10kb.npz")


if __name__ == "__main__":
    main()
