#!/usr/bin/env python3
"""SpQN on the 10 kb imputed-vs-predicted Pearson matrix.

Same Spatial Quantile Normalization as the compartment analysis (cell count
as the covariate, 10 groups, window 20, reference at 0.8 toward high cell
count). This matrix is not a TF-TF similarity: entry (i, j) is imputed_i vs
pred_j, so the result is not symmetrized and the diagonal is put back to the
original matched Pearson.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from build_bin1k_correlations import plot_matrix, write_tsv

ROOT = Path("/work/users/d/y/dyy12/XuLab")
SRC = ROOT / "downstream" / "tf_hitag_pearson" / "bin10kb" / "imputed_vs_pred_pearson.tsv"
META = ROOT / "data" / "TF1000cells.meta.csv"
OUT_TSV = SRC.with_name("imputed_vs_pred_spqn.tsv")
OUT_PNG = SRC.with_name("imputed_vs_pred_spqn.png")


def load_corr(path: Path):
    lines = path.read_text().splitlines()
    labels = lines[0].split("\t")[1:]
    rows, data = [], []
    for ln in lines[1:]:
        parts = ln.split("\t")
        rows.append(parts[0])
        data.append([float(x) for x in parts[1:]])
    if rows != labels:
        raise SystemExit("row and column TF order differ")
    return labels, np.array(data, dtype=float)


def cell_counts(path: Path) -> dict[str, int]:
    cc: dict[str, int] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            k = row["TF"].strip()
            cc[k] = cc.get(k, 0) + 1
    return cc


def spqn_keep_diag(R, feat_mean, n_group=10, w=20, ref_frac=0.8):
    """compartment_rpkm_correlation.spqn, without symmetrizing or resetting the diagonal."""
    n = R.shape[0]
    w = min(w, n)
    n_group = max(2, min(n_group, n - w + 1))
    order = np.argsort(feat_mean)
    Ro = R[np.ix_(order, order)].astype(float)
    dstep = (n - w) / (n_group - 1) if n_group > 1 else 0
    ystart = [int(round(k * dstep)) for k in range(n_group)]
    xbnd = np.linspace(0, n, n_group + 1).astype(int)

    def yvals(i, j):
        sub = Ro[ystart[i]:ystart[i] + w, ystart[j]:ystart[j] + w]
        mask = ~np.eye(*sub.shape, dtype=bool) if ystart[i] == ystart[j] else np.ones(sub.shape, bool)
        return np.sort(sub[mask].ravel())

    r = int(round(ref_frac * (n_group - 1)))
    ref = yvals(r, r)
    refq = np.linspace(0, 1, len(ref)) if len(ref) else np.array([0.0, 1.0])
    out = Ro.copy()
    for i in range(n_group):
        for j in range(n_group):
            Ys = yvals(i, j)
            if len(Ys) == 0:
                continue
            xi, xj = slice(xbnd[i], xbnd[i + 1]), slice(xbnd[j], xbnd[j + 1])
            block = Ro[xi, xj]
            q = np.searchsorted(Ys, block.ravel(), side="right") / len(Ys)
            out[xi, xj] = np.interp(q, refq, ref).reshape(block.shape)
    inv = np.argsort(order)
    out = out[np.ix_(inv, inv)]
    np.fill_diagonal(out, np.diag(R))
    return out


def row_col_confound(R, depth):
    off = R.copy()
    np.fill_diagonal(off, np.nan)
    row_mean = np.nanmean(off, axis=1)
    col_mean = np.nanmean(off, axis=0)
    return (
        float(np.corrcoef(depth, row_mean)[0, 1]),
        float(np.corrcoef(depth, col_mean)[0, 1]),
    )


def main() -> int:
    labels, R = load_corr(SRC)
    cc = cell_counts(META)
    missing = [t for t in labels if t not in cc]
    if missing:
        raise SystemExit(f"no cell count for: {', '.join(missing)}")
    depth = np.array([cc[t] for t in labels], dtype=float)
    Rn = spqn_keep_diag(R, depth)
    c_row, c_col = row_col_confound(R, depth)
    n_row, n_col = row_col_confound(Rn, depth)
    off = ~np.eye(len(labels), dtype=bool)
    print(f"n={len(labels)}", flush=True)
    print(f"before  diag_median={np.median(np.diag(R)):.3f}  "
          f"off_median={np.median(R[off]):.3f}  "
          f"confound row={c_row:+.3f} col={c_col:+.3f}", flush=True)
    print(f"after   diag_median={np.median(np.diag(Rn)):.3f}  "
          f"off_median={np.median(Rn[off]):.3f}  "
          f"confound row={n_row:+.3f} col={n_col:+.3f}", flush=True)
    print(f"range before {R.min():.3f}..{R.max():.3f}  after {Rn.min():.3f}..{Rn.max():.3f}", flush=True)
    write_tsv(OUT_TSV, labels, labels, Rn)
    plot_matrix(OUT_PNG, labels, labels, Rn, "Imputed vs Pred, SpQN", numbers=True)
    print(f"[done] {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
