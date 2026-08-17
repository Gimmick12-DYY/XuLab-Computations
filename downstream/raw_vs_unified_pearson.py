#!/usr/bin/env python3
"""Pearson correlation between raw mm and unified-imputed matrices per TF.

Primary metrics (always):
  r_bin_totals   corr(raw row-sums, imputed row-sums) across all genomic bins
  r_cell_totals  corr(raw col-sums, imputed col-sums) across all cells

Profile metrics (on final pos/neg bins when available, else random bin sample):
  r_per_cell_median / mean   median/mean of per-cell Pearson across bins
  r_per_bin_median  / mean   median/mean of per-bin Pearson across cells

Raw is used as counts; imputed is typically binary (0/1). For profile metrics
raw is binarized (>0) so both vectors live on the same scale.

Usage:
  python3 downstream/raw_vs_unified_pearson.py
  python3 downstream/raw_vs_unified_pearson.py --tfs znf282 maz
  python3 downstream/raw_vs_unified_pearson.py --out downstream/raw_vs_unified_pearson.tsv
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

DEFAULT_TFS = [
    "ctcf", "maz", "nfya", "zbtb7a", "zic2", "znf777", "znf282",
]


def _read_lines(path: Path) -> list[str]:
    op = gzip.open if path.suffix == ".gz" or path.name.endswith(".gz") else open
    with op(path, "rt") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_aligned(mm_dir: Path, impute_dir: Path) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    raw = sio.mmread(str(mm_dir / "matrix.mtx.gz")).tocsr()
    imp = sp.load_npz(impute_dir / "matrix_csr.npz").tocsr()
    if raw.shape != imp.shape:
        raise SystemExit(f"shape mismatch raw{raw.shape} vs impute{imp.shape} in {mm_dir}")
    # barcodes should already match; cheap sanity check on counts
    n_raw = len(_read_lines(mm_dir / "barcodes.tsv.gz"))
    n_imp = len(_read_lines(impute_dir / "barcodes.tsv"))
    if n_raw != raw.shape[1] or n_imp != imp.shape[1]:
        raise SystemExit(
            f"barcode count mismatch: mm={n_raw}/{raw.shape[1]} impute={n_imp}/{imp.shape[1]}"
        )
    return raw, imp


def _final_bin_idx(tf: str) -> np.ndarray | None:
    tsv = ROOT / f"bins_{tf}_final" / "pos_neg_bins.tsv"
    if not tsv.exists():
        return None
    idx = []
    with tsv.open() as f:
        header = f.readline()
        cols = header.rstrip("\n").split("\t")
        # Prefer post-blacklist index if present, else first index-like col
        if "bin_idx_post_blacklist" in cols:
            j = cols.index("bin_idx_post_blacklist")
        elif "bin_idx_orig" in cols:
            j = cols.index("bin_idx_orig")
        else:
            j = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > j and parts[j]:
                idx.append(int(parts[j]))
    if not idx:
        return None
    return np.unique(np.asarray(idx, dtype=np.int64))


def _per_axis_pearson(
    a: sp.csr_matrix,
    b: sp.csr_matrix,
    axis: int,
    sample_n: int | None,
    seed: int,
) -> np.ndarray:
    """Pearson along rows (axis=1) or columns (axis=0) for two aligned sparse mats.

    Densifies one vector at a time over the shared feature length of that axis.
    For axis=1 (per-row / per-bin): vectors length = n_cols (cells) — cheap.
    For axis=0 (per-col / per-cell): vectors length = n_rows (bins) — only use
    on a row-subset matrix.
    """
    rng = np.random.default_rng(seed)
    if axis == 1:
        # per bin across cells
        a = a.tocsr()
        b = b.tocsr()
        n = a.shape[0]
        out = np.full(n, np.nan, dtype=np.float64)
        take = np.arange(n)
        if sample_n is not None and 0 < sample_n < n:
            take = np.sort(rng.choice(n, size=sample_n, replace=False))
        for i in take:
            out[i] = _safe_pearson(a.getrow(i).toarray().ravel(), b.getrow(i).toarray().ravel())
        return out
    # per cell across bins
    a = a.tocsc()
    b = b.tocsc()
    n = a.shape[1]
    out = np.full(n, np.nan, dtype=np.float64)
    take = np.arange(n)
    if sample_n is not None and 0 < sample_n < n:
        take = np.sort(rng.choice(n, size=sample_n, replace=False))
    for j in take:
        out[j] = _safe_pearson(a.getcol(j).toarray().ravel(), b.getcol(j).toarray().ravel())
    return out


def analyze_tf(tf: str, work_root: Path, seed: int, bin_sample: int) -> dict:
    mm = work_root / tf / "mm"
    impute = work_root / tf / "impute"
    if not (mm / "matrix.mtx.gz").exists():
        raise SystemExit(f"[{tf}] missing {mm / 'matrix.mtx.gz'}")
    if not (impute / "matrix_csr.npz").exists():
        raise SystemExit(f"[{tf}] missing {impute / 'matrix_csr.npz'}")

    print(f"[{tf}] loading matrices...", flush=True)
    raw, imp = _load_aligned(mm, impute)
    n_bins, n_cells = raw.shape

    raw_bin_tot = np.asarray(raw.sum(axis=1)).ravel()
    imp_bin_tot = np.asarray(imp.sum(axis=1)).ravel()
    raw_cell_tot = np.asarray(raw.sum(axis=0)).ravel()
    imp_cell_tot = np.asarray(imp.sum(axis=0)).ravel()

    out: dict = {
        "tf": tf,
        "n_bins": n_bins,
        "n_cells": n_cells,
        "raw_nnz": int(raw.nnz),
        "impute_nnz": int(imp.nnz),
        "r_bin_totals": _safe_pearson(raw_bin_tot, imp_bin_tot),
        "r_cell_totals": _safe_pearson(raw_cell_tot, imp_cell_tot),
    }

    # Profile metrics on final bins (or sampled bins)
    bin_idx = _final_bin_idx(tf)
    scope = "final_posneg"
    if bin_idx is None:
        scope = f"random_{bin_sample}"
        rng = np.random.default_rng(seed)
        bin_idx = np.sort(rng.choice(n_bins, size=min(bin_sample, n_bins), replace=False))
    else:
        # drop out-of-range just in case
        bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]

    out["profile_scope"] = scope
    out["n_profile_bins"] = int(bin_idx.size)

    print(
        f"[{tf}] profile on {bin_idx.size} bins ({scope}); "
        f"per-cell + per-bin Pearson...",
        flush=True,
    )
    raw_sub = (raw[bin_idx] > 0).astype(np.float32)
    imp_sub = (imp[bin_idx] > 0).astype(np.float32)

    per_cell = _per_axis_pearson(raw_sub, imp_sub, axis=0, sample_n=None, seed=seed)
    per_bin = _per_axis_pearson(raw_sub, imp_sub, axis=1, sample_n=None, seed=seed)

    def _summ(arr: np.ndarray, prefix: str) -> None:
        ok = arr[np.isfinite(arr)]
        out[f"{prefix}_n"] = int(ok.size)
        out[f"{prefix}_median"] = float(np.median(ok)) if ok.size else float("nan")
        out[f"{prefix}_mean"] = float(np.mean(ok)) if ok.size else float("nan")

    _summ(per_cell, "r_per_cell")
    _summ(per_bin, "r_per_bin")
    ok_c = per_cell[np.isfinite(per_cell)]
    out["frac_cells_r_gt_0.99"] = float(np.mean(ok_c > 0.99)) if ok_c.size else float("nan")

    # Also: Pearson of bin totals restricted to profile bins
    out["r_bin_totals_profile"] = _safe_pearson(raw_bin_tot[bin_idx], imp_bin_tot[bin_idx])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tfs", nargs="*", default=None, help="TF ids (lowercase). Default: slide panel.")
    ap.add_argument(
        "--work-root",
        type=Path,
        default=REPO / "unified" / "work",
        help="Root containing <tf>/{mm,impute}",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "raw_vs_unified_pearson.tsv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--bin-sample",
        type=int,
        default=20000,
        help="If final bins missing, sample this many bins for profile metrics.",
    )
    args = ap.parse_args()
    tfs = [t.lower() for t in (args.tfs or DEFAULT_TFS)]

    rows = []
    for tf in tfs:
        try:
            rows.append(analyze_tf(tf, args.work_root, args.seed, args.bin_sample))
        except SystemExit as e:
            print(f"ERROR {e}", file=sys.stderr)
            return 1

    cols = [
        "tf", "n_bins", "n_cells", "raw_nnz", "impute_nnz",
        "r_bin_totals", "r_cell_totals", "r_bin_totals_profile",
        "profile_scope", "n_profile_bins",
        "r_per_cell_n", "r_per_cell_median", "r_per_cell_mean", "frac_cells_r_gt_0.99",
        "r_per_bin_n", "r_per_bin_median", "r_per_bin_mean",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r[c]:.6g}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in cols
            ) + "\n")

    # Human-readable summary
    print("\n=== Pearson raw vs unified ===")
    hdr = (
        f"{'TF':<8} {'r_binTot':>9} {'r_cellTot':>9} "
        f"{'r_cellMed':>9} {'r_binMed':>9} {'f_r>0.99':>8}  scope"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['tf']:<8} {r['r_bin_totals']:9.4f} {r['r_cell_totals']:9.4f} "
            f"{r['r_per_cell_median']:9.4f} {r['r_per_bin_median']:9.4f} "
            f"{r['frac_cells_r_gt_0.99']:8.3f}  "
            f"{r['profile_scope']} (n={r['n_profile_bins']})"
        )
    print(f"\nWrote {args.out}")

    # JSON sidecar for plotting later
    js = args.out.with_suffix(".json")
    js.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
