#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 11_per_cell_fragmentation.py
#
# Per-cell view of how many regions are "lit up" before vs. after imputation.
# For binarised scATAC-seq this is the per-cell fragment count (number of
# accessible bins). After cisTopic imputation every (region, cell) carries a
# positive P(r|c), so we count entries above a series of thresholds.
#
# Outputs (under <work_dir>/eval/):
#   <out_name>.tsv   per-cell counts: original nnz, imputed nnz at each threshold
#   <out_name>.json  summary (median / mean / min / max / fold-vs-original)
#   <out_name>.png   overlaid histogram + CDF of per-cell fragmentation
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths  # noqa: E402

log = logging.getLogger("11_frag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [x.rstrip("\n") for x in fh]


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--imputed-h5", type=str, default=None,
                    help="Path to imputed Prc HDF5 (default: paths.impute/imputed_Prc_hdf5_all.h5).")
    ap.add_argument("--mm-dir", type=str, default=None,
                    help="Directory with original matrix.mtx.gz + barcodes.tsv.gz (default: paths.mm).")
    ap.add_argument("--thresholds", type=str,
                    default="1e-7,1e-6,3e-6,1e-5,3e-5,1e-4",
                    help="Comma-separated thresholds in P(r|c) probability space.")
    ap.add_argument("--chunk-rows", type=int, default=10000)
    ap.add_argument("--out-name", type=str, default="per_cell_fragmentation")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    eval_dir = Path(cfg["paths"]["work_dir"]) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    mm_dir = Path(args.mm_dir) if args.mm_dir else Path(cfg["paths"]["mm"])
    mtx_path = mm_dir / "matrix.mtx.gz"
    bar_path = mm_dir / "barcodes.tsv.gz"
    for p in (mtx_path, bar_path):
        if not p.exists():
            log.error("Missing %s", p)
            return 1

    h5_path = (Path(args.imputed_h5) if args.imputed_h5
               else Path(cfg["paths"]["impute"]) / "imputed_Prc_hdf5_all.h5")
    if not h5_path.exists():
        log.error("Missing imputed HDF5: %s", h5_path)
        return 1

    thresholds = np.array(
        sorted(float(x) for x in args.thresholds.split(",") if x.strip()),
        dtype=np.float64,
    )

    # -- original per-cell stats --------------------------------------------
    log.info("Reading original matrix %s", mtx_path)
    orig = sio.mmread(mtx_path).tocsc()  # regions x cells
    orig_bars = read_lines_gz(bar_path)
    if orig.shape[1] != len(orig_bars):
        log.error("Original mtx columns (%d) != barcodes (%d).", orig.shape[1], len(orig_bars))
        return 1

    # CSC indptr diff is per-column nnz (number of nonzero regions per cell).
    orig_per_cell_nnz = np.diff(orig.indptr).astype(np.int64)
    orig_total_per_cell = np.asarray(orig.sum(axis=0)).ravel().astype(np.int64)
    log.info(
        "Original: %d regions x %d cells, nnz=%d. Per-cell nonzero regions: "
        "median=%d, mean=%.1f, min=%d, max=%d",
        *orig.shape, int(orig.nnz),
        int(np.median(orig_per_cell_nnz)), float(orig_per_cell_nnz.mean()),
        int(orig_per_cell_nnz.min()), int(orig_per_cell_nnz.max()),
    )

    # -- imputed per-cell stats ---------------------------------------------
    log.info("Reading imputed %s", h5_path)
    with h5py.File(h5_path, "r") as h5:
        ds = h5["Prc"]
        Rf, Cf = ds.shape

        if "barcodes" in h5:
            imp_bars = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in h5["barcodes"][:]]
        else:
            log.warning("HDF5 has no 'barcodes' dataset; assuming column order matches mm/barcodes.tsv.gz.")
            if Cf != len(orig_bars):
                log.error("Cannot align: HDF5 cols=%d vs mm barcodes=%d.", Cf, len(orig_bars))
                return 1
            imp_bars = list(orig_bars)

        # Detect stored scaling using the entire matrix (chunked column-sum
        # mean). Sub-sampling rows underestimates scale for the new full-shape
        # HDF5 because most rows are zero-padded.
        log.info("Scanning HDF5 to detect scale_factor and accumulate per-cell stats...")
        col_sums = np.zeros(Cf, dtype=np.float64)
        for i0 in range(0, Rf, args.chunk_rows):
            i1 = min(Rf, i0 + args.chunk_rows)
            col_sums += ds[i0:i1, :].sum(axis=0, dtype=np.float64)
        scale = float(col_sums.mean())
        if scale <= 0:
            log.warning("Detected scale <= 0; falling back to 1.0.")
            scale = 1.0
        log.info("Detected per-cell sum mean (HDF5 scale): %.4g", scale)

        # Pre-scale thresholds into HDF5-stored space so we can compare
        # float32 chunks directly without dividing every entry.
        threshold_h5 = (thresholds * scale).astype(np.float32)
        T = len(thresholds)
        per_cell_nnz = np.zeros((T, Cf), dtype=np.int64)

        for i0 in range(0, Rf, args.chunk_rows):
            i1 = min(Rf, i0 + args.chunk_rows)
            block = ds[i0:i1, :]  # float32
            for ti, t_h5 in enumerate(threshold_h5):
                per_cell_nnz[ti] += np.count_nonzero(block > t_h5, axis=0)
            if (i0 // args.chunk_rows) % 20 == 0:
                log.info("  rows %d/%d", i1, Rf)

    # -- align imputed cells to original cell positions ---------------------
    orig_lookup = {b: i for i, b in enumerate(orig_bars)}
    missing = [b for b in imp_bars if b not in orig_lookup]
    if missing:
        log.error("%d imputed barcodes not present in original (showing 3): %s",
                  len(missing), missing[:3])
        return 1
    align_idx = np.array([orig_lookup[b] for b in imp_bars], dtype=np.int64)
    orig_nnz_aligned = orig_per_cell_nnz[align_idx]
    orig_total_aligned = orig_total_per_cell[align_idx]
    n_dropped = len(orig_bars) - Cf
    if n_dropped:
        log.info("Note: %d cells were dropped during 02_build filtering.", n_dropped)

    # -- TSV ---------------------------------------------------------------
    tsv_path = eval_dir / f"{args.out_name}.tsv"
    log.info("Writing %s", tsv_path)
    with open(tsv_path, "w") as fh:
        cols = (["barcode", "orig_nonzero_regions", "orig_total_counts", "imputed_sum"]
                + [f"imputed_nnz_t={t:g}" for t in thresholds])
        fh.write("\t".join(cols) + "\n")
        for c in range(Cf):
            row = [imp_bars[c],
                   str(int(orig_nnz_aligned[c])),
                   str(int(orig_total_aligned[c])),
                   f"{col_sums[c]:.6g}"]
            row += [str(int(per_cell_nnz[ti, c])) for ti in range(T)]
            fh.write("\t".join(row) + "\n")

    # -- JSON summary ------------------------------------------------------
    def _summary(v: np.ndarray) -> dict:
        return {
            "median": int(np.median(v)),
            "mean":   float(v.mean()),
            "min":    int(v.min()),
            "max":    int(v.max()),
            "p10":    int(np.percentile(v, 10)),
            "p90":    int(np.percentile(v, 90)),
        }

    orig_med = max(int(np.median(orig_nnz_aligned)), 1)
    summary = {
        "imputed_h5": str(h5_path),
        "scale_in_h5": scale,
        "n_cells": int(Cf),
        "n_regions_full": int(Rf),
        "n_cells_dropped_during_filter": int(n_dropped),
        "original_per_cell_nonzero": _summary(orig_nnz_aligned),
        "imputed_per_cell_at_threshold": [
            {
                "threshold": float(t),
                **_summary(per_cell_nnz[ti]),
                "fold_vs_original_median": float(np.median(per_cell_nnz[ti]) / orig_med),
            }
            for ti, t in enumerate(thresholds)
        ],
    }
    json_path = eval_dir / f"{args.out_name}.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Wrote %s", json_path)

    # -- plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    upper = max(int(orig_nnz_aligned.max()), int(per_cell_nnz.max()), 10)
    bins = np.logspace(0, np.log10(upper), 60)

    axes[0].hist(np.clip(orig_nnz_aligned, 1, None), bins=bins,
                 alpha=0.55, label="original", color="black")
    cmap = plt.get_cmap("viridis")
    for ti, t in enumerate(thresholds):
        axes[0].hist(np.clip(per_cell_nnz[ti], 1, None), bins=bins,
                     alpha=0.45, label=f"imputed > {t:g}",
                     color=cmap(ti / max(len(thresholds) - 1, 1)))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("nonzero regions per cell")
    axes[0].set_ylabel("# cells")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Per-cell fragmentation: original vs imputed")

    s_orig = np.sort(orig_nnz_aligned)
    axes[1].plot(np.clip(s_orig, 1, None),
                 np.linspace(0, 1, len(s_orig), endpoint=False),
                 label="original", color="black")
    for ti, t in enumerate(thresholds):
        s = np.sort(per_cell_nnz[ti])
        axes[1].plot(np.clip(s, 1, None),
                     np.linspace(0, 1, len(s), endpoint=False),
                     label=f"imputed > {t:g}",
                     color=cmap(ti / max(len(thresholds) - 1, 1)))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("nonzero regions per cell")
    axes[1].set_ylabel("CDF (fraction of cells)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("CDF: per-cell fragmentation")

    fig.suptitle(f"Per-cell fragmentation  (n_cells={Cf}, R_full={Rf})")
    fig.tight_layout()
    png_path = eval_dir / f"{args.out_name}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", png_path)

    # console quick-look
    log.info("Original median per-cell nnz: %d", orig_med)
    for ti, t in enumerate(thresholds):
        med = int(np.median(per_cell_nnz[ti]))
        log.info("  threshold %.0e -> imputed median nnz = %d (%.1fx original)",
                 t, med, med / orig_med)
    return 0


if __name__ == "__main__":
    sys.exit(main())
