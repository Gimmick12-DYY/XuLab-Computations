#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 08_compare_imputed.py
#
# Side-by-side comparison of the cisTopic-imputed accessibility P(r|c) against
# the *original* observed matrix (the one written by 01_export_rds_to_mm.R).
#
# Two reporting spaces:
#
#   1. filtered space: regions that survived the 02_build filter
#      (e.g. 597,236 x 10,410). This matches what LDA actually modelled and is
#      what the cisTopic paper evaluates on.
#
#   2. lifted space: original universe (e.g. 3,031,053 x 10,410). The imputed
#      matrix is mapped back into the full row index by region name; rows that
#      were filtered out are treated as predicted-zero (i.e. uninformative).
#      This is the strict, "fair" view because both matrices have identical
#      dimensions.
#
# Outputs (under <work_dir>/eval/):
#   compare_imputed.json           summary metrics for both spaces
#   compare_imputed_regions.tsv    per-region AUROC / mean / nnz (filtered)
#   compare_imputed_cells.tsv      per-cell  AUROC / mean / nnz (filtered)
#   compare_imputed_summary.png    diagnostic figure
#
# Optional: --regions-bed <peaks.bed> restricts the comparison to a BED file
# of windows (e.g. ChIP-seq peaks for a TF). This is the hook for future
# TF-bin-wise analysis: each row of the original matrix that overlaps a peak
# is treated as a "TF bin", and metrics are reported on that subset.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import h5py  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths  # noqa: E402

log = logging.getLogger("08_compare")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# --- helpers -----------------------------------------------------------------
def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC that returns NaN for trivial vectors (all 0 or all 1)."""
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, s))


def _parse_bed(path: Path) -> list[tuple[str, int, int]]:
    """Tiny BED parser that ignores comment/track lines."""
    out: list[tuple[str, int, int]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln or ln.startswith(("#", "track", "browser")):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            out.append((parts[0], int(parts[1]), int(parts[2])))
    return out


def _bin_overlap_mask(
    region_names: list[str], bed: list[tuple[str, int, int]]
) -> np.ndarray:
    """Return a boolean mask over `region_names` (chr:start-end strings) marking
    bins that overlap any BED interval. O(N + M) via per-chromosome bucket sort."""
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for c, s, e in bed:
        by_chrom.setdefault(c, []).append((s, e))
    for c in by_chrom:
        by_chrom[c].sort()

    mask = np.zeros(len(region_names), dtype=bool)
    for i, name in enumerate(region_names):
        try:
            chrom, span = name.split(":")
            bs, be = span.split("-")
            bs_i, be_i = int(bs), int(be)
        except ValueError:
            continue
        ivs = by_chrom.get(chrom)
        if not ivs:
            continue
        # linear scan -- BEDs of TF peaks are typically <1e6 entries.
        for s, e in ivs:
            if s >= be_i:
                break
            if e > bs_i:
                mask[i] = True
                break
    return mask


def _per_axis_auroc(
    y_dense: np.ndarray,
    s_dense: np.ndarray,
    axis: int,
    sample_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """AUROC along `axis`. axis=1 -> per-row (region) over cells; axis=0 -> per-col (cell) over regions."""
    if axis == 1:
        n = y_dense.shape[0]
    elif axis == 0:
        n = y_dense.shape[1]
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if sample_n and sample_n < n:
        idx = rng.choice(n, size=sample_n, replace=False)
    else:
        idx = np.arange(n)
    out = np.full(n, np.nan, dtype=np.float64)
    for k in idx:
        if axis == 1:
            yk = y_dense[k]
            sk = s_dense[k]
        else:
            yk = y_dense[:, k]
            sk = s_dense[:, k]
        out[k] = _safe_auroc(yk, sk)
    return out


# --- main --------------------------------------------------------------------
def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--imputed-h5", type=str, default=None,
                    help="Path to imputed HDF5 (default: <impute>/imputed_Prc_hdf5_all.h5).")
    ap.add_argument("--mm-dir", type=str, default=None,
                    help="Directory containing the original matrix.mtx.gz / regions.tsv.gz / "
                         "barcodes.tsv.gz (default: paths.mm from config).")
    ap.add_argument("--regions-bed", type=str, default=None,
                    help="Optional BED file of windows (e.g. TF peaks). When given, all "
                         "metrics are restricted to bins that overlap the BED.")
    ap.add_argument("--region-sample", type=int, default=20000,
                    help="Number of regions to sub-sample for per-region AUROC (memory).")
    ap.add_argument("--cell-sample", type=int, default=0,
                    help="Number of cells to sub-sample for per-cell AUROC (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", type=str, default="compare_imputed")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    rng = np.random.default_rng(args.seed)

    mm_dir = Path(args.mm_dir) if args.mm_dir else Path(cfg["paths"]["mm"])
    mtx_path = mm_dir / "matrix.mtx.gz"
    reg_path = mm_dir / "regions.tsv.gz"
    bar_path = mm_dir / "barcodes.tsv.gz"
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            log.error("Missing original input: %s", p)
            return 1

    impute_dir = Path(cfg["paths"]["impute"])
    h5_path = Path(args.imputed_h5) if args.imputed_h5 else impute_dir / "imputed_Prc_hdf5_all.h5"
    if not h5_path.exists():
        log.error(
            "Imputed HDF5 not found: %s\n"
            "Run 05_impute.py with impute.mode set to hdf5_all or hdf5_variable first.",
            h5_path,
        )
        return 1

    # -- load original (regions x cells, sparse, binarised) -------------------
    log.info("Reading original matrix: %s", mtx_path)
    orig = sio.mmread(mtx_path).tocsr()
    orig_regs = _read_lines_gz(reg_path)
    orig_bars = _read_lines_gz(bar_path)
    if orig.shape != (len(orig_regs), len(orig_bars)):
        log.error("Shape mismatch: matrix %s vs regions/barcodes %s/%s",
                  orig.shape, len(orig_regs), len(orig_bars))
        return 1
    log.info("Original: %d regions x %d cells, nnz=%d (density=%.3g)",
             *orig.shape, orig.nnz, orig.nnz / float(orig.shape[0] * orig.shape[1]))
    # Binarise to match what LDA modelled.
    orig_bin = (orig > 0).astype(np.int8)

    # -- load imputed (filtered regions x cells, dense float32) ---------------
    log.info("Reading imputed HDF5: %s", h5_path)
    with h5py.File(h5_path, "r") as h5:
        imp_regs = [s.decode() if isinstance(s, bytes) else str(s) for s in h5["regions"][:]]
        imp_bars = [s.decode() if isinstance(s, bytes) else str(s) for s in h5["barcodes"][:]]
        # Read the entire dense matrix; for very large datasets we could chunk.
        Prc = h5["Prc"][...].astype(np.float32, copy=False)
        h5_attrs = dict(h5["Prc"].attrs) if hasattr(h5["Prc"], "attrs") else {}
        h5_attrs.update(dict(h5.attrs))
    log.info("Imputed: %s (%.1f GiB float32)", Prc.shape, Prc.nbytes / 1024**3)

    # -- align indices --------------------------------------------------------
    reg_to_orig = {r: i for i, r in enumerate(orig_regs)}
    bar_to_orig = {b: i for i, b in enumerate(orig_bars)}
    try:
        row_idx = np.array([reg_to_orig[r] for r in imp_regs], dtype=np.int64)
        col_idx = np.array([bar_to_orig[b] for b in imp_bars], dtype=np.int64)
    except KeyError as e:
        log.error("Imputed name not found in original: %s", e)
        return 1
    log.info("Imputed regions cover %d / %d of original (%.2f%%); cells cover %d / %d (%.2f%%).",
             len(imp_regs), len(orig_regs), 100.0 * len(imp_regs) / len(orig_regs),
             len(imp_bars), len(orig_bars), 100.0 * len(imp_bars) / len(orig_bars))

    # Original sub-matrix that exactly matches the imputed orientation/order.
    orig_aligned = orig_bin[row_idx][:, col_idx].toarray().astype(np.int8)
    assert orig_aligned.shape == Prc.shape

    # Optional BED subset (TF-bin-wise hook).
    if args.regions_bed:
        bed_path = Path(args.regions_bed)
        log.info("Restricting to BED windows: %s", bed_path)
        bed = _parse_bed(bed_path)
        full_mask = _bin_overlap_mask(orig_regs, bed)
        log.info("BED overlaps %d / %d of original bins (%.3f%%).",
                 int(full_mask.sum()), len(full_mask), 100.0 * full_mask.mean())
        # Filtered-space mask: only over imputed rows.
        imp_mask = full_mask[row_idx]
        log.info("BED overlaps %d / %d of imputed (filtered) bins (%.3f%%).",
                 int(imp_mask.sum()), len(imp_mask), 100.0 * imp_mask.mean())
        orig_aligned = orig_aligned[imp_mask]
        Prc = Prc[imp_mask]
        kept_imp_regs = [imp_regs[i] for i in np.where(imp_mask)[0]]
        bed_kept_full_rows = full_mask  # for lifted-space stats below
    else:
        kept_imp_regs = imp_regs
        bed_kept_full_rows = None

    # -- summary stats --------------------------------------------------------
    obs_density_filt = float(orig_aligned.mean())
    pred_mean_filt = float(Prc.mean())
    log.info("Filtered space: observed density=%.5f, predicted mean=%.5g",
             obs_density_filt, pred_mean_filt)

    # Sub-sample (region, cell) pairs for global AUROC/AUPRC to avoid OOM.
    R_f, C_f = orig_aligned.shape
    pair_n = min(2_000_000, R_f * C_f)
    pr = rng.integers(0, R_f, size=pair_n, dtype=np.int64)
    pc = rng.integers(0, C_f, size=pair_n, dtype=np.int64)
    yp = orig_aligned[pr, pc]
    sp_ = Prc[pr, pc]
    metrics_filt = {
        "n_regions": int(R_f),
        "n_cells": int(C_f),
        "observed_density": obs_density_filt,
        "predicted_mean": pred_mean_filt,
        "global_auroc": _safe_auroc(yp, sp_),
        "global_auprc": _safe_auprc(yp, sp_),
        "pearson_pairs": float(np.corrcoef(yp.astype(np.float32), sp_)[0, 1]),
        "n_pairs_sampled": int(pair_n),
    }

    # Per-region / per-cell AUROC (sub-sampled if asked).
    log.info("Per-region AUROC over %d regions (sample=%s)...",
             R_f, args.region_sample or "all")
    region_auroc = _per_axis_auroc(orig_aligned, Prc, axis=1,
                                   sample_n=args.region_sample, rng=rng)
    log.info("Per-cell   AUROC over %d cells (sample=%s)...",
             C_f, args.cell_sample or "all")
    cell_auroc = _per_axis_auroc(orig_aligned, Prc, axis=0,
                                 sample_n=args.cell_sample, rng=rng)

    metrics_filt["per_region_auroc_mean"] = float(np.nanmean(region_auroc))
    metrics_filt["per_region_auroc_median"] = float(np.nanmedian(region_auroc))
    metrics_filt["per_cell_auroc_mean"] = float(np.nanmean(cell_auroc))
    metrics_filt["per_cell_auroc_median"] = float(np.nanmedian(cell_auroc))

    # -- lifted-space metrics (zeros for filtered-out rows) -------------------
    # We avoid materialising the full 3M x 10K dense predicted matrix. Instead:
    # global AUROC over a uniform sample of (r, c) pairs from the *original*
    # universe. predicted = Prc[map(r), c] if r is imputed, else 0.0.
    log.info("Lifted-space sampling over the full %d-region universe...", orig.shape[0])
    R_o, C_o = orig.shape
    # If a BED was supplied, restrict the lifted sample to BED-overlapping rows
    # so that the comparison stays on the same bin set.
    if bed_kept_full_rows is not None:
        sample_pool = np.where(bed_kept_full_rows)[0]
        if sample_pool.size == 0:
            log.warning("BED produced no original-space bins; skipping lifted metrics.")
            metrics_lifted = None
        else:
            ridx = rng.choice(sample_pool, size=min(2_000_000, sample_pool.size * C_o),
                              replace=True)
            metrics_lifted = True
    else:
        ridx = rng.integers(0, R_o, size=2_000_000, dtype=np.int64)
        metrics_lifted = True

    if metrics_lifted:
        cidx = rng.integers(0, C_o, size=ridx.size, dtype=np.int64)
        # Original observed value at sampled pairs.
        y_lift = np.asarray(orig_bin[ridx, cidx]).ravel().astype(np.int8)
        # Predicted: 0 unless the row is in the imputed (filtered) set.
        # Build a full-size lookup: orig_row -> imputed row index, or -1.
        # IMPORTANT: BED restriction (when used) applied to imputed rows above,
        # so we must mirror it here -- restrict the lookup to kept_imp_regs.
        lookup = -np.ones(R_o, dtype=np.int64)
        if bed_kept_full_rows is not None:
            kept_mask = bed_kept_full_rows[row_idx]
            kept_orig_rows = row_idx[kept_mask]
        else:
            kept_orig_rows = row_idx
        # Prc rows are 0..len(kept_orig_rows)-1 in the order they survive any BED filter
        # (which mirrors how Prc was sub-indexed earlier with `Prc = Prc[imp_mask]`).
        lookup[kept_orig_rows] = np.arange(kept_orig_rows.size)
        col_lookup = -np.ones(C_o, dtype=np.int64)
        col_lookup[col_idx] = np.arange(len(imp_bars))

        ir = lookup[ridx]
        ic = col_lookup[cidx]
        s_lift = np.zeros(ridx.size, dtype=np.float32)
        ok = (ir >= 0) & (ic >= 0)
        s_lift[ok] = Prc[ir[ok], ic[ok]]
        metrics_lifted = {
            "n_regions": int(R_o if bed_kept_full_rows is None
                              else int(bed_kept_full_rows.sum())),
            "n_cells": int(C_o),
            "observed_density": float(orig.nnz) / float(R_o * C_o)
                                if bed_kept_full_rows is None
                                else float(np.asarray(
                                    orig_bin[np.where(bed_kept_full_rows)[0]].sum()
                                )) / float(int(bed_kept_full_rows.sum()) * C_o),
            "global_auroc": _safe_auroc(y_lift, s_lift),
            "global_auprc": _safe_auprc(y_lift, s_lift),
            "n_pairs_sampled": int(ridx.size),
            "fraction_pairs_zero_predicted": float((s_lift == 0).mean()),
        }

    # -- write tables / json / png -------------------------------------------
    eval_dir = Path(cfg["paths"]["work_dir"]) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_json = eval_dir / f"{args.out_name}.json"
    out_reg = eval_dir / f"{args.out_name}_regions.tsv"
    out_cell = eval_dir / f"{args.out_name}_cells.tsv"
    out_png = eval_dir / f"{args.out_name}_summary.png"

    summary = {
        "config": str(args.config),
        "mm_dir": str(mm_dir),
        "imputed_h5": str(h5_path),
        "regions_bed": args.regions_bed,
        "h5_attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in h5_attrs.items()},
        "filtered_space": metrics_filt,
        "lifted_space": metrics_lifted,
        "n_imputed_regions_kept": int(len(kept_imp_regs)),
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    with open(out_reg, "w") as fh:
        fh.write("region\tobs_density\tpred_mean\tauroc\n")
        # write a sample to keep the file small.
        sub = np.where(~np.isnan(region_auroc))[0]
        if sub.size > 50_000:
            sub = rng.choice(sub, size=50_000, replace=False)
        for i in sub:
            fh.write(f"{kept_imp_regs[i]}\t{orig_aligned[i].mean():.6f}\t"
                     f"{Prc[i].mean():.6g}\t{region_auroc[i]:.6f}\n")

    with open(out_cell, "w") as fh:
        fh.write("cell\tobs_density\tpred_mean\tauroc\n")
        for i in range(C_f):
            if np.isnan(cell_auroc[i]):
                continue
            fh.write(f"{imp_bars[i]}\t{orig_aligned[:, i].mean():.6f}\t"
                     f"{Prc[:, i].mean():.6g}\t{cell_auroc[i]:.6f}\n")

    # Diagnostic figure: AUROC distributions + observed-vs-pred density scatter.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    valid_r = region_auroc[~np.isnan(region_auroc)]
    valid_c = cell_auroc[~np.isnan(cell_auroc)]
    axes[0].hist(valid_r, bins=50, color="#3a7", alpha=0.85)
    axes[0].axvline(0.5, color="k", lw=0.6, ls=":")
    axes[0].set_title(f"Per-region AUROC (n={valid_r.size})")
    axes[0].set_xlabel("AUROC")
    axes[1].hist(valid_c, bins=50, color="#37a", alpha=0.85)
    axes[1].axvline(0.5, color="k", lw=0.6, ls=":")
    axes[1].set_title(f"Per-cell AUROC (n={valid_c.size})")
    axes[1].set_xlabel("AUROC")
    # Density-vs-pred scatter (region-level).
    obs_rd = orig_aligned.mean(axis=1)
    pred_rd = Prc.mean(axis=1)
    sub = rng.choice(obs_rd.size, size=min(20_000, obs_rd.size), replace=False)
    axes[2].scatter(obs_rd[sub], pred_rd[sub], s=2, alpha=0.4, color="#a33")
    axes[2].set_xlabel("Observed region density (binarised)")
    axes[2].set_ylabel("Predicted region mean P(r|c)")
    axes[2].set_title("Region-level: observed vs predicted")
    fig.suptitle(
        f"cisTopic imputation vs original  ({R_f}x{C_f} filtered, "
        f"AUROC={metrics_filt['global_auroc']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    log.info("Wrote: %s", out_json)
    log.info("Wrote: %s", out_reg)
    log.info("Wrote: %s", out_cell)
    log.info("Wrote: %s", out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
