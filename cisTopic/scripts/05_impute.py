#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Build cisTopic's imputed accessibility matrix as a drop-in replacement for
# the raw mm matrix:
#
#     imp[r, c] = (phi @ theta)[r, c] * scale_factor   if imp > threshold
#               = 0                                    otherwise
#
# The output is a scipy CSR at the FULL mm dimensions (n_mm_bins, n_mm_cells)
# so a downstream bigWig / browser workflow can treat it exactly like the raw
# matrix (same row/col labels, same shape). The threshold is picked INSIDE
# this step from the configured impute.threshold_mode (default
# `sparsity_match:2`, i.e. each pipeline calls 2x as many entries as raw saw).
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz   scipy CSR float32 (n_mm_bins, n_mm_cells), nnz only on
#                    modeled rows where imp > threshold
#   regions.tsv      full mm region order
#   barcodes.tsv     full mm barcode order
#   meta.json        threshold + origin + nnz + shape + rank_k + scale_factor
#   factors.npz      legacy phi + theta + per_cell_factor scalar
#                    (skipped iff impute.drop_factors_npz)
#   cell_topic_theta.{parquet,npy}, region_topic_phi.{parquet,npy}: legacy
#                    artefacts kept for backwards compatibility
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("05_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _parse_threshold_mode(s: str) -> tuple[str, float]:
    s = (s or "sparsity_match:1").strip()
    if s == "sparsity_match":
        return "sparsity_match", 1.0
    if s.startswith("sparsity_match:"):
        try:
            return "sparsity_match", float(s.split(":", 1)[1])
        except ValueError as e:
            raise SystemExit(f"Cannot parse multiplier from {s!r}: {e}")
    if s.startswith("quantile:"):
        try:
            q = float(s.split(":", 1)[1])
        except ValueError as e:
            raise SystemExit(f"Cannot parse quantile from {s!r}: {e}")
        if not (0.0 <= q <= 1.0):
            raise SystemExit(f"quantile must be in [0,1], got {q}")
        return "quantile", q
    if s.startswith("fixed:"):
        try:
            return "fixed", float(s.split(":", 1)[1])
        except ValueError as e:
            raise SystemExit(f"Cannot parse fixed threshold from {s!r}: {e}")
    raise SystemExit(
        f"Unknown impute.threshold_mode {s!r}; "
        f"expected sparsity_match[:K] | quantile:Q | fixed:T"
    )


def _sample_imp_values(phi: np.ndarray, theta: np.ndarray, scale: float,
                       n_sample: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n_sample imputed values uniformly from (modeled_row, cell) pairs.

    imp[r, c] = (phi[r] @ theta[:, c]) * scale.
    """
    R, _ = phi.shape
    _, C = theta.shape
    n_sample = int(min(int(n_sample), int(R) * int(C)))
    if n_sample <= 0:
        return np.zeros(0, dtype=np.float64)
    rows = rng.integers(0, R, size=n_sample, dtype=np.int64)
    cols = rng.integers(0, C, size=n_sample, dtype=np.int64)
    return np.einsum("ik,ki->i",
                     phi[rows].astype(np.float64),
                     theta[:, cols].astype(np.float64)) * float(scale)


def _pick_threshold(phi: np.ndarray, theta: np.ndarray,
                    per_cell_factor: float, mode: str, param: float,
                    target_nnz_modeled: int,
                    sample_size: int,
                    rng: np.random.Generator) -> tuple[float, str]:
    """Resolve the impute.threshold_mode to a numeric cutoff."""
    if mode == "fixed":
        return float(param), f"fixed:{float(param):.6g}"

    sample = _sample_imp_values(phi, theta, per_cell_factor, sample_size, rng)
    if sample.size == 0:
        return 0.0, "empty_sample"

    R, C = phi.shape[0], theta.shape[1]
    total_entries = int(R) * int(C)

    if mode == "sparsity_match":
        K = float(param)
        target = int(round(target_nnz_modeled * K))
        target = max(1, min(target, total_entries))
        target_fraction = target / total_entries
        q = max(0.0, min(1.0, 1.0 - target_fraction))
        t = float(np.quantile(sample, q))
        return t, (f"sparsity_match:{K} "
                   f"(target_calls={target}, q={q:.6g}, sample={sample.size})")

    if mode == "quantile":
        q = float(param)
        t = float(np.quantile(sample, q))
        return t, f"quantile:{q} -> t={t:.4g}"

    raise SystemExit(f"Unknown mode {mode!r}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--scale-factor", type=float, default=None)
    ap.add_argument("--no-rescale", action="store_true")
    ap.add_argument("--threshold-mode", type=str, default=None,
                    help="Override impute.threshold_mode "
                         "(sparsity_match[:K] | quantile:Q | fixed:T).")
    ap.add_argument("--no-align-to-mm", dest="align_to_mm",
                    action="store_false", default=None)
    ap.add_argument("--binarize-output", action="store_true", default=None,
                    help="Store 1 instead of the imputed magnitude at every "
                         "above-threshold entry.")
    ap.add_argument("--drop-factors-npz", action="store_true", default=None,
                    help="Skip saving the legacy factors.npz.")
    ap.add_argument("--chunk-rows", type=int, default=2000)
    ap.add_argument("--sample-size", type=int, default=None,
                    help="Random-sample size for threshold estimation.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp = cfg.setdefault("impute", {})
    if args.scale_factor is not None:
        imp["scale_factor"] = args.scale_factor
    if args.threshold_mode is not None:
        imp["threshold_mode"] = args.threshold_mode
    if args.align_to_mm is False:
        imp["align_to_mm"] = False
    if args.binarize_output is True:
        imp["binarize_output"] = True
    if args.drop_factors_npz is True:
        imp["drop_factors_npz"] = True
    if args.sample_size is not None:
        imp["threshold_sample_size"] = int(args.sample_size)

    desired_scale = float(imp.get("scale_factor", 1_000_000))
    threshold_mode_str = str(imp.get("threshold_mode", "sparsity_match:1"))
    align_to_mm = bool(imp.get("align_to_mm", True))
    binarize_output = bool(imp.get("binarize_output", False))
    drop_factors_npz = bool(imp.get("drop_factors_npz", False))
    sample_size = int(imp.get("threshold_sample_size", 4_000_000))
    rng = np.random.default_rng(int(args.seed))
    mode, mode_param = _parse_threshold_mode(threshold_mode_str)

    # ------------------------------------------------------------------
    # Load CistopicObject
    # ------------------------------------------------------------------
    obj_path = Path(cfg["paths"]["obj"]) / "cistopic_obj.pkl"
    if not obj_path.exists():
        log.error("CistopicObject not found: %s", obj_path)
        return 1
    with open(obj_path, "rb") as fh:
        cistopic_obj = pickle.load(fh)
    if cistopic_obj.selected_model is None:
        log.error("No model attached to the CistopicObject; "
                  "run 04_select_model.py first.")
        return 1

    theta = np.asarray(cistopic_obj.selected_model.cell_topic, dtype=np.float32)   # K x C
    phi   = np.asarray(cistopic_obj.selected_model.topic_region, dtype=np.float32) # R x K
    modeled_cells   = list(cistopic_obj.cell_names)
    modeled_regions = list(cistopic_obj.region_names)
    K = theta.shape[0]
    log.info("theta: %s (K x C), phi: %s (R x K)", theta.shape, phi.shape)

    out_dir = Path(cfg["paths"]["impute"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-cell factor (paper-faithful scalar). phi/theta are row/col stochastic
    # so sum_r (phi @ theta)[:, c] ~= 1; multiplying by scale_factor gives a
    # CPM-like reference scale.
    # ------------------------------------------------------------------
    if args.no_rescale:
        log.info("Storing factors verbatim (--no-rescale).")
        per_cell_factor = float(1.0)
        rescaled = False
    else:
        col_sums = (phi.sum(axis=0) @ theta).astype(np.float64)
        sum_per_cell_mean = float(col_sums.mean()) if col_sums.size else 0.0
        if sum_per_cell_mean <= 0:
            log.warning("phi @ theta per-cell sum mean <= 0; storing verbatim.")
            per_cell_factor = float(1.0)
            rescaled = False
        else:
            per_cell_factor = float(desired_scale / sum_per_cell_mean)
            log.info("Per-cell factor (scalar) = %.4g  (target scale=%.0f)",
                     per_cell_factor, desired_scale)
            rescaled = True

    # ------------------------------------------------------------------
    # mm alignment: figure out modeled -> mm row / col indices.
    # ------------------------------------------------------------------
    mm_dir = Path(cfg["paths"]["mm"])
    mm_reg_path = mm_dir / "regions.tsv.gz"
    mm_bar_path = mm_dir / "barcodes.tsv.gz"
    if align_to_mm and (not mm_reg_path.exists() or not mm_bar_path.exists()):
        log.error("align_to_mm: true but %s or %s missing.",
                  mm_reg_path, mm_bar_path)
        return 1
    if align_to_mm:
        full_regions  = _read_lines_gz(mm_reg_path)
        full_barcodes = _read_lines_gz(mm_bar_path)
        log.info("Full mm shape: %d regions x %d cells",
                 len(full_regions), len(full_barcodes))
        name_to_full_row = {r: i for i, r in enumerate(full_regions)}
        bar_to_full_col  = {b: i for i, b in enumerate(full_barcodes)}
        try:
            modeled_to_full_row = np.array(
                [name_to_full_row[r] for r in modeled_regions], dtype=np.int64)
            modeled_to_full_col = np.array(
                [bar_to_full_col[b] for b in modeled_cells], dtype=np.int64)
        except KeyError as e:
            log.error("Modeled region/cell %r missing from mm.", e)
            return 1
        n_full_rows, n_full_cols = len(full_regions), len(full_barcodes)
    else:
        full_regions  = list(modeled_regions)
        full_barcodes = list(modeled_cells)
        modeled_to_full_row = np.arange(len(modeled_regions), dtype=np.int64)
        modeled_to_full_col = np.arange(len(modeled_cells),   dtype=np.int64)
        n_full_rows, n_full_cols = len(full_regions), len(full_barcodes)

    # ------------------------------------------------------------------
    # Target call count for sparsity_match: raw nonzeros on the modeled rows
    # x modeled cells subspace. We read the raw mm to count.
    # ------------------------------------------------------------------
    target_nnz_modeled = 0
    if mode == "sparsity_match":
        mm_mtx = mm_dir / "matrix.mtx.gz"
        if not mm_mtx.exists():
            log.error("threshold_mode=sparsity_match requires %s.", mm_mtx)
            return 1
        log.info("Reading raw mm matrix for sparsity-match target count.")
        raw = sio.mmread(str(mm_mtx)).tocsr()
        raw = (raw > 0).astype(np.int8)
        if not align_to_mm:
            # If we didn't align names already, we still need them now.
            full_regions  = _read_lines_gz(mm_reg_path)
            full_barcodes = _read_lines_gz(mm_bar_path)
            name_to_full_row = {r: i for i, r in enumerate(full_regions)}
            bar_to_full_col  = {b: i for i, b in enumerate(full_barcodes)}
            modeled_to_full_row = np.array(
                [name_to_full_row[r] for r in modeled_regions], dtype=np.int64)
            modeled_to_full_col = np.array(
                [bar_to_full_col[b] for b in modeled_cells], dtype=np.int64)
        sub = raw[modeled_to_full_row][:, modeled_to_full_col]
        target_nnz_modeled = int(sub.nnz)
        log.info("Raw nonzeros on modeled subspace: %d", target_nnz_modeled)

    # ------------------------------------------------------------------
    # Pick threshold
    # ------------------------------------------------------------------
    threshold, threshold_origin = _pick_threshold(
        phi, theta, per_cell_factor,
        mode, mode_param,
        target_nnz_modeled,
        sample_size, rng,
    )
    log.info("Threshold = %.6g  (%s)", threshold, threshold_origin)

    # ------------------------------------------------------------------
    # Stream phi @ theta in row chunks; build a COO at full-mm shape with
    # only above-threshold entries.
    # ------------------------------------------------------------------
    Rm = phi.shape[0]
    chunk = max(1, int(args.chunk_rows))
    rows_acc: list[np.ndarray] = []
    cols_acc: list[np.ndarray] = []
    vals_acc: list[np.ndarray] = []
    n_kept = 0
    log.info("Streaming phi @ theta in chunks of %d rows ...", chunk)
    for s in range(0, Rm, chunk):
        e = min(s + chunk, Rm)
        block = (phi[s:e].astype(np.float32, copy=False) @ theta).astype(
            np.float32, copy=False) * np.float32(per_cell_factor)
        mask = block > threshold
        if not mask.any():
            continue
        local_rows, local_cols = np.where(mask)
        full_rows = modeled_to_full_row[s + local_rows]
        full_cols = modeled_to_full_col[local_cols]
        if binarize_output:
            vals = np.ones(local_rows.shape[0], dtype=np.float32)
        else:
            vals = block[local_rows, local_cols].astype(np.float32, copy=False)
        rows_acc.append(full_rows)
        cols_acc.append(full_cols)
        vals_acc.append(vals)
        n_kept += int(local_rows.size)

    if n_kept == 0:
        log.warning("No entries survived the threshold. Writing empty matrix.")
        data = np.zeros(0, dtype=np.float32)
        rows_full = np.zeros(0, dtype=np.int64)
        cols_full = np.zeros(0, dtype=np.int64)
    else:
        rows_full = np.concatenate(rows_acc)
        cols_full = np.concatenate(cols_acc)
        data      = np.concatenate(vals_acc)
    log.info("Kept %d nonzero entries (sparsity = %.3g) at threshold %.6g",
             n_kept, n_kept / max(int(n_full_rows) * int(n_full_cols), 1),
             threshold)

    csr = sparse.coo_matrix(
        (data, (rows_full, cols_full)),
        shape=(n_full_rows, n_full_cols), dtype=np.float32,
    ).tocsr()
    csr.sum_duplicates()

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    csr_path     = out_dir / "matrix_csr.npz"
    regions_out  = out_dir / "regions.tsv"
    barcodes_out = out_dir / "barcodes.tsv"
    meta_out     = out_dir / "meta.json"

    log.info("Writing %s (shape=%s, nnz=%d)", csr_path, csr.shape, csr.nnz)
    sparse.save_npz(csr_path, csr)
    regions_out.write_text("\n".join(full_regions) + "\n")
    barcodes_out.write_text("\n".join(full_barcodes) + "\n")

    # Legacy factors.npz: keep the soft predictive distribution available so
    # the "factored" loader downstream can still reconstruct raw P(r|c).
    factors_path = out_dir / "factors.npz"
    if drop_factors_npz:
        log.info("drop_factors_npz: true -> NOT writing %s", factors_path)
        if factors_path.exists():
            factors_path.unlink()
    else:
        log.info("Writing legacy %s", factors_path)
        np.savez_compressed(
            factors_path,
            W=phi,
            H=theta,
            per_cell_factor=np.float32(per_cell_factor),
        )

    # Legacy parquet/npy artefacts (unchanged).
    pd.DataFrame(theta, columns=modeled_cells,
                 index=[f"Topic{i+1}" for i in range(K)]).to_parquet(
                     out_dir / "cell_topic_theta.parquet")
    pd.DataFrame(phi, index=modeled_regions,
                 columns=[f"Topic{i+1}" for i in range(K)]).to_parquet(
                     out_dir / "region_topic_phi.parquet")
    np.save(out_dir / "cell_topic_theta.npy", theta)
    np.save(out_dir / "region_topic_phi.npy", phi)

    summary = {
        "pipeline":          "cisTopic",
        "format":            "csr_full_mm" if align_to_mm else "csr_modeled",
        "matrix_path":       str(csr_path),
        "regions_path":      str(regions_out),
        "barcodes_path":     str(barcodes_out),
        "shape":             [int(n_full_rows), int(n_full_cols)],
        "n_modeled":         int(Rm),
        "n_cells":           int(theta.shape[1]),
        "rank_k":            int(K),
        "scale_factor":      desired_scale,
        "per_cell_factor":   float(per_cell_factor),
        "rescaled":          bool(rescaled),
        "threshold":         float(threshold),
        "threshold_mode":    threshold_mode_str,
        "threshold_origin":  threshold_origin,
        "target_nnz_modeled_subspace": int(target_nnz_modeled),
        "nnz":               int(csr.nnz),
        "density":           float(csr.nnz / max(int(n_full_rows) * int(n_full_cols), 1)),
        "align_to_mm":       bool(align_to_mm),
        "binarize_output":   bool(binarize_output),
        "drop_factors_npz":  bool(drop_factors_npz),
        "factors_path":      None if drop_factors_npz else str(factors_path),
        "source_obj":        str(obj_path),
    }
    meta_out.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
