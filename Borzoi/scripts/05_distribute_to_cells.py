#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_distribute_to_cells.py
#
# Bridge the per-bin Borzoi prior to a per-cell matrix.
#
# Mode B (default and only mode shipped today):
#   - For each bin b:
#       * if raw_csr[b].nnz >= observed_k : use cell_graph_csr[b] as-is
#         (the cell-axis posterior from downstream/cell_graph_kneighbour.py).
#       * else                            : use bin_score[b] * cell_factor[c]
#         for every cell c, thresholded with sparsity_match.
#   - The thresholding only applies to the Borzoi-prior path. The cell-graph
#     rows are already sparse + thresholded by their own pipeline.
#
# cell_factor[c] = clip(raw_nnz_per_cell[c] / median(raw_nnz_per_cell),
#                       distribute.cell_factor_clip[0],
#                       distribute.cell_factor_clip[1])
#
# Inputs:
#   <work>/mm/matrix.mtx.gz
#   <work>/predict/bin_score.npy
#   <work>/predict/bin_score_mm_row_idx.npy
#   paths.cell_graph_csr   (CSR from downstream/cell_graph_kneighbour.py)
#
# Outputs (under <work>/predict/):
#   predictions_csr.npz   sparse CSR (n_regions_full, n_cells)
#   cell_factors.npy      (n_cells,) float32
#   distribute_meta.json  splits, thresholds, counts
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("05_dist")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_threshold_mode(spec: str) -> tuple[str, float]:
    if ":" in spec:
        mode, val = spec.split(":", 1)
        return mode.strip(), float(val)
    return spec.strip(), 0.0


def pick_threshold(values: np.ndarray, mode: str, param: float,
                   raw_nnz: int, n_bins_zero: int, n_cells: int) -> float:
    if mode == "fixed":
        return float(param)
    if mode == "quantile":
        q = float(np.clip(param, 0.0, 1.0))
        return float(np.quantile(values, q))
    if mode == "sparsity_match":
        K = float(param)
        target = K * raw_nnz
        denom = float(max(1, n_bins_zero * n_cells))
        keep_frac = min(max(target / denom, 1e-9), 1.0)
        return float(np.quantile(values, 1.0 - keep_frac))
    raise SystemExit(f"Unknown threshold_mode: {mode}")


def main() -> int:
    ap = base_parser(__doc__ or "")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    d_cfg = cfg.setdefault("distribute", {})
    mode_name        = str(d_cfg.get("mode", "B")).upper()
    observed_k       = int(d_cfg.get("observed_k", 1))
    clip_lo, clip_hi = list(d_cfg.get("cell_factor_clip", [0.1, 10.0]))
    thr_mode_str     = str(d_cfg.get("threshold_mode", "sparsity_match:3"))
    thr_sample       = int(d_cfg.get("threshold_sample_size", 4_000_000))
    if mode_name != "B":
        log.error("Only mode B is implemented in this version (got %r).", mode_name)
        return 4

    mm_dir = Path(cfg["paths"]["mm"])
    log.info("Loading raw matrix %s", mm_dir / "matrix.mtx.gz")
    raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr()
    n_bins_full, n_cells = raw.shape
    raw_nnz = int(raw.nnz)
    log.info("Raw: %d regions x %d cells, nnz=%d", n_bins_full, n_cells, raw_nnz)

    # Per-row nnz for the observed/unobserved split
    row_nnz = np.asarray((raw != 0).sum(axis=1)).ravel().astype(np.int64)
    observed_rows = np.where(row_nnz >= observed_k)[0]
    zero_rows = np.where(row_nnz < observed_k)[0]
    log.info("Bins observed (nnz >= %d): %d ;  unobserved: %d",
             observed_k, len(observed_rows), len(zero_rows))

    # Per-cell library factor
    col_nnz = np.asarray((raw != 0).sum(axis=0)).ravel().astype(np.float32)
    med = float(np.median(col_nnz[col_nnz > 0])) if (col_nnz > 0).any() else 1.0
    if med == 0:
        med = 1.0
    cell_factor = np.clip(col_nnz / med, clip_lo, clip_hi).astype(np.float32)
    np.save(Path(cfg["paths"]["predict"]) / "cell_factors.npy", cell_factor)
    log.info("cell_factor: clip=[%g, %g], median(raw_nnz/cell)=%g, range=[%g, %g]",
             clip_lo, clip_hi, med, float(cell_factor.min()), float(cell_factor.max()))

    # Borzoi prior aligned to mm rows
    score = np.load(Path(cfg["paths"]["predict"]) / "bin_score.npy").astype(np.float32)
    score_mm_row = np.load(Path(cfg["paths"]["predict"]) / "bin_score_mm_row_idx.npy")
    score_aligned = np.zeros(n_bins_full, dtype=np.float32)
    score_aligned[score_mm_row] = score
    log.info("Borzoi prior aligned: %d / %d bins have a score",
             int(np.count_nonzero(score_mm_row >= 0)), n_bins_full)

    # Cell-graph CSR (mode B observed path)
    cg_path = Path(cfg["paths"].get("cell_graph_csr", ""))
    if not cg_path.is_file():
        log.error("Mode B requires paths.cell_graph_csr (got %s). Run "
                  "downstream/cell_graph_kneighbour.py first.", cg_path)
        return 5
    log.info("Loading cell-graph CSR %s", cg_path)
    cg = sp.load_npz(cg_path).tocsr()
    if cg.shape != (n_bins_full, n_cells):
        log.error("cell_graph_csr shape %s != raw shape %s", cg.shape, (n_bins_full, n_cells))
        return 6
    log.info("cell_graph_csr: nnz=%d", cg.nnz)

    # ---- threshold estimation for the Borzoi-prior fill ---------------------
    thr_mode, thr_param = parse_threshold_mode(thr_mode_str)
    if len(zero_rows) == 0:
        threshold = 1.0  # nothing to threshold
        log.info("No zero rows; skipping threshold estimation")
    elif thr_mode == "fixed":
        threshold = float(thr_param)
        log.info("Threshold (fixed): %g", threshold)
    else:
        sample_bins = max(1, min(len(zero_rows), thr_sample // max(n_cells, 1)))
        rng = np.random.default_rng(int(cfg.get("train", {}).get("random_seed", 555)))
        sample_idx = rng.choice(zero_rows, size=sample_bins, replace=False)
        log.info("Pass 1: sampling %d zero-rows (~%d values) for %s threshold",
                 sample_bins, sample_bins * n_cells, thr_mode)
        sample_vals = np.empty(sample_bins * n_cells, dtype=np.float32)
        cf = cell_factor
        for k, bi in enumerate(sample_idx):
            sample_vals[k * n_cells:(k + 1) * n_cells] = score_aligned[bi] * cf
        threshold = pick_threshold(
            sample_vals, thr_mode, thr_param,
            raw_nnz=raw_nnz, n_bins_zero=len(zero_rows), n_cells=n_cells,
        )
        log.info("Threshold (%s:%g) = %.6f  [sample mean=%.4f max=%.4f]",
                 thr_mode, thr_param, threshold,
                 float(sample_vals.mean()), float(sample_vals.max()))

    # ---- assemble output sparse CSR -----------------------------------------
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []

    # Observed path: copy cell-graph rows directly.
    if len(observed_rows) > 0:
        cg_sub = cg[observed_rows]               # (n_obs, n_cells) CSR
        coo = cg_sub.tocoo()
        row_chunks.append(observed_rows[coo.row].astype(np.int64))
        col_chunks.append(coo.col.astype(np.int64))
        val_chunks.append(coo.data.astype(np.float32))
        log.info("Observed path: copied %d entries from cell-graph CSR", coo.nnz)

    # Prior path: bin_score[b] * cell_factor[c], thresholded.
    cf = cell_factor
    kept_prior = 0
    BATCH = 4096
    for b0 in range(0, len(zero_rows), BATCH):
        sel = zero_rows[b0:b0 + BATCH]
        # outer product (len(sel), n_cells); but score is per row only
        sc = score_aligned[sel][:, None] * cf[None, :]    # (len(sel), n_cells)
        keep = sc > threshold
        if keep.any():
            r, c = np.where(keep)
            row_chunks.append(sel[r].astype(np.int64))
            col_chunks.append(c.astype(np.int64))
            val_chunks.append(sc[r, c].astype(np.float32))
            kept_prior += int(keep.sum())
    log.info("Prior path: kept %d entries above threshold %.4f", kept_prior, threshold)

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        vals = np.concatenate(val_chunks)
    else:
        rows = np.zeros(0, dtype=np.int64); cols = rows.copy()
        vals = np.zeros(0, dtype=np.float32)

    out = sp.coo_matrix((vals, (rows, cols)), shape=(n_bins_full, n_cells)).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()

    out_path = Path(cfg["paths"]["predict"]) / "predictions_csr.npz"
    sp.save_npz(out_path, out)

    meta = {
        "mode":            mode_name,
        "observed_k":      observed_k,
        "n_observed_bins": int(len(observed_rows)),
        "n_zero_bins":     int(len(zero_rows)),
        "n_cells":         int(n_cells),
        "threshold_mode":  thr_mode_str,
        "threshold":       float(threshold),
        "cell_factor_clip": [clip_lo, clip_hi],
        "cell_graph_csr":  str(cg_path),
        "n_entries_observed_path": int(sum(len(c) for c in val_chunks[:1] if val_chunks)),
        "n_entries_prior_path":    int(kept_prior),
        "out_nnz":         int(out.nnz),
        "predictions_csr": str(out_path),
    }
    (Path(cfg["paths"]["predict"]) / "distribute_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    log.info("Wrote %s (nnz=%d)", out_path, out.nnz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
