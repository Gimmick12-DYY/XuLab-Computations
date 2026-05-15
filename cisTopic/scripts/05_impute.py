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
# so a downstream bigWig / browser workflow can use the same row/col labels and
# shape as the raw mm. The threshold is picked INSIDE this step from
# impute.threshold_mode (default `sparsity_match:3`: #(imp > t) ≈ K · raw_nnz on
# the modeled subspace).
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz   thresholded imputed values only (no max with raw)
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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("05_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


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


def _parse_regions(regions: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse 'chr:start-end' into chrom-string array, start (int), end (int)."""
    n = len(regions)
    chroms = np.empty(n, dtype=object)
    starts = np.empty(n, dtype=np.int64)
    ends   = np.empty(n, dtype=np.int64)
    for i, r in enumerate(regions):
        m = _REGION_RE.match(r)
        if m is None:
            raise SystemExit(f"Cannot parse region {r!r} as chr:start-end")
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i]   = int(m.group(3))
    return chroms, starts, ends


def _load_target_bed_mask(bed_path: Path | str | None,
                          chroms: np.ndarray,
                          starts: np.ndarray,
                          ends: np.ndarray) -> np.ndarray | None:
    """Boolean mask over regions of bins overlapping any interval in the BED.

    Returns None when bed_path is falsy (caller treats None as "no restriction").
    """
    if not bed_path:
        return None
    p = Path(bed_path)
    if not p.exists():
        log.warning("propagate.target_bed not found: %s; ignoring restriction.", p)
        return None
    log.info("Loading propagate.target_bed from %s", p)
    bed_by_chrom: dict[str, list[tuple[int, int]]] = {}
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(str(p), "rt") as fh:  # type: ignore[arg-type]
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith(("#", "track", "browser")):
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            try:
                bed_by_chrom.setdefault(parts[0], []).append(
                    (int(parts[1]), int(parts[2]))
                )
            except ValueError:
                continue

    mask = np.zeros(len(chroms), dtype=bool)
    unique_chroms = np.unique(chroms)
    for chrom in unique_chroms:
        ivs = bed_by_chrom.get(str(chrom))
        if not ivs:
            continue
        ivs.sort()
        starts_iv = np.fromiter((iv[0] for iv in ivs), dtype=np.int64, count=len(ivs))
        ends_iv   = np.fromiter((iv[1] for iv in ivs), dtype=np.int64, count=len(ivs))
        bin_idx   = np.where(chroms == chrom)[0]
        bs        = starts[bin_idx]
        be        = ends[bin_idx]
        # For each bin [bs, be), find intervals with start < be (could overlap),
        # then check ends > bs among them. Vectorise per bin via searchsorted.
        hi_arr = np.searchsorted(starts_iv, be, side="left")
        # Walk the bins; keep simple loop -- typical genome has <1M bins per chrom.
        for k, (b_s, b_e, hi) in enumerate(zip(bs, be, hi_arr)):
            if hi == 0:
                continue
            if (ends_iv[:hi] > b_s).any():
                mask[bin_idx[k]] = True
    return mask


def _propagate_to_zero_bins(csr_in: sparse.csr_matrix,
                            regions: list[str],
                            propagate_cfg: dict,
                            ) -> tuple[sparse.csr_matrix, dict]:
    """Add (target_row, cell) entries to ``csr_in`` at currently-zero rows that
    have at least ``min_neighbors`` imp-positive sources within ``window_bp``
    on the same chromosome (per cell). Propagated value is
    ``weight · mean(neighbour values in cell c)``.

    The point: LDA / NMF cannot create signal at never-observed bins because
    they were never in the training vocabulary. This step fills those bins
    using the genomic structure of CTCF binding.
    """
    if not propagate_cfg.get("enabled", False):
        return csr_in, {"enabled": False, "n_propagated_bins": 0,
                         "n_new_entries": 0}

    window_bp     = int(propagate_cfg.get("window_bp", 5000))
    min_neighbors = int(propagate_cfg.get("min_neighbors", 2))
    weight        = float(propagate_cfg.get("weight", 0.5))
    chunk_targets = int(propagate_cfg.get("chunk_targets", 5000))
    target_bed    = propagate_cfg.get("target_bed", None)

    log.info(
        "Neighbour propagation: window_bp=%d, min_neighbors=%d, weight=%.4g, "
        "target_bed=%s, chunk_targets=%d",
        window_bp, min_neighbors, weight, target_bed, chunk_targets,
    )

    chroms, starts, ends = _parse_regions(regions)
    n_total_rows = int(csr_in.shape[0])

    has_signal       = np.diff(csr_in.indptr) > 0
    target_eligible  = ~has_signal

    if target_bed:
        bed_mask = _load_target_bed_mask(target_bed, chroms, starts, ends)
        if bed_mask is not None:
            n_pre = int(target_eligible.sum())
            target_eligible &= bed_mask
            log.info(
                "Target candidates restricted by BED: %d zero-imp -> %d eligible",
                n_pre, int(target_eligible.sum()),
            )

    n_target_total = int(target_eligible.sum())
    n_source_total = int(has_signal.sum())
    log.info(
        "Propagation candidates: %d eligible target rows, %d source rows",
        n_target_total, n_source_total,
    )
    if n_target_total == 0 or n_source_total < min_neighbors:
        log.info("Nothing to propagate (no eligible targets or too few sources).")
        return csr_in, {
            "enabled": True, "n_propagated_bins": 0, "n_new_entries": 0,
            "n_target_total": n_target_total, "n_source_total": n_source_total,
            "window_bp": window_bp, "min_neighbors": min_neighbors,
            "weight": weight,
            "target_bed": str(target_bed) if target_bed else None,
        }

    new_rows: list[np.ndarray] = []
    new_cols: list[np.ndarray] = []
    new_vals: list[np.ndarray] = []
    n_propagated_bins = 0

    for chrom in np.unique(chroms):
        in_chrom       = (chroms == chrom)
        chrom_targets  = np.where(in_chrom & target_eligible)[0]
        chrom_sources  = np.where(in_chrom & has_signal)[0]
        if chrom_targets.size == 0 or chrom_sources.size < min_neighbors:
            continue

        src_starts = starts[chrom_sources]
        src_order  = np.argsort(src_starts, kind="stable")
        src_starts_sorted = src_starts[src_order]
        sources_sorted    = chrom_sources[src_order]

        # cisTopic's csr_in is already thresholded -> every nonzero IS an
        # imp-positive source. Densify the chromosome's source rows once.
        src_vals_dense  = csr_in[sources_sorted].toarray().astype(np.float32, copy=False)
        src_indicator   = (src_vals_dense > 0).astype(np.float32, copy=False)

        for ts in range(0, chrom_targets.size, chunk_targets):
            te = min(ts + chunk_targets, chrom_targets.size)
            chunk_target_rows   = chrom_targets[ts:te]
            chunk_target_starts = starts[chunk_target_rows]
            n_chunk = te - ts

            edge_t: list[int] = []
            edge_s: list[int] = []
            for i, t_start in enumerate(chunk_target_starts):
                lo = int(np.searchsorted(src_starts_sorted,
                                          int(t_start) - window_bp, side="left"))
                hi = int(np.searchsorted(src_starts_sorted,
                                          int(t_start) + window_bp, side="right"))
                n_n = hi - lo
                if n_n < min_neighbors:
                    continue
                edge_t.extend([i] * n_n)
                edge_s.extend(range(lo, hi))

            if not edge_t:
                continue

            edge_t_arr = np.asarray(edge_t, dtype=np.int64)
            edge_s_arr = np.asarray(edge_s, dtype=np.int64)
            ones_arr   = np.ones(len(edge_t), dtype=np.float32)

            # Indicator-only matrix: P_i[i, k] = 1 if source k is in target i's window.
            P_i = sparse.coo_matrix(
                (ones_arr, (edge_t_arr, edge_s_arr)),
                shape=(n_chunk, sources_sorted.size), dtype=np.float32,
            ).tocsr()

            # Per (target, cell): sum of imp-positive source values, count of
            # imp-positive sources actually contributing. mean = sum / count.
            prop_sum   = (P_i @ src_vals_dense).astype(np.float32, copy=False)
            prop_count = (P_i @ src_indicator).astype(np.float32, copy=False)
            keep = prop_count >= float(min_neighbors)

            with np.errstate(divide="ignore", invalid="ignore"):
                prop_mean = np.where(
                    keep,
                    prop_sum / np.maximum(prop_count, 1.0),
                    0.0,
                )
            propagated = (prop_mean.astype(np.float32, copy=False)
                          * np.float32(weight))

            for ti in range(n_chunk):
                vals = propagated[ti]
                nz = np.where(vals > 0.0)[0]
                if nz.size == 0:
                    continue
                target_row_full = int(chunk_target_rows[ti])
                new_rows.append(np.full(nz.size, target_row_full, dtype=np.int64))
                new_cols.append(nz.astype(np.int64))
                new_vals.append(vals[nz].astype(np.float32, copy=False))
                n_propagated_bins += 1

        del src_vals_dense, src_indicator

    if not new_rows:
        log.info("Propagation produced 0 new entries.")
        return csr_in, {
            "enabled": True, "n_propagated_bins": 0, "n_new_entries": 0,
            "n_target_total": n_target_total, "n_source_total": n_source_total,
            "window_bp": window_bp, "min_neighbors": min_neighbors,
            "weight": weight,
            "target_bed": str(target_bed) if target_bed else None,
        }

    new_rows_arr = np.concatenate(new_rows)
    new_cols_arr = np.concatenate(new_cols)
    new_vals_arr = np.concatenate(new_vals)
    log.info(
        "Propagation: added %d new (region, cell) entries across %d new bins",
        int(new_vals_arr.size), n_propagated_bins,
    )

    orig_coo = csr_in.tocoo()
    all_rows = np.concatenate([orig_coo.row, new_rows_arr])
    all_cols = np.concatenate([orig_coo.col, new_cols_arr])
    all_vals = np.concatenate([orig_coo.data, new_vals_arr])
    csr_out = sparse.coo_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=csr_in.shape, dtype=np.float32,
    ).tocsr()
    csr_out.sum_duplicates()

    return csr_out, {
        "enabled": True,
        "n_propagated_bins": n_propagated_bins,
        "n_new_entries": int(new_vals_arr.size),
        "n_target_total": n_target_total,
        "n_source_total": n_source_total,
        "mean_new_entries_per_propagated_bin":
            float(new_vals_arr.size / max(n_propagated_bins, 1)),
        "window_bp": window_bp, "min_neighbors": min_neighbors,
        "weight": weight,
        "target_bed": str(target_bed) if target_bed else None,
    }


def _pick_threshold(phi: np.ndarray, theta: np.ndarray,
                    per_cell_factor: float, mode: str, param: float,
                    sparsity_match_target: int,
                    sample_size: int,
                    rng: np.random.Generator) -> tuple[float, str]:
    """Resolve the impute.threshold_mode to a numeric cutoff.

    For sparsity_match, ``sparsity_match_target`` is the desired number of
    (modeled row, cell) pairs with imp > t (replacement: K · raw_nnz on subspace).
    """
    if mode == "fixed":
        return float(param), f"fixed:{float(param):.6g}"

    sample = _sample_imp_values(phi, theta, per_cell_factor, sample_size, rng)
    if sample.size == 0:
        return 0.0, "empty_sample"

    R, C = phi.shape[0], theta.shape[1]
    total_entries = int(R) * int(C)

    if mode == "sparsity_match":
        K = float(param)
        target = int(sparsity_match_target)
        if target <= 0:
            t_hi = float(np.max(sample)) * (1.0 + 1e-9)
            return t_hi, (
                f"sparsity_match:{K} target_calls=0 "
                f"(t above sampled imp; sample={sample.size})"
            )
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
    n_topics = int(theta.shape[0])
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
    # Target call count for sparsity_match (raw nnz on modeled subspace).
    # ------------------------------------------------------------------
    target_nnz_modeled = 0
    mm_mtx = mm_dir / "matrix.mtx.gz"
    if mode == "sparsity_match":
        if not mm_mtx.exists():
            log.error("Need %s for sparsity_match.", mm_mtx)
            return 1
        log.info("Reading raw mm matrix for sparsity-match target count.")
        raw_cnt = sio.mmread(str(mm_mtx)).tocsr()
        raw_bin = (raw_cnt > 0).astype(np.int8)
        sub = raw_bin[modeled_to_full_row][:, modeled_to_full_col]
        target_nnz_modeled = int(sub.nnz)
        log.info("Raw nonzeros on modeled subspace: %d", target_nnz_modeled)
        del raw_bin, sub, raw_cnt

    if mode == "sparsity_match":
        match_k = float(mode_param)
        sparsity_match_target = max(1, int(round(match_k * target_nnz_modeled)))
        log.info(
            "sparsity_match: target #(imp>t) = %d (K*raw_nnz, K=%.6g)",
            sparsity_match_target, match_k,
        )
    else:
        sparsity_match_target = 0  # unused for non-sparsity_match modes

    # ------------------------------------------------------------------
    # Pick threshold
    # ------------------------------------------------------------------
    threshold, threshold_origin = _pick_threshold(
        phi, theta, per_cell_factor,
        mode, mode_param,
        sparsity_match_target,
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

    csr_imp_only = sparse.coo_matrix(
        (data, (rows_full, cols_full)),
        shape=(n_full_rows, n_full_cols), dtype=np.float32,
    ).tocsr()
    csr_imp_only.sum_duplicates()
    nnz_thr = int(csr_imp_only.nnz)

    # ------------------------------------------------------------------
    # Neighbour propagation: introduce signal at currently-zero rows
    # (LDA / NMF can never reach them on their own).
    #
    # If propagate.target_bed is null, auto-detect a CTCF motif BED at
    # <repo-root>/downstream/cache/CTCF_motif_hg38.bed (run
    # downstream/fetch_ctcf_motif_bed.sh once to populate it). With the
    # motif BED present, propagation is restricted to bins where biology
    # says CTCF could plausibly bind, regardless of whether LDA saw signal
    # there.
    # ------------------------------------------------------------------
    propagate_cfg = dict(imp.get("propagate", {}) or {})
    if propagate_cfg.get("target_bed", None) is None:
        wd_path = Path(cfg["paths"]["work_dir"]).resolve()
        for anc in [wd_path, *wd_path.parents]:
            cand = anc / "downstream" / "cache" / "CTCF_motif_hg38.bed"
            if cand.is_file():
                propagate_cfg["target_bed"] = str(cand)
                log.info("propagate.target_bed auto-detected -> %s", cand)
                break
    csr, propagation_stats = _propagate_to_zero_bins(
        csr_imp_only, full_regions, propagate_cfg,
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    csr_path        = out_dir / "matrix_csr.npz"
    csr_imp_only_path = out_dir / "matrix_csr_imputed_only.npz"
    regions_out     = out_dir / "regions.tsv"
    barcodes_out    = out_dir / "barcodes.tsv"
    meta_out        = out_dir / "meta.json"

    log.info("Writing %s (shape=%s, nnz=%d) -- imp only, pre-propagation",
             csr_imp_only_path, csr_imp_only.shape, csr_imp_only.nnz)
    sparse.save_npz(csr_imp_only_path, csr_imp_only)

    log.info("Writing %s (shape=%s, nnz=%d) -- after propagation",
             csr_path, csr.shape, csr.nnz)
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
                 index=[f"Topic{i+1}" for i in range(n_topics)]).to_parquet(
                     out_dir / "cell_topic_theta.parquet")
    pd.DataFrame(phi, index=modeled_regions,
                 columns=[f"Topic{i+1}" for i in range(n_topics)]).to_parquet(
                     out_dir / "region_topic_phi.parquet")
    np.save(out_dir / "cell_topic_theta.npy", theta)
    np.save(out_dir / "region_topic_phi.npy", phi)

    summary = {
        "pipeline":          "cisTopic",
        "format":            "csr_full_mm" if align_to_mm else "csr_modeled",
        "matrix_path":       str(csr_path),
        "matrix_imputed_only_path": str(csr_imp_only_path),
        "regions_path":      str(regions_out),
        "barcodes_path":     str(barcodes_out),
        "shape":             [int(n_full_rows), int(n_full_cols)],
        "n_modeled":         int(Rm),
        "n_cells":           int(theta.shape[1]),
        "rank_k":            n_topics,
        "scale_factor":      desired_scale,
        "per_cell_factor":   float(per_cell_factor),
        "rescaled":          bool(rescaled),
        "threshold":         float(threshold),
        "threshold_mode":    threshold_mode_str,
        "threshold_origin":  threshold_origin,
        "target_nnz_modeled_subspace": int(target_nnz_modeled),
        "sparsity_match_target_imp_gt_t": int(sparsity_match_target),
        "nnz_imp_gt_threshold_only": int(nnz_thr),
        "nnz":               int(csr.nnz),
        "density":           float(csr.nnz / max(int(n_full_rows) * int(n_full_cols), 1)),
        "align_to_mm":       bool(align_to_mm),
        "binarize_output":   bool(binarize_output),
        "drop_factors_npz":  bool(drop_factors_npz),
        "factors_path":      None if drop_factors_npz else str(factors_path),
        "source_obj":        str(obj_path),
        "propagation":       propagation_stats,
    }
    meta_out.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
