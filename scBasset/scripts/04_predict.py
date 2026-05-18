#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 04_predict.py
#
# Genome-wide prediction with the trained scBasset model. Two passes:
#   pass 1: predict on a random subset of bins -> sample of sigmoid values used
#           to estimate the sparsity_match / quantile threshold
#   pass 2: predict on every bin in seqs.h5 in batches; for each (bin, cell)
#           with pred > threshold accumulate a sparse triplet, then save the
#           result at full mm dimensions.
#
# This is the step that fills bins observed in **zero** cells: predictions
# depend on the bin's sequence + each cell's learned embedding, nothing else.
#
# Inputs:
#   <work>/seqs/seqs.h5         X, mm_row_idx, regions, trainable, ...
#   <work>/seqs/matrix_train.npz (used for the raw nnz count when threshold_mode = sparsity_match)
#   <work>/mm/matrix.mtx.gz      authoritative raw nnz for sparsity_match
#   <work>/model/scbasset.keras  trained model
#
# Outputs (under <work>/predict/):
#   predictions_csr.npz   sparse CSR (n_regions_full, n_cells) of pred > threshold
#   meta.json             threshold, threshold_mode, nnz, predict_all_bins flag, ...
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


def parse_regions(names: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends   = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(names):
        m = _REGION_RE.match(str(name).strip())
        if not m:
            raise SystemExit(f"Bad region name row {i}: {name!r}")
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i]   = int(m.group(3))
    return chroms, starts, ends


def load_bed_intervals(bed_path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    opener = gzip.open if str(bed_path).endswith(".gz") else open
    with opener(bed_path, "rt") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            out[parts[0]].append((int(parts[1]), int(parts[2])))
    return out


def overlap_mask(
    chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
    intervals: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    """True at rows whose [start, end) overlaps any interval on the same chrom."""
    hit = np.zeros(len(chroms), dtype=bool)
    by_chr: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by_chr[str(c)].append(i)
    for chrom, idx_list in by_chr.items():
        iv = intervals.get(chrom) or intervals.get(chrom.replace("chr", "")) or []
        if not iv:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        s_sub, e_sub = starts[idx], ends[idx]
        local = np.zeros(len(idx), dtype=bool)
        for bs, be in iv:
            local |= np.maximum(s_sub, bs) < np.minimum(e_sub, be)
        hit[idx[local]] = True
    return hit


def resolve_predict_bed(predict_bed_cfg, work_dir: Path) -> Path | None:
    """Returns BED path or None. 'off' disables; null auto-detects cached motif BED."""
    if isinstance(predict_bed_cfg, str):
        if predict_bed_cfg.strip().lower() == "off":
            return None
        p = Path(predict_bed_cfg).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"predict_bed not found: {p}")
        return p
    # null -> auto-detect <repo>/downstream/cache/CTCF_motif_hg38.bed
    wd = work_dir.resolve()
    for anc in [wd, *wd.parents]:
        cand = anc / "downstream" / "cache" / "CTCF_motif_hg38.bed"
        if cand.is_file():
            return cand
    return None

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

log = logging.getLogger("04_predict")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_threshold_mode(spec: str) -> tuple[str, float]:
    if ":" in spec:
        mode, val = spec.split(":", 1)
        return mode.strip(), float(val)
    return spec.strip(), 0.0


def pick_threshold(values: np.ndarray, mode: str, param: float,
                   raw_nnz: int, n_bins_full: int, n_cells: int) -> float:
    """Mirror the cisTopic 05_impute / PUscOpen 06_impute threshold vocab."""
    if mode == "fixed":
        return float(param)
    if mode == "quantile":
        # param = Q in (0,1); keep top (1-Q) fraction of preds
        q = float(np.clip(param, 0.0, 1.0))
        return float(np.quantile(values, q))
    if mode == "sparsity_match":
        # pick t so that #(pred > t) ~= param * raw_nnz (on the predicted subspace)
        K = float(param)
        target_nnz = K * raw_nnz
        total_cells_in_sample = values.size
        # fraction above threshold we want: target / (predict subspace size)
        # predict subspace = n_bins_predicted * n_cells, but `values` is a uniform
        # sample of it -- so the quantile threshold transfers directly.
        # Approx fraction of all (bin, cell) entries to keep:
        keep_frac = target_nnz / float(n_bins_full * n_cells)
        keep_frac = min(max(keep_frac, 1e-9), 1.0)
        return float(np.quantile(values, 1.0 - keep_frac))
    raise SystemExit(f"Unknown threshold_mode: {mode}")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threshold-mode", type=str, default=None,
                    help="sparsity_match:K | quantile:Q | fixed:T")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    p_cfg = cfg.setdefault("predict", {})
    if args.batch_size is not None:    p_cfg["batch_size"]     = args.batch_size
    if args.threshold_mode is not None: p_cfg["threshold_mode"] = args.threshold_mode

    batch_size = int(p_cfg.get("batch_size", 256))
    predict_all = bool(p_cfg.get("predict_all_bins", True))
    threshold_mode_str = str(p_cfg.get("threshold_mode", "sparsity_match:3"))
    threshold_sample_size = int(p_cfg.get("threshold_sample_size", 4_000_000))
    mode, param = parse_threshold_mode(threshold_mode_str)

    seqs_dir   = Path(cfg["paths"]["seqs"])
    model_dir  = Path(cfg["paths"]["model"])
    out_dir    = Path(cfg["paths"]["predict"])
    mm_dir     = Path(cfg["paths"]["mm"])
    h5_path    = seqs_dir / "seqs.h5"
    model_path = model_dir / "scbasset.keras"
    for p in (h5_path, model_path):
        if not p.exists():
            log.error("Missing %s", p); return 1

    log.info("Loading mm matrix (for raw_nnz / full shape) %s", mm_dir)
    raw_csr = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr()
    n_bins_full, n_cells_full = raw_csr.shape
    raw_nnz = int(raw_csr.nnz)
    log.info("Raw: %d regions x %d cells, nnz=%d", n_bins_full, n_cells_full, raw_nnz)

    log.info("TensorFlow import ...")
    import tensorflow as tf
    log.info("TF %s, GPUs: %s", tf.__version__, tf.config.list_physical_devices("GPU"))
    model = tf.keras.models.load_model(model_path, compile=False)

    with h5py.File(h5_path, "r") as h5:
        n_bins_h5 = int(h5["X"].shape[0])
        seq_length = int(h5.attrs["seq_length"])
        mm_row_idx = np.asarray(h5["mm_row_idx"])
        trainable  = np.asarray(h5["trainable"])
        regions_h5 = np.asarray([s.decode("ascii") if isinstance(s, bytes) else s
                                 for s in h5["regions"][:]])
    assert int(model.input_shape[1]) == seq_length, (
        f"model expects seq_length={model.input_shape[1]}, h5 has {seq_length}"
    )
    assert int(model.output_shape[-1]) == n_cells_full, (
        f"model expects n_cells={model.output_shape[-1]}, mm has {n_cells_full}"
    )

    pred_rows = np.arange(n_bins_h5) if predict_all else np.where(trainable)[0]
    log.info("Predicting on %d / %d bins (predict_all_bins=%s)",
             len(pred_rows), n_bins_h5, predict_all)

    # ---- predict_bed: restrict prediction to motif-containing bins ----------
    seqs_cfg = cfg.setdefault("seqs", {})
    predict_bed_path = resolve_predict_bed(
        seqs_cfg.get("predict_bed", None),
        Path(cfg["paths"]["work_dir"]),
    )
    motif_filter_applied = False
    if predict_bed_path is None:
        log.info("predict_bed: no BED applied (auto-detect found nothing, or set to 'off')")
    else:
        log.info("predict_bed: restricting prediction to bins overlapping %s",
                 predict_bed_path)
        bed_iv = load_bed_intervals(predict_bed_path)
        rchrom, rstart, rend = parse_regions(regions_h5[pred_rows])
        hit = overlap_mask(rchrom, rstart, rend, bed_iv)
        n_before = len(pred_rows)
        pred_rows = pred_rows[hit]
        motif_filter_applied = True
        log.info("predict_bed: %d / %d bins survive motif filter",
                 len(pred_rows), n_before)
        if len(pred_rows) == 0:
            log.error("No bins survived predict_bed filter; aborting.")
            return 4

    # ---------------- pass 1: threshold estimation ----------------------------
    if mode == "fixed":
        threshold = float(param)
        log.info("Threshold (fixed): %g", threshold)
    else:
        # Sample at most ~threshold_sample_size / n_cells bins; predict per-cell.
        sample_bins = max(1, min(len(pred_rows),
                                 threshold_sample_size // max(n_cells_full, 1)))
        rng = np.random.default_rng(int(cfg.get("train", {}).get("random_seed", 555)))
        sample_idx = np.sort(rng.choice(pred_rows, size=sample_bins, replace=False))
        log.info("Pass 1: sampling %d bins (~%d values) for %s threshold",
                 sample_bins, sample_bins * n_cells_full, mode)
        sample_vals = _predict_in_batches(model, h5_path, sample_idx, batch_size)
        threshold = pick_threshold(
            sample_vals.ravel(), mode, param,
            raw_nnz=raw_nnz, n_bins_full=n_bins_full, n_cells=n_cells_full,
        )
        log.info("Threshold (%s:%g) = %.6f  [sample min=%.4f max=%.4f mean=%.4f]",
                 mode, param, threshold,
                 float(sample_vals.min()), float(sample_vals.max()), float(sample_vals.mean()))
        del sample_vals

    # ---------------- pass 2: predict + threshold per chunk -------------------
    log.info("Pass 2: predicting + thresholding across all chunks")
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []
    seen = 0
    for b0 in range(0, len(pred_rows), batch_size):
        sel = pred_rows[b0:b0 + batch_size]
        preds = _predict_in_batches(model, h5_path, sel, batch_size)
        # preds is (len(sel), n_cells_full) float32 in [0,1]
        keep = preds > threshold
        if keep.any():
            bin_pos, cell_pos = np.where(keep)
            row_chunks.append(mm_row_idx[sel[bin_pos]].astype(np.int64))
            col_chunks.append(cell_pos.astype(np.int64))
            val_chunks.append(preds[keep].astype(np.float32))
        seen += len(sel)
        if (b0 // batch_size) % 50 == 0:
            kept = sum(len(c) for c in row_chunks)
            log.info("  ... %d / %d bins, kept %d entries", seen, len(pred_rows), kept)

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        vals = np.concatenate(val_chunks)
    else:
        rows = np.zeros(0, dtype=np.int64); cols = rows.copy()
        vals = np.zeros(0, dtype=np.float32)

    log.info("Assembling CSR at full mm dimensions (%d x %d), nnz=%d",
             n_bins_full, n_cells_full, len(vals))
    out = sp.coo_matrix((vals, (rows, cols)),
                        shape=(n_bins_full, n_cells_full)).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()

    out_path = out_dir / "predictions_csr.npz"
    sp.save_npz(out_path, out)
    meta = {
        "shape":              [int(n_bins_full), int(n_cells_full)],
        "nnz":                int(out.nnz),
        "raw_nnz":            raw_nnz,
        "threshold_mode":     threshold_mode_str,
        "threshold":          float(threshold),
        "predict_all_bins":   predict_all,
        "n_bins_predicted":   int(len(pred_rows)),
        "predict_bed":        str(predict_bed_path) if predict_bed_path else None,
        "predict_bed_applied": motif_filter_applied,
        "model_keras":        str(model_path),
        "h5":                 str(h5_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s (nnz=%d)", out_path, out.nnz)
    return 0


def _predict_in_batches(model, h5_path: Path, bin_idx_in_h5: np.ndarray,
                        batch_size: int) -> np.ndarray:
    """Predict for the given h5-row indices and return (len(idx), n_cells) float32."""
    import tensorflow as tf
    n = len(bin_idx_in_h5)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    out: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as h5:
        X_ds = h5["X"]
        for b0 in range(0, n, batch_size):
            sel = np.sort(bin_idx_in_h5[b0:b0 + batch_size])
            xb = X_ds[sel.tolist()].astype(np.float32)
            pb = model(xb, training=False).numpy()
            out.append(pb)
    return np.concatenate(out, axis=0)


if __name__ == "__main__":
    sys.exit(main())
