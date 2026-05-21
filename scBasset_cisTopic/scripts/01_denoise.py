#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 01_denoise.py
#
# LDA-validated denoising of scBasset's imputed matrix, using cisTopic LDA
# topics trained on the RAW matrix. Optional element-wise max with cisTopic's
# standalone output at the end.
#
# Inputs (from sources.*):
#   scbasset_impute_dir/
#     matrix_csr.npz                  scBasset's mm-aligned CSR
#     regions.tsv / barcodes.tsv      mm row/col order
#   cistopic_impute_dir/
#     factors.npz                     W=phi (n_modeled, K), H=theta (K, n_cells_modeled)
#     region_topic_phi.parquet        index = modeled_regions
#     cell_topic_theta.parquet        columns = modeled_cells
#     matrix_csr.npz                  cisTopic's mm-aligned CSR (used iff
#                                     combine.union_with_cistopic == True)
#     regions.tsv / barcodes.tsv      full mm; must match scBasset's
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz   CSR (float32) of the final denoised matrix
#   regions.tsv      full mm region order (same as input)
#   barcodes.tsv     full mm barcode order (same as input)
#   meta.json        threshold, per-source nnz, validated/dropped counts, ...
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("01_denoise")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines(p: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in p.read_text().splitlines() if ln.strip()]


def _parse_mode(spec: str) -> tuple[str, float]:
    s = (spec or "off").strip().lower()
    if s in ("off", "none"):
        return "off", 0.0
    for prefix in ("quantile", "global_quantile", "fixed"):
        if s.startswith(prefix + ":"):
            try:
                return prefix, float(s.split(":", 1)[1])
            except ValueError as e:
                raise SystemExit(f"Bad denoise.mode {spec!r}: {e}")
    raise SystemExit(
        f"Unknown denoise.mode {spec!r}. Use: off | quantile:Q | global_quantile:Q | fixed:T"
    )


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--mode", type=str, default=None,
                    help="off | quantile:Q | global_quantile:Q | fixed:T")
    ap.add_argument("--no-union-with-cistopic", action="store_true")
    ap.add_argument("--no-binarize", action="store_true")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    src = cfg.setdefault("sources", {})
    den = cfg.setdefault("denoise", {})
    cmb = cfg.setdefault("combine", {})
    if args.mode is not None: den["mode"] = args.mode
    if args.no_union_with_cistopic: cmb["union_with_cistopic"] = False
    if args.no_binarize: cmb["binarize_output"] = False

    scb_dir = Path(src.get("scbasset_impute_dir", "")).expanduser().resolve()
    cis_dir = Path(src.get("cistopic_impute_dir", "")).expanduser().resolve()
    mode, param = _parse_mode(str(den.get("mode", "off")))
    sample_size = int(den.get("threshold_sample_size", 5000))
    keep_truly_new = bool(den.get("keep_truly_new_bins", True))
    union_with_cis = bool(cmb.get("union_with_cistopic", True))
    binarise = bool(cmb.get("binarize_output", True))
    for d in (scb_dir, cis_dir):
        if not d.is_dir():
            log.error("Missing source dir: %s", d); return 1

    # ---- Load scBasset ------------------------------------------------------
    log.info("Loading scBasset CSR from %s", scb_dir)
    scb = sp.load_npz(scb_dir / "matrix_csr.npz").tocsr().astype(np.float32)
    scb_regions  = read_lines(scb_dir / "regions.tsv")
    scb_barcodes = read_lines(scb_dir / "barcodes.tsv")
    n_mm, n_cells = scb.shape
    assert scb.shape == (len(scb_regions), len(scb_barcodes))
    log.info("  scbasset: %d regions x %d cells, nnz=%d", n_mm, n_cells, scb.nnz)

    # ---- Load cisTopic ------------------------------------------------------
    log.info("Loading cisTopic factors + regions/barcodes from %s", cis_dir)
    cis_regions  = read_lines(cis_dir / "regions.tsv")
    cis_barcodes = read_lines(cis_dir / "barcodes.tsv")
    if cis_regions != scb_regions or cis_barcodes != scb_barcodes:
        log.error(
            "scBasset and cisTopic mm region/barcode orders must match line-by-line. "
            "Both must use align_to_mm=True on the same raw mm. "
            "(scb_regions=%d, cis_regions=%d)", len(scb_regions), len(cis_regions),
        )
        return 2

    with np.load(cis_dir / "factors.npz") as z:
        phi = np.asarray(z["W"], dtype=np.float32)        # (n_modeled, K)
        theta = np.asarray(z["H"], dtype=np.float32)      # (K, n_modeled_cells)
    df_phi   = pd.read_parquet(cis_dir / "region_topic_phi.parquet")
    df_theta = pd.read_parquet(cis_dir / "cell_topic_theta.parquet")
    modeled_regions = df_phi.index.tolist()
    modeled_cells   = list(df_theta.columns)
    assert phi.shape[0] == len(modeled_regions), \
        f"phi rows {phi.shape[0]} != modeled_regions {len(modeled_regions)}"
    assert theta.shape[1] == len(modeled_cells), \
        f"theta cols {theta.shape[1]} != modeled_cells {len(modeled_cells)}"
    K = phi.shape[1]
    log.info("  cistopic: %d modeled regions, %d modeled cells, K=%d topics",
             len(modeled_regions), len(modeled_cells), K)

    # ---- Map modeled regions/cells to mm indices ----------------------------
    mm_region_to_idx = {r: i for i, r in enumerate(scb_regions)}
    mm_cell_to_idx   = {b: i for i, b in enumerate(scb_barcodes)}

    lda_to_mm_row = np.fromiter(
        (mm_region_to_idx.get(r, -1) for r in modeled_regions),
        dtype=np.int64, count=len(modeled_regions),
    )
    if (lda_to_mm_row < 0).any():
        n_missing = int((lda_to_mm_row < 0).sum())
        log.error("%d modeled regions are not in scBasset's mm region list; aborting.",
                  n_missing)
        return 3

    cell_lda_to_mm = np.fromiter(
        (mm_cell_to_idx.get(b, -1) for b in modeled_cells),
        dtype=np.int64, count=len(modeled_cells),
    )
    if (cell_lda_to_mm < 0).any():
        log.error("%d modeled cells are missing from scBasset's barcode list; aborting.",
                  int((cell_lda_to_mm < 0).sum())); return 3

    if not np.array_equal(cell_lda_to_mm, np.arange(n_cells, dtype=np.int64)):
        # Reorder theta columns so they line up with mm column order.
        log.info("Reordering theta columns to match scBasset's barcode order")
        inv = np.empty(n_cells, dtype=np.int64)
        inv[cell_lda_to_mm] = np.arange(n_cells)
        theta = theta[:, inv]

    in_lda = np.zeros(n_mm, dtype=bool)
    in_lda[lda_to_mm_row] = True
    n_lda = int(in_lda.sum())
    n_truly_new = int(n_mm - n_lda)
    log.info("Bins in LDA vocabulary: %d  truly-new in scBasset: %d", n_lda, n_truly_new)

    if mode == "off":
        log.info("denoise.mode=off — pass scBasset through unchanged (no LDA filtering).")

    # ---- Estimate per-cell / global threshold from a sample of LDA bins -----
    threshold_per_cell: np.ndarray | None = None
    threshold_global: float = 0.0
    if mode != "off":
        rng = np.random.default_rng(555)
        n_sample = int(min(len(modeled_regions), max(1, sample_size)))
        log.info("Estimating LDA score threshold (mode=%s:%g) from %d sampled modeled bins",
                 mode, param, n_sample)
        sample_idx = np.sort(rng.choice(len(modeled_regions), size=n_sample, replace=False))
        scores_sample = phi[sample_idx] @ theta      # (n_sample, n_cells)
        if mode == "quantile":
            threshold_per_cell = np.quantile(scores_sample, float(param), axis=0).astype(np.float32)
            log.info("Per-cell threshold quantiles (Q=%g): min=%.4g median=%.4g max=%.4g",
                     param, float(threshold_per_cell.min()),
                     float(np.median(threshold_per_cell)), float(threshold_per_cell.max()))
        elif mode == "global_quantile":
            threshold_global = float(np.quantile(scores_sample.ravel(), float(param)))
            log.info("Global threshold (Q=%g) = %.6g", param, threshold_global)
        elif mode == "fixed":
            threshold_global = float(param)
            log.info("Fixed threshold = %.6g", threshold_global)

    # ---- Per-chunk: score and filter scBasset entries on LDA-vocab bins -----
    out_rows: list[np.ndarray] = []
    out_cols: list[np.ndarray] = []
    out_vals: list[np.ndarray] = []

    n_sb_modeled = 0      # scBasset entries on LDA-vocab bins
    n_kept_modeled = 0    # of those, how many pass the filter
    chunk = 1000
    log.info("Filtering scBasset entries on %d LDA-vocab bins (chunks of %d)",
             n_lda, chunk)
    for c0 in range(0, len(modeled_regions), chunk):
        c1 = min(c0 + chunk, len(modeled_regions))
        mm_rows_chunk = lda_to_mm_row[c0:c1]                       # (chunk,)
        sb_chunk = scb[mm_rows_chunk].tocsr()                      # (chunk, n_cells)
        if sb_chunk.nnz == 0:
            continue
        coo = sb_chunk.tocoo()
        n_sb_modeled += int(coo.nnz)

        if mode == "off":
            keep_mask = np.ones(coo.nnz, dtype=bool)
        else:
            scores_chunk = phi[c0:c1] @ theta                      # (chunk, n_cells)
            if mode == "quantile":
                assert threshold_per_cell is not None
                # Per-cell threshold
                keep_mask = scores_chunk[coo.row, coo.col] >= threshold_per_cell[coo.col]
            else:
                # global_quantile / fixed -> single scalar threshold
                keep_mask = scores_chunk[coo.row, coo.col] >= threshold_global

        n_kept_modeled += int(keep_mask.sum())
        if keep_mask.any():
            kept_full_rows = mm_rows_chunk[coo.row[keep_mask]]
            out_rows.append(kept_full_rows)
            out_cols.append(coo.col[keep_mask])
            out_vals.append(coo.data[keep_mask])

        if (c0 // chunk) % 20 == 0:
            log.info("  ... %d / %d modeled bins, kept %d / %d so far",
                     c1, len(modeled_regions), n_kept_modeled, n_sb_modeled)

    drop_frac = 1.0 - (n_kept_modeled / max(n_sb_modeled, 1))
    log.info("LDA validation: kept %d / %d scBasset entries on modeled bins (%.1f%% dropped)",
             n_kept_modeled, n_sb_modeled, 100.0 * drop_frac)

    # ---- Truly-new bins: keep as-is (or drop) -------------------------------
    n_truly_new_entries = 0
    if keep_truly_new and n_truly_new > 0:
        log.info("Keeping all scBasset entries on %d truly-new (non-LDA) bins", n_truly_new)
        truly_new_idx = np.where(~in_lda)[0]
        sb_tn = scb[truly_new_idx].tocsr().tocoo()
        n_truly_new_entries = int(sb_tn.nnz)
        if sb_tn.nnz > 0:
            out_rows.append(truly_new_idx[sb_tn.row])
            out_cols.append(sb_tn.col)
            out_vals.append(sb_tn.data)
    elif n_truly_new > 0:
        log.info("Dropping truly-new bins (denoise.keep_truly_new_bins=false): "
                 "%d scBasset entries removed", int((scb[~in_lda]).nnz))

    # ---- Assemble filtered scBasset CSR -------------------------------------
    if out_rows:
        rows = np.concatenate(out_rows)
        cols = np.concatenate(out_cols)
        vals = np.concatenate(out_vals)
    else:
        rows = np.zeros(0, dtype=np.int64); cols = rows.copy()
        vals = np.zeros(0, dtype=np.float32)

    filtered = sp.coo_matrix((vals.astype(np.float32), (rows, cols)),
                              shape=scb.shape).tocsr()
    filtered.sum_duplicates()
    filtered.eliminate_zeros()
    log.info("Filtered scBasset nnz=%d (modeled-bin nnz=%d, truly-new nnz=%d)",
             filtered.nnz, n_kept_modeled, n_truly_new_entries)

    # ---- Optional union with cisTopic standalone ----------------------------
    cis_nnz_total = -1
    if union_with_cis:
        log.info("Loading cisTopic CSR for union step")
        cis = sp.load_npz(cis_dir / "matrix_csr.npz").tocsr().astype(np.float32)
        if cis.shape != scb.shape:
            log.error("cisTopic CSR shape %s != scBasset shape %s; cisTopic must use align_to_mm=true",
                      cis.shape, scb.shape); return 4
        cis_nnz_total = int(cis.nnz)
        log.info("  cistopic nnz=%d", cis_nnz_total)
        out = filtered.maximum(cis).tocsr()
    else:
        out = filtered

    if binarise:
        out = (out > 0).astype(np.float32).tocsr()

    out.eliminate_zeros()
    log.info("Output nnz=%d (gain vs filtered: %+d)", out.nnz, out.nnz - filtered.nnz)

    # ---- Persist ------------------------------------------------------------
    impute_dir = Path(cfg["paths"]["impute"])
    matrix_path = impute_dir / "matrix_csr.npz"
    sp.save_npz(matrix_path, out)
    (impute_dir / "regions.tsv").write_text("\n".join(scb_regions) + "\n")
    (impute_dir / "barcodes.tsv").write_text("\n".join(scb_barcodes) + "\n")

    meta = {
        "pipeline":                  "scBasset+cisTopic (LDA-denoise + union)",
        "scbasset_impute_dir":       str(scb_dir),
        "cistopic_impute_dir":       str(cis_dir),
        "shape":                     [int(n_mm), int(n_cells)],
        "denoise_mode":              f"{mode}:{param}" if mode != "off" else "off",
        "threshold_sample_size":     sample_size,
        "n_modeled_bins":            int(n_lda),
        "n_truly_new_bins_in_mm":    int(n_truly_new),
        "scbasset_nnz_total":        int(scb.nnz),
        "scbasset_nnz_on_modeled":   int(n_sb_modeled),
        "kept_modeled_nnz":          int(n_kept_modeled),
        "kept_truly_new_nnz":        int(n_truly_new_entries),
        "filtered_scbasset_nnz":     int(filtered.nnz),
        "union_with_cistopic":       bool(union_with_cis),
        "cistopic_nnz_total":        int(cis_nnz_total),
        "binarize_output":           bool(binarise),
        "output_nnz":                int(out.nnz),
    }
    (impute_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s", matrix_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
