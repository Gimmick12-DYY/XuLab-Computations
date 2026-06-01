#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Persist the final scBasset imputed matrix in the standard cross-pipeline
# schema. Takes the thresholded predictions from 04_predict.py, optionally
# binarises them, optionally unions with raw, and writes a full-mm-aligned
# CSR + regions/barcodes + meta.json -- same shape and conventions as every
# other pipeline's <work>/impute/.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("05_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines_gz(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--no-binarize", action="store_true",
                    help="Store sigmoid probabilities (default: store 1.0 at every above-threshold entry).")
    ap.add_argument("--no-keep-raw", action="store_true",
                    help="Do not union with raw; output predictions only.")
    ap.add_argument("--weight", type=float, default=None,
                    help="Multiplier on prediction values (only if not binarising).")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp = cfg.setdefault("impute", {})
    if args.no_binarize: imp["binarize_output"] = False
    if args.no_keep_raw: imp["keep_raw"]        = False
    if args.weight is not None: imp["weight"]   = args.weight

    binarise = bool(imp.get("binarize_output", True))
    keep_raw = bool(imp.get("keep_raw", True))
    weight   = float(imp.get("weight", 1.0))

    mm_dir      = Path(cfg["paths"]["mm"])
    predict_dir = Path(cfg["paths"]["predict"])
    impute_dir  = Path(cfg["paths"]["impute"])
    pred_path   = predict_dir / "predictions_csr.npz"
    if not pred_path.exists():
        log.error("Missing %s -- run 04_predict.py first", pred_path); return 1

    log.info("Loading mm matrix from %s", mm_dir)
    raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr().astype(np.float32)
    regions  = np.asarray(read_lines_gz(mm_dir / "regions.tsv.gz"))
    barcodes = np.asarray(read_lines_gz(mm_dir / "barcodes.tsv.gz"))
    n_reg, n_cells = raw.shape

    log.info("Loading predictions from %s", pred_path)
    pred = sp.load_npz(pred_path).tocsr().astype(np.float32)
    if pred.shape != (n_reg, n_cells):
        log.error("Predictions shape %s != mm shape %s", pred.shape, (n_reg, n_cells))
        return 2
    log.info("Predictions: nnz=%d", pred.nnz)

    if binarise:
        pred = (pred > 0).astype(np.float32).tocsr()
    elif weight != 1.0:
        pred = pred.copy()
        pred.data *= weight

    if keep_raw:
        # Union via element-wise max so raw entries (~1.0 after cast) win when
        # both predictions and raw fire at the same (r, c).
        out = raw.maximum(pred).tocsr()
    else:
        out = pred.tocsr()
    out.eliminate_zeros()

    matrix_path = impute_dir / "matrix_csr.npz"
    sp.save_npz(matrix_path, out)
    (impute_dir / "regions.tsv").write_text("\n".join(regions.tolist()) + "\n")
    (impute_dir / "barcodes.tsv").write_text("\n".join(barcodes.tolist()) + "\n")

    pred_meta = json.loads((predict_dir / "meta.json").read_text())
    meta = {
        "pipeline":          "scBasset_TF",
        "shape":             [int(n_reg), int(n_cells)],
        "raw_nnz":           int(raw.nnz),
        "predicted_nnz":     int(pred.nnz),
        "output_nnz":        int(out.nnz),
        "gain_vs_raw":       int(out.nnz) - int(raw.nnz),
        "binarize_output":   binarise,
        "keep_raw":          keep_raw,
        "weight":            weight,
        "threshold_mode":    pred_meta.get("threshold_mode"),
        "threshold":         pred_meta.get("threshold"),
        "predict_all_bins":  pred_meta.get("predict_all_bins"),
        "predictions_path":  str(pred_path),
        "mm_dir":            str(mm_dir),
    }
    (impute_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s (nnz=%d) -- gain over raw: %+d entries",
             matrix_path, out.nnz, out.nnz - raw.nnz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
