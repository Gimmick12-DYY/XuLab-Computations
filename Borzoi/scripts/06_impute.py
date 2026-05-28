#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 06_impute.py
#
# Assemble the final cross-pipeline output schema. Mirrors scBasset/05_impute.py
# and cisTopic/05_impute.py.
#
# Inputs:
#   <work>/mm/{matrix.mtx.gz, regions.tsv.gz, barcodes.tsv.gz}
#   <work>/predict/predictions_csr.npz
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz       sparse CSR (n_regions, n_cells)
#   regions.tsv          mm region order
#   barcodes.tsv         mm barcode order
#   meta.json            params + counts + pipeline label
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

log = logging.getLogger("06_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines_gz(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main() -> int:
    ap = base_parser(__doc__ or "")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    i_cfg = cfg.setdefault("impute", {})
    binarize = bool(i_cfg.get("binarize_output", True))
    keep_raw = bool(i_cfg.get("keep_raw", True))
    weight   = float(i_cfg.get("weight", 1.0))

    mm_dir   = Path(cfg["paths"]["mm"])
    pred_p   = Path(cfg["paths"]["predict"]) / "predictions_csr.npz"
    out_dir  = Path(cfg["paths"]["impute"])
    if not pred_p.exists():
        log.error("Missing %s -- run 05_distribute_to_cells.py first", pred_p)
        return 1

    log.info("Loading raw %s", mm_dir / "matrix.mtx.gz")
    raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr()
    raw_bin = (raw != 0).astype(np.float32)

    log.info("Loading predictions %s", pred_p)
    pred = sp.load_npz(pred_p).tocsr().astype(np.float32)
    if pred.shape != raw.shape:
        log.error("predictions shape %s != raw shape %s", pred.shape, raw.shape)
        return 2

    if binarize:
        pred = (pred > 0).astype(np.float32)
    else:
        pred = pred * weight

    if keep_raw:
        out = raw_bin.maximum(pred)
    else:
        out = pred
    out = out.tocsr()
    out.eliminate_zeros()

    regions  = read_lines_gz(mm_dir / "regions.tsv.gz")
    barcodes = read_lines_gz(mm_dir / "barcodes.tsv.gz")
    assert out.shape == (len(regions), len(barcodes))

    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "matrix_csr.npz", out)
    (out_dir / "regions.tsv").write_text("\n".join(regions) + "\n")
    (out_dir / "barcodes.tsv").write_text("\n".join(barcodes) + "\n")

    meta = {
        "pipeline":        "Borzoi",
        "n_regions":       int(out.shape[0]),
        "n_cells":         int(out.shape[1]),
        "nnz_raw":         int(raw_bin.nnz),
        "nnz_predictions": int(pred.nnz),
        "nnz_out":         int(out.nnz),
        "binarize_output": binarize,
        "keep_raw":        keep_raw,
        "weight":          weight,
        "sources": {
            "predict_meta":     str(Path(cfg["paths"]["predict"]) / "predict_meta.json"),
            "bin_score_meta":   str(Path(cfg["paths"]["predict"]) / "bin_score_meta.json"),
            "distribute_meta":  str(Path(cfg["paths"]["predict"]) / "distribute_meta.json"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s (nnz=%d)", out_dir / "matrix_csr.npz", out.nnz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
