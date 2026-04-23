#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_build_cistopic_obj.py
#
# Load the Matrix Market files written by 01_export_rds_to_mm.R, apply the
# per-region / per-cell filters defined in the config, and pickle a
# pycisTopic CistopicObject that every subsequent step consumes.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("02_build")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines_gz(path: str) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--min-counts-per-region", type=int, default=None)
    ap.add_argument("--min-cells-per-region",  type=int, default=None)
    ap.add_argument("--min-regions-per-cell",  type=int, default=None)
    ap.add_argument("--no-binarize", action="store_true")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    flt = cfg.setdefault("filter", {})
    for k, v in (
        ("min_counts_per_region", args.min_counts_per_region),
        ("min_cells_per_region",  args.min_cells_per_region),
        ("min_regions_per_cell",  args.min_regions_per_cell),
    ):
        if v is not None:
            flt[k] = v
    if args.no_binarize:
        flt["binarize_input"] = False

    mm_dir   = Path(cfg["paths"]["mm"])
    mtx_path = mm_dir / "matrix.mtx.gz"
    reg_path = mm_dir / "regions.tsv.gz"
    bar_path = mm_dir / "barcodes.tsv.gz"
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            log.error("Missing input: %s (run 01_export_rds_to_mm.R first)", p)
            return 1

    log.info("Reading %s", mtx_path)
    mat = sio.mmread(mtx_path).tocsr()           # regions x cells

    regions  = np.asarray(read_lines_gz(str(reg_path)))
    barcodes = np.asarray(read_lines_gz(str(bar_path)))
    assert mat.shape == (len(regions), len(barcodes)), (
        f"shape mismatch: matrix {mat.shape}, regions {len(regions)}, barcodes {len(barcodes)}"
    )
    log.info("Loaded %d regions x %d cells, nnz=%d", *mat.shape, mat.nnz)

    # -- filter regions -------------------------------------------------------
    min_counts = int(flt.get("min_counts_per_region", 0))
    min_cells  = int(flt.get("min_cells_per_region", 0))
    if min_counts > 0 or min_cells > 0:
        row_sum   = np.asarray(mat.sum(axis=1)).ravel()
        row_nnz   = np.diff(mat.indptr)            # non-zero entries per row (region)
        keep_rows = (row_sum >= min_counts) & (row_nnz >= min_cells)
        log.info(
            "Region filter (counts >= %d, cells >= %d): keeping %d / %d (%.2f%%)",
            min_counts, min_cells, int(keep_rows.sum()), len(keep_rows),
            100.0 * keep_rows.mean(),
        )
        mat     = mat[keep_rows]
        regions = regions[keep_rows]

    # -- binarize -------------------------------------------------------------
    if flt.get("binarize_input", True):
        log.info("Binarising input matrix (count > 0 -> 1).")
        mat = mat.astype(bool).astype(np.int32)

    # -- filter cells ---------------------------------------------------------
    min_regs = int(flt.get("min_regions_per_cell", 0))
    if min_regs > 0:
        col_nnz   = np.diff(mat.tocsc().indptr)
        keep_cols = col_nnz >= min_regs
        log.info(
            "Cell filter (regions >= %d): keeping %d / %d cells.",
            min_regs, int(keep_cols.sum()), len(keep_cols),
        )
        mat      = mat[:, keep_cols]
        barcodes = barcodes[keep_cols]

    log.info("Post-filter matrix: %d regions x %d cells, nnz=%d", *mat.shape, mat.nnz)

    # -- build CistopicObject -------------------------------------------------
    try:
        from pycisTopic.cistopic_class import create_cistopic_object
    except ImportError as e:
        log.error("pycisTopic is not importable: %s", e)
        return 2

    log.info("Creating CistopicObject...")
    cistopic_obj = create_cistopic_object(
        fragment_matrix=sp.csr_matrix(mat),
        cell_names=list(barcodes),
        region_names=list(regions),
        project="cisTopic_TF",
        tag_cells=False,
    )

    out = Path(cfg["paths"]["obj"]) / "cistopic_obj.pkl"
    log.info("Pickling CistopicObject -> %s", out)
    with open(out, "wb") as fh:
        pickle.dump(cistopic_obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "n_regions_in":   int(len(regions)),
        "n_cells_in":     int(len(barcodes)),
        "nnz":            int(mat.nnz),
        "binarized":      bool(flt.get("binarize_input", True)),
        "obj_path":       str(out),
    }
    (Path(cfg["paths"]["obj"]) / "cistopic_obj.meta.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n"
    )
    log.info("Done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
