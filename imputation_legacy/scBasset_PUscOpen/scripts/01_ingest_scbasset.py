#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 01_ingest_scbasset.py
#
# Convert scBasset's imputed CSR into a fresh mm/ directory that the standard
# PUscOpen pipeline (02_build_mask.R through 06_impute.py) can consume unchanged.
#
# Inputs (from cascade.source_impute_dir):
#   matrix_csr.npz   (n_bins, n_cells) sparse CSR, binary or sigmoid floats
#   regions.tsv      bin names (chr:start-end), one per row
#   barcodes.tsv     cell barcodes, one per column
#
# Outputs (under <work>/mm/):
#   matrix.mtx.gz    Matrix Market, gzipped, binary int32
#   regions.tsv.gz   one bin name per line
#   barcodes.tsv.gz  one barcode per line
#   matrix_info.tsv  tiny metadata dump (n_regions, n_cells, nnz, source)
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("01_ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines_maybe_gz(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def _parse_output_mode(spec: str | None, fallback: str = "pseudocount") -> tuple[str, int]:
    """Return (mode, scale). mode in {binary, float, pseudocount}; scale unused unless pseudocount."""
    s = (spec or fallback).strip().lower()
    if s in ("binary", "binarise", "binarize"):
        return "binary", 0
    if s == "float":
        return "float", 0
    if s.startswith("pseudocount"):
        if ":" in s:
            try:
                return "pseudocount", int(s.split(":", 1)[1])
            except ValueError as e:
                raise SystemExit(f"Bad pseudocount spec {spec!r}: {e}")
        return "pseudocount", -1   # caller fills from scale_factor
    raise SystemExit(f"Unknown cascade.output_mode {spec!r} (use binary | float | pseudocount[:K])")


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--source-impute-dir", type=str, default=None,
                    help="Override cascade.source_impute_dir from the config.")
    ap.add_argument("--output-mode", type=str, default=None,
                    help="binary | float | pseudocount[:K]  (overrides cascade.output_mode)")
    ap.add_argument("--scale-factor", type=int, default=None,
                    help="Pseudocount multiplier (overrides cascade.scale_factor; default 10).")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    casc = cfg.setdefault("cascade", {})
    if args.source_impute_dir is not None:
        casc["source_impute_dir"] = args.source_impute_dir
    if args.output_mode is not None:
        casc["output_mode"] = args.output_mode
    if args.scale_factor is not None:
        casc["scale_factor"] = args.scale_factor

    src_dir = Path(casc.get("source_impute_dir", "")).expanduser().resolve()
    mode, mode_scale = _parse_output_mode(casc.get("output_mode"), fallback="pseudocount")
    scale = int(mode_scale if mode_scale > 0 else casc.get("scale_factor", 10))
    if not src_dir.is_dir():
        log.error("cascade.source_impute_dir is not a directory: %s", src_dir)
        return 1

    mat_path = src_dir / "matrix_csr.npz"
    reg_path = src_dir / "regions.tsv"
    bar_path = src_dir / "barcodes.tsv"
    for p in (mat_path, reg_path, bar_path):
        if not p.exists():
            log.error("Missing %s under source_impute_dir", p); return 1

    log.info("Loading scBasset CSR from %s", mat_path)
    mat = sp.load_npz(mat_path).tocsr()
    regions  = read_lines_maybe_gz(reg_path)
    barcodes = read_lines_maybe_gz(bar_path)
    if mat.shape != (len(regions), len(barcodes)):
        log.error("Shape mismatch: matrix %s vs regions/barcodes %d/%d",
                  mat.shape, len(regions), len(barcodes))
        return 2

    src_nnz = int(mat.nnz)
    looks_binary = bool(np.all(mat.data == mat.data.astype(np.int64))) and bool(np.all(mat.data <= 1))

    if mode in ("pseudocount", "float") and looks_binary:
        log.warning(
            "cascade.output_mode = %s but scBasset's matrix is already binary "
            "(no sub-1.0 values). Falling back to 'binary' mode. To use %s, "
            "re-run scBasset with impute.binarize_output: false so its output "
            "carries sigmoid probabilities.", mode, mode)
        mode = "binary"

    if mode == "binary":
        mat = (mat > 0).astype(np.int32).tocsr()
        info_mode_str = "binary"
    elif mode == "float":
        mat = mat.astype(np.float32).tocsr()
        info_mode_str = "float"
    else:  # pseudocount
        # Convention (only valid when scBasset was run with binarize_output: false
        # AND keep_raw: true): values in (0, 1) are imputed sigmoid probabilities;
        # values >= 1 are original raw counts preserved by the keep_raw union.
        # We scale the sigmoid range into integer counts [1, scale] and leave the
        # already-integer raw counts untouched, so the mm matrix has the same
        # dtype and value range as the original raw data.
        out = mat.copy()
        out.data = out.data.astype(np.float64)
        imputed = out.data < 1.0
        out.data[imputed] = np.maximum(1.0, np.round(out.data[imputed] * scale))
        out.data = out.data.astype(np.int32)
        # Sanity: count how many entries each rule touched
        n_imputed = int(imputed.sum())
        log.info("pseudocount(scale=%d): rescaled %d imputed sigmoid entries; "
                 "preserved %d raw-count entries", scale, n_imputed, src_nnz - n_imputed)
        mat = out
        info_mode_str = f"pseudocount:{scale}"
    log.info("Loaded %d regions x %d cells, source nnz=%d, mode=%s",
             *mat.shape, src_nnz, info_mode_str)

    mm_dir = Path(cfg["paths"]["mm"])
    mm_dir.mkdir(parents=True, exist_ok=True)
    mtx_path = mm_dir / "matrix.mtx"
    log.info("Writing %s.gz", mtx_path)
    sio.mmwrite(str(mtx_path), mat)
    subprocess.run(["gzip", "-f", str(mtx_path)], check=True)

    with gzip.open(mm_dir / "regions.tsv.gz", "wt") as fh:
        fh.write("\n".join(regions) + "\n")
    with gzip.open(mm_dir / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(barcodes) + "\n")

    (mm_dir / "matrix_info.tsv").write_text(
        "\n".join([
            f"n_regions\t{int(mat.shape[0])}",
            f"n_cells\t{int(mat.shape[1])}",
            f"nnz\t{int(mat.nnz)}",
            f"source_impute_dir\t{src_dir}",
            f"source_nnz\t{src_nnz}",
            f"mode\t{info_mode_str}",
        ]) + "\n"
    )
    (mm_dir / "ingest_meta.json").write_text(json.dumps({
        "source_impute_dir": str(src_dir),
        "source_nnz":        src_nnz,
        "output_shape":      [int(mat.shape[0]), int(mat.shape[1])],
        "output_nnz":        int(mat.nnz),
        "output_mode":       info_mode_str,
    }, indent=2) + "\n")

    log.info("Done. mm/ ready for PUscOpen 02_build_mask.R")
    return 0


if __name__ == "__main__":
    sys.exit(main())
