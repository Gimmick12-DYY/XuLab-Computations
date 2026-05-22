#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 01_union.py
#
# Element-wise max of two already-combined imputed matrices:
#   * sources.scbasset_cistopic_impute_dir  (scBasset + cisTopic union/denoise)
#   * sources.scbasset_puscopen_impute_dir  (scBasset + PUscOpen cascade)
#
#     out[r, c] = max(scbasset_cistopic[r, c], scbasset_puscopen[r, c])
#
# Purely additive: every entry called by either input survives. Both inputs
# must be mm-aligned (same regions.tsv + barcodes.tsv); the two scBasset-
# combined pipelines both write that.
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz   CSR (float32) of the union
#   regions.tsv      full mm region order (copied from sources)
#   barcodes.tsv     full mm barcode order (copied from sources)
#   meta.json        per-source nnz, output nnz, gain vs each source
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("01_union")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines(p: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in p.read_text().splitlines() if ln.strip()]


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--scbasset-cistopic-impute-dir", type=str, default=None)
    ap.add_argument("--scbasset-puscopen-impute-dir", type=str, default=None)
    ap.add_argument("--no-binarize", action="store_true",
                    help="Keep the element-wise-max float values instead of binarising.")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    src = cfg.setdefault("sources", {})
    cmb = cfg.setdefault("combine", {})
    if args.scbasset_cistopic_impute_dir is not None:
        src["scbasset_cistopic_impute_dir"] = args.scbasset_cistopic_impute_dir
    if args.scbasset_puscopen_impute_dir is not None:
        src["scbasset_puscopen_impute_dir"] = args.scbasset_puscopen_impute_dir
    if args.no_binarize:
        cmb["binarize_output"] = False

    cis_dir = Path(src.get("scbasset_cistopic_impute_dir", "")).expanduser().resolve()
    pus_dir = Path(src.get("scbasset_puscopen_impute_dir", "")).expanduser().resolve()
    method  = str(cmb.get("method", "max")).lower()
    binarise = bool(cmb.get("binarize_output", True))
    if method != "max":
        log.error("Unsupported combine.method=%r (only 'max' is implemented).", method)
        return 1
    for d in (cis_dir, pus_dir):
        if not d.is_dir():
            log.error("Missing source dir: %s", d); return 1

    log.info("Loading scbasset_cistopic CSR from %s", cis_dir)
    cis = sp.load_npz(cis_dir / "matrix_csr.npz").tocsr().astype(np.float32)
    cis_regions  = read_lines(cis_dir / "regions.tsv")
    cis_barcodes = read_lines(cis_dir / "barcodes.tsv")

    log.info("Loading scbasset_puscopen CSR from %s", pus_dir)
    pus = sp.load_npz(pus_dir / "matrix_csr.npz").tocsr().astype(np.float32)
    pus_regions  = read_lines(pus_dir / "regions.tsv")
    pus_barcodes = read_lines(pus_dir / "barcodes.tsv")

    if cis_regions != pus_regions or cis_barcodes != pus_barcodes:
        log.error(
            "Region / barcode order mismatch between sources. Both pipelines "
            "must align_to_mm against the same raw mm (regions/barcodes "
            "identical line-by-line). scbasset_cistopic=%d regions, "
            "scbasset_puscopen=%d regions.",
            len(cis_regions), len(pus_regions),
        )
        return 2
    if cis.shape != pus.shape:
        log.error("Shape mismatch: scbasset_cistopic %s vs scbasset_puscopen %s",
                  cis.shape, pus.shape)
        return 2
    n_reg, n_cells = cis.shape
    log.info("Both inputs aligned: %d regions x %d cells", n_reg, n_cells)
    log.info("  scbasset_cistopic nnz=%d", cis.nnz)
    log.info("  scbasset_puscopen nnz=%d", pus.nnz)

    log.info("Computing element-wise max ...")
    out = cis.maximum(pus).tocsr()
    out.eliminate_zeros()
    log.info("Union nnz=%d  (gain vs scbasset_cistopic: %+d; gain vs scbasset_puscopen: %+d)",
             out.nnz, out.nnz - cis.nnz, out.nnz - pus.nnz)

    if binarise:
        out = (out > 0).astype(np.float32).tocsr()
        out.eliminate_zeros()
        log.info("Binarised union: nnz=%d (== n_unique_(region,cell)_calls)", out.nnz)

    impute_dir = Path(cfg["paths"]["impute"])
    matrix_path = impute_dir / "matrix_csr.npz"
    sp.save_npz(matrix_path, out)
    (impute_dir / "regions.tsv").write_text("\n".join(cis_regions) + "\n")
    (impute_dir / "barcodes.tsv").write_text("\n".join(cis_barcodes) + "\n")

    meta = {
        "pipeline":                       "scbasset_cistopic ∪ scbasset_puscopen",
        "method":                          "max",
        "binarize_output":                 binarise,
        "shape":                           [int(n_reg), int(n_cells)],
        "scbasset_cistopic_impute_dir":    str(cis_dir),
        "scbasset_puscopen_impute_dir":    str(pus_dir),
        "scbasset_cistopic_nnz":           int(cis.nnz),
        "scbasset_puscopen_nnz":           int(pus.nnz),
        "output_nnz":                      int(out.nnz),
        "gain_vs_scbasset_cistopic":       int(out.nnz) - int(cis.nnz),
        "gain_vs_scbasset_puscopen":       int(out.nnz) - int(pus.nnz),
    }
    (impute_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s", matrix_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
