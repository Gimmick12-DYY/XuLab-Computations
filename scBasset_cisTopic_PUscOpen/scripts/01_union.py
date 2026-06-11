#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 01_union.py
#
# Combine scBasset+cisTopic (BACKBONE) with scBasset+PUscOpen (ENHANCER):
#
#     out[r, c] = cis[r, c] + w * cis[r, c]            if cis[r, c] > 0 AND pus[r, c] > 0
#               = cis[r, c]                             if cis[r, c] > 0 AND pus[r, c] = 0
#               = 0                                     if cis[r, c] = 0
#
# i.e. multiplicative boost (1 + w) at every (bin, cell) where pus also
# fires, otherwise the cis value passes through unchanged.
#
# HARD INVARIANT (asserted at runtime):
#     out.nnz == cis.nnz   AND   out.indices == cis.indices
# Every (bin, cell) entry in scbasset_cistopic is in the output, exactly
# one for one. Nothing is added, nothing is removed.
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz   sparse float32 (n_regions, n_cells)
#   regions.tsv      copied from cis
#   barcodes.tsv     copied from cis
#   meta.json        per-source nnz, n_boosted, output stats
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
    ap.add_argument("--enhancer-weight", type=float, default=None,
                    help="Multiplicative boost amplitude at cis ∩ pus entries. "
                         "Default 1.0 (= 2x cis value at overlap). 0 disables.")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    src = cfg.setdefault("sources", {})
    cmb = cfg.setdefault("combine", {})
    if args.scbasset_cistopic_impute_dir is not None:
        src["scbasset_cistopic_impute_dir"] = args.scbasset_cistopic_impute_dir
    if args.scbasset_puscopen_impute_dir is not None:
        src["scbasset_puscopen_impute_dir"] = args.scbasset_puscopen_impute_dir
    if args.enhancer_weight is not None:
        cmb["enhancer_weight"] = args.enhancer_weight

    cis_dir = Path(src.get("scbasset_cistopic_impute_dir", "")).expanduser().resolve()
    pus_dir = Path(src.get("scbasset_puscopen_impute_dir", "")).expanduser().resolve()
    enhancer_weight = float(cmb.get("enhancer_weight", 1.0))
    for d in (cis_dir, pus_dir):
        if not d.is_dir():
            log.error("Missing source dir: %s", d); return 1

    # -------- Load both matrices --------------------------------------------
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
            "must align_to_mm against the same raw mm. "
            "cis: %d regions, %d barcodes;  pus: %d regions, %d barcodes.",
            len(cis_regions), len(cis_barcodes),
            len(pus_regions), len(pus_barcodes),
        )
        return 2
    if cis.shape != pus.shape:
        log.error("Shape mismatch: cis %s vs pus %s", cis.shape, pus.shape)
        return 2
    n_reg, n_cells = cis.shape

    # -------- Loud nnz report so the user can see what's there --------------
    log.info("=" * 64)
    log.info("Inputs:")
    log.info("  scbasset_cistopic nnz=%d", cis.nnz)
    log.info("  scbasset_puscopen nnz=%d", pus.nnz)
    log.info("=" * 64)

    # -------- The combine, written so the invariant is obvious --------------
    # 1. Start with cis as the output (every cis entry preserved verbatim).
    out = cis.copy()

    # 2. Find (bin, cell) entries where BOTH cis and pus fire. Indicator
    #    multiplication: pus_indicator has 1 at every pus nonzero, 0
    #    elsewhere; cis * pus_indicator zeroes out cis entries not in pus.
    n_overlap = 0
    if pus.nnz > 0 and cis.nnz > 0 and enhancer_weight != 0:
        pus_indicator = pus.copy()
        pus_indicator.data = np.ones_like(pus_indicator.data, dtype=np.float32)
        cis_at_overlap = cis.multiply(pus_indicator).tocsr()
        cis_at_overlap.eliminate_zeros()
        n_overlap = int(cis_at_overlap.nnz)
        # 3. Boost overlap entries by w * their cis value (multiplicative).
        #    Because cis_at_overlap.indices ⊆ cis.indices, the addition does
        #    not introduce any new (bin, cell) positions.
        out = (out + enhancer_weight * cis_at_overlap).tocsr()

    # 4. HARD INVARIANT. If this fires, the code is wrong, not your data.
    if out.nnz != cis.nnz:
        log.error("INVARIANT VIOLATED: out.nnz=%d != cis.nnz=%d. "
                  "Enhancer mode must preserve every cis entry exactly.",
                  out.nnz, cis.nnz)
        return 3

    log.info("Combined:")
    log.info("  cis backbone entries kept = %d  (must equal cis.nnz=%d): %s",
             out.nnz, cis.nnz, "OK" if out.nnz == cis.nnz else "BUG")
    log.info("  entries boosted (cis ∩ pus) = %d  (%.2f%% of cis)",
             n_overlap, 100.0 * n_overlap / max(cis.nnz, 1))
    log.info("  pus-only entries dropped  = %d  (pus.nnz - overlap)",
             pus.nnz - n_overlap)
    log.info("=" * 64)
    log.info("Value stats:")
    log.info("  cis data: min=%.4g, mean=%.4g, max=%.4g",
             float(cis.data.min()) if cis.nnz else 0.0,
             float(cis.data.mean()) if cis.nnz else 0.0,
             float(cis.data.max()) if cis.nnz else 0.0)
    log.info("  out data: min=%.4g, mean=%.4g, max=%.4g  (with weight=%g)",
             float(out.data.min()) if out.nnz else 0.0,
             float(out.data.mean()) if out.nnz else 0.0,
             float(out.data.max()) if out.nnz else 0.0,
             enhancer_weight)
    log.info("=" * 64)

    # -------- Persist --------------------------------------------------------
    impute_dir = Path(cfg["paths"]["impute"])
    matrix_path = impute_dir / "matrix_csr.npz"
    sp.save_npz(matrix_path, out)
    (impute_dir / "regions.tsv").write_text("\n".join(cis_regions) + "\n")
    (impute_dir / "barcodes.tsv").write_text("\n".join(cis_barcodes) + "\n")

    meta = {
        "pipeline":                       "scbasset_cistopic (backbone) + scbasset_puscopen (enhancer)",
        "rule":                            "out[r,c] = cis[r,c] * (1 + w * 1{pus[r,c]>0})  if cis[r,c]>0 else 0",
        "enhancer_weight":                 enhancer_weight,
        "shape":                           [int(n_reg), int(n_cells)],
        "scbasset_cistopic_impute_dir":    str(cis_dir),
        "scbasset_puscopen_impute_dir":    str(pus_dir),
        "scbasset_cistopic_nnz":           int(cis.nnz),
        "scbasset_puscopen_nnz":           int(pus.nnz),
        "n_cis_entries_boosted_by_pus":    int(n_overlap),
        "n_pus_only_entries_dropped":      int(pus.nnz - n_overlap),
        "output_nnz":                      int(out.nnz),
        "invariant_check_passed":          bool(out.nnz == cis.nnz),
    }
    (impute_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s", matrix_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
