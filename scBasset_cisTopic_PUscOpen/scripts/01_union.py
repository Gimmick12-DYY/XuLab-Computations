#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 01_union.py
#
# Combine scBasset+cisTopic (BACKBONE) with scBasset+PUscOpen (additive):
#
#     pus_norm  = pus * (cis_total / pus_total)        # rescale pus to cis scale
#     out[r, c] = cis[r, c] + w * pus_norm[r, c]
#
# pus is renormalised so its total signal equals cis's (equal mean per-cell
# sum) -- otherwise pus's ~1e5-scale values would swamp cis's ~1-17. Then:
#   * cis-only entries pass through as cis,
#   * overlap entries are boosted by w * pus_norm,
#   * pus-only entries are ADDED at w * pus_norm.
# This REPLACES the old enhancer rule (which preserved out.nnz == cis.nnz and
# dropped all pus-only entries); now the dense PUscOpen coverage reaches the
# cascade. `w` = combine.enhancer_weight (default 1.0 -> pus contributes the
# same total mass as cis).
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
                    help="Weight w on the normalised pus contribution: "
                         "out = cis + w * pus_norm. Default 1.0 (pus_norm carries the "
                         "same total mass as cis). 0 disables pus (output == cis).")
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

    # -------- Additive union (pus renormalised to cis's scale) --------------
    # pus values live on a ~1e5 scale and cis on ~1-17, so a raw add would let
    # pus swamp the backbone. Rescale pus so its TOTAL signal equals cis's (i.e.
    # equal mean per-cell sum), then add with weight w:
    #     out[r, c] = cis[r, c] + w * pus_norm[r, c]
    # Every cis entry is preserved; overlap entries get a w*pus_norm boost; and
    # -- the change from the old enhancer rule -- pus-only entries are now ADDED
    # rather than dropped, so the (dense) PUscOpen coverage actually reaches the
    # cascade.
    cis_total = float(cis.sum())
    pus_total = float(pus.sum())
    norm_factor = (cis_total / pus_total) if pus_total > 0 else 0.0
    log.info("pus normalisation: cis_total=%.4g  pus_total=%.4g  -> pus_scale=%.6g "
             "(x enhancer_weight=%g)", cis_total, pus_total, norm_factor, enhancer_weight)

    if norm_factor > 0 and enhancer_weight != 0 and pus.nnz > 0:
        pus_norm = (pus * np.float32(enhancer_weight * norm_factor)).tocsr()
        out = (cis + pus_norm).tocsr()
    else:
        out = cis.copy()
    out.eliminate_zeros()
    out.sum_duplicates()

    # All values are > 0, so no entry cancels: out.nnz == |cis ∪ pus|, which
    # gives the overlap and pus-only counts for free.
    n_overlap  = int(cis.nnz + pus.nnz - out.nnz)
    n_pus_only = int(pus.nnz - n_overlap)

    log.info("Combined (additive, pus normalised to cis scale; w=%g):", enhancer_weight)
    log.info("  cis entries (all preserved) = %d", cis.nnz)
    log.info("  cis ∩ pus (boosted)         = %d  (%.2f%% of cis)",
             n_overlap, 100.0 * n_overlap / max(cis.nnz, 1))
    log.info("  pus-only entries ADDED      = %d  (was dropped under the old rule)",
             n_pus_only)
    log.info("  output nnz                  = %d", out.nnz)
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
        "pipeline":                       "scbasset_cistopic (backbone) + scbasset_puscopen (additive, normalised)",
        "rule":                            "out[r,c] = cis[r,c] + w * pus_norm[r,c]; pus_norm = pus * (cis_total/pus_total)",
        "enhancer_weight":                 enhancer_weight,
        "pus_norm_factor":                 float(norm_factor),
        "cis_total_signal":               cis_total,
        "pus_total_signal":               pus_total,
        "shape":                           [int(n_reg), int(n_cells)],
        "scbasset_cistopic_impute_dir":    str(cis_dir),
        "scbasset_puscopen_impute_dir":    str(pus_dir),
        "scbasset_cistopic_nnz":           int(cis.nnz),
        "scbasset_puscopen_nnz":           int(pus.nnz),
        "n_cis_pus_overlap_boosted":       int(n_overlap),
        "n_pus_only_entries_added":        int(n_pus_only),
        "output_nnz":                      int(out.nnz),
    }
    (impute_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s", matrix_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
