#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 01_union.py
#
# Combine two already-combined imputed matrices into a final cross-source CSR:
#   * sources.scbasset_cistopic_impute_dir  (scBasset + cisTopic union/denoise)
#   * sources.scbasset_puscopen_impute_dir  (scBasset + PUscOpen cascade)
#
# Two combine modes (combine.method):
#
#   enhancer (DEFAULT) -- cis-anchored
#       cisTopic side decides which (bin, cell) entries exist; PUscOpen only
#       AMPLIFIES the cis signal at overlapping entries. Pus-only entries
#       (where cis is zero) are DROPPED.
#
#           out[r, c] = cis[r, c] + w * pus_normalized[r, c]   if cis[r, c] > 0
#                     = 0                                       otherwise
#
#       Guarantees: every (bin, cell) entry in scbasset_cistopic survives
#       unchanged; entries where pus also fires get boosted; pus-only
#       entries are filtered out. The user-facing intent is "cisTopic is
#       the backbone; PUscOpen is an enhancer".
#
#   max -- element-wise max (symmetric union)
#           out[r, c] = max(cis[r, c], pus[r, c])
#       Every entry from either input survives. Use when you want pure
#       additive coverage (and accept pus-only entries into the output).
#
# Both inputs must be mm-aligned (same regions.tsv + barcodes.tsv); the two
# scBasset-combined pipelines both write that.
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
    ap.add_argument("--binarize", action="store_true",
                    help="Threshold the union at 0 and store 1.0 at every nonzero entry.")
    ap.add_argument("--no-normalize-inputs", action="store_true",
                    help="Skip the per-input mean-nonzero rescale before max.")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    src = cfg.setdefault("sources", {})
    cmb = cfg.setdefault("combine", {})
    if args.scbasset_cistopic_impute_dir is not None:
        src["scbasset_cistopic_impute_dir"] = args.scbasset_cistopic_impute_dir
    if args.scbasset_puscopen_impute_dir is not None:
        src["scbasset_puscopen_impute_dir"] = args.scbasset_puscopen_impute_dir
    if args.binarize:
        cmb["binarize_output"] = True
    if args.no_normalize_inputs:
        cmb["normalize_inputs"] = False

    cis_dir = Path(src.get("scbasset_cistopic_impute_dir", "")).expanduser().resolve()
    pus_dir = Path(src.get("scbasset_puscopen_impute_dir", "")).expanduser().resolve()
    method  = str(cmb.get("method", "enhancer")).lower()
    enhancer_weight = float(cmb.get("enhancer_weight", 1.0))
    binarise = bool(cmb.get("binarize_output", False))
    normalize_inputs = bool(cmb.get("normalize_inputs", True))
    if method not in ("enhancer", "max"):
        log.error("Unsupported combine.method=%r (use 'enhancer' or 'max').", method)
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
    log.info("  scbasset_cistopic nnz=%d  data[min=%.4g, mean=%.4g, max=%.4g]",
             cis.nnz, float(cis.data.min()) if cis.nnz else 0.0,
             float(cis.data.mean()) if cis.nnz else 0.0,
             float(cis.data.max()) if cis.nnz else 0.0)
    log.info("  scbasset_puscopen nnz=%d  data[min=%.4g, mean=%.4g, max=%.4g]",
             pus.nnz, float(pus.data.min()) if pus.nnz else 0.0,
             float(pus.data.mean()) if pus.nnz else 0.0,
             float(pus.data.max()) if pus.nnz else 0.0)

    # Per-input rescale to unit mean-of-nonzeros so the two value scales are
    # comparable before max. Without this step the binary scbasset_cistopic
    # entries (=1.0) lose every contest against scbasset_puscopen's scOpen-
    # NMF floats (can be >> 1.0) -- the union becomes "puscopen wherever
    # puscopen fired, cistopic only where puscopen didn't". With the rescale,
    # each input's *typical* nonzero entry has the same magnitude (1.0), and
    # peaks (entries above each input's mean) translate proportionally into
    # the union -- so BigWig per-bin sums vary by both #cells and per-cell
    # confidence instead of being uniform.
    cis_scale = pus_scale = 1.0
    if normalize_inputs:
        if cis.nnz > 0:
            cis_scale = float(cis.data.mean())
            if cis_scale > 0:
                cis = cis.multiply(1.0 / cis_scale).tocsr()
                log.info("Normalised scbasset_cistopic by mean(nnz)=%.6g", cis_scale)
        if pus.nnz > 0:
            pus_scale = float(pus.data.mean())
            if pus_scale > 0:
                pus = pus.multiply(1.0 / pus_scale).tocsr()
                log.info("Normalised scbasset_puscopen by mean(nnz)=%.6g", pus_scale)
    else:
        log.info("normalize_inputs=false: union will be dominated by whichever "
                 "input has larger raw magnitudes.")

    n_boosted = 0
    if method == "enhancer":
        # Cis-anchored: every cis entry is preserved; pus values at the SAME
        # (r, c) are added (scaled by enhancer_weight); pus-only entries are
        # dropped. Implementation: pus_at_cis = pus * indicator(cis>0), so
        # entries where cis=0 contribute 0 even if pus is large.
        log.info("Combine mode = enhancer (cis backbone, pus amplifier, weight=%g)",
                 enhancer_weight)
        out = cis.copy().astype(np.float32)
        if pus.nnz > 0 and cis.nnz > 0:
            cis_indicator = cis.copy().astype(np.float32)
            cis_indicator.data = np.ones_like(cis_indicator.data, dtype=np.float32)
            pus_at_cis = pus.multiply(cis_indicator).tocsr()
            pus_at_cis.eliminate_zeros()
            n_boosted = int(pus_at_cis.nnz)
            out = (out + enhancer_weight * pus_at_cis).tocsr()
        log.info("Enhancer: %d cis backbone entries preserved; %d boosted by pus",
                 cis.nnz, n_boosted)
    else:  # method == "max"
        log.info("Combine mode = max (symmetric union)")
        out = cis.maximum(pus).tocsr()
        log.info("Max union nnz=%d  (gain vs cis: %+d; gain vs pus: %+d)",
                 out.nnz, out.nnz - cis.nnz, out.nnz - pus.nnz)

    out.eliminate_zeros()
    log.info("Output data[min=%.4g, mean=%.4g, max=%.4g]",
             float(out.data.min()) if out.nnz else 0.0,
             float(out.data.mean()) if out.nnz else 0.0,
             float(out.data.max()) if out.nnz else 0.0)

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
        "pipeline":                        "scbasset_cistopic ⊕ scbasset_puscopen",
        "method":                          method,
        "enhancer_weight":                 enhancer_weight if method == "enhancer" else None,
        "normalize_inputs":                bool(normalize_inputs),
        "scbasset_cistopic_input_mean_nnz": float(cis_scale),
        "scbasset_puscopen_input_mean_nnz": float(pus_scale),
        "binarize_output":                 binarise,
        "shape":                           [int(n_reg), int(n_cells)],
        "scbasset_cistopic_impute_dir":    str(cis_dir),
        "scbasset_puscopen_impute_dir":    str(pus_dir),
        "scbasset_cistopic_nnz":           int(cis.nnz),
        "scbasset_puscopen_nnz":           int(pus.nnz),
        "n_cis_entries_boosted_by_pus":    int(n_boosted) if method == "enhancer" else None,
        "output_nnz":                      int(out.nnz),
        "gain_vs_scbasset_cistopic":       int(out.nnz) - int(cis.nnz),
        "gain_vs_scbasset_puscopen":       int(out.nnz) - int(pus.nnz),
    }
    (impute_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s", matrix_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
