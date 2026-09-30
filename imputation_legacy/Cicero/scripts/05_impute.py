#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 05_impute.py
#
# Cicero co-accessibility propagation. Loads the Peak1/Peak2/coaccess table
# from 03_run_cicero.R, turns it into a sparse symmetric bin-bin adjacency
# C, and propagates raw signal cell-by-cell:
#
#     prop_sum[r2, c]   = sum_{r1 ~ r2 in cell c, raw[r1, c] > 0} coaccess(r1, r2)
#     prop_count[r2, c] = #{r1 ~ r2 with raw[r1, c] > 0}
#     value[r2, c]      = weight * prop_sum / prop_count    (count >= min_links)
#
# where r1 ~ r2 means coaccess(r1, r2) >= coaccess_threshold. By default we
# only fill bins where raw[r2, c] = 0 (keeps raw values untouched), then
# union with raw. Output is a full mm-aligned CSR — same regions / barcodes
# as the input mm — so it is a drop-in for the raw matrix everywhere the
# downstream comparators expect.
#
# Outputs (under <work>/impute/):
#   matrix_csr.npz   sparse CSR (n_regions x n_cells)
#   regions.tsv      full mm region order (one per line)
#   barcodes.tsv     full mm barcode order (one per line)
#   meta.json        nnz / shape / threshold / weight / source
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("05_impute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines_gz(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


_REGION_US_RE = re.compile(r"^([^:_]+)[_:]?(\d+)[-_](\d+)$")


def to_underscore(name: str) -> str:
    """chr1:100-200 -> chr1_100_200 (must match 02_build_cds.R to_underscore)."""
    m = _REGION_US_RE.match(name)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return name


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--coaccess-threshold", type=float, default=None)
    ap.add_argument("--min-links",          type=int,   default=None)
    ap.add_argument("--weight",             type=float, default=None)
    ap.add_argument("--no-only-zero-targets", action="store_true",
                    help="Allow propagated values to overwrite raw entries (max(raw, prop)).")
    ap.add_argument("--no-keep-raw", action="store_true",
                    help="Output propagated values only; do not union with raw.")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    imp = cfg.setdefault("impute", {})
    if args.coaccess_threshold is not None: imp["coaccess_threshold"] = args.coaccess_threshold
    if args.min_links is not None:          imp["min_links"]          = args.min_links
    if args.weight is not None:             imp["weight"]             = args.weight
    if args.no_only_zero_targets:           imp["only_zero_targets"]  = False
    if args.no_keep_raw:                    imp["keep_raw"]           = False

    coaccess_thr = float(imp.get("coaccess_threshold", 0.05))
    min_links    = int(imp.get("min_links", 1))
    weight       = float(imp.get("weight", 1.0))
    only_zero    = bool(imp.get("only_zero_targets", True))
    keep_raw     = bool(imp.get("keep_raw", True))

    work_dir   = Path(cfg["paths"]["work_dir"])
    mm_dir     = Path(cfg["paths"]["mm"])
    impute_dir = Path(cfg["paths"]["impute"])
    coaccess_path = work_dir / "cicero" / "coaccess.tsv.gz"
    if not coaccess_path.exists():
        log.error("Missing co-accessibility table: %s — run 03_run_cicero.R first", coaccess_path)
        return 1

    log.info("Loading raw mm from %s", mm_dir)
    raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr().astype(np.float32)
    regions  = np.asarray(read_lines_gz(mm_dir / "regions.tsv.gz"))
    barcodes = np.asarray(read_lines_gz(mm_dir / "barcodes.tsv.gz"))
    n_reg, n_cells = raw.shape
    assert raw.shape == (len(regions), len(barcodes)), (
        f"shape mismatch: {raw.shape} vs ({len(regions)}, {len(barcodes)})"
    )
    log.info("Raw matrix: %d regions x %d cells, nnz=%d", n_reg, n_cells, raw.nnz)

    # Precompute lookup that accepts both colon-form and underscore-form names.
    region_to_idx: dict[str, int] = {}
    for i, r in enumerate(regions):
        region_to_idx[r] = i
        region_to_idx[to_underscore(r)] = i

    log.info("Loading co-accessibility table from %s", coaccess_path)
    df = pd.read_csv(coaccess_path, sep="\t")
    needed = {"Peak1", "Peak2", "coaccess"}
    if not needed.issubset(df.columns):
        log.error("Expected columns %s, got %s", needed, list(df.columns))
        return 2

    df = df.dropna(subset=["coaccess"])
    df = df[df["coaccess"] >= coaccess_thr]
    df = df[df["Peak1"].astype(str) != df["Peak2"].astype(str)]
    log.info("Links surviving threshold (coaccess >= %g): %d", coaccess_thr, len(df))
    if len(df) == 0:
        log.error("No links above threshold; nothing to propagate.")
        return 3

    rows = df["Peak1"].map(region_to_idx).to_numpy()
    cols = df["Peak2"].map(region_to_idx).to_numpy()
    scr  = df["coaccess"].to_numpy(dtype=np.float32)
    valid = pd.notna(rows) & pd.notna(cols)
    n_unmapped = int((~valid).sum())
    if n_unmapped:
        log.warning("Dropping %d / %d links whose peak names are not in mm/regions.tsv.gz",
                    n_unmapped, len(df))
    rows = rows[valid].astype(np.int64)
    cols = cols[valid].astype(np.int64)
    scr  = scr[valid]

    # Build symmetric adjacency (max in case both directions are present).
    C = sp.coo_matrix((scr, (rows, cols)), shape=(n_reg, n_reg)).tocsr()
    C = C.maximum(C.T)
    C.setdiag(0.0)
    C.eliminate_zeros()
    n_unique_links = int(C.nnz // 2)
    n_linked_peaks = int(np.unique(np.concatenate([
        np.repeat(np.arange(n_reg), np.diff(C.indptr)),
        C.indices,
    ])).size) if C.nnz else 0
    log.info("Adjacency: %d unique links touching %d peaks", n_unique_links, n_linked_peaks)

    # Indicator copy (same sparsity pattern, all-ones data).
    C_i = C.copy()
    C_i.data = np.ones_like(C_i.data, dtype=np.float32)

    raw_bool = (raw > 0).astype(np.float32).tocsr()

    log.info("prop_sum   = C   @ raw_bool ...")
    prop_sum = (C   @ raw_bool).tocsr()
    log.info("prop_count = C_i @ raw_bool ...")
    prop_count = (C_i @ raw_bool).tocsr()

    # Both share the same sparsity pattern (same multiplications), but
    # double-check before zip-iterating data arrays.
    assert prop_sum.shape == prop_count.shape == raw.shape
    if prop_sum.nnz != prop_count.nnz or not np.array_equal(prop_sum.indices, prop_count.indices):
        # Re-align by adding a zero matrix of the union pattern.
        union = (prop_sum + prop_count).tocsr()
        union.data[:] = 0.0
        prop_sum   = (prop_sum   + union).tocsr()
        prop_count = (prop_count + union).tocsr()

    # Drop entries with too few supporting links.
    if min_links > 1:
        keep = prop_count.data >= min_links
        prop_sum.data   = prop_sum.data   * keep
        prop_count.data = prop_count.data * keep
        prop_sum.eliminate_zeros()
        prop_count.eliminate_zeros()

    # value = weight * sum / count (count > 0 by construction now)
    safe_count = prop_count.data.copy()
    safe_count[safe_count == 0] = 1.0
    prop_value = sp.csr_matrix(
        (weight * prop_sum.data / safe_count,
         prop_sum.indices.copy(),
         prop_sum.indptr.copy()),
        shape=prop_sum.shape,
    )

    # Restrict to bins where raw[r, c] == 0
    if only_zero:
        raw_pos = (raw > 0).astype(np.float32)
        prop_value = prop_value - prop_value.multiply(raw_pos)
        prop_value.eliminate_zeros()

    propagated_nnz = int(prop_value.nnz)
    log.info("Propagated entries (post-filters): %d", propagated_nnz)

    # Combine with raw
    if keep_raw:
        raw_f = raw.astype(np.float32).tocsr()
        if only_zero:
            out = (raw_f + prop_value).tocsr()
        else:
            out = raw_f.maximum(prop_value).tocsr()
    else:
        out = prop_value.tocsr()
    out.eliminate_zeros()

    # Persist
    impute_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = impute_dir / "matrix_csr.npz"
    sp.save_npz(matrix_path, out)
    (impute_dir / "regions.tsv").write_text("\n".join(regions.tolist()) + "\n")
    (impute_dir / "barcodes.tsv").write_text("\n".join(barcodes.tolist()) + "\n")

    meta = {
        "pipeline": "Cicero",
        "shape":               [int(n_reg), int(n_cells)],
        "raw_nnz":             int(raw.nnz),
        "propagated_nnz":      propagated_nnz,
        "output_nnz":          int(out.nnz),
        "gain_vs_raw":         int(out.nnz) - int(raw.nnz),
        "coaccess_threshold":  coaccess_thr,
        "min_links":           min_links,
        "weight":              weight,
        "only_zero_targets":   only_zero,
        "keep_raw":            keep_raw,
        "n_links_used":        n_unique_links,
        "n_links_unmapped":    n_unmapped,
        "coaccess_table":      str(coaccess_path),
        "mm_dir":              str(mm_dir),
    }
    (impute_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s (nnz=%d) — gain over raw: %+d entries",
             matrix_path, out.nnz, out.nnz - raw.nnz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
