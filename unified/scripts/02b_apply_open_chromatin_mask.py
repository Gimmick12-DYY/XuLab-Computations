#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02b_apply_open_chromatin_mask.py
#
# Post-hoc open-chromatin mask for an existing unified impute CSR.
# Does NOT re-run the GPU model: reconstructs "new" imputed calls as
#   imputed_only = out AND NOT raw
# zeros those on closed (non-ATAC) bins, then re-unions with raw:
#   out' = raw.maximum(imputed_only_masked)
#
# Default writes in-place under <work>/impute/ (backs up prior CSR to
# matrix_csr.npz.pre_openmask). Use --out-dir to write a sibling directory.
#
# Usage:
#   python scripts/02b_apply_open_chromatin_mask.py --config configs/maz.yaml
#   python scripts/02b_apply_open_chromatin_mask.py --work-dir work/ctcf
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths
from _regions import open_chromatin_bin_mask

log = logging.getLogger("02b_openmask")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_lines(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Write masked CSR here (default: overwrite impute/ with backup).",
    )
    ap.add_argument(
        "--open-chromatin-bed",
        type=str,
        default=None,
        help="Override impute.open_chromatin_bed from config.",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="When overwriting impute/, do not keep matrix_csr.npz.pre_openmask.",
    )
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    i_cfg = cfg.setdefault("impute", {})
    open_bed = args.open_chromatin_bed or i_cfg.get("open_chromatin_bed")
    mask_imputed_only = bool(i_cfg.get("open_chromatin_mask_imputed_only", True))

    mm_dir = Path(cfg["paths"]["mm"])
    impute_dir = Path(cfg["paths"]["impute"])
    out_dir = Path(args.out_dir) if args.out_dir else impute_dir

    csr_path = impute_dir / "matrix_csr.npz"
    if not csr_path.is_file():
        log.error("Missing %s", csr_path)
        return 1
    if not (mm_dir / "matrix.mtx.gz").is_file():
        log.error("Missing raw mm %s", mm_dir / "matrix.mtx.gz")
        return 1

    regions_path = impute_dir / "regions.tsv"
    if not regions_path.is_file():
        regions_path = mm_dir / "regions.tsv.gz"
    regions = read_lines(regions_path)

    log.info("Loading imputed CSR %s", csr_path)
    out = sp.load_npz(csr_path).tocsr()
    log.info("Loading raw mm")
    raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr()
    if out.shape != raw.shape:
        log.error("Shape mismatch impute %s vs raw %s", out.shape, raw.shape)
        return 1
    if len(regions) != out.shape[0]:
        log.error("regions length %d != n_bins %d", len(regions), out.shape[0])
        return 1

    raw_bin = (raw != 0).astype(np.float32).tocsr()
    open_mask = open_chromatin_bin_mask(regions, open_bed)
    if open_mask is None:
        log.error("open_chromatin_bed is unset/off; nothing to do")
        return 1

    n_open = int(open_mask.sum())
    log.info(
        "Open chromatin: %d / %d bins (%.2f%%) from %s",
        n_open, out.shape[0], 100.0 * n_open / out.shape[0], open_bed,
    )

    # Imputed-only = entries in out that are absent from raw.
    out_coo = out.tocoo()
    raw_at = np.asarray(raw_bin[out_coo.row, out_coo.col]).ravel()
    is_new = raw_at == 0
    if mask_imputed_only:
        keep = is_new & open_mask[out_coo.row]
    else:
        # Hard mask: keep any out entry only if bin is open.
        keep = open_mask[out_coo.row]
    pred_masked = sp.coo_matrix(
        (np.ones(keep.sum(), dtype=np.float32),
         (out_coo.row[keep], out_coo.col[keep])),
        shape=out.shape,
    ).tocsr()
    n_pred_only = int(is_new.sum())
    log.info(
        "Imputed-only nnz %d; kept after open-chromatin mask %d (dropped %d)",
        n_pred_only, int(pred_masked.nnz),
        n_pred_only - int(pred_masked.nnz) if mask_imputed_only
        else int(out.nnz) - int(pred_masked.nnz),
    )

    if mask_imputed_only:
        # Preserve raw everywhere; add only open-chromatin imputed calls.
        new_out = raw_bin.maximum(pred_masked).tocsr()
    else:
        new_out = pred_masked.tocsr()
    new_out.eliminate_zeros()

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "matrix_csr.npz"
    if dest.resolve() == csr_path.resolve() and not args.no_backup:
        bak = impute_dir / "matrix_csr.npz.pre_openmask"
        if not bak.is_file():
            log.info("Backing up %s -> %s", csr_path, bak)
            shutil.copy2(csr_path, bak)
        else:
            log.info("Backup already exists: %s", bak)

    sp.save_npz(dest, new_out)

    # Copy labels if writing to a new dir
    for name in ("regions.tsv", "barcodes.tsv"):
        src = impute_dir / name
        if src.is_file() and (out_dir / name) != src:
            shutil.copy2(src, out_dir / name)

    meta_path = impute_dir / "meta.json"
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
    meta.update({
        "open_chromatin_bed": str(open_bed),
        "open_chromatin_n_open_bins": n_open,
        "open_chromatin_mask_imputed_only": mask_imputed_only,
        "open_chromatin_applied_posthoc": True,
        "predicted_nnz_pre_mask": n_pred_only,
        "predicted_nnz": int(pred_masked.nnz),
        "output_nnz": int(new_out.nnz),
        "raw_nnz": int(raw_bin.nnz),
        "gain_vs_raw": int(new_out.nnz) - int(raw_bin.nnz),
        "shape": [int(new_out.shape[0]), int(new_out.shape[1])],
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info(
        "Wrote %s  nnz %d -> %d  gain_vs_raw %+d",
        dest, int(out.nnz), int(new_out.nnz), int(new_out.nnz) - int(raw_bin.nnz),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
