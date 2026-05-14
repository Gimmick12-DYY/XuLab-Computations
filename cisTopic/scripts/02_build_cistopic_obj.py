#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_build_cistopic_obj.py
#
# Load the Matrix Market files written by 01_export_rds_to_mm.R, apply the
# per-region / per-cell filters defined in the config, and pickle a
# pycisTopic CistopicObject that every subsequent step consumes.
#
# When filter.exclude_blacklist is true (default), bins overlapping the ENCODE
# hg38 blacklist v2 are removed first — same BED and overlap convention as
# downstream/prepare_pos_neg_bins.R and Bin_posVSneg.ipynb.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import logging
import pickle
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import sklearn.preprocessing as skp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("02_build")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")
_BLACKLIST_URL_DEFAULT = (
    "https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/"
    "lists/hg38-blacklist.v2.bed.gz"
)


def read_lines_gz(path: str) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def default_blacklist_bed_gz(work_dir: Path) -> Path:
    """Prefer an existing XuLab/downstream/cache file; else infer default path for download."""
    wd = work_dir.resolve()
    for anc in [wd, *wd.parents]:
        cand = anc / "downstream" / "cache" / "hg38-blacklist.v2.bed.gz"
        if cand.is_file():
            return cand
    if len(wd.parents) >= 3:
        return wd.parents[2] / "downstream" / "cache" / "hg38-blacklist.v2.bed.gz"
    return wd / "hg38-blacklist.v2.bed.gz"


def ensure_blacklist_file(path: Path, url: str) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return path
    log.info("Downloading blacklist %s -> %s", url, path)
    req = urllib.request.Request(url, headers={"User-Agent": "cisTopic-pipeline/02_build"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path


def load_blacklist_intervals0(bed_gz: Path) -> dict[str, list[tuple[int, int]]]:
    """BED chrom, 0-based start, end exclusive -> half-open intervals grouped by chrom."""
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    opener = gzip.open if str(bed_gz).endswith(".gz") else open
    with opener(bed_gz, "rt") as fh:  # type: ignore[arg-type]
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom, s, e = parts[0], int(parts[1]), int(parts[2])
            if e <= s:
                continue
            out[chrom].append((s, e))
    return out


def parse_region_intervals0(region_names: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse chr:start-end as half-open 0-based [start, end)."""
    n = len(region_names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(region_names):
        m = _REGION_RE.match(str(name).strip())
        if not m:
            raise SystemExit(f"Bad region name row {i}: {name!r} (expected chr:start-end)")
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i] = int(m.group(3))
        if ends[i] <= starts[i]:
            raise SystemExit(f"Empty or inverted interval row {i}: {name!r}")
    return chroms, starts, ends


def mask_blacklist_overlaps(
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    blacklist: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    """Boolean mask: True = drop row (overlaps blacklist)."""
    drop = np.zeros(len(chroms), dtype=bool)
    by_chr: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by_chr[str(c)].append(i)
    for chrom, idx_list in by_chr.items():
        bl_iv = blacklist.get(chrom) or blacklist.get(chrom.replace("chr", "")) or []
        if not bl_iv:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        s_sub = starts[idx]
        e_sub = ends[idx]
        local_drop = np.zeros(len(idx), dtype=bool)
        for bs, be in bl_iv:
            local_drop |= np.maximum(s_sub, bs) < np.minimum(e_sub, be)
        drop[idx[local_drop]] = True
    return drop


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--min-counts-per-region", type=int, default=None)
    ap.add_argument("--min-cells-per-region",  type=int, default=None)
    ap.add_argument("--min-regions-per-cell",  type=int, default=None)
    ap.add_argument("--no-binarize", action="store_true")
    ap.add_argument("--no-blacklist", action="store_true",
                    help="Skip ENCODE blacklist filtering even if enabled in config.")
    ap.add_argument("--blacklist-bed", type=str, default=None,
                    help="Override path to hg38-blacklist.v2.bed.gz (or uncompressed .bed).")
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
    if args.no_blacklist:
        flt["exclude_blacklist"] = False

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

    # -- blacklist (match downstream/prepare_pos_neg_bins.R + Bin_posVSneg.ipynb) ---
    n_bl_drop = 0
    if flt.get("exclude_blacklist", False):
        wd = Path(cfg["paths"]["work_dir"])
        if args.blacklist_bed:
            bl_path = Path(args.blacklist_bed).expanduser().resolve()
        elif flt.get("blacklist_bed_gz"):
            bl_path = Path(str(flt["blacklist_bed_gz"])).expanduser().resolve()
        else:
            bl_path = default_blacklist_bed_gz(wd)
        url = str(flt.get("blacklist_url") or _BLACKLIST_URL_DEFAULT)
        bl_path = ensure_blacklist_file(bl_path, url)
        bl_iv = load_blacklist_intervals0(bl_path)
        rchrom, rstart, rend = parse_region_intervals0(regions)
        drop = mask_blacklist_overlaps(rchrom, rstart, rend, bl_iv)
        n_bl_drop = int(drop.sum())
        if n_bl_drop:
            keep = ~drop
            n_tot = int(drop.size)
            mat = mat[keep]
            regions = regions[keep]
            log.info(
                "Blacklist %s: dropped %d / %d regions (%.2f%%).",
                bl_path,
                n_bl_drop,
                n_tot,
                100.0 * n_bl_drop / max(n_tot, 1),
            )
        else:
            log.info("Blacklist %s: no overlapping regions in this matrix.", bl_path)
    else:
        log.info("Blacklist filtering disabled (exclude_blacklist=false).")

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
        "blacklist_dropped": int(n_bl_drop),
        "exclude_blacklist": bool(flt.get("exclude_blacklist", False)),
        "obj_path":       str(out),
    }
    (Path(cfg["paths"]["obj"]) / "cistopic_obj.meta.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n"
    )
    log.info("Done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
