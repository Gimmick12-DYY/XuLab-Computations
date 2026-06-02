#!/usr/bin/env python
# -----------------------------------------------------------------------------
# build_peak_matrix.py
#
# Build a binary peak x cell sparse matrix RDS from a CUT&Tag fragment BED.
# The output is a dgCMatrix saved as .rds that drops directly into
# paths.input_rds of the scBasset / scBasset_TF pipelines.
#
# Inputs:
#   --fragments TF.CTCF.bed.gz       5-col BED: chr, start, end, barcode, count
#   --out      CTCF_peak_matrix.rds  final dgCMatrix RDS
#
# Steps (1:1 with the design plan):
#   1. Tool check (macs2 / bedtools / Rscript on PATH).
#   2. Blacklist auto-discovery + download (reuses repo cache convention).
#   3. MACS2 callpeak with CUT&Tag params:
#        --nomodel --shift -75 --extsize 150 -q 0.01 --keep-dup all
#   4. Peak filter: chr1..22 + chrX/Y, drop blacklist overlaps.
#   5. bedtools intersect (streamed) -> (peak_idx, barcode) pairs.
#   6. Build binary CSR.
#   7. Filter: peaks observed in < min_cells_per_peak cells, then cells with
#      < min_peaks_per_cell peaks. Order matters.
#   8. Write mm intermediate (matrix.mtx.gz + regions.tsv.gz + barcodes.tsv.gz).
#   9. Call save_peak_matrix.R to materialize the dgCMatrix RDS.
#  10. Print final dims + sparsity.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

log = logging.getLogger("build_peak_matrix")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHROM_WHITELIST = {f"chr{c}" for c in [str(i) for i in range(1, 23)] + ["X", "Y"]}
BLACKLIST_URL = (
    "https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/"
    "lists/hg38-blacklist.v2.bed.gz"
)


# ---------- env / setup ------------------------------------------------------

def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"required tool not on PATH: {name}")


def default_blacklist_path(work_dir: Path) -> Path:
    """Mirror scBasset_TF/scripts/02_prepare_seqs.py:78-86 — probe
    <repo>/downstream/cache/hg38-blacklist.v2.bed.gz from any ancestor."""
    wd = work_dir.resolve()
    for anc in [wd, *wd.parents]:
        cand = anc / "downstream" / "cache" / "hg38-blacklist.v2.bed.gz"
        if cand.is_file():
            return cand
    return work_dir / "hg38-blacklist.v2.bed.gz"


def ensure_blacklist(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading blacklist %s -> %s", BLACKLIST_URL, path)
    req = urllib.request.Request(
        BLACKLIST_URL, headers={"User-Agent": "build_peak_matrix"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        path.write_bytes(resp.read())
    return path


# ---------- MACS2 ------------------------------------------------------------

def run_macs2(fragments: Path, work_dir: Path, tf_name: str) -> Path:
    """Run MACS2 callpeak in CUT&Tag mode. Returns the narrowPeak path."""
    name = f"{tf_name}_pseudo"
    cmd = [
        "macs2", "callpeak",
        "-t", str(fragments),
        "-f", "BED",
        "-g", "hs",
        "-n", name,
        "--outdir", str(work_dir),
        "--nomodel",
        "--shift", "-75",
        "--extsize", "150",
        "-q", "0.01",
        "--keep-dup", "all",
    ]
    log.info("MACS2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    peaks_path = work_dir / f"{name}_peaks.narrowPeak"
    if not peaks_path.is_file():
        sys.exit(f"MACS2 did not produce expected output: {peaks_path}")
    return peaks_path


# ---------- peak filtering ---------------------------------------------------

def load_intervals(bed_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Load BED into {chrom: [(start, end), ...]} (per-chrom unsorted)."""
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    opener = gzip.open if str(bed_path).endswith(".gz") else open
    with opener(bed_path, "rt") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                out[parts[0]].append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return out


def overlap_mask(
    chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
    intervals: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    """True at rows whose [start, end) overlaps any interval on the same chrom.
    Same vectorized per-chrom scan as scBasset_TF/scripts/02_prepare_seqs.py."""
    hit = np.zeros(len(chroms), dtype=bool)
    by_chr: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by_chr[str(c)].append(i)
    for chrom, idx_list in by_chr.items():
        iv = intervals.get(chrom) or intervals.get(chrom.replace("chr", "")) or []
        if not iv:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        s_sub, e_sub = starts[idx], ends[idx]
        local = np.zeros(len(idx), dtype=bool)
        for bs, be in iv:
            local |= np.maximum(s_sub, bs) < np.minimum(e_sub, be)
        hit[idx[local]] = True
    return hit


def filter_peaks(
    narrow_peak: Path, blacklist: Path, work_dir: Path
) -> tuple[Path, list[tuple[str, int, int]]]:
    """Filter narrowPeak by chrom whitelist + blacklist. Returns (bed path, peaks)."""
    chroms_l, starts_l, ends_l = [], [], []
    with open(narrow_peak) as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in CHROM_WHITELIST:
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            chroms_l.append(chrom)
            starts_l.append(start)
            ends_l.append(end)

    if not chroms_l:
        sys.exit("no peaks remain after chrom whitelist; aborting.")

    chroms = np.asarray(chroms_l, dtype=object)
    starts = np.asarray(starts_l, dtype=np.int64)
    ends   = np.asarray(ends_l,   dtype=np.int64)
    log.info("Peaks after chrom whitelist: %d", len(chroms))

    bl_iv = load_intervals(blacklist)
    bl_hit = overlap_mask(chroms, starts, ends, bl_iv)
    keep = ~bl_hit
    log.info("Peaks after blacklist filter: %d (dropped %d)",
             int(keep.sum()), int(bl_hit.sum()))
    if not keep.any():
        sys.exit("no peaks remain after blacklist filter; aborting.")

    out_bed = work_dir / "peaks_filtered.bed"
    peaks: list[tuple[str, int, int]] = []
    with open(out_bed, "w") as fh:
        for c, s, e, k in zip(chroms.tolist(), starts.tolist(), ends.tolist(), keep.tolist()):
            if k:
                fh.write(f"{c}\t{s}\t{e}\n")
                peaks.append((c, s, e))
    log.info("Wrote %s (%d peaks)", out_bed, len(peaks))
    return out_bed, peaks


# ---------- fragment -> (peak, cell) assignment ------------------------------

def build_binary_matrix(
    fragments: Path, peaks_bed: Path, peaks: list[tuple[str, int, int]]
) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Stream `bedtools intersect` and accumulate (peak_idx, cell_idx) pairs."""
    log.info("Streaming bedtools intersect -> sparse matrix")
    peak_to_idx = {(c, s, e): i for i, (c, s, e) in enumerate(peaks)}
    region_names = [f"{c}:{s}-{e}" for c, s, e in peaks]

    cmd = [
        "bedtools", "intersect",
        "-a", str(fragments),
        "-b", str(peaks_bed),
        "-wa", "-wb",
    ]
    log.info("bedtools: %s", " ".join(cmd))

    barcode_to_idx: dict[str, int] = {}
    pairs: set[tuple[int, int]] = set()
    n_lines = 0
    n_unmatched = 0

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        for ln in proc.stdout:
            n_lines += 1
            parts = ln.rstrip("\n").split("\t")
            # 5-col BED (-a) + 3-col BED (-b) = 8 fields; the BED can have
            # extra columns but we only need [3] (barcode), [5], [6], [7].
            if len(parts) < 8:
                continue
            barcode = parts[3]
            peak_chrom = parts[5]
            try:
                peak_start = int(parts[6])
                peak_end = int(parts[7])
            except ValueError:
                continue
            peak_idx = peak_to_idx.get((peak_chrom, peak_start, peak_end))
            if peak_idx is None:
                n_unmatched += 1
                continue
            cell_idx = barcode_to_idx.get(barcode)
            if cell_idx is None:
                cell_idx = len(barcode_to_idx)
                barcode_to_idx[barcode] = cell_idx
            pairs.add((peak_idx, cell_idx))
    finally:
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"bedtools intersect exited with {proc.returncode}")

    n_peaks = len(peaks)
    n_cells = len(barcode_to_idx)
    log.info("Streamed %d intersect lines (%d unmatched peak coords)",
             n_lines, n_unmatched)
    log.info("Unique (peak, cell) pairs: %d ; barcodes: %d", len(pairs), n_cells)

    if not pairs:
        sys.exit("no (peak, cell) pairs after intersect; aborting.")

    rows = np.fromiter((p for p, _ in pairs), dtype=np.int64, count=len(pairs))
    cols = np.fromiter((c for _, c in pairs), dtype=np.int64, count=len(pairs))
    data = np.ones(len(pairs), dtype=np.int8)
    mat = sp.coo_matrix((data, (rows, cols)), shape=(n_peaks, n_cells)).tocsr()

    barcodes: list[str | None] = [None] * n_cells
    for bc, i in barcode_to_idx.items():
        barcodes[i] = bc
    barcodes_out: list[str] = [b for b in barcodes if b is not None]
    assert len(barcodes_out) == n_cells

    return mat, region_names, barcodes_out


# ---------- filtering --------------------------------------------------------

def filter_matrix(
    mat: sp.csr_matrix,
    regions: list[str],
    barcodes: list[str],
    min_cells_per_peak: int,
    min_peaks_per_cell: int,
) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Drop low-evidence peaks first, then low-coverage cells."""
    cells_per_peak = np.asarray((mat > 0).sum(axis=1)).ravel()
    keep_peaks = cells_per_peak >= min_cells_per_peak
    log.info("Peak filter (>= %d cells): %d -> %d",
             min_cells_per_peak, mat.shape[0], int(keep_peaks.sum()))
    mat = mat[keep_peaks]
    regions = [r for r, k in zip(regions, keep_peaks.tolist()) if k]

    peaks_per_cell = np.asarray((mat > 0).sum(axis=0)).ravel()
    keep_cells = peaks_per_cell >= min_peaks_per_cell
    log.info("Cell filter (>= %d peaks): %d -> %d",
             min_peaks_per_cell, mat.shape[1], int(keep_cells.sum()))
    mat = mat[:, keep_cells]
    barcodes = [b for b, k in zip(barcodes, keep_cells.tolist()) if k]

    if mat.shape[0] == 0 or mat.shape[1] == 0:
        sys.exit("matrix collapsed to empty after filtering; aborting.")

    return mat.tocsr(), regions, barcodes


# ---------- mm intermediate + R RDS ------------------------------------------

def write_mm(
    mat: sp.csr_matrix, regions: list[str], barcodes: list[str], work_dir: Path
) -> Path:
    mm_dir = work_dir / "mm"
    mm_dir.mkdir(parents=True, exist_ok=True)
    mtx_path = mm_dir / "matrix.mtx"
    log.info("Writing %s", mtx_path)
    sio.mmwrite(str(mtx_path), mat.astype(np.int32))
    subprocess.run(["gzip", "-f", str(mtx_path)], check=True)

    with gzip.open(mm_dir / "regions.tsv.gz", "wt") as fh:
        fh.write("\n".join(regions) + "\n")
    with gzip.open(mm_dir / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(barcodes) + "\n")
    return mm_dir


def save_as_rds(mm_dir: Path, out_path: Path) -> None:
    r_script = Path(__file__).resolve().parent / "save_peak_matrix.R"
    if not r_script.is_file():
        sys.exit(f"missing companion script: {r_script}")
    cmd = ["Rscript", str(r_script), "--mm-dir", str(mm_dir), "--out", str(out_path)]
    log.info("RDS: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------- main -------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fragments", required=True, type=Path,
                    help="5-col BED(.gz): chr, start, end, barcode, count")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .rds path (dgCMatrix)")
    ap.add_argument("--tf-name", default="CTCF",
                    help="Used in MACS2 sample name. Default: CTCF.")
    ap.add_argument("--blacklist", type=Path, default=None,
                    help="hg38 blacklist BED. Auto-discovered/downloaded if None.")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="Intermediate files. Default: <out>.work/")
    ap.add_argument("--min-cells-per-peak", type=int, default=2)
    ap.add_argument("--min-peaks-per-cell", type=int, default=200)
    args = ap.parse_args()

    for tool in ("macs2", "bedtools", "Rscript", "gzip"):
        require_tool(tool)

    fragments = args.fragments.expanduser().resolve()
    out_path  = args.out.expanduser().resolve()
    if not fragments.is_file():
        sys.exit(f"fragments not found: {fragments}")
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else out_path.parent / f"{out_path.stem}.work"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("work_dir: %s", work_dir)

    bl_path = args.blacklist or default_blacklist_path(work_dir)
    blacklist = ensure_blacklist(bl_path)
    log.info("blacklist: %s", blacklist)

    # 1. peak call
    narrow = run_macs2(fragments, work_dir, args.tf_name)

    # 2. filter
    peaks_bed, peaks = filter_peaks(narrow, blacklist, work_dir)

    # 3. assign + build binary
    mat, regions, barcodes = build_binary_matrix(fragments, peaks_bed, peaks)

    # 4. filter peaks + cells
    mat, regions, barcodes = filter_matrix(
        mat, regions, barcodes,
        min_cells_per_peak=args.min_cells_per_peak,
        min_peaks_per_cell=args.min_peaks_per_cell,
    )

    # 5. mm + RDS
    mm_dir = write_mm(mat, regions, barcodes, work_dir)
    save_as_rds(mm_dir, out_path)

    # 6. report
    n_rows, n_cols = mat.shape
    n_pos = int(mat.nnz)
    n_total = int(n_rows) * int(n_cols)
    sparsity = 1.0 - (n_pos / n_total) if n_total else 0.0
    print()
    print("=" * 64)
    print(f"  Output:   {out_path}")
    print(f"  Matrix:   {n_rows:,} peaks x {n_cols:,} cells")
    print(f"  nnz:      {n_pos:,}")
    print(f"  Sparsity: {sparsity:.6f}   ({(1.0 - sparsity) * 100:.4f}% nonzero)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
