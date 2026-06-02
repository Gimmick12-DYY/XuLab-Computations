#!/usr/bin/env python
# -----------------------------------------------------------------------------
# build_peak_matrix.py
#
# Standard MACS2 peak matrix builder. Turns a CUT&Tag fragment BED into a
# binary peak x cell sparse matrix RDS for the scBasset / scBasset_TF
# pipelines.
#
# Pipeline:
#   1. MACS2 callpeak (CUT&Tag params)
#        --nomodel --shift -75 --extsize 150 -q 0.01 --keep-dup all
#   2. bedtools intersect fragments x peaks (streamed)
#   3. Build sparse (peak x cell) binary matrix
#   4. Write mm intermediate -> R subprocess -> dgCMatrix RDS
#
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

log = logging.getLogger("build_peak_matrix")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"required tool not on PATH: {name}")


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


def parse_peaks(narrow_peak: Path, work_dir: Path) -> tuple[Path, list[tuple[str, int, int]]]:
    """Read MACS2 narrowPeak, emit a 3-col BED unchanged. No filtering."""
    peaks: list[tuple[str, int, int]] = []
    with open(narrow_peak) as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            peaks.append((parts[0], start, end))
    if not peaks:
        sys.exit("MACS2 produced no peaks; aborting.")

    out_bed = work_dir / "peaks.bed"
    with open(out_bed, "w") as fh:
        for c, s, e in peaks:
            fh.write(f"{c}\t{s}\t{e}\n")
    log.info("Peaks parsed: %d (wrote %s)", len(peaks), out_bed)
    return out_bed, peaks


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
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="Intermediate files. Default: <out>.work/")
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

    narrow = run_macs2(fragments, work_dir, args.tf_name)
    peaks_bed, peaks = parse_peaks(narrow, work_dir)
    mat, regions, barcodes = build_binary_matrix(fragments, peaks_bed, peaks)

    mm_dir = write_mm(mat, regions, barcodes, work_dir)
    save_as_rds(mm_dir, out_path)

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
