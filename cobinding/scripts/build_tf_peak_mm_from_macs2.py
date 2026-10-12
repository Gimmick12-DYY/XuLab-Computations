#!/usr/bin/env python3
"""Cell-by-peak matrix from a MACS2 narrowPeak and the fragment BED MACS2 used.

Peaks come from bigwig/macs2/TF.*/TF.*_peaks.narrowPeak. The MACS2 xls in
that same directory names the treatment file, bigwig/fragment/TF.*.bed.gz,
whose fourth column is the cell barcode. A cell is counted in a peak when
any of its tags overlaps the peak. The matrix is then binarized, because
Cicero binarizes before scoring.

The narrowPeak was called at p = 0.01. Peaks with q > 0.05 are dropped
using the -log10(q) column in that same file. That is the q <= 0.05 set.
"""

from __future__ import annotations

import argparse
import gzip
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def load_peaks(path: Path, max_q: float) -> list[tuple[str, int, int]]:
    min_logq = -math.log10(max_q) if max_q < 1 else 0.0
    peaks = []
    seen: set[tuple[str, int, int]] = set()
    n_dup = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            logq = float(parts[8])
            if logq < min_logq:
                continue
            key = (parts[0], int(parts[1]), int(parts[2]))
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            peaks.append(key)
    print(f"[pmat] dropped {n_dup:,} duplicate intervals", flush=True)
    if not peaks:
        print(f"[pmat] no peaks with q <= {max_q} in {path}", flush=True)
        raise SystemExit(2)
    return peaks


def count_overlaps(fragment_bed: Path, peaks: list[tuple[str, int, int]], bin_size: int = 10_000):
    # Peak index -> the 10 kb bins the peak touches. A tag only checks the
    # bins it touches, so a chromosome with tens of thousands of peaks stays cheap.
    bins: dict[str, dict[int, list[tuple[int, int, int]]]] = defaultdict(lambda: defaultdict(list))
    for i, (chrom, start, end) in enumerate(peaks):
        last = max(start, end - 1) // bin_size
        for b in range(start // bin_size, last + 1):
            bins[chrom][b].append((start, end, i))

    rows_i: list[int] = []
    cols_i: list[int] = []
    barcodes: dict[str, int] = {}
    n_tags = 0
    with gzip.open(fragment_bed, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split("\t")
            chrom = parts[0]
            chrom_bins = bins.get(chrom)
            if not chrom_bins:
                continue
            n_tags += 1
            start, end = int(parts[1]), int(parts[2])
            if end <= start:
                continue
            cell = None
            seen: set[int] = set()
            last = (end - 1) // bin_size
            for b in range(start // bin_size, last + 1):
                for pstart, pend, peak_i in chrom_bins.get(b, ()):
                    if peak_i in seen or pend <= start or pstart >= end:
                        continue
                    seen.add(peak_i)
                    if cell is None:
                        cell = barcodes.get(parts[3])
                        if cell is None:
                            cell = len(barcodes)
                            barcodes[parts[3]] = cell
                    rows_i.append(cell)
                    cols_i.append(peak_i)
    return rows_i, cols_i, barcodes, n_tags


def write_mm(path: Path, matrix: sp.spmatrix, regions, barcodes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(path / "matrix.mtx", matrix.T.tocoo())
    with open(path / "matrix.mtx", "rb") as src, gzip.open(path / "matrix.mtx.gz", "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    (path / "matrix.mtx").unlink()

    def write_lines(name: str, values) -> None:
        with gzip.open(path / name, "wt") as out:
            for value in values:
                out.write(f"{value}\n")

    write_lines("regions.tsv.gz", regions)
    write_lines("barcodes.tsv.gz", barcodes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--peaks", type=Path, required=True)
    ap.add_argument("--fragments", type=Path, required=True)
    ap.add_argument("--out-mm-dir", type=Path, required=True)
    ap.add_argument("--max-q", type=float, default=0.05)
    args = ap.parse_args()

    peaks = load_peaks(args.peaks, args.max_q)
    print(f"[pmat] {args.tf} peaks q<={args.max_q}: {len(peaks):,}", flush=True)
    print(f"[pmat] fragments {args.fragments}", flush=True)
    row, col, barcodes, n_tags = count_overlaps(args.fragments, peaks)
    print(f"[pmat] tags on peak chromosomes {n_tags:,}  cells {len(barcodes):,}  hits {len(row):,}",
          flush=True)
    if not barcodes or not row:
        raise SystemExit(f"{args.tf} matrix is empty")

    # Binarize. Cicero does this again; doing it here keeps the file small.
    mat = sp.coo_matrix(
        (np.ones(len(row), dtype=np.int32), (row, col)),
        shape=(len(barcodes), len(peaks)),
    ).tocsr()
    mat.data[:] = 1
    print(f"[pmat] cells={mat.shape[0]:,} peaks={mat.shape[1]:,} nnz={mat.nnz:,}", flush=True)
    regions = [f"{chrom}:{start}-{end}" for chrom, start, end in peaks]
    names = [bc for bc, _ in sorted(barcodes.items(), key=lambda kv: kv[1])]
    write_mm(args.out_mm_dir, mat, regions, names)
    print(f"[pmat] wrote {args.out_mm_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
