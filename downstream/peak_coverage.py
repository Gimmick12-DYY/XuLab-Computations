#!/usr/bin/env python
# -----------------------------------------------------------------------------
# peak_coverage.py
#
# Companion to compare_pos_neg.py's BIN-level coverage / AUROC: for each
# imputation pipeline, CALL PEAKS from the imputed matrix with MACS2, then
# compute the fraction of reference positive/negative bins covered by any
# called peak.
#
# Per-pipeline workflow:
#   1. Load the (region x cell) matrix from the input dir.
#        - mm/         (raw)           : matrix.mtx.gz + regions.tsv.gz + barcodes.tsv.gz
#        - impute/     (csr_full)      : matrix_csr.npz + regions.tsv + barcodes.tsv
#        - impute/     (factored)      : factors.npz + regions.tsv (cisTopic / scOpen)
#        - impute/     (dense)         : matrix.npy + regions.tsv (FITS / MAGIC)
#   2. Pseudo-bulk: per-bin signal = sum across cells (binary or continuous).
#   3. Rescale so the total simulated "fragment" count equals --target-fragments
#      (default 50,000,000 -- a typical CUT&Tag sample depth). This
#      normalizes across pipelines whose continuous-vs-binary signal scales
#      are wildly different.
#   4. Emit a synthetic 3-col BED: each bin contributes
#      `floor(scaled_signal)` lines of (chrom, start, end).
#   5. Run MACS2 callpeak with the standard CUT&Tag params:
#        --nomodel --shift -75 --extsize 150 -q 0.01 --keep-dup all
#   6. Parse the narrowPeak output -> list of called peak intervals.
#
# Coverage step:
#   - Reference bins come from prepare_pos_neg_bins.R via the bins TSV
#     (`bin_name = chr:start-end`, `label ∈ {1, 0}` for pos / neg).
#   - For each reference bin, mark it covered if ANY called peak overlaps.
#   - positive_peak_coverage = #pos covered / #pos
#     negative_peak_coverage = #neg covered / #neg
#
# Outputs (under --out-dir):
#   peak_coverage.tsv               one row per input
#   <label>/                        per-pipeline subdir with synthetic BED,
#                                   MACS2 output, called peak BED.
#
# Usage:
#   python downstream/peak_coverage.py \
#     --input raw=/work/.../mm \
#     --input cisTopic=/work/.../cisTopic/work/ctcf/impute \
#     --input PUscOpen=/work/.../PUscOpen/work/ctcf/impute \
#     --input scBasset=/work/.../scBasset/work/ctcf/impute \
#     --input scBasset_PUscOpen=/work/.../scBasset_PUscOpen/work/ctcf/impute \
#     --input scBasset_cisTopic=/work/.../scBasset_cisTopic/work/ctcf/impute \
#     --input scBasset_cisTopic_PUscOpen=/work/.../scBasset_cisTopic_PUscOpen/work/ctcf/impute \
#     --bins-tsv /work/.../downstream/bins/pos_neg_bins.tsv \
#     --out-dir  /work/.../downstream/peak_coverage_ctcf
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

log = logging.getLogger("peak_coverage")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------- I/O helpers ------------------------------------------------------

def read_lines(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def parse_region_names(names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 'chr:start-end' strings into chroms / starts / ends arrays."""
    n = len(names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends   = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(names):
        try:
            c, rest = name.split(":", 1)
            s, e = rest.split("-")
            chroms[i] = c
            starts[i] = int(s)
            ends[i]   = int(e)
        except (ValueError, AttributeError):
            sys.exit(f"bad region name at row {i}: {name!r}")
    return chroms, starts, ends


# ---------- per-input loader -------------------------------------------------

def _per_cell_factor_from_npz(z: np.lib.npyio.NpzFile) -> tuple[bool, np.ndarray | None, float]:
    """Return (is_vector, vector_or_none, scalar)."""
    if "per_cell_factor" not in z.files:
        return False, None, 1.0
    pcf_raw = z["per_cell_factor"]
    if getattr(pcf_raw, "ndim", 0) > 0:
        return True, np.asarray(pcf_raw, dtype=np.float64).ravel(), 1.0
    return False, None, float(pcf_raw)


def load_per_bin_signal_factored(path: Path) -> tuple[np.ndarray, list[str], str]:
    factors_path = path / "factors.npz"
    regions_path = path / "regions.tsv"
    log.info("  loading factored: %s + %s", factors_path, regions_path)
    regions = read_lines(regions_path)
    with np.load(factors_path) as z:
        W = np.asarray(z["W"], dtype=np.float64)
        H = np.asarray(z["H"], dtype=np.float64)
        pcf_is_array, pcf_arr, pcf_scalar = _per_cell_factor_from_npz(z)
        if W.shape[1] != H.shape[0]:
            sys.exit(f"rank mismatch in {factors_path}: W{W.shape} vs H{H.shape}")
        if pcf_is_array:
            assert pcf_arr is not None
            if pcf_arr.shape[0] != H.shape[1]:
                sys.exit(
                    f"per_cell_factor length {pcf_arr.shape[0]} != H.shape[1] {H.shape[1]} "
                    f"in {factors_path}"
                )
            signal = W @ (H @ pcf_arr)
        else:
            signal = (W @ H.sum(axis=1)) * pcf_scalar
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.shape[0] != len(regions):
        sys.exit(
            f"factored row count {signal.shape[0]} != regions {len(regions)} in {path}"
        )
    return signal, regions, "factored"


def load_per_bin_signal_dense(path: Path) -> tuple[np.ndarray, list[str], str]:
    matrix_path = path / "matrix.npy"
    regions_path = path / "regions.tsv"
    log.info("  loading dense: %s + %s", matrix_path, regions_path)
    arr = np.load(matrix_path, mmap_mode="r")
    signal = np.asarray(arr.sum(axis=1)).ravel().astype(np.float64)
    regions = read_lines(regions_path)
    if signal.shape[0] != len(regions):
        sys.exit(
            f"dense row count {signal.shape[0]} != regions {len(regions)} in {path}"
        )
    return signal, regions, "dense"


def load_per_bin_signal(path: Path) -> tuple[np.ndarray, list[str], str]:
    """Return (per_bin_signal float64, region_names, kind)."""
    if (path / "matrix_csr.npz").is_file() and (path / "regions.tsv").is_file():
        log.info("  loading impute CSR: %s", path / "matrix_csr.npz")
        mat = sp.load_npz(path / "matrix_csr.npz").tocsr()
        signal = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
        regions = read_lines(path / "regions.tsv")
        return signal, regions, "csr_full"

    if (path / "factors.npz").is_file() and (path / "regions.tsv").is_file():
        return load_per_bin_signal_factored(path)

    if (path / "matrix.npy").is_file() and (path / "regions.tsv").is_file():
        return load_per_bin_signal_dense(path)

    if (path / "matrix.mtx.gz").is_file():
        log.info("  loading raw mm: %s", path / "matrix.mtx.gz")
        mat = sio.mmread(str(path / "matrix.mtx.gz")).tocsr()
        signal = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
        regions = read_lines(path / "regions.tsv.gz")
        return signal, regions, "mm"

    sys.exit(
        f"no matrix found in {path} "
        "(expected matrix.mtx.gz, matrix_csr.npz, factors.npz, or matrix.npy)"
    )


# ---------- synthetic BED ----------------------------------------------------

def generate_synthetic_bed(
    signal: np.ndarray, regions: list[str],
    target_fragments: int, out_bed: Path,
) -> int:
    """Write a synthetic 3-col BED. Each bin emits floor(scaled_signal)
    repeated lines at the bin's coordinates. Returns the total line count."""
    chroms, starts, ends = parse_region_names(regions)
    total_signal = float(signal.sum())
    if total_signal <= 0:
        sys.exit(f"per-bin signal sum is zero; cannot synthesize fragments")

    scale = target_fragments / total_signal
    counts = np.floor(signal * scale).astype(np.int64)
    n_total = int(counts.sum())
    log.info("  scaling: total_signal=%.3g  scale=%.6g  n_synthetic_fragments=%d",
             total_signal, scale, n_total)
    if n_total == 0:
        sys.exit("scaled fragment count is zero; raise --target-fragments")

    keep = counts > 0
    chroms_k = chroms[keep]
    starts_k = starts[keep]
    ends_k   = ends[keep]
    counts_k = counts[keep]
    log.info("  bins contributing fragments: %d / %d",
             int(keep.sum()), len(signal))

    chroms_r = np.repeat(chroms_k, counts_k)
    starts_r = np.repeat(starts_k, counts_k)
    ends_r   = np.repeat(ends_k,   counts_k)

    # Buffered write to avoid 50M-line list allocation.
    CHUNK = 1_000_000
    with open(out_bed, "w") as fh:
        for i in range(0, n_total, CHUNK):
            cs = chroms_r[i:i + CHUNK]
            ss = starts_r[i:i + CHUNK].tolist()
            es = ends_r[i:i + CHUNK].tolist()
            fh.write("\n".join(f"{c}\t{s}\t{e}" for c, s, e in zip(cs, ss, es)))
            fh.write("\n")
    log.info("  wrote synthetic BED: %s (%d lines)", out_bed, n_total)
    return n_total


# ---------- MACS2 ------------------------------------------------------------

def run_macs2(bed_path: Path, work_dir: Path, name: str) -> Path:
    """Run MACS2 in CUT&Tag mode. Returns the narrowPeak path."""
    cmd = [
        "macs2", "callpeak",
        "-t", str(bed_path),
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
    log.info("  MACS2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    peaks_path = work_dir / f"{name}_peaks.narrowPeak"
    if not peaks_path.is_file():
        sys.exit(f"MACS2 did not produce expected output: {peaks_path}")
    return peaks_path


def load_called_peaks(narrow: Path) -> list[tuple[str, int, int]]:
    """Parse MACS2 narrowPeak -> list of (chrom, start, end)."""
    peaks: list[tuple[str, int, int]] = []
    with open(narrow) as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                peaks.append((parts[0], int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return peaks


# ---------- reference bins ---------------------------------------------------

def load_reference_bins(bins_tsv: Path) -> tuple[list, list]:
    """Parse pos_neg_bins.tsv. Returns (positives, negatives) where each is
    a list of (chrom, start, end). Accepts label in {1/0} or {pos/neg}."""
    pos: list[tuple[str, int, int]] = []
    neg: list[tuple[str, int, int]] = []
    with open(bins_tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            col_label = header.index("label")
        except ValueError:
            sys.exit(f"bins TSV {bins_tsv} missing 'label' column. header={header}")
        try:
            col_name = header.index("bin_name")
        except ValueError:
            sys.exit(f"bins TSV {bins_tsv} missing 'bin_name' column. header={header}")

        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) <= max(col_label, col_name):
                continue
            label = parts[col_label].strip()
            name = parts[col_name].strip()
            try:
                chrom, rest = name.split(":", 1)
                s, e = rest.split("-")
                entry = (chrom, int(s), int(e))
            except (ValueError, AttributeError):
                continue
            if label in ("1", "pos", "positive"):
                pos.append(entry)
            elif label in ("0", "neg", "negative"):
                neg.append(entry)
    log.info("Reference bins: %d positives, %d negatives", len(pos), len(neg))
    return pos, neg


# ---------- overlap ----------------------------------------------------------

def coverage_fraction(
    reference_bins: list[tuple[str, int, int]],
    called_peaks: list[tuple[str, int, int]],
) -> tuple[int, int]:
    """Return (#bins overlapping any peak, total #bins)."""
    if not reference_bins:
        return 0, 0
    peaks_by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chrom, s, e in called_peaks:
        peaks_by_chrom[chrom].append((s, e))
    for chrom in peaks_by_chrom:
        peaks_by_chrom[chrom].sort()
    starts_by_chrom: dict[str, np.ndarray] = {
        c: np.asarray([s for s, _ in lst], dtype=np.int64)
        for c, lst in peaks_by_chrom.items()
    }
    ends_by_chrom: dict[str, np.ndarray] = {
        c: np.asarray([e for _, e in lst], dtype=np.int64)
        for c, lst in peaks_by_chrom.items()
    }

    n_covered = 0
    for chrom, start, end in reference_bins:
        if chrom not in starts_by_chrom:
            continue
        s_arr = starts_by_chrom[chrom]
        e_arr = ends_by_chrom[chrom]
        # candidates: peaks with start < end
        i = int(np.searchsorted(s_arr, end))
        # any of [0..i) with end > start ?
        if i > 0 and np.any(e_arr[:i] > start):
            n_covered += 1
    return n_covered, len(reference_bins)


# ---------- main -------------------------------------------------------------

def parse_input_arg(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--input must be label=path; got {s!r}")
    label, p = s.split("=", 1)
    return label, Path(p).expanduser().resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", action="append", required=True,
                    type=parse_input_arg,
                    metavar="LABEL=PATH", dest="inputs",
                    help="Repeatable. PATH is mm/ (raw) or impute/ (csr_full).")
    ap.add_argument("--bins-tsv", required=True, type=Path,
                    help="prepare_pos_neg_bins.R output (label + bin_name).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--target-fragments", type=int, default=50_000_000,
                    help="Total simulated fragments per pipeline (default 50M, typical CUT&Tag).")
    ap.add_argument("--out-name", default="peak_coverage")
    args = ap.parse_args()

    if shutil.which("macs2") is None:
        sys.exit("macs2 not found on PATH; activate the data_prep env or load the cluster module")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading reference bins from %s", args.bins_tsv)
    pos_bins, neg_bins = load_reference_bins(args.bins_tsv)

    rows: list[dict] = []
    for label, path in args.inputs:
        log.info("=== %s :: %s ===", label, path)
        if not path.is_dir():
            log.warning("  skip: %s is not a directory", path)
            continue
        pipe_dir = args.out_dir / label
        pipe_dir.mkdir(parents=True, exist_ok=True)

        signal, regions, kind = load_per_bin_signal(path)
        log.info("  matrix: %d regions, total_signal=%.3g, kind=%s",
                 len(regions), float(signal.sum()), kind)

        bed_path = pipe_dir / "synthetic.bed"
        n_frag = generate_synthetic_bed(signal, regions, args.target_fragments, bed_path)

        narrow = run_macs2(bed_path, pipe_dir, name=f"{label}_pseudo")
        peaks = load_called_peaks(narrow)
        log.info("  called peaks: %d", len(peaks))

        # Save peaks BED next to MACS2 outputs for visual inspection.
        bed_called = pipe_dir / f"{label}_called_peaks.bed"
        with open(bed_called, "w") as fh:
            for c, s, e in peaks:
                fh.write(f"{c}\t{s}\t{e}\n")

        n_pos_cov, n_pos = coverage_fraction(pos_bins, peaks)
        n_neg_cov, n_neg = coverage_fraction(neg_bins, peaks)
        pos_cov = (100.0 * n_pos_cov / n_pos) if n_pos else float("nan")
        neg_cov = (100.0 * n_neg_cov / n_neg) if n_neg else float("nan")
        log.info("  pos cov: %d / %d (%.2f%%) ; neg cov: %d / %d (%.2f%%)",
                 n_pos_cov, n_pos, pos_cov, n_neg_cov, n_neg, neg_cov)

        rows.append({
            "run":                       label,
            "kind":                      kind,
            "n_bins_in_matrix":          int(len(regions)),
            "total_signal":              float(signal.sum()),
            "target_fragments":          int(args.target_fragments),
            "n_synthetic_fragments":     int(n_frag),
            "n_called_peaks":            int(len(peaks)),
            "called_peak_total_bp":      int(sum(e - s for _, s, e in peaks)),
            "n_pos":                     int(n_pos),
            "n_neg":                     int(n_neg),
            "n_pos_covered":             int(n_pos_cov),
            "n_neg_covered":             int(n_neg_cov),
            "positive_peak_coverage_pct": round(pos_cov, 4),
            "negative_peak_coverage_pct": round(neg_cov, 4),
            "input_path":                str(path),
            "narrowpeak":                str(narrow),
            "called_peaks_bed":          str(bed_called),
        })

    # Write the TSV summary
    tsv_path = args.out_dir / f"{args.out_name}.tsv"
    if rows:
        cols = list(rows[0].keys())
        with open(tsv_path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        log.info("Wrote %s (%d rows)", tsv_path, len(rows))
    else:
        log.warning("No rows produced; nothing to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
