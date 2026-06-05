#!/usr/bin/env python
# -----------------------------------------------------------------------------
# peak_coverage.py
#
# Companion to compare_pos_neg.py's BIN-level coverage / AUROC: for each
# imputation pipeline, CALL PEAKS from the imputed matrix with MACS3, then
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
#   5. Run MACS3 callpeak with the standard CUT&Tag params:
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
#                                   MACS3 output, called peak BED.
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
    """Write a synthetic 3-col BED. Each bin emits floor(scaled_signal) 1-bp
    lines at the bin's MIDPOINT. This is the critical fix: with MACS3's
    `--shift -75 --extsize 150`, a fragment at (chrom, start, start+1) becomes
    a 150 bp peak window centered at start. If we emit at the bin's start
    (left edge), the resulting peak straddles bin boundaries and bleeds into
    the previous bin -- inflating peak coverage above bin coverage. Emitting
    at the bin midpoint keeps the peak fully contained inside the source bin
    (1 kb bins >> 150 bp peak width), so peak coverage <= bin coverage by
    construction.

    Returns the total line count.
    """
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

    # Bin midpoints (anchor for MACS3 fragments). 1 bp wide so MACS3's
    # --extsize 150 picks up the full 150 bp window centered at the midpoint.
    centers = (starts + ends) // 2

    keep = counts > 0
    chroms_k  = chroms[keep]
    centers_k = centers[keep]
    counts_k  = counts[keep]
    log.info("  bins contributing fragments: %d / %d (emitted at bin midpoint)",
             int(keep.sum()), len(signal))

    chroms_r  = np.repeat(chroms_k, counts_k)
    centers_r = np.repeat(centers_k, counts_k)

    # Buffered write to avoid huge in-memory line list.
    CHUNK = 1_000_000
    with open(out_bed, "w") as fh:
        for i in range(0, n_total, CHUNK):
            cs = chroms_r[i:i + CHUNK]
            ms = centers_r[i:i + CHUNK].tolist()
            fh.write("\n".join(f"{c}\t{m}\t{m+1}" for c, m in zip(cs, ms)))
            fh.write("\n")
    log.info("  wrote synthetic BED: %s (%d lines)", out_bed, n_total)
    return n_total


# ---------- MACS3 ------------------------------------------------------------

def run_macs3(bed_path: Path, work_dir: Path, name: str) -> Path:
    """Run MACS3 in CUT&Tag mode. Returns the narrowPeak path."""
    cmd = [
        "macs3", "callpeak",
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
    log.info("  MACS3: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    peaks_path = work_dir / f"{name}_peaks.narrowPeak"
    if not peaks_path.is_file():
        sys.exit(f"MACS3 did not produce expected output: {peaks_path}")
    return peaks_path


def load_called_peaks(narrow: Path) -> list[tuple[str, int, int]]:
    """Parse MACS3 narrowPeak -> list of (chrom, start, end)."""
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

def write_ref_bed(reference_bins: list[tuple[str, int, int]], out_bed: Path) -> None:
    """Write reference bin set to a 3-col BED, sorted by (chrom, start)."""
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    sorted_bins = sorted(reference_bins, key=lambda r: (r[0], r[1]))
    with open(out_bed, "w") as fh:
        for chrom, s, e in sorted_bins:
            fh.write(f"{chrom}\t{s}\t{e}\n")


def coverage_via_bedtools(
    ref_bed: Path,
    called_peaks_bed: Path,
    min_overlap_frac: float = 0.0,
    covered_out_bed: Path | None = None,
    pairs_out_tsv: Path | None = None,
) -> tuple[int, int]:
    """Use `bedtools intersect -u` to count reference bins overlapping any
    called peak. `-u` is the "soft unique" semantics — each reference bin is
    returned at most once regardless of how many peaks hit it.

    Optional artifacts:
      covered_out_bed:  if given, write the BED of reference bins that ARE
                        covered (i.e. the `-u` output). 3-col BED, subset of
                        ref_bed. Useful for IGV / cross-pipeline diffs.
      pairs_out_tsv:    if given, write the per-overlap pairing as a TSV
                        with columns: ref_chrom, ref_start, ref_end,
                        peak_chrom, peak_start, peak_end, overlap_bp.
                        Generated via `bedtools intersect -wo`.

    `min_overlap_frac > 0` adds `-f <frac>` (require that fraction of the
    REFERENCE bin be covered). 0 = any overlap counts (the default and the
    semantic the bin-coverage metric uses).
    """
    n_total = sum(1 for ln in ref_bed.read_text().splitlines() if ln.strip())
    if n_total == 0:
        return 0, 0

    # 1. -u to count + write the covered subset BED
    cmd = ["bedtools", "intersect",
           "-a", str(ref_bed),
           "-b", str(called_peaks_bed),
           "-u"]
    if min_overlap_frac > 0:
        cmd += ["-f", f"{min_overlap_frac:.6g}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    covered_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    n_covered = len(covered_lines)

    if covered_out_bed is not None:
        covered_out_bed.parent.mkdir(parents=True, exist_ok=True)
        covered_out_bed.write_text("\n".join(covered_lines) + ("\n" if covered_lines else ""))

    # 2. (optional) -wo for the full per-overlap pairing
    if pairs_out_tsv is not None:
        cmd_wo = ["bedtools", "intersect",
                  "-a", str(ref_bed),
                  "-b", str(called_peaks_bed),
                  "-wo"]
        if min_overlap_frac > 0:
            cmd_wo += ["-f", f"{min_overlap_frac:.6g}"]
        proc_wo = subprocess.run(cmd_wo, capture_output=True, text=True, check=True)
        pairs_out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(pairs_out_tsv, "w") as fh:
            fh.write("ref_chrom\tref_start\tref_end\tpeak_chrom\tpeak_start\tpeak_end\toverlap_bp\n")
            for ln in proc_wo.stdout.splitlines():
                if ln.strip():
                    fh.write(ln + "\n")

    return n_covered, n_total


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
    ap.add_argument("--min-overlap-frac", type=float, default=0.0,
                    help="Pass-through to `bedtools intersect -f`. "
                         "0 (default) = any overlap counts (same semantic as bin coverage). "
                         "Set >0 to require that fraction of the reference bin be covered.")
    ap.add_argument("--emit-pairs", action="store_true",
                    help="Also emit per-pipeline overlap TSV (bedtools -wo): "
                         "one row per (ref bin, called peak) pair with overlap bp.")
    args = ap.parse_args()

    for tool in ("macs3", "bedtools"):
        if shutil.which(tool) is None:
            sys.exit(
                f"{tool} not found on PATH; activate the data_prep env "
                f"(pip macs3 + conda bedtools) or another env with both."
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading reference bins from %s", args.bins_tsv)
    pos_bins, neg_bins = load_reference_bins(args.bins_tsv)

    # Write reference bins to BED once; bedtools intersect uses them per run.
    ref_pos_bed = args.out_dir / "_ref_pos.bed"
    ref_neg_bed = args.out_dir / "_ref_neg.bed"
    write_ref_bed(pos_bins, ref_pos_bed)
    write_ref_bed(neg_bins, ref_neg_bed)
    log.info("Wrote reference BEDs: %s, %s", ref_pos_bed, ref_neg_bed)

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

        narrow = run_macs3(bed_path, pipe_dir, name=f"{label}_pseudo")
        peaks = load_called_peaks(narrow)
        log.info("  called peaks: %d", len(peaks))

        # Save peaks BED (sorted) next to MACS3 outputs for visual inspection
        # AND for bedtools intersect.
        bed_called = pipe_dir / f"{label}_called_peaks.bed"
        sorted_peaks = sorted(peaks, key=lambda r: (r[0], r[1]))
        with open(bed_called, "w") as fh:
            for c, s, e in sorted_peaks:
                fh.write(f"{c}\t{s}\t{e}\n")

        covered_pos_bed = pipe_dir / f"{label}_pos_covered_bins.bed"
        covered_neg_bed = pipe_dir / f"{label}_neg_covered_bins.bed"
        pairs_pos_tsv = pipe_dir / f"{label}_pos_overlap_pairs.tsv" if args.emit_pairs else None
        pairs_neg_tsv = pipe_dir / f"{label}_neg_overlap_pairs.tsv" if args.emit_pairs else None

        n_pos_cov, n_pos = coverage_via_bedtools(
            ref_pos_bed, bed_called,
            min_overlap_frac=args.min_overlap_frac,
            covered_out_bed=covered_pos_bed,
            pairs_out_tsv=pairs_pos_tsv,
        )
        n_neg_cov, n_neg = coverage_via_bedtools(
            ref_neg_bed, bed_called,
            min_overlap_frac=args.min_overlap_frac,
            covered_out_bed=covered_neg_bed,
            pairs_out_tsv=pairs_neg_tsv,
        )
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
            "min_overlap_frac":          float(args.min_overlap_frac),
            "n_pos":                     int(n_pos),
            "n_neg":                     int(n_neg),
            "n_pos_covered":             int(n_pos_cov),
            "n_neg_covered":             int(n_neg_cov),
            "positive_peak_coverage_pct": round(pos_cov, 4),
            "negative_peak_coverage_pct": round(neg_cov, 4),
            "input_path":                str(path),
            "narrowpeak":                str(narrow),
            "called_peaks_bed":          str(bed_called),
            "covered_pos_bed":           str(covered_pos_bed),
            "covered_neg_bed":           str(covered_neg_bed),
            "pos_overlap_pairs_tsv":     str(pairs_pos_tsv) if pairs_pos_tsv else "",
            "neg_overlap_pairs_tsv":     str(pairs_neg_tsv) if pairs_neg_tsv else "",
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
