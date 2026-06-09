#!/usr/bin/env python
# -----------------------------------------------------------------------------
# peak_coverage.py
#
# Per-pipeline workflow:
#   1. Load the matrix from each input dir.
#   2. Per-bin signal = sum across cells.
#   3. PER-BIN scaling: count_r = floor(signal_r * frags_at_ref / ref_val),
#      where ref_val is a within-method quantile of the nonzero signal. This is
#      NOT normalized to a fixed total (the old scheme); the absolute scale
#      cancels in MACS3's fold, so per-total vs per-bin only changes which
#      low-signal bins survive floor(). Per-total normalization made that
#      survival threshold grow with the number of signaled bins, penalizing
#      dense (well-imputed) matrices -- per-bin scaling removes that.
#   4. Write synthetic BED: fragments clustered within --spread-bp of each
#      bin's midpoint.
#   5. MACS3 callpeak in CUT&Tag mode (defaults):
#        --nomodel --shift -100 --extsize 200 -q 0.05 --keep-dup all --nolambda
#      --nolambda scores each bin against a single background, not the local
#      1k/5k/10k lambda -- without it a method that signals most of the genome
#      (scBasset, ~77% of bins) inflates its own local background and rejects
#      peaks that are obviously real in IGV.
#      That background is NOT the full hs genome (which is mostly empty, so the
#      bar is ~0 and every signaled bin passes -> capture==100%, peak calling a
#      pass-through). Instead -g is set per-method to n_frag*extsize/bar_frags,
#      where bar_frags is the pile-up of a bin at --bg-quantile of the track's
#      nonzero signal. So a bin is called only if it beats the track's own
#      typical signaled level -> peak calling discriminates within the track.
#   6. bedtools intersect the narrowPeak against the reference pos/neg bins.
#
# Outputs (under --out-dir):
#   peak_coverage.tsv                       one row per input
#   _ref_pos.bed, _ref_neg.bed              reference BEDs (shared)
#   <label>/synthetic.bed                   synthetic fragments fed to MACS3
#   <label>/<label>_pseudo_peaks.narrowPeak MACS3 output
#   <label>/<label>_called_peaks.bed        sorted 3-col BED for bedtools intersect
#   <label>/<label>_pos_covered_bins.bed    pos bins overlapping any peak
#   <label>/<label>_neg_covered_bins.bed    neg bins overlapping any peak
#
# Usage:
#   python downstream/peak_coverage.py \
#     --input raw=/work/.../mm \
#     --input scBasset=/work/.../scBasset/work/ctcf/impute \
#     ... \
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
from scipy.stats import rankdata

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


def load_per_bin_signal_factored(path: Path) -> tuple[np.ndarray, list[str], str, int]:
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
        n_cells = int(H.shape[1])
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.shape[0] != len(regions):
        sys.exit(
            f"factored row count {signal.shape[0]} != regions {len(regions)} in {path}"
        )
    return signal, regions, "factored", n_cells


def load_per_bin_signal_dense(path: Path) -> tuple[np.ndarray, list[str], str, int]:
    matrix_path = path / "matrix.npy"
    regions_path = path / "regions.tsv"
    log.info("  loading dense: %s + %s", matrix_path, regions_path)
    arr = np.load(matrix_path, mmap_mode="r")
    signal = np.asarray(arr.sum(axis=1)).ravel().astype(np.float64)
    n_cells = int(arr.shape[1])
    regions = read_lines(regions_path)
    if signal.shape[0] != len(regions):
        sys.exit(
            f"dense row count {signal.shape[0]} != regions {len(regions)} in {path}"
        )
    return signal, regions, "dense", n_cells


def load_per_bin_signal(path: Path) -> tuple[np.ndarray, list[str], str, int]:
    """Return (per_bin_signal float64, region_names, kind, n_cells)."""
    if (path / "matrix_csr.npz").is_file() and (path / "regions.tsv").is_file():
        log.info("  loading impute CSR: %s", path / "matrix_csr.npz")
        mat = sp.load_npz(path / "matrix_csr.npz").tocsr()
        signal = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
        regions = read_lines(path / "regions.tsv")
        return signal, regions, "csr_full", int(mat.shape[1])

    if (path / "factors.npz").is_file() and (path / "regions.tsv").is_file():
        return load_per_bin_signal_factored(path)

    if (path / "matrix.npy").is_file() and (path / "regions.tsv").is_file():
        return load_per_bin_signal_dense(path)

    if (path / "matrix.mtx.gz").is_file():
        log.info("  loading raw mm: %s", path / "matrix.mtx.gz")
        mat = sio.mmread(str(path / "matrix.mtx.gz")).tocsr()
        signal = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
        regions = read_lines(path / "regions.tsv.gz")
        return signal, regions, "mm", int(mat.shape[1])

    sys.exit(
        f"no matrix found in {path} "
        "(expected matrix.mtx.gz, matrix_csr.npz, factors.npz, or matrix.npy)"
    )


# ---------- synthetic BED ----------------------------------------------------

# ---------- synthetic BED generation -----------------------------------------

def generate_synthetic_bed(
    signal: np.ndarray, regions: list[str],
    out_bed: Path,
    *,
    spread_bp: int = 150,
    ref_quantile: float = 0.5,
    frags_at_ref: float = 20.0,
    max_frags_per_bin: int = 500,
    max_total_fragments: int = 150_000_000,
    seed: int = 42,
) -> tuple[int, float, float]:
    """Write a synthetic 3-col BED for MACS3 callpeak.

    Each bin emits floor(signal_r * scale) fragments at RANDOM positions
    centered on the bin midpoint within a `spread_bp` window.

    SCALING (per-bin, not per-total). The fragment count for a bin is
        count_r = floor(signal_r * scale),  scale = frags_at_ref / ref_val
    where ref_val = the `ref_quantile` quantile of the NONZERO signal. This
    replaces the old `scale = target_fragments / total_signal`. The absolute
    scale CANCELS in MACS3's fold test (it multiplies both the pile-up and the
    background lambda), so it does not change relative peak significance -- its
    only jobs are (a) floor() survival and (b) total file size. The old
    per-total scaling made the floor survival threshold `1/scale =
    total_signal/target` GROW with the number of signaled bins, so dense
    (well-imputed) matrices had their low-signal CTCF bins categorically zeroed
    before peak calling. Anchoring to a within-method quantile makes the
    threshold `ref_val / frags_at_ref` independent of how many bins are
    signaled -- removing that density penalty. Per-bin counts are clipped to
    `max_frags_per_bin` (so a single outlier bin, e.g. PUscOpen, can't blow up
    the file) and the whole set is uniformly downscaled if the total would
    exceed `max_total_fragments` (fold-neutral under --nolambda).

    `spread_bp` is the dial from commit 0e0900a: small -> tight pile-up ->
    higher fold; large (~1 kb bin) -> MACS3 rejects diffuse bins. With
    --nolambda the fold is `signal_r / genome_avg_signal`, so spread_bp only
    sets pile-up width (and the minimum-peak width via --extsize), not the
    background -- the self-raised-local-lambda penalty on dense matrices is
    gone.

    The peak-calling background bar (which depends on bg_quantile, bg_scale and
    extsize) is computed OUTSIDE this function (see `bar_frags_for` /
    `effective_genome_size`) so it can be swept without regenerating the BED --
    the synthetic fragments depend only on the scaling args, not on q / bg_scale
    / extsize.

    Returns (total fragments written, scale, downscale) so the caller can derive
    the background bar for any bg_scale.
    """
    chroms, starts, ends = parse_region_names(regions)
    nz = signal > 0
    if not nz.any():
        sys.exit("per-bin signal is all zero; cannot synthesize fragments")

    ref_val = float(np.quantile(signal[nz], ref_quantile))
    if ref_val <= 0:                      # quantile landed on a zero/tiny bin
        ref_val = float(signal[nz].min())
    scale = frags_at_ref / ref_val
    raw = np.minimum(signal * scale, float(max_frags_per_bin))
    predicted = float(raw.sum())
    downscale = 1.0
    if predicted > max_total_fragments:
        downscale = max_total_fragments / predicted
        raw = raw * downscale
    counts = np.floor(raw).astype(np.int64)
    n_total = int(counts.sum())
    floor_thresh = ref_val / frags_at_ref / max(downscale, 1e-12)
    log.info("  scaling(per-bin): ref_q=%.3g ref_val=%.3g scale=%.6g "
             "floor_thresh=%.3g downscale=%.3g n_synthetic_fragments=%d",
             ref_quantile, ref_val, scale, floor_thresh, downscale, n_total)
    if n_total == 0:
        sys.exit("scaled fragment count is zero; raise --frags-at-ref")

    keep = counts > 0
    chroms_k = chroms[keep]
    starts_k = starts[keep]
    ends_k   = ends[keep]
    counts_k = counts[keep]
    log.info("  bins contributing fragments: %d / %d (cluster within %d bp around midpoint)",
             int(keep.sum()), len(signal), spread_bp)

    # Cluster fragments in a `spread_bp` window centered on each bin's midpoint.
    # Cap the window to the bin's own width so we don't drop fragments outside
    # of small/edge bins.
    centers_k    = (starts_k + ends_k) // 2
    bin_widths_k = (ends_k - starts_k).astype(np.int64)
    half_window  = np.minimum(spread_bp // 2, bin_widths_k // 2)

    chroms_r     = np.repeat(chroms_k, counts_k)
    centers_r    = np.repeat(centers_k, counts_k)
    half_r       = np.repeat(half_window, counts_k)

    rng = np.random.default_rng(seed)
    # Random offset in [-half, +half], inclusive
    offsets = rng.integers(-half_r, half_r + 1)
    positions = centers_r + offsets

    CHUNK = 1_000_000
    with open(out_bed, "w") as fh:
        for i in range(0, n_total, CHUNK):
            cs = chroms_r[i:i + CHUNK]
            ps = positions[i:i + CHUNK].tolist()
            fh.write("\n".join(f"{c}\t{p}\t{p+1}" for c, p in zip(cs, ps)))
            fh.write("\n")
    log.info("  wrote synthetic BED: %s (%d lines)", out_bed, n_total)
    return n_total, scale, downscale


def bar_frags_for(bar_signal: float, scale: float, downscale: float,
                  max_frags_per_bin: int, bg_scale: float) -> int:
    """Background bar in fragments = pile-up of a bin at `bar_signal` (the
    bg_quantile of the nonzero signal), under the same clip+downscale transform
    as the synthetic counts, times the continuous `bg_scale` multiplier.
    Returns 0 when disabled (-> caller falls back to the full hs genome)."""
    if bar_signal <= 0 or bg_scale <= 0:
        return 0
    bar_raw = min(bar_signal * scale, float(max_frags_per_bin)) * downscale * bg_scale
    return max(1, int(np.floor(bar_raw)))


def effective_genome_size(n_frag: int, extsize: int, bar_frags: int) -> str | int:
    """Effective genome size that makes the --nolambda genome-wide background
    equal to `bar_frags` pile-up over an extsize window. bar_frags == 0 -> 'hs'."""
    if bar_frags <= 0:
        return "hs"
    return max(1_000_000, int(n_frag * extsize / bar_frags))


# ---------- MACS3 ------------------------------------------------------------

def run_macs3(
    bed_path: Path, work_dir: Path, name: str,
    qvalue: float = 0.05, extsize: int = 200, nolambda: bool = True,
    effective_genome_size: str | int = "hs",
) -> Path:
    """Run MACS3 in CUT&Tag mode. Returns the narrowPeak path.

    `extsize` sets the pile-up (and minimum peak) width; `--shift` is pinned
    to -extsize/2 so the extended tag stays centered on the cut site. Wider
    extsize -> wider called peaks -> more likely to overlap a ~1 kb reference
    bin, so it lifts coverage on top of the span/q knobs. `qvalue` is the
    MACS3 significance cutoff; loosening it (0.01 -> 0.05) admits the
    marginal CTCF pile-ups that the smoothed pipelines produce.

    `nolambda` (default True) passes MACS3 `--nolambda`, which tests each bin
    against the GENOME-WIDE background only, not the local 1k/5k/10k lambda.
    This is the fix for the dense-imputation bias: with local lambda, a method
    that signals most of the genome (scBasset signals ~77% of bins) inflates
    the background in the 10 kb window around every CTCF site, collapsing the
    fold and rejecting genuine peaks that are obviously real in IGV. With a
    single genome-wide background the denominator is identical for every bin,
    so a CTCF bin is called iff its own signal exceeds the method's genome
    average -- which is what the eye reads off the browser track.
    """
    shift = -(extsize // 2)
    cmd = [
        "macs3", "callpeak",
        "-t", str(bed_path),
        "-f", "BED",
        "-g", str(effective_genome_size),
        "-n", name,
        "--outdir", str(work_dir),
        "--nomodel",
        "--shift", str(shift),
        "--extsize", str(extsize),
        "-q", f"{qvalue:g}",
        "--keep-dup", "all",
    ]
    if nolambda:
        cmd.append("--nolambda")
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


# ---------- AUROC + table formatting -----------------------------------------

def signal_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUROC of per-bin signal discriminating positive vs negative reference
    bins (Mann-Whitney U; tie-corrected via average ranks). Zeros included, so
    it scores the full reference universe. Equivalent to
    P(signal at a random pos bin > signal at a random neg bin)."""
    n1, n0 = len(pos_scores), len(neg_scores)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos_scores, neg_scores]))
    r1 = float(ranks[:n1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _fmt_count(v: float) -> str:
    """K/M-abbreviated count, e.g. 76 -> '76', 9800 -> '9.8K', 1.04e6 -> '1.04M'."""
    if not np.isfinite(v):
        return "NA"
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:.2f}M"
    if av >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:.0f}"


def format_summary_table(rows: list[dict], macs_q: float) -> str:
    """Render the slide-style fixed-width table. `raw` rows are pushed last."""
    ordered = [r for r in rows if r["run"] != "raw"] + [r for r in rows if r["run"] == "raw"]

    cols = [
        # (header, key, formatter, align)
        ("Pipeline",            lambda r: str(r["run"]),                                  "<"),
        ("Valued bins",         lambda r: f'{r["n_nonzero_bins"]:,}',                     ">"),
        ("Mean UMI/cell",       lambda r: _fmt_count(r["mean_umi_per_cell"]),             ">"),
        ("Pos.cov %",           lambda r: f'{r["pos_signal_ceiling_pct"]:.1f}',           ">"),
        ("Neg.cov %",           lambda r: f'{r["neg_signal_ceiling_pct"]:.1f}',           ">"),
        ("Pos in peaks %",      lambda r: f'{r["positive_peak_coverage_pct"]:.3g}',       ">"),
        ("Neg in peaks %",      lambda r: f'{r["negative_peak_coverage_pct"]:.3g}',       ">"),
        (f"Peaks (q={macs_q:g})", lambda r: f'{r["n_called_peaks"]:,}',                   ">"),
        ("AUROC",               lambda r: f'{r["auroc"]:.4f}',                            ">"),
    ]

    cells = [[fmt(r) for _, fmt, _ in cols] for r in ordered]
    widths = [max(len(h), *(len(row[i]) for row in cells)) if cells else len(h)
              for i, (h, _, _) in enumerate(cols)]

    def line(values: list[str]) -> str:
        return "  ".join(f"{v:{cols[i][2]}{widths[i]}}" for i, v in enumerate(values))

    header = line([h for h, _, _ in cols])
    sep = "-" * len(header)
    body = "\n".join(line(row) for row in cells)
    return f"{header}\n{sep}\n{body}"


def _parse_num_list(s: str | None, fallback: float, want_int: bool = False) -> list:
    """Parse a comma list like '0.01,0.05,0.1' -> [0.01,0.05,0.1]. None/'' -> [fallback]."""
    if not s:
        return [int(fallback) if want_int else float(fallback)]
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(float(tok)) if want_int else float(tok))
    return out or [int(fallback) if want_int else float(fallback)]


def format_q_sweep_block(rows: list[dict], q_list: list[float],
                         bg_scale: float, extsize: int) -> str:
    """Arrow-style table for one (bg_scale, extsize): rows = pipelines, with
    Peaks / Pos-in-peaks / Neg-in-peaks shown as 'v1 -> v2 -> v3' across q_list.
    Mirrors the slide format. `raw` rows pushed last."""
    sub = [r for r in rows
           if abs(r["bg_scale"] - bg_scale) < 1e-9 and r["extsize"] == extsize]
    runs = list(dict.fromkeys(r["run"] for r in sub))
    runs = [r for r in runs if r != "raw"] + [r for r in runs if r == "raw"]

    def cell(run: str, key: str, fmt: str) -> str:
        by_q = {r["macs_q"]: r for r in sub if r["run"] == run}
        return " -> ".join(
            (format(by_q[q][key], fmt) if q in by_q else "NA") for q in q_list)

    def first(run: str, key: str, fmt: str) -> str:
        for r in sub:
            if r["run"] == run:
                return format(r[key], fmt)
        return "NA"

    qhead = "->".join(f"{q:g}" for q in q_list)
    cols = [
        ("Pipeline",            lambda run: run,                                              "<"),
        (f"Peaks (q={qhead})",  lambda run: cell(run, "n_called_peaks", ",d"),                ">"),
        ("Pos in peaks %",      lambda run: cell(run, "positive_peak_coverage_pct", ".3g"),   ">"),
        ("Neg in peaks %",      lambda run: cell(run, "negative_peak_coverage_pct", ".3g"),   ">"),
        ("Pos ceiling %",       lambda run: first(run, "pos_signal_ceiling_pct", ".1f"),      ">"),
        ("AUROC",               lambda run: first(run, "auroc", ".4f"),                       ">"),
    ]
    cells = [[fn(run) for _, fn, _ in cols] for run in runs]
    widths = [max(len(h), *(len(row[i]) for row in cells)) if cells else len(h)
              for i, (h, _, _) in enumerate(cols)]

    def line(vals: list[str]) -> str:
        return "  ".join(f"{v:{cols[i][2]}{widths[i]}}" for i, v in enumerate(vals))

    title = f"=== bg_scale={bg_scale:g}  extsize={extsize}  (q sweep: {qhead}) ==="
    header = line([h for h, _, _ in cols])
    sep = "-" * len(header)
    body = "\n".join(line(row) for row in cells)
    return f"{title}\n{header}\n{sep}\n{body}"


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
    ap.add_argument("--target-fragments", type=int, default=150_000_000,
                    dest="max_total_fragments",
                    help="Safety CAP on total simulated fragments per pipeline (default 150M). "
                         "Scaling is now per-bin (see --frags-at-ref), not normalized to this "
                         "total; the set is only uniformly downscaled if it would exceed the cap "
                         "(fold-neutral under --nolambda). Kept under the old flag name for "
                         "backward compatibility with the sbatch wrapper.")
    ap.add_argument("--ref-quantile", type=float, default=0.5,
                    help="Quantile of the NONZERO per-bin signal used as the scaling anchor "
                         "(default 0.5 = median). The floor() survival threshold is "
                         "ref_val/frags-at-ref, independent of how many bins are signaled -- "
                         "this removes the density penalty the old total-normalization imposed "
                         "on well-imputed (dense) matrices.")
    ap.add_argument("--frags-at-ref", type=float, default=20.0,
                    help="Fragments emitted by a bin whose signal equals the --ref-quantile "
                         "anchor (default 20). Larger -> more low-signal bins survive floor(). "
                         "Absolute value cancels in MACS3's fold, so it only affects floor "
                         "survival and file size.")
    ap.add_argument("--max-frags-per-bin", type=int, default=500,
                    help="Per-bin cap on synthetic fragments (default 500) so a single outlier "
                         "bin (e.g. PUscOpen's concentrated signal) can't dominate the file.")
    ap.add_argument("--bg-quantile", type=float, default=0.5,
                    help="Quantile of the NONZERO signal used as the peak-calling background "
                         "bar (default 0.5 = median). A bin is called only if its signal beats "
                         "this quantile of its own track -- this makes peak calling discriminate "
                         "WITHIN the track instead of passing every signaled bin (capture==100%%). "
                         "Higher -> stricter (more filtering, lower coverage); 0 -> disable (use "
                         "the full hs genome, the old pass-through behaviour).")
    ap.add_argument("--bg-scale", type=float, default=1.0,
                    help="Continuous multiplier on the background bar (default 1.0). Use this "
                         "instead of --bg-quantile when the nonzero signal is piled at 1 (median "
                         "== 1), where quantiles all collapse to the same bar. <1 lowers the bar "
                         "(recovers coverage), >1 raises it (stricter). Sweep to maximize the "
                         "pos-neg coverage gap.")
    ap.add_argument("--nolambda", action=argparse.BooleanOptionalAction, default=True,
                    help="Pass MACS3 --nolambda (default ON): score each bin against the "
                         "genome-wide background only, not the local 1k/5k/10k lambda. This is "
                         "the fix for dense-imputation bias -- a method that signals most of the "
                         "genome no longer inflates its own local background and rejects real "
                         "CTCF peaks. Use --no-nolambda to restore local-lambda behavior.")
    ap.add_argument("--macs-q", type=float, default=0.05,
                    help="MACS3 callpeak q-value cutoff (default 0.05). Looser than the "
                         "0.01 default to admit the marginal CTCF pile-ups from smoothed "
                         "(imputed) matrices that otherwise fall just under threshold.")
    ap.add_argument("--spread-bp", type=int, default=150,
                    help="Width (bp) of the cluster window around each bin's midpoint that "
                         "fragments are spread across. With --nolambda this sets pile-up width "
                         "only (background is genome-wide), not the fold; keep it <= --extsize. "
                         "Default 150.")
    ap.add_argument("--extsize", type=int, default=200,
                    help="MACS3 --extsize (bp); --shift is pinned to -extsize/2. Default 200 "
                         "(vs the 150 CUT&Tag standard) widens called peaks so they overlap "
                         "the ~1 kb reference bins more readily, raising coverage.")
    ap.add_argument("--q-sweep", type=str, default=None,
                    help="Comma list of MACS3 q cutoffs to sweep, e.g. '0.01,0.05,0.1'. "
                         "The synthetic BED is generated ONCE per pipeline and reused across "
                         "the whole grid. Overrides --macs-q. Default: just --macs-q.")
    ap.add_argument("--bg-scale-sweep", type=str, default=None,
                    help="Comma list of bg_scale multipliers to sweep, e.g. '1.0,0.9,0.8'. "
                         "Lower -> lower background bar -> more coverage (watch neg). "
                         "Overrides --bg-scale. Default: just --bg-scale.")
    ap.add_argument("--extsize-sweep", type=str, default=None,
                    help="Comma list of extsize values to sweep, e.g. '200,300'. "
                         "Overrides --extsize. Default: just --extsize.")
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

    # Reference bin names ("chr:start-end") for the signal-ceiling diagnostic.
    # The ceiling = fraction of reference bins with ANY nonzero matrix signal
    # *before* peak calling. It is the hard upper bound on coverage: a bin with
    # no signal can never be in a peak. Comparing ceiling vs. realized coverage
    # tells us whether a low number is the method's fault (low ceiling -> the
    # imputation never put signal there) or the peak-caller's (high ceiling but
    # low coverage -> spread/floor/q is dropping it).
    pos_names = {f"{c}:{s}-{e}" for c, s, e in pos_bins}
    neg_names = {f"{c}:{s}-{e}" for c, s, e in neg_bins}

    # Sweep grid: q x bg_scale x extsize. Single values reproduce the old
    # behavior. The synthetic BED is generated once per pipeline (it depends
    # only on the scaling args), then reused across every combo.
    q_list  = _parse_num_list(args.q_sweep,        args.macs_q)
    bg_list = _parse_num_list(args.bg_scale_sweep, args.bg_scale)
    ext_list = _parse_num_list(args.extsize_sweep, args.extsize, want_int=True)
    combos = [(q, bg, ext) for ext in ext_list for bg in bg_list for q in q_list]
    sweeping = len(combos) > 1
    if sweeping:
        log.info("Sweep: %d combos per pipeline  (q=%s  bg_scale=%s  extsize=%s)",
                 len(combos), q_list, bg_list, ext_list)

    rows: list[dict] = []
    for label, path in args.inputs:
        log.info("=== %s :: %s ===", label, path)
        if not path.is_dir():
            log.warning("  skip: %s is not a directory", path)
            continue
        pipe_dir = args.out_dir / label
        pipe_dir.mkdir(parents=True, exist_ok=True)

        signal, regions, kind, n_cells = load_per_bin_signal(path)
        nnz = int(np.count_nonzero(signal))
        total_signal = float(signal.sum())
        mean_umi_per_cell = total_signal / n_cells if n_cells else float("nan")
        log.info("  matrix: %d regions x %d cells, total_signal=%.3g, mean_umi/cell=%.4g, "
                 "nnz=%d (%.1f%% of bins), kind=%s",
                 len(regions), n_cells, total_signal, mean_umi_per_cell, nnz,
                 100.0 * nnz / len(regions) if regions else float("nan"), kind)

        # Nonzero-signal distribution: tells us whether the matrix has dynamic
        # range to discriminate (median >> 1) or is near-binary (median ~ 1, so
        # peak calling can only ever be near-pass-through; bg_scale can't create
        # contrast that isn't in the data).
        if nnz:
            _nzv = signal[signal > 0]
            _pc = np.percentile(_nzv, [50, 75, 90, 99, 99.9])
            log.info("  nonzero signal pctiles: p50=%.3g p75=%.3g p90=%.3g p99=%.3g p99.9=%.3g "
                     "max=%.3g mean=%.3g", *_pc, float(_nzv.max()), float(_nzv.mean()))

        # Per-bin signal at reference bins (0 where the matrix has no signal).
        # Drives both the signal ceiling and the AUROC.
        sig_by_name = {name: sig for name, sig in zip(regions, signal) if sig > 0}
        pos_scores = np.fromiter((sig_by_name.get(nm, 0.0) for nm in pos_names),
                                 dtype=np.float64, count=len(pos_names))
        neg_scores = np.fromiter((sig_by_name.get(nm, 0.0) for nm in neg_names),
                                 dtype=np.float64, count=len(neg_names))
        n_pos_sig = int(np.count_nonzero(pos_scores))
        n_neg_sig = int(np.count_nonzero(neg_scores))
        pos_ceiling = 100.0 * n_pos_sig / len(pos_names) if pos_names else float("nan")
        neg_ceiling = 100.0 * n_neg_sig / len(neg_names) if neg_names else float("nan")
        auroc = signal_auroc(pos_scores, neg_scores)
        log.info("  signal ceiling: pos %d/%d (%.2f%%) ; neg %d/%d (%.2f%%) ; AUROC=%.4f "
                 "-- ceiling = max coverage if peak-calling were perfect",
                 n_pos_sig, len(pos_names), pos_ceiling,
                 n_neg_sig, len(neg_names), neg_ceiling, auroc)

        # 1. Synthetic BED -- ONCE per pipeline (independent of q/bg_scale/extsize)
        bed_path = pipe_dir / "synthetic.bed"
        n_frag, scale, downscale = generate_synthetic_bed(
            signal, regions, bed_path,
            spread_bp=args.spread_bp,
            ref_quantile=args.ref_quantile,
            frags_at_ref=args.frags_at_ref,
            max_frags_per_bin=args.max_frags_per_bin,
            max_total_fragments=args.max_total_fragments,
        )
        bar_signal = (float(np.quantile(signal[signal > 0], args.bg_quantile))
                      if args.bg_quantile and args.bg_quantile > 0 else 0.0)

        # 2. Sweep MACS3 over the (q, bg_scale, extsize) grid, reusing the BED.
        for (q, bg_scale, extsize) in combos:
            bar_frags = bar_frags_for(bar_signal, scale, downscale,
                                      args.max_frags_per_bin, bg_scale)
            g_eff = effective_genome_size(n_frag, extsize, bar_frags)
            tag = "" if not sweeping else f"_q{q:g}_bg{bg_scale:g}_ext{extsize}"
            if sweeping:
                log.info("  [combo q=%g bg_scale=%g extsize=%d] bar_frags=%d g_eff=%s",
                         q, bg_scale, extsize, bar_frags, g_eff)

            narrow = run_macs3(
                bed_path, pipe_dir, name=f"{label}_pseudo{tag}",
                qvalue=q, extsize=extsize, nolambda=args.nolambda,
                effective_genome_size=g_eff,
            )
            peaks = load_called_peaks(narrow)
            log.info("  called peaks: %d", len(peaks))

            bed_called = pipe_dir / f"{label}_called_peaks{tag}.bed"
            sorted_peaks = sorted(peaks, key=lambda r: (r[0], r[1]))
            with open(bed_called, "w") as fh:
                for c, s, e in sorted_peaks:
                    fh.write(f"{c}\t{s}\t{e}\n")

            covered_pos_bed = pipe_dir / f"{label}_pos_covered_bins{tag}.bed"
            covered_neg_bed = pipe_dir / f"{label}_neg_covered_bins{tag}.bed"
            pairs_pos_tsv = pipe_dir / f"{label}_pos_overlap_pairs{tag}.tsv" if args.emit_pairs else None
            pairs_neg_tsv = pipe_dir / f"{label}_neg_overlap_pairs{tag}.tsv" if args.emit_pairs else None

            n_pos_cov, n_pos = coverage_via_bedtools(
                ref_pos_bed, bed_called,
                min_overlap_frac=args.min_overlap_frac,
                covered_out_bed=covered_pos_bed, pairs_out_tsv=pairs_pos_tsv,
            )
            n_neg_cov, n_neg = coverage_via_bedtools(
                ref_neg_bed, bed_called,
                min_overlap_frac=args.min_overlap_frac,
                covered_out_bed=covered_neg_bed, pairs_out_tsv=pairs_neg_tsv,
            )
            pos_cov = (100.0 * n_pos_cov / n_pos) if n_pos else float("nan")
            neg_cov = (100.0 * n_neg_cov / n_neg) if n_neg else float("nan")
            pos_capture = (100.0 * n_pos_cov / n_pos_sig) if n_pos_sig else float("nan")
            log.info("  -> q=%g bg_scale=%g extsize=%d : pos %.2f%% ; neg %.2f%% ; "
                     "capture %.2f%% ; peaks %d", q, bg_scale, extsize,
                     pos_cov, neg_cov, pos_capture, len(peaks))

            rows.append({
                "run":                       label,
                "kind":                      kind,
                "n_cells":                   int(n_cells),
                "n_bins_in_matrix":          int(len(regions)),
                "n_nonzero_bins":            nnz,
                "total_signal":              total_signal,
                "mean_umi_per_cell":         round(mean_umi_per_cell, 6),
                "n_pos_with_signal":         int(n_pos_sig),
                "n_neg_with_signal":         int(n_neg_sig),
                "pos_signal_ceiling_pct":    round(pos_ceiling, 4),
                "neg_signal_ceiling_pct":    round(neg_ceiling, 4),
                "auroc":                     round(auroc, 6),
                "max_total_fragments":       int(args.max_total_fragments),
                "ref_quantile":              float(args.ref_quantile),
                "frags_at_ref":              float(args.frags_at_ref),
                "bg_quantile":               float(args.bg_quantile),
                "bg_scale":                  float(bg_scale),
                "bg_bar_frags":              int(bar_frags),
                "effective_genome_size":     str(g_eff),
                "max_frags_per_bin":         int(args.max_frags_per_bin),
                "nolambda":                  bool(args.nolambda),
                "spread_bp":                 int(args.spread_bp),
                "macs_q":                    float(q),
                "extsize":                   int(extsize),
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
                "positive_capture_pct":       round(pos_capture, 4),
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

        if sweeping:
            # One arrow-style q-sweep block per (bg_scale, extsize).
            blocks = []
            for ext in ext_list:
                for bg in bg_list:
                    blocks.append(format_q_sweep_block(rows, q_list, bg, ext))
            table = "\n\n".join(blocks)
            table_path = args.out_dir / f"{args.out_name}_sweep_table.txt"
            table_path.write_text(table + "\n")
            log.info("Wrote %s", table_path)
            log.info("Sweep tables:\n%s", table)
        else:
            # Slide-style single-setting summary table.
            table = format_summary_table(rows, args.macs_q)
            table_path = args.out_dir / f"{args.out_name}_table.txt"
            table_path.write_text(table + "\n")
            log.info("Wrote %s", table_path)
            log.info("Summary table (q=%g):\n%s", args.macs_q, table)
    else:
        log.warning("No rows produced; nothing to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
