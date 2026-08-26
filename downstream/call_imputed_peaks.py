#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# call_imputed_peaks.py
#
# Call peaks on an imputed matrix at MACS3 q < 0.05 -- the same pseudobulk +
# MACS3 method used by peak_coverage.py (the project's peak-calling standard) --
# and emit a per-TF narrowPeak/BED for motif enrichment. This replaces the old
# fixed-prevalence peak definition, which produced non-comparable, often
# degenerate peak sets (1 .. 10k peaks across TFs).
#
# Pipeline (reuses peak_coverage.py):
#   per-bin signal = sum across cells  ->  synthetic fragment BED
#   ->  macs3 callpeak --nomodel --shift -extsize/2 --extsize 200 -q 0.05
#       --keep-dup all --nolambda  (effective genome size from --bg-quantile)
#
# Outputs:
#   --out-bed          called peaks, 6-col BED (chrom start end name signal .)
#   <out-dir>/<tf>.synthetic.bed, <tf>_qcut_peaks.narrowPeak   (intermediates)
#   --bg-out           (with --open-bed) accessibility-matched background BED
#
# Exit 3 if fewer than --min-peaks are called (too sparse for valid enrichment).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

_DOWN = Path(__file__).resolve().parent
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
from peak_coverage import (  # noqa: E402  (vetted peak-calling standard)
    bar_frags_for,
    effective_genome_size,
    generate_synthetic_bed,
    load_per_bin_signal,
    normalize_per_bin_signal,
    run_macs3,
)

_UNIFIED = _DOWN.parent / "unified" / "scripts"
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))
from _regions import bins_overlap_bed, load_bed_intervals, parse_region_names  # noqa: E402


def parse_narrowpeak(path: Path) -> list[tuple[str, int, int, float]]:
    """narrowPeak -> [(chrom, start, end, signalValue)]."""
    out: list[tuple[str, int, int, float]] = []
    with open(path) as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                sig = float(p[6]) if len(p) > 6 else 0.0
                out.append((p[0], int(p[1]), int(p[2]), sig))
            except ValueError:
                continue
    return out


def merge_sorted_bins(chroms, starts, ends, idx) -> list[tuple[str, int, int]]:
    order = idx[np.lexsort((starts[idx], chroms[idx]))]
    out: list[tuple[str, int, int]] = []
    for i in order:
        c, s, e = str(chroms[i]), int(starts[i]), int(ends[i])
        if out and out[-1][0] == c and s <= out[-1][2]:
            pc, ps, pe = out[-1]
            out[-1] = (pc, ps, max(pe, e))
        else:
            out.append((c, s, e))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impute-dir", type=Path, required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--out-bed", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="dir for MACS intermediates")
    ap.add_argument("--min-peaks", type=int, default=0,
                    help="exit 3 (no BED) if fewer than this many peaks are called")
    # MACS / pseudobulk knobs -- defaults match peak_coverage.py.
    ap.add_argument("--macs-q", type=float, default=0.05)
    ap.add_argument("--extsize", type=int, default=200)
    ap.add_argument("--spread-bp", type=int, default=150)
    ap.add_argument("--ref-quantile", type=float, default=0.5)
    ap.add_argument("--frags-at-ref", type=float, default=20.0)
    ap.add_argument("--max-frags-per-bin", type=int, default=500)
    ap.add_argument("--target-fragments", type=int, default=150_000_000, dest="max_total_fragments")
    ap.add_argument("--bg-quantile", type=float, default=0.5)
    ap.add_argument("--bg-scale", type=float, default=1.0)
    ap.add_argument("--no-nolambda", action="store_true", help="disable MACS3 --nolambda")
    # Consensus-subtracted (TF-specific) peaks: call on residual = max(0, norm(signal)
    # - consensus) instead of the raw pseudobulk. Isolates signal enriched ABOVE the
    # shared accessibility landscape. Build the consensus with build_consensus_accessibility.py.
    ap.add_argument("--consensus-npy", type=Path, default=None,
                    help="consensus accessibility vector; subtract before peak calling")
    ap.add_argument("--consensus-normalize", choices=["rank", "none"], default="rank",
                    help="per-track normalization; MUST match build_consensus_accessibility.py")
    # accessibility-matched background (optional).
    ap.add_argument("--open-bed", type=Path, default=None)
    ap.add_argument("--bg-out", type=Path, default=None)
    ap.add_argument("--flank-bp", type=int, default=500)
    args = ap.parse_args()

    if not (args.impute_dir / "matrix_csr.npz").is_file():
        raise SystemExit(f"missing matrix_csr.npz in {args.impute_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    signal, regions, kind, n_cells = load_per_bin_signal(args.impute_dir)
    print(f"[{args.tf}] {len(regions)} bins x {n_cells} cells (kind={kind}), "
          f"nonzero={int(np.count_nonzero(signal))}", flush=True)

    if args.consensus_npy:
        if not args.consensus_npy.is_file():
            raise SystemExit(f"consensus not found: {args.consensus_npy}")
        cons = np.load(args.consensus_npy)
        if cons.shape[0] != signal.shape[0]:
            raise SystemExit(f"consensus len {cons.shape[0]} != bins {signal.shape[0]}")
        s_norm = normalize_per_bin_signal(signal, args.consensus_normalize)
        residual = s_norm - cons
        np.clip(residual, 0.0, None, out=residual)
        n_pos = int(np.count_nonzero(residual))
        print(f"[{args.tf}] consensus-subtracted ({args.consensus_normalize}): "
              f"{n_pos} bins above consensus (of {int(np.count_nonzero(s_norm))} signaled) "
              f"-> TF-specific peak calling", flush=True)
        if n_pos == 0:
            raise SystemExit(f"{args.tf}: residual all zero after consensus subtraction")
        signal = residual

    synth = args.out_dir / f"{args.tf}.synthetic.bed"
    n_frag, scale, downscale = generate_synthetic_bed(
        signal, regions, synth,
        spread_bp=args.spread_bp, ref_quantile=args.ref_quantile,
        frags_at_ref=args.frags_at_ref, max_frags_per_bin=args.max_frags_per_bin,
        max_total_fragments=args.max_total_fragments,
    )
    bar_signal = (float(np.quantile(signal[signal > 0], args.bg_quantile))
                  if args.bg_quantile and args.bg_quantile > 0 else 0.0)
    bar_frags = bar_frags_for(bar_signal, scale, downscale, args.max_frags_per_bin, args.bg_scale)
    g_eff = effective_genome_size(n_frag, args.extsize, bar_frags)

    narrow = run_macs3(synth, args.out_dir, name=f"{args.tf}_qcut",
                       qvalue=args.macs_q, extsize=args.extsize,
                       nolambda=not args.no_nolambda, effective_genome_size=g_eff)
    peaks = parse_narrowpeak(narrow)
    print(f"[{args.tf}] MACS3 q<{args.macs_q:g}: {len(peaks)} peaks", flush=True)

    if args.min_peaks and len(peaks) < args.min_peaks:
        print(f"[{args.tf}] INSUFFICIENT: {len(peaks)} peaks < --min-peaks "
              f"{args.min_peaks}; not writing BED", flush=True)
        raise SystemExit(3)
    if not peaks:
        raise SystemExit(f"{args.tf}: MACS3 called 0 peaks")

    args.out_bed.parent.mkdir(parents=True, exist_ok=True)
    peaks_sorted = sorted(peaks, key=lambda r: (r[0], r[1]))
    with args.out_bed.open("w") as fh:
        for k, (c, s, e, sig) in enumerate(peaks_sorted):
            fh.write(f"{c}\t{s}\t{e}\t{args.tf}_p{k}\t{sig:.6g}\t.\n")
    print(f"[{args.tf}] wrote {len(peaks_sorted)} peaks -> {args.out_bed}", flush=True)

    if args.open_bed:
        if args.bg_out is None:
            raise SystemExit("--bg-out required with --open-bed")
        if not args.open_bed.is_file():
            raise SystemExit(f"open-bed not found: {args.open_bed}")
        chroms, starts, ends = parse_region_names(regions)
        open_mask = bins_overlap_bed(chroms, starts, ends, load_bed_intervals(args.open_bed))
        peak_iv: dict[str, list[tuple[int, int]]] = {}
        for c, s, e, _ in peaks:
            peak_iv.setdefault(c, []).append((s, e))
        peak_mask = bins_overlap_bed(chroms, starts, ends, peak_iv)
        bin_w = int(np.median(ends - starts)) or 1000
        flank_bins = max(0, int(np.ceil(args.flank_bp / bin_w)))
        excl = peak_mask.copy()
        for shift in range(1, flank_bins + 1):
            l = np.zeros(len(regions), dtype=bool)
            r = np.zeros(len(regions), dtype=bool)
            l[:-shift] = peak_mask[shift:] & (chroms[:-shift] == chroms[shift:])
            r[shift:] = peak_mask[:-shift] & (chroms[shift:] == chroms[:-shift])
            excl |= l | r
        bg_idx = np.flatnonzero(open_mask & ~excl)
        if bg_idx.size == 0:
            raise SystemExit(f"{args.tf}: empty background (open minus peaks)")
        bg = merge_sorted_bins(chroms, starts, ends, bg_idx)
        with args.bg_out.open("w") as fh:
            for k, (c, s, e) in enumerate(bg):
                fh.write(f"{c}\t{s}\t{e}\t{args.tf}_bg{k}\t0\t.\n")
        print(f"[{args.tf}] wrote {len(bg)} background intervals -> {args.bg_out} "
              f"(open bins={int(open_mask.sum())}, flank_bins={flank_bins})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
