#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# call_raw_peaks.py
#
# Call MACS3 q<0.05 peaks for ONE TF from its RAW scCUT&Tag pseudobulk
# (unified/work/<tf>/mm) and write a per-TF 6-col BED. Uses the project's vetted
# peak-calling standard (downstream/peak_coverage.py): per-bin signal = sum across
# cells -> synthetic fragment BED -> MACS3 --nomodel --nolambda -q 0.05.
#
# RAW (not imputed): the imputed matrices are open-chromatin masked, which forces
# every TF into the same open bins and inflates co-occupancy. Co-occupancy /
# complex analysis must use raw binding. (See tf_complex/README.md.)
#
# Field-standard co-occupancy (Partridge et al. 2020, Nature) is built on called
# PEAKS, not genome-wide signal bins -- peaks are less accessibility-driven. This
# script produces the per-TF peak sets that build_peak_matrix.py unions into loci.
#
# Output:
#   --out-bed   called peaks, 6-col BED (chrom start end name signal .)
# Exit 3 if fewer than --min-peaks are called.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_DOWN = Path(__file__).resolve().parents[2] / "downstream"
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
from peak_coverage import (  # noqa: E402  (vetted peak-calling standard)
    bar_frags_for,
    effective_genome_size,
    generate_synthetic_bed,
    load_per_bin_signal,
    run_macs3,
)


def parse_narrowpeak(path: Path):
    out = []
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="per-TF dir containing mm/ (raw) and/or impute/ (imputed)")
    ap.add_argument("--source", choices=["raw", "imputed"], default="raw",
                    help="raw = mm/matrix.mtx.gz (default); imputed = impute/matrix_csr.npz "
                         "(open-chromatin-masked). Imputed densifies sparse low-cell TFs -- "
                         "test whether that reduces the cell-count/complexity confound, at the "
                         "cost of the shared-accessibility wash.")
    ap.add_argument("--tf", required=True)
    ap.add_argument("--out-bed", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="dir for MACS intermediates")
    ap.add_argument("--min-peaks", type=int, default=100)
    # MACS / pseudobulk knobs -- defaults match peak_coverage.py / call_imputed_peaks.py
    ap.add_argument("--macs-q", type=float, default=0.05)
    ap.add_argument("--extsize", type=int, default=200)
    ap.add_argument("--spread-bp", type=int, default=150)
    ap.add_argument("--ref-quantile", type=float, default=0.5)
    ap.add_argument("--frags-at-ref", type=float, default=20.0)
    ap.add_argument("--max-frags-per-bin", type=int, default=500)
    ap.add_argument("--target-fragments", type=int, default=150_000_000, dest="max_total_fragments")
    ap.add_argument("--bg-quantile", type=float, default=0.5)
    ap.add_argument("--bg-scale", type=float, default=1.0)
    ap.add_argument("--no-nolambda", action="store_true")
    args = ap.parse_args()

    sub = "mm" if args.source == "raw" else "impute"
    src = args.raw_dir / sub
    have = (src / "matrix.mtx.gz").is_file() or (src / "matrix_csr.npz").is_file()
    if not have:
        raise SystemExit(f"missing matrix in {src} (source={args.source})")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    signal, regions, kind, n_cells = load_per_bin_signal(src)
    print(f"[{args.tf}] {len(regions)} bins x {n_cells} cells (kind={kind}), "
          f"nonzero={int(np.count_nonzero(signal))}", flush=True)

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

    if len(peaks) < args.min_peaks:
        print(f"[{args.tf}] INSUFFICIENT: {len(peaks)} < --min-peaks {args.min_peaks}", flush=True)
        raise SystemExit(3)

    args.out_bed.parent.mkdir(parents=True, exist_ok=True)
    peaks_sorted = sorted(peaks, key=lambda r: (r[0], r[1]))
    with args.out_bed.open("w") as fh:
        for k, (c, s, e, sig) in enumerate(peaks_sorted):
            fh.write(f"{c}\t{s}\t{e}\t{args.tf}_p{k}\t{sig:.6g}\t.\n")
    print(f"[{args.tf}] wrote {len(peaks_sorted)} peaks -> {args.out_bed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
