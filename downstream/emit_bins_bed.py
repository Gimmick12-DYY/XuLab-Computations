#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# emit_bins_bed.py
#
# "bins" region set for motif enrichment: the TF's top binding BINS taken DIRECTLY
# from the (imputed or raw) bin matrix -- no MACS peak-calling. Each bin is written
# at its native width. Optionally emits an OCR-matched background (open bins minus
# these bins, +flank), same as call_imputed_peaks, so BG_MODE=ocr works unchanged.
#
# Output:
#   --out-bed   top-N bins as 6-col BED (chrom start end name signal .)
#   --bg-out    (with --open-bed) open bins minus these bins (+flank)
# Exit 3 if fewer than --min-peaks bins.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_DOWN = Path(__file__).resolve().parent
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
from peak_coverage import load_per_bin_signal  # noqa: E402

_UNIFIED = _DOWN.parent / "unified" / "scripts"
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))
from _regions import bins_overlap_bed, load_bed_intervals, parse_region_names  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-dir", type=Path, required=True,
                    help="dir with the bin matrix (impute/ matrix_csr.npz or mm/matrix.mtx.gz)")
    ap.add_argument("--tf", required=True)
    ap.add_argument("--out-bed", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=20000, help="top-N bins by signal (0 = all nonzero)")
    ap.add_argument("--min-peaks", type=int, default=200)
    ap.add_argument("--open-bed", type=Path, default=None)
    ap.add_argument("--bg-out", type=Path, default=None)
    ap.add_argument("--flank-bp", type=int, default=500)
    args = ap.parse_args()

    signal, regions, kind, ncells = load_per_bin_signal(args.matrix_dir)
    chroms, starts, ends = parse_region_names(regions)
    nz = np.flatnonzero(signal > 0)
    if args.top_n and args.top_n > 0 and nz.size > args.top_n:
        keep = nz[np.argpartition(signal[nz], -args.top_n)[-args.top_n:]]
    else:
        keep = nz
    print(f"[{args.tf}] {len(regions)} bins (kind={kind}); {nz.size} nonzero -> {keep.size} bin regions", flush=True)
    if keep.size < args.min_peaks:
        print(f"[{args.tf}] INSUFFICIENT: {keep.size} < --min-peaks {args.min_peaks}", flush=True)
        raise SystemExit(3)

    order = keep[np.lexsort((starts[keep], chroms[keep]))]
    args.out_bed.parent.mkdir(parents=True, exist_ok=True)
    with args.out_bed.open("w") as f:
        for k, i in enumerate(order):
            f.write(f"{chroms[i]}\t{int(starts[i])}\t{int(ends[i])}\t{args.tf}_b{k}\t{signal[i]:.6g}\t.\n")
    print(f"[{args.tf}] wrote {order.size} bin regions -> {args.out_bed}", flush=True)

    if args.open_bed and args.bg_out:
        if not args.open_bed.is_file():
            raise SystemExit(f"open-bed not found: {args.open_bed}")
        open_mask = bins_overlap_bed(chroms, starts, ends, load_bed_intervals(args.open_bed))
        reg_mask = np.zeros(len(regions), dtype=bool); reg_mask[keep] = True
        bin_w = int(np.median(ends - starts)) or 1000
        flank = max(0, int(np.ceil(args.flank_bp / bin_w)))
        excl = reg_mask.copy()
        for sh in range(1, flank + 1):
            l = np.zeros(len(regions), bool); r = np.zeros(len(regions), bool)
            l[:-sh] = reg_mask[sh:] & (chroms[:-sh] == chroms[sh:])
            r[sh:] = reg_mask[:-sh] & (chroms[sh:] == chroms[:-sh])
            excl |= l | r
        bg = np.flatnonzero(open_mask & ~excl)
        bg = bg[np.lexsort((starts[bg], chroms[bg]))]
        with args.bg_out.open("w") as f:
            for k, i in enumerate(bg):
                f.write(f"{chroms[i]}\t{int(starts[i])}\t{int(ends[i])}\t{args.tf}_bg{k}\t0\t.\n")
        print(f"[{args.tf}] wrote {bg.size} background bins -> {args.bg_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
