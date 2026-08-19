#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_imputed_peaks.py
#
# Turn a (open-chromatin-masked) imputed matrix into a per-TF "peak" BED that
# HOMER / AME can consume for motif enrichment.
#
# Input: an impute dir with matrix_csr.npz (bins x cells, binarized after the
# open-chromatin mask + keep_raw union) and regions.tsv (chr:start-end per bin).
#
# A bin is a called peak when it is imputed-bound in enough cells:
#     n_cells_bound = rowsum(matrix_csr)         (binary -> count of cells)
#     prevalence    = n_cells_bound / n_cells
#     keep bin iff  n_cells_bound >= --min-cells AND prevalence >= --min-frac
#     (then optionally cap to the strongest --top-n bins by prevalence)
# Adjacent kept bins are merged into single intervals (unless --no-merge).
#
# Optional accessibility-matched BACKGROUND (recommended HOMER -bg): open-chromatin
# bins that are NOT called peaks (and not within --flank-bp of one). This controls
# for accessibility so enriched motifs reflect TF specificity, not just openness.
#
# Outputs (6-col BED: chrom start end name prevalence strand):
#   --out-bed   called peaks
#   --bg-out    background (only when --open-bed is given)
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_UNIFIED_SCRIPTS = REPO / "unified" / "scripts"
if str(_UNIFIED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_UNIFIED_SCRIPTS))
from _regions import bins_overlap_bed, load_bed_intervals, parse_region_names  # noqa: E402


def read_regions(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def row_prevalence(csr_path: Path) -> tuple[np.ndarray, int]:
    """Return (n_cells_bound per bin, n_cells). Binary matrix -> rowsum = cells."""
    from scipy import sparse

    sm = sparse.load_npz(csr_path)
    n_cells = int(sm.shape[1])
    counts = np.asarray(sm.sum(axis=1)).ravel().astype(np.float64)
    return counts, n_cells


def merge_sorted_bins(chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                      score: np.ndarray, idx: np.ndarray) -> list[tuple[str, int, int, float]]:
    """Merge overlapping/bookended kept bins; interval score = max prevalence."""
    order = idx[np.lexsort((starts[idx], chroms[idx]))]
    out: list[tuple[str, int, int, float]] = []
    for i in order:
        c, s, e, v = str(chroms[i]), int(starts[i]), int(ends[i]), float(score[i])
        if out and out[-1][0] == c and s <= out[-1][2]:
            pc, ps, pe, pv = out[-1]
            out[-1] = (pc, ps, max(pe, e), max(pv, v))
        else:
            out.append((c, s, e, v))
    return out


def write_bed(path: Path, intervals, name_prefix: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for k, (c, s, e, v) in enumerate(intervals):
            fh.write(f"{c}\t{s}\t{e}\t{name_prefix}_{k}\t{v:.6g}\t.\n")
    return len(intervals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impute-dir", type=Path, required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--out-bed", type=Path, required=True)
    ap.add_argument("--min-cells", type=float, default=2,
                    help="min cells with imputed signal per bin (default 2)")
    ap.add_argument("--min-frac", type=float, default=0.02,
                    help="min fraction of cells bound per bin (default 0.02)")
    ap.add_argument("--top-n", type=int, default=0,
                    help="cap to strongest N bins by prevalence (0 = no cap)")
    ap.add_argument("--no-merge", action="store_true", help="keep 1 kb bins, no merge")
    ap.add_argument("--open-bed", type=Path, default=None,
                    help="open-chromatin BED; enables accessibility-matched background")
    ap.add_argument("--bg-out", type=Path, default=None,
                    help="background BED path (with --open-bed)")
    ap.add_argument("--flank-bp", type=int, default=500,
                    help="exclude background bins within this many bp of a peak")
    args = ap.parse_args()

    csr = args.impute_dir / "matrix_csr.npz"
    reg = args.impute_dir / "regions.tsv"
    for f in (csr, reg):
        if not f.is_file():
            raise SystemExit(f"missing {f}")

    names = read_regions(reg)
    chroms, starts, ends = parse_region_names(names)
    counts, n_cells = row_prevalence(csr)
    if counts.size != len(names):
        raise SystemExit(f"{args.tf}: matrix rows {counts.size} != regions {len(names)}")
    frac = counts / max(n_cells, 1)

    keep = (counts >= args.min_cells) & (frac >= args.min_frac)
    n_keep0 = int(keep.sum())
    idx = np.flatnonzero(keep)
    if args.top_n and idx.size > args.top_n:
        strongest = idx[np.argsort(frac[idx])[::-1][: args.top_n]]
        idx = np.sort(strongest)
    print(f"[{args.tf}] n_cells={n_cells} bins_kept={idx.size} "
          f"(pre-topN {n_keep0}; min_cells={args.min_cells} min_frac={args.min_frac} "
          f"top_n={args.top_n or 'off'})", flush=True)
    if idx.size == 0:
        raise SystemExit(f"{args.tf}: no bins passed the peak thresholds")

    if args.no_merge:
        peaks = [(str(chroms[i]), int(starts[i]), int(ends[i]), float(frac[i])) for i in idx]
    else:
        peaks = merge_sorted_bins(chroms, starts, ends, frac, idx)
    n_peaks = write_bed(args.out_bed, peaks, f"{args.tf}_imp")
    print(f"[{args.tf}] wrote {n_peaks} peaks -> {args.out_bed}", flush=True)

    if args.open_bed:
        if args.bg_out is None:
            raise SystemExit("--bg-out required with --open-bed")
        if not args.open_bed.is_file():
            raise SystemExit(f"open-bed not found: {args.open_bed}")
        open_mask = bins_overlap_bed(chroms, starts, ends, load_bed_intervals(args.open_bed))
        # Exclude peak bins and their flank neighbours (flank in whole bins).
        peak_mask = np.zeros(len(names), dtype=bool)
        peak_mask[idx] = True
        bin_w = int(np.median(ends - starts)) or 1000
        flank_bins = max(0, int(np.ceil(args.flank_bp / bin_w)))
        excl = peak_mask.copy()
        for shift in range(1, flank_bins + 1):
            same_l = np.zeros(len(names), dtype=bool)
            same_r = np.zeros(len(names), dtype=bool)
            same_l[:-shift] = peak_mask[shift:] & (chroms[:-shift] == chroms[shift:])
            same_r[shift:] = peak_mask[:-shift] & (chroms[shift:] == chroms[:-shift])
            excl |= same_l | same_r
        bg_idx = np.flatnonzero(open_mask & ~excl)
        if bg_idx.size == 0:
            raise SystemExit(f"{args.tf}: empty background (open bins minus peaks)")
        bg = merge_sorted_bins(chroms, starts, ends, frac, bg_idx)
        n_bg = write_bed(args.bg_out, bg, f"{args.tf}_bg")
        print(f"[{args.tf}] wrote {n_bg} background intervals -> {args.bg_out} "
              f"(open bins={int(open_mask.sum())}, flank_bins={flank_bins})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
