#!/usr/bin/env python
"""Build the union open-chromatin mask across the three HEK293T ATAC peak sets.

Merges peak intervals from all inputs into one sorted, non-overlapping BED. That
union BED is a drop-in replacement for `open_chromatin_bed` in the unified config
(the imputer / downstream overlap bins against intervals, not against a bin list).

Also reports, over the unified bin universe, how many open bins the union covers
and how much each source contributes (bins unique to each source vs shared).

Usage:
  python build_union_mask.py \
    --mask omniatac=/work/.../data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz \
    --mask gse283384=/work/.../data/HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz \
    --mask gse152177=/work/.../data/HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz \
    --regions /work/.../unified/work/ctcf/impute/regions.tsv \
    --out-bed /work/.../data/HEK293T_ATAC_union3.bed.gz \
    --out-dir /work/.../atac_validation/results
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

from _maskutil import bin_mask_for_peaks, load_bed_intervals, load_region_names, region_arrays


def parse_masks(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--mask expects name=path, got {it!r}")
        name, path = it.split("=", 1)
        if not Path(path).is_file():
            raise SystemExit(f"mask file not found for {name}: {path}")
        out[name] = path
    return out


def merge_intervals(all_ivs: dict[str, list[tuple[int, int]]]) -> list[tuple[str, int, int]]:
    """Merge overlapping/bookended intervals per chromosome."""
    merged: list[tuple[str, int, int]] = []
    for chrom in sorted(all_ivs):
        ivs = sorted(all_ivs[chrom])
        cs, ce = ivs[0]
        for s, e in ivs[1:]:
            if s <= ce:  # overlap or bookend
                ce = max(ce, e)
            else:
                merged.append((chrom, cs, ce))
                cs, ce = s, e
        merged.append((chrom, cs, ce))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", action="append", required=True, metavar="name=peaks")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out-bed", required=True, help="union BED(.gz); drop-in open_chromatin_bed")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    masks = parse_masks(args.mask)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Interval union -> BED.
    pooled: dict[str, list[tuple[int, int]]] = {}
    per_source_bp: dict[str, int] = {}
    for name, path in masks.items():
        ivs = load_bed_intervals(Path(path))
        bp = 0
        for chrom, lst in ivs.items():
            pooled.setdefault(chrom, []).extend(lst)
            bp += sum(e - s for s, e in lst)
        per_source_bp[name] = bp
        print(f"[union] {name}: {sum(len(v) for v in ivs.values()):,} intervals, {bp:,} bp")

    merged = merge_intervals(pooled)
    union_bp = sum(e - s for _, s, e in merged)
    out_bed = Path(args.out_bed)
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(out_bed).endswith(".gz") else open
    with opener(out_bed, "wt") as fh:
        for chrom, s, e in merged:
            fh.write(f"{chrom}\t{s}\t{e}\n")
    print(f"[union] merged -> {len(merged):,} intervals, {union_bp:,} bp -> {out_bed}")

    # 2) Bin-level accounting over the unified universe.
    names = load_region_names(args.regions)
    chroms, starts, ends = region_arrays(names)
    order = list(masks.keys())
    bin_masks = {n: bin_mask_for_peaks(chroms, starts, ends, masks[n]) for n in order}
    union_mask = bin_mask_for_peaks(chroms, starts, ends, out_bed)

    with open(out_dir / "union_summary.tsv", "w") as fh:
        fh.write("item\tvalue\n")
        fh.write(f"n_bins_universe\t{len(names)}\n")
        for n in order:
            m = bin_masks[n]
            fh.write(f"open_bins_{n}\t{int(m.sum())}\n")
            fh.write(f"peaks_bp_{n}\t{per_source_bp[n]}\n")
        fh.write(f"open_bins_union\t{int(union_mask.sum())}\n")
        fh.write(f"union_intervals\t{len(merged)}\n")
        fh.write(f"union_bp\t{union_bp}\n")
        # bins gained over OmniATAC alone (if present)
        if "omniatac" in bin_masks:
            gained = int(np.count_nonzero(union_mask & ~bin_masks["omniatac"]))
            fh.write(f"union_bins_gained_over_omniatac\t{gained}\n")
        # bins unique to each source
        for n in order:
            others = np.zeros(len(names), dtype=bool)
            for k in order:
                if k != n:
                    others |= bin_masks[k]
            uniq = int(np.count_nonzero(bin_masks[n] & ~others))
            fh.write(f"open_bins_unique_to_{n}\t{uniq}\n")

    print(f"[union] open bins: " +
          ", ".join(f"{n}={int(bin_masks[n].sum()):,}" for n in order) +
          f", union={int(union_mask.sum()):,}")
    print(f"[union] wrote {out_dir/'union_summary.tsv'}")
    print(f"[union] use this as open_chromatin_bed:\n  {out_bed}")


if __name__ == "__main__":
    main()
