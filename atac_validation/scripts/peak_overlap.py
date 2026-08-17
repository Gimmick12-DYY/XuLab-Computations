#!/usr/bin/env python
"""Pairwise concordance of ATAC open-chromatin masks at 1 kb bin resolution.

Given several narrowPeak/BED files (the current OmniATAC IDR mask + the two
validation datasets), compute for the shared unified bin universe:

  * n_open bins per mask,
  * pairwise Jaccard (bin-level),
  * directional recovery: fraction of mask A's open bins also open in mask B.

Bin overlap uses the same logic as the imputer's open-chromatin mask
(`unified/scripts/_regions.py`), so numbers line up with downstream masking.

Usage:
  python peak_overlap.py \
    --regions  /work/.../unified/work/ctcf/impute/regions.tsv \
    --mask omniatac=/work/.../data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz \
    --mask gse283384=/work/.../data/HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz \
    --mask gse152177=/work/.../data/HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz \
    --out-dir  /work/.../atac_validation/results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _maskutil import bin_mask_for_peaks, jaccard, load_region_names, region_arrays


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True, help="unified regions.tsv (bin universe)")
    ap.add_argument("--mask", action="append", required=True, metavar="name=peaks",
                    help="repeatable; narrowPeak/BED(.gz) per mask")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plot", action="store_true", help="also write a Jaccard heatmap PNG")
    args = ap.parse_args()

    masks = parse_masks(args.mask)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[peak_overlap] loading bin universe from {args.regions}")
    names = load_region_names(args.regions)
    chroms, starts, ends = region_arrays(names)
    n_bins = len(names)
    print(f"[peak_overlap] {n_bins:,} bins")

    bin_masks: dict[str, np.ndarray] = {}
    for name, path in masks.items():
        m = bin_mask_for_peaks(chroms, starts, ends, path)
        bin_masks[name] = m
        print(f"[peak_overlap] {name}: {int(m.sum()):,} open bins  ({m.mean()*100:.2f}%)")

    order = list(masks.keys())

    # Per-mask summary.
    with open(out_dir / "mask_sizes.tsv", "w") as fh:
        fh.write("mask\tn_open_bins\tpct_of_universe\tpeaks_file\n")
        for name in order:
            m = bin_masks[name]
            fh.write(f"{name}\t{int(m.sum())}\t{m.mean()*100:.4f}\t{masks[name]}\n")

    # Pairwise metrics (directional recovery is A-covered-by-B).
    rows = []
    for a in order:
        for b in order:
            ma, mb = bin_masks[a], bin_masks[b]
            inter = int(np.count_nonzero(ma & mb))
            na = int(ma.sum())
            rows.append({
                "mask_a": a, "mask_b": b,
                "n_a": na, "n_b": int(mb.sum()),
                "intersection": inter,
                "union": int(np.count_nonzero(ma | mb)),
                "jaccard": round(jaccard(ma, mb), 4),
                "frac_a_in_b": round(inter / na, 4) if na else 0.0,
            })
    with open(out_dir / "mask_overlap.tsv", "w") as fh:
        cols = ["mask_a", "mask_b", "n_a", "n_b", "intersection", "union",
                "jaccard", "frac_a_in_b"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Jaccard matrix (readable console + optional heatmap).
    J = np.array([[jaccard(bin_masks[a], bin_masks[b]) for b in order] for a in order])
    print("\n[peak_overlap] bin-level Jaccard matrix:")
    print("           " + "  ".join(f"{n:>10.10}" for n in order))
    for i, a in enumerate(order):
        print(f"{a:>10.10} " + "  ".join(f"{J[i,j]:>10.3f}" for j in range(len(order))))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(order),) * 2)
            im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
            ax.set_yticks(range(len(order)), order)
            for i in range(len(order)):
                for j in range(len(order)):
                    ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
                            color="white" if J[i, j] < 0.6 else "black", fontsize=9)
            fig.colorbar(im, ax=ax, label="bin Jaccard")
            ax.set_title("HEK293T ATAC open-bin concordance")
            fig.tight_layout()
            fig.savefig(out_dir / "mask_jaccard_heatmap.png", dpi=150)
            print(f"[peak_overlap] wrote {out_dir/'mask_jaccard_heatmap.png'}")
        except Exception as e:  # pragma: no cover
            print(f"[peak_overlap] plot skipped: {e}")

    print(f"\n[peak_overlap] wrote {out_dir/'mask_overlap.tsv'} and mask_sizes.tsv")


if __name__ == "__main__":
    main()
