#!/usr/bin/env python3
"""Mirrored chromosome tracks: Cicero pairs above, Peakachu loops below."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import union_5k_10k_loops_vs_pairs as U


HG38 = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895,
}
CHROMS = list(HG38)
PAIR = "#536B88"
MATCH = "#E69F00"
LOOP5 = "#D81B60"
LOOP10 = "#0072B2"
INK = "#242830"
BG = "#FBFBF9"


def parse_peak(value: str) -> tuple[str, int, int]:
    chrom, coords = value.split(":", 1)
    start, end = coords.split("-", 1)
    return chrom, int(start), int(end)


def make_curve(chrom: str, start: float, end: float, y: float, side: int,
               tall: bool = False) -> np.ndarray:
    x1, x2 = sorted((start / 1e6, end / 1e6))
    span = max(x2 - x1, 0.001)
    height = min(0.50, 0.09 + 0.25 * math.sqrt(span))
    t = np.linspace(0, 1, 9)
    x = x1 + (x2 - x1) * t
    yy = y + side * height * np.sin(np.pi * t)
    return np.column_stack((x, yy))


def pair_curves(df: pd.DataFrame, y_of: dict[str, float],
                tall: bool = False) -> list[np.ndarray]:
    curves = []
    for row in df.itertuples(index=False):
        p1, p2 = parse_peak(str(row.peak1)), parse_peak(str(row.peak2))
        if p1[0] != p2[0] or p1[0] not in y_of:
            continue
        curves.append(make_curve(
            p1[0], (p1[1] + p1[2]) / 2, (p2[1] + p2[2]) / 2, y_of[p1[0]], 1, tall
        ))
    return curves


def matched_loop_curves(df: pd.DataFrame, y_of: dict[str, float]) -> list[np.ndarray]:
    seen: set[tuple] = set()
    curves = []
    for row in df.itertuples(index=False):
        key = (str(row.chrom1), int(row.start1), int(row.end1),
               str(row.chrom2), int(row.start2), int(row.end2))
        if key in seen or key[0] not in y_of or key[0] != key[3]:
            continue
        seen.add(key)
        curves.append(make_curve(
            key[0], (key[1] + key[2]) / 2, (key[4] + key[5]) / 2, y_of[key[0]], -1, True
        ))
    return curves


def loop_curves(loops: list[dict], y_of: dict[str, float]) -> list[np.ndarray]:
    curves = []
    for loop in loops:
        a, b = loop["a"], loop["b"]
        if a[0] != b[0] or a[0] not in y_of:
            continue
        curves.append(make_curve(
            a[0], (a[1] + a[2]) / 2, (b[1] + b[2]) / 2, y_of[a[0]], -1
        ))
    return curves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--matched-pairs", type=Path, required=True)
    ap.add_argument("--fine", type=Path, required=True)
    ap.add_argument("--coarse", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs, sep="\t")
    matched = pd.read_csv(args.matched_pairs, sep="\t")
    fine = U.load_bedpe(args.fine, "5000")
    coarse = U.load_bedpe(args.coarse, "10000")

    row_gap = 1.35
    y_of = {chrom: (len(CHROMS) - 1 - i) * row_gap for i, chrom in enumerate(CHROMS)}
    fig, ax = plt.subplots(figsize=(21, 18), facecolor=BG)
    ax.set_facecolor(BG)

    for i, chrom in enumerate(CHROMS):
        y = y_of[chrom]
        if i % 2:
            ax.axhspan(y - 0.61, y + 0.61, color="#F1F2F3", zorder=0)
        ax.plot([0, HG38[chrom] / 1e6], [y, y], color="#343B45", lw=3.8,
                solid_capstyle="round", zorder=4)
        ax.text(-4.2, y, chrom, ha="right", va="center", fontsize=9,
                fontweight="bold", color=INK)

    all_pair_curves = pair_curves(pairs, y_of)
    matched_curves = pair_curves(matched, y_of)
    fine_curves = loop_curves(fine, y_of)
    coarse_curves = loop_curves(coarse, y_of)
    ax.add_collection(LineCollection(all_pair_curves, colors=PAIR, linewidths=0.42,
                                     alpha=0.14, zorder=1, rasterized=True))
    ax.add_collection(LineCollection(matched_curves, colors=MATCH, linewidths=0.85,
                                     alpha=0.72, zorder=3, rasterized=True))
    ax.add_collection(LineCollection(fine_curves, colors=LOOP5, linewidths=0.38,
                                     alpha=0.11, zorder=1, rasterized=True))
    ax.add_collection(LineCollection(coarse_curves, colors=LOOP10, linewidths=0.38,
                                     alpha=0.10, zorder=1, rasterized=True))

    max_mb = max(HG38.values()) / 1e6
    ax.text(max_mb + 3, y_of["chr1"] + 0.34, "Cicero pairs ↑", color=PAIR,
            fontsize=9, fontweight="bold", va="center")
    ax.text(max_mb + 3, y_of["chr1"] - 0.34, "Peakachu loops ↓", color=LOOP5,
            fontsize=9, fontweight="bold", va="center")
    ax.set_xlim(-8, max_mb + 29)
    ax.set_ylim(-0.75, y_of["chr1"] + 0.78)
    ax.set_xlabel("hg38 genomic position (Mb)", fontsize=10, color=INK)
    ax.set_yticks([])
    ax.grid(axis="x", color="#D9DDE2", lw=0.7, alpha=0.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#ABB1B9")
    ax.tick_params(axis="x", colors="#626873", labelsize=8)
    ax.set_title(
        "Genome-wide RBBP4 coordinated-binding pairs and Peakachu loops\n"
        "Pairs above chromosomes · loops below chromosomes · score ≥0.5 · ±30 kb matching",
        loc="left", fontsize=17, fontweight="bold", color=INK, pad=15,
    )
    handles = [
        Line2D([0], [0], color=PAIR, lw=2, label=f"all Cicero pairs ({len(pairs):,})"),
        Line2D([0], [0], color=MATCH, lw=2,
               label=f"loop-supported Cicero pairs ({len(matched):,})"),
        Line2D([0], [0], color=LOOP5, lw=2,
               label=f"Peakachu 5 kb loops ({len(fine):,})"),
        Line2D([0], [0], color=LOOP10, lw=2,
               label=f"Peakachu 10 kb loops ({len(coarse):,})"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, ncol=2)
    fig.savefig(args.out, dpi=220, bbox_inches="tight", facecolor=BG)
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
