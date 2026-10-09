#!/usr/bin/env python3
"""Plot loop-supported Cicero cliques as small networks and by chromosome."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd


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
COL5 = "#D81B60"
COL10 = "#0072B2"
PROX = "#E69F00"
DIST = "#56B4E9"
GREY = "#C5CAD3"
INK = "#20242A"
BG = "#FAFAF8"


def parse_peak(value: str) -> tuple[str, int, int]:
    chrom, coords = value.split(":", 1)
    start, end = coords.split("-", 1)
    return chrom, int(start), int(end)


def edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def save_both(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=BG)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_clique_cards(cliques, node_types, hits, out: Path, subtitle: str) -> None:
    hit_by_clique: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    for row in hits.itertuples(index=False):
        hit_by_clique[str(row.clique_id)][edge_key(str(row.peak1), str(row.peak2))] = str(
            row.resolution
        )

    selected = []
    for row in cliques.itertuples(index=False):
        cid = str(row.clique_id)
        if len(hit_by_clique[cid]) > 1:
            selected.append(row)
    selected.sort(key=lambda r: (-len(hit_by_clique[str(r.clique_id)]), str(r.chromosome),
                                 int(parse_peak(str(r.regions).split(";")[0])[1])))

    ncol = 6
    nrow = math.ceil(len(selected) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 2.75 * nrow), facecolor=BG)
    axes = np.asarray(axes).reshape(-1)

    for ax, row in zip(axes, selected):
        ax.set_facecolor(BG)
        cid = str(row.clique_id)
        peaks = sorted(str(row.regions).split(";"), key=lambda p: parse_peak(p)[1])
        n = len(peaks)
        theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
        pos = {p: np.array([math.cos(t), math.sin(t)]) for p, t in zip(peaks, theta)}

        for a, b in combinations(peaks, 2):
            xy = np.vstack((pos[a], pos[b]))
            ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=1.4, zorder=1)
        for (a, b), res in hit_by_clique[cid].items():
            if a not in pos or b not in pos:
                continue
            xy = np.vstack((pos[a], pos[b]))
            ax.plot(xy[:, 0], xy[:, 1], color="white", lw=7.0, zorder=2)
            ax.plot(xy[:, 0], xy[:, 1], color=COL5 if res == "5000" else COL10,
                    lw=4.0, linestyle="-" if res == "5000" else "--", zorder=3)

        for i, p in enumerate(peaks, 1):
            typ = node_types.get(p, "distal")
            marker = "s" if typ == "proximal" else "o"
            ax.scatter(*pos[p], s=125, marker=marker,
                       color=PROX if typ == "proximal" else DIST,
                       edgecolor=INK, linewidth=0.8, zorder=4)
            ax.text(*(pos[p] * 1.24), str(i), ha="center", va="center",
                    fontsize=7.5, color=INK)

        starts = [parse_peak(p)[1] for p in peaks]
        chrom = parse_peak(peaks[0])[0]
        ax.set_title(f"{cid}  ·  {len(hit_by_clique[cid])}/{n*(n-1)//2} loop-edges",
                     fontsize=9.5, fontweight="bold", color=INK, pad=2)
        ax.text(0, -1.42, f"{chrom}  {min(starts)/1e6:.2f}–{max(starts)/1e6:.2f} Mb",
                ha="center", va="top", fontsize=7.5, color="#626873")
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.58, 1.38)
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PROX,
               markeredgecolor=INK, markersize=8, label="proximal region"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DIST,
               markeredgecolor=INK, markersize=8, label="distal region"),
        Line2D([0], [0], color=GREY, lw=2, label="Cicero clique edge"),
        Line2D([0], [0], color=COL5, lw=4, label="5 kb loop-supported edge"),
        Line2D([0], [0], color=COL10, lw=4, ls="--", label="10 kb loop-supported edge"),
    ]
    fig.suptitle(
        f"RBBP4 cliques with multiple loop-supported edges (n={len(selected)})\n{subtitle}",
        x=0.02, y=0.998, ha="left", va="top", fontsize=17, fontweight="bold", color=INK,
    )
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.997),
               frameon=False, ncol=2, fontsize=9)
    fig.subplots_adjust(top=0.955, hspace=0.35, wspace=0.18)
    save_both(fig, out)


def arc(ax, x1: float, x2: float, y: float, color: str, dashed: bool, width: float) -> None:
    span = max(x2 - x1, 1.0)
    height = min(0.44, 0.10 + 0.50 * math.sqrt(span / 250.0))
    verts = [(x1, y), ((x1 + x2) / 2, y - height), (x2, y)]
    patch = PathPatch(
        MplPath(verts, [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]),
        facecolor="none", edgecolor=color, lw=width, alpha=0.80,
        linestyle="--" if dashed else "-", capstyle="round", zorder=3,
    )
    ax.add_patch(patch)


def plot_chromosomes(hits, out: Path, subtitle: str) -> None:
    counts_per_clique = Counter(str(x) for x in hits["clique_id"])
    by_chr = Counter(str(x) for x in hits["chrom1"])
    cliques_by_chr: dict[str, set[str]] = defaultdict(set)
    for row in hits.itertuples(index=False):
        cliques_by_chr[str(row.chrom1)].add(str(row.clique_id))

    fig, ax = plt.subplots(figsize=(19, 16), facecolor=BG)
    ax.set_facecolor(BG)
    y_of = {chrom: i for i, chrom in enumerate(CHROMS)}

    for i, chrom in enumerate(CHROMS):
        y = float(i)
        if i % 2:
            ax.axhspan(y - 0.48, y + 0.48, color="#F1F2F3", zorder=0)
        length_mb = HG38[chrom] / 1e6
        ax.plot([0, length_mb], [y, y], color="#7D838C", lw=4.2,
                solid_capstyle="round", zorder=1)
        ax.text(length_mb + 3.0, y, f"{by_chr[chrom]} edges · "
                f"{len(cliques_by_chr[chrom])} cliques",
                va="center", fontsize=8, color="#626873")

    for row in hits.itertuples(index=False):
        chrom = str(row.chrom1)
        if chrom not in y_of:
            continue
        p1, p2 = parse_peak(str(row.peak1)), parse_peak(str(row.peak2))
        x1 = ((p1[1] + p1[2]) / 2) / 1e6
        x2 = ((p2[1] + p2[2]) / 2) / 1e6
        if x1 > x2:
            x1, x2 = x2, x1
        cid = str(row.clique_id)
        res = str(row.resolution)
        color = COL5 if res == "5000" else COL10
        width = 2.2 if counts_per_clique[cid] > 1 else 1.25
        y = float(y_of[chrom])
        arc(ax, x1, x2, y, color, res != "5000", width)
        ax.scatter([x1, x2], [y, y], s=13 if width > 2 else 8, color=color,
                   edgecolor="white", linewidth=0.25, zorder=4)
        ax.scatter([(x1 + x2) / 2], [y - 0.13], marker="D",
                   s=22 if width > 2 else 13, color=color,
                   edgecolor=INK if width > 2 else "white",
                   linewidth=0.45, zorder=5)

    ax.set_yticks(range(len(CHROMS)), CHROMS, fontsize=9)
    ax.set_ylim(len(CHROMS) - 0.25, -0.9)
    ax.set_xlim(-2, max(HG38.values()) / 1e6 + 37)
    ax.set_xlabel("hg38 genomic position (Mb)", fontsize=10, color=INK)
    ax.tick_params(axis="x", colors="#626873", labelsize=8)
    ax.tick_params(axis="y", length=0, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#AEB3BA")
    ax.grid(axis="x", color="#DEE1E5", lw=0.7, alpha=0.65)
    ax.set_title(
        f"Chromosomal map of loop-supported clique edges (n={len(hits)})\n{subtitle}",
        loc="left", fontsize=17, fontweight="bold", color=INK, pad=14,
    )
    handles = [
        Line2D([0], [0], color=COL5, lw=3, label="5 kb Peakachu support"),
        Line2D([0], [0], color=COL10, lw=3, ls="--", label="10 kb Peakachu support"),
        Line2D([0], [0], color=INK, lw=2.2, label="thicker arc: clique has >1 loop-edge"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    save_both(fig, out)


def plot_circos(cliques, hits, out: Path, subtitle: str) -> None:
    """Published-study-style whole-genome Circos interaction map."""
    gap = math.radians(1.6)
    usable = 2 * math.pi - gap * len(CHROMS)
    genome = sum(HG38.values())
    spans: dict[str, tuple[float, float]] = {}
    cursor = math.pi / 2
    for chrom in CHROMS:
        width = usable * HG38[chrom] / genome
        spans[chrom] = (cursor, cursor - width)
        cursor -= width + gap

    def theta(chrom: str, pos: float) -> float:
        start, end = spans[chrom]
        return start + (end - start) * pos / HG38[chrom]

    def xy(t: float, radius: float) -> tuple[float, float]:
        return radius * math.cos(t), radius * math.sin(t)

    hit_map = {
        (str(r.clique_id), edge_key(str(r.peak1), str(r.peak2))): str(r.resolution)
        for r in hits.itertuples(index=False)
    }
    hit_count = Counter(str(x) for x in hits["clique_id"])
    all_edges = []
    regions = set()
    for row in cliques.itertuples(index=False):
        cid = str(row.clique_id)
        peaks = str(row.regions).split(";")
        regions.update(peaks)
        all_edges.extend((cid, a, b) for a, b in combinations(peaks, 2))

    fig, ax = plt.subplots(figsize=(17, 17), facecolor=BG)
    ax.set_facecolor(BG)
    ring_colors = ("#495867", "#8493A3")
    for i, chrom in enumerate(CHROMS):
        start, end = spans[chrom]
        ts = np.linspace(start, end, 120)
        ax.plot(np.cos(ts), np.sin(ts), color=ring_colors[i % 2], lw=8,
                solid_capstyle="butt", zorder=5)
        mid = (start + end) / 2
        lx, ly = xy(mid, 1.075)
        ax.text(lx, ly, chrom.replace("chr", ""), ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=INK)

    for peak in regions:
        chrom, start, end = parse_peak(peak)
        if chrom not in spans:
            continue
        t = theta(chrom, (start + end) / 2)
        x1, y1 = xy(t, 0.955)
        x2, y2 = xy(t, 0.985)
        ax.plot([x1, x2], [y1, y2], color="#B9BEC6", lw=0.45, alpha=0.8, zorder=6)

    def chord(a: str, b: str, color: str, lw: float, alpha: float,
              dashed: bool = False, zorder: int = 1) -> None:
        p1, p2 = parse_peak(a), parse_peak(b)
        if p1[0] not in spans or p2[0] not in spans:
            return
        t1 = theta(p1[0], (p1[1] + p1[2]) / 2)
        t2 = theta(p2[0], (p2[1] + p2[2]) / 2)
        x1, y1 = xy(t1, 0.945)
        x2, y2 = xy(t2, 0.945)
        mid = math.atan2(math.sin(t1) + math.sin(t2), math.cos(t1) + math.cos(t2))
        cx, cy = xy(mid, 0.45)
        path = MplPath(
            [(x1, y1), (cx, cy), (x2, y2)],
            [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
        )
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw,
                               alpha=alpha, linestyle="--" if dashed else "-",
                               capstyle="round", zorder=zorder))

    for cid, a, b in all_edges:
        chord(a, b, "#9EA5AE", 0.35, 0.13, zorder=1)
    for row in hits.itertuples(index=False):
        cid, a, b = str(row.clique_id), str(row.peak1), str(row.peak2)
        res = str(row.resolution)
        chord(a, b, COL5 if res == "5000" else COL10,
              1.9 if hit_count[cid] > 1 else 1.15, 0.82,
              dashed=res != "5000", zorder=3)
        for peak in (a, b):
            chrom, start, end = parse_peak(peak)
            t = theta(chrom, (start + end) / 2)
            x, y = xy(t, 0.945)
            ax.scatter([x], [y], s=12, color=COL5 if res == "5000" else COL10,
                       edgecolor="white", linewidth=0.3, zorder=7)

    ax.text(0, 0.08, "RBBP4", ha="center", va="center", fontsize=24,
            fontweight="bold", color=INK)
    ax.text(0, -0.04, f"{len(cliques):,} cliques", ha="center", fontsize=12, color="#5C626B")
    ax.text(0, -0.12, f"{len(hits):,} loop-supported edges", ha="center",
            fontsize=12, color="#5C626B")
    ax.set_title("Genome-wide RBBP4 clique and loop-edge map\n" + subtitle,
                 loc="left", fontsize=17, fontweight="bold", color=INK, pad=16)
    handles = [
        Line2D([0], [0], color="#9EA5AE", lw=1, alpha=0.5, label="all clique edges"),
        Line2D([0], [0], color=COL5, lw=2.5, label="5 kb loop-supported edge"),
        Line2D([0], [0], color=COL10, lw=2.5, ls="--",
               label="10 kb loop-supported edge"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    save_both(fig, out)


def plot_all_pairs_circos(edges, pair_hits, out: Path, subtitle: str) -> None:
    """Circos map containing every called Cicero pair plus highlighted loop pairs."""
    gap = math.radians(1.6)
    usable = 2 * math.pi - gap * len(CHROMS)
    genome = sum(HG38.values())
    spans: dict[str, tuple[float, float]] = {}
    cursor = math.pi / 2
    for chrom in CHROMS:
        width = usable * HG38[chrom] / genome
        spans[chrom] = (cursor, cursor - width)
        cursor -= width + gap

    def theta(chrom: str, pos: float) -> float:
        start, end = spans[chrom]
        return start + (end - start) * pos / HG38[chrom]

    def xy(t: float, radius: float) -> np.ndarray:
        return np.array([radius * math.cos(t), radius * math.sin(t)])

    def curves(rows, radius: float = 0.94, control_radius: float = 0.48):
        out_curves = []
        ts = np.linspace(0, 1, 9)[:, None]
        for row in rows:
            p1, p2 = parse_peak(str(row.peak1)), parse_peak(str(row.peak2))
            if p1[0] not in spans or p2[0] not in spans:
                continue
            t1 = theta(p1[0], (p1[1] + p1[2]) / 2)
            t2 = theta(p2[0], (p2[1] + p2[2]) / 2)
            a, b = xy(t1, radius), xy(t2, radius)
            mid = math.atan2(math.sin(t1) + math.sin(t2), math.cos(t1) + math.cos(t2))
            c = xy(mid, control_radius)
            curve = (1 - ts) ** 2 * a + 2 * (1 - ts) * ts * c + ts ** 2 * b
            out_curves.append(curve)
        return out_curves

    fig, ax = plt.subplots(figsize=(17, 17), facecolor=BG)
    ax.set_facecolor(BG)
    ring_colors = ("#495867", "#8493A3")
    for i, chrom in enumerate(CHROMS):
        start, end = spans[chrom]
        angles = np.linspace(start, end, 120)
        ax.plot(np.cos(angles), np.sin(angles), color=ring_colors[i % 2], lw=8,
                solid_capstyle="butt", zorder=6)
        mid = (start + end) / 2
        lx, ly = xy(mid, 1.075)
        ax.text(lx, ly, chrom.replace("chr", ""), ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=INK)

    # Inner endpoint-density track, analogous to density tracks in published Circos maps.
    bins_per_chrom = 80
    endpoint_counts: dict[str, np.ndarray] = {
        chrom: np.zeros(bins_per_chrom, dtype=int) for chrom in CHROMS
    }
    for row in edges.itertuples(index=False):
        for value in (str(row.peak1), str(row.peak2)):
            chrom, start, end = parse_peak(value)
            if chrom in endpoint_counts:
                idx = min(bins_per_chrom - 1,
                          int(((start + end) / 2) / HG38[chrom] * bins_per_chrom))
                endpoint_counts[chrom][idx] += 1
    max_count = max(int(x.max()) for x in endpoint_counts.values())
    density_segments = []
    for chrom, counts in endpoint_counts.items():
        for i, count in enumerate(counts):
            if count == 0:
                continue
            t = theta(chrom, (i + 0.5) / bins_per_chrom * HG38[chrom])
            outer = 0.915
            inner = outer - 0.095 * math.sqrt(count / max_count)
            density_segments.append([xy(t, inner), xy(t, outer)])
    ax.add_collection(LineCollection(density_segments, colors="#596B7C", linewidths=1.0,
                                     alpha=0.72, zorder=5))

    all_curves = curves(edges.itertuples(index=False))
    ax.add_collection(LineCollection(all_curves, colors="#7D858F", linewidths=0.28,
                                     alpha=0.055, zorder=1, rasterized=True))

    hit5 = pair_hits[pair_hits["resolution"].astype(str) == "5000"]
    hit10 = pair_hits[pair_hits["resolution"].astype(str) != "5000"]
    curves5 = curves(hit5.itertuples(index=False), control_radius=0.43)
    curves10 = curves(hit10.itertuples(index=False), control_radius=0.43)
    ax.add_collection(LineCollection(curves5, colors=COL5, linewidths=0.85,
                                     alpha=0.62, zorder=3))
    ax.add_collection(LineCollection(curves10, colors=COL10, linewidths=0.85,
                                     alpha=0.62, linestyles="dashed", zorder=3))

    ax.text(0, 0.09, "RBBP4", ha="center", va="center", fontsize=24,
            fontweight="bold", color=INK)
    ax.text(0, -0.035, f"{len(edges):,} Cicero pairs", ha="center",
            fontsize=12, color="#5C626B")
    ax.text(0, -0.12, f"{len(pair_hits):,} loop-supported pairs", ha="center",
            fontsize=12, color="#5C626B")
    ax.set_title("Genome-wide coordinated-binding and loop map\n" + subtitle,
                 loc="left", fontsize=17, fontweight="bold", color=INK, pad=16)
    handles = [
        Line2D([0], [0], color="#7D858F", lw=1.2, alpha=0.5,
               label="all Cicero coordinated-binding pairs"),
        Line2D([0], [0], color="#596B7C", lw=3, label="pair-endpoint density"),
        Line2D([0], [0], color=COL5, lw=2.5, label="5 kb loop-supported pair"),
        Line2D([0], [0], color=COL10, lw=2.5, ls="--",
               label="10 kb loop-supported pair"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    save_both(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cliques", type=Path, required=True)
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--loop-edges", type=Path, required=True)
    ap.add_argument("--all-pairs", type=Path)
    ap.add_argument("--pair-loop-edges", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subtitle", default="Peakachu 5 kb ∪ 10 kb, score ≥0.5, ±30 kb")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cliques = pd.read_csv(args.cliques, sep="\t")
    nodes = pd.read_csv(args.nodes, sep="\t")
    hits = pd.read_csv(args.loop_edges, sep="\t")
    node_types = dict(zip(nodes["peak"].astype(str), nodes["types"].astype(str)))

    plot_clique_cards(
        cliques, node_types, hits,
        args.out_dir / "RBBP4_multi_loop_cliques.png", args.subtitle,
    )
    plot_chromosomes(
        hits, args.out_dir / "RBBP4_loop_edges_by_chromosome.png", args.subtitle,
    )
    plot_circos(
        cliques, hits, args.out_dir / "RBBP4_clique_loop_edges_circos.png", args.subtitle,
    )
    if args.all_pairs and args.pair_loop_edges:
        edges = pd.read_csv(args.all_pairs, sep="\t")
        pair_hits = pd.read_csv(args.pair_loop_edges, sep="\t")
        plot_all_pairs_circos(
            edges, pair_hits,
            args.out_dir / "RBBP4_all_cicero_pairs_loop_circos.png",
            args.subtitle,
        )
    print(f"wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
