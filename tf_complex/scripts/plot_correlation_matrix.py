#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# plot_correlation_matrix.py
#
# Clustered TF x TF correlation heatmap.
#
# Clustering always uses the raw Pearson matrix (average linkage on d = 1-r).
# Colour mapping is separate and does not change the tree or the TSV:
#   percentile  global empirical CDF of unique off-diagonal r  (default)
#   r           raw Pearson, floor = 2nd percentile (or --vmin)
#   clip        raw Pearson, floor = 0.92 (or --vmin)
#
# Layout (inches, square cells):
#   title + colorbar | dendrogram | heatmap | cell-count bars
#
#   python tf_complex/scripts/plot_correlation_matrix.py \
#     --sim-tsv tf_complex/results_compartment_rpkm_domain_sm2_1Mb/tf_similarity_genome.tsv \
#     --out     tf_complex/results_compartment_rpkm_domain_sm2_1Mb/compartment_rpkm_genome.png \
#     --color   percentile
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


def load_similarity_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    labels = header[1:]
    R = np.loadtxt(path, delimiter="\t", skiprows=1, usecols=range(1, len(header)))
    return labels, np.asarray(R, dtype=float)


def load_cell_counts(path: Path | None) -> dict[str, int]:
    if path is None or not Path(path).is_file():
        return {}
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return {}
    hdr = [h.strip() for h in rows[0]]
    if "TF" in hdr:
        i = hdr.index("TF")
        cc: dict[str, int] = {}
        for r in rows[1:]:
            if len(r) > i and r[i].strip():
                k = r[i].strip().lower()
                cc[k] = cc.get(k, 0) + 1
        return cc
    cc = {}
    for r in rows:
        if len(r) >= 2 and r[1].strip().isdigit():
            cc[r[0].strip().lower()] = int(r[1])
    return cc


def linkage_from_pearson(R: np.ndarray, optimize_leaves: bool = True):
    """Average-linkage tree on d = 1-r.

    If optimize_leaves, rotate forks (optimal leaf order) so adjacent leaves
    are as similar as possible. Topology is unchanged.
    """
    D = 1.0 - R
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    y = squareform(D, checks=False)
    Z = linkage(y, method="average")
    if optimize_leaves:
        Z = optimal_leaf_ordering(Z, y)
    return Z


def global_percentile(R: np.ndarray) -> np.ndarray:
    """q_ij = F_emp(r_ij) from the unique upper-triangle pairs. One mapping, not per-row."""
    tri = R[np.triu_indices(R.shape[0], k=1)]
    s = np.sort(tri)
    Q = np.searchsorted(s, R, side="right") / float(s.size)
    np.fill_diagonal(Q, 1.0)
    return Q


def color_matrix(R: np.ndarray, mode: str = "percentile", vmin: float | None = None):
    """Return (values to paint, vmin, vmax, colorbar label). Does not touch clustering."""
    if mode == "percentile":
        return global_percentile(R), 0.0, 1.0, "Percentile of\nTF–TF Pearson r"
    if mode == "clip":
        lo = 0.92 if vmin is None else float(vmin)
        return R, lo, 1.0, f"pearson r\n(clip {lo:.2f}–1)"
    lo = float(np.percentile(R, 2)) if vmin is None else float(vmin)
    return R, lo, 1.0, "pearson cor"


def write_ordered_tsv(path: Path, M: np.ndarray, labels: list[str], order: list[int]) -> Path:
    """Write M in heatmap axis order. Row/col 0 = top-left of the PNG (origin=upper)."""
    names = [labels[i] for i in order]
    Mo = M[np.ix_(order, order)]
    path = Path(path)
    with path.open("w") as f:
        f.write("TF\t" + "\t".join(names) + "\n")
        for i, lab in enumerate(names):
            f.write(lab + "\t" + "\t".join(f"{Mo[i, j]:.4f}" for j in range(len(names))) + "\n")
    print(f"[tsv] wrote {path}")
    return path


def plot_correlation_matrix(
    R: np.ndarray,
    labels: list[str],
    out: Path,
    *,
    cells: dict[str, int] | None = None,
    color: str = "percentile",
    vmin: float | None = None,
    title: str = "",
    Z=None,
    cmap: str = "RdYlBu_r",
    cell_in: float = 0.145,
    dpi: int = 160,
    write_tsv: bool = True,
    optimize_leaves: bool = True,
) -> Path:
    """Draw dendrogram | heatmap | optional cell-count bars. Save `out`.

    `R` must be the raw Pearson matrix. `color` only changes the heatmap paint.
    If write_tsv, also write `out` with .tsv: same row/column order as the PNG
    (first TF = top-left; the 1-diagonal runs down-right, like the file).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    if Z is None:
        Z = linkage_from_pearson(R, optimize_leaves=optimize_leaves)
    order = dendrogram(Z, no_plot=True)["leaves"]
    names = [labels[i].upper() for i in order]
    P, vmin, vmax, cbar_lab = color_matrix(R, color, vmin)

    heat_in = n * cell_in
    dend_in, gap_in, bar_in = 1.7, 1.35, 3.4
    left_in, right_in, bot_in, top_in = 1.55, 0.35, 2.15, 0.35
    if not cells:
        bar_in = 0.0
        gap_in = 1.15
    fig_w = left_in + dend_in + heat_in + gap_in + bar_in + right_in
    fig_h = bot_in + heat_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def box(x, w):
        return [x / fig_w, bot_in / fig_h, w / fig_w, heat_in / fig_h]

    x = left_in
    axd = fig.add_axes(box(x, dend_in)); x += dend_in
    axh = fig.add_axes(box(x, heat_in)); x += heat_in + gap_in
    axb = fig.add_axes(box(x, bar_in)) if cells else None
    cax = fig.add_axes([0.038, 0.30, 0.012, 0.24])

    if title:
        fig.text(0.014, 0.80, title, fontsize=12, va="top")

    dendrogram(Z, orientation="left", ax=axd, no_labels=True, link_color_func=lambda k: "#555")
    axd.set_xticks([]); axd.set_yticks([])
    # SciPy puts leaves[0] at y=0; invert so leaf 0 is at the TOP (matches origin=upper).
    axd.set_ylim(10 * n, 0)
    for s in axd.spines.values():
        s.set_visible(False)

    Ro = P[np.ix_(order, order)]
    im = axh.imshow(Ro, aspect="equal", origin="upper", cmap=cmap,
                    vmin=vmin, vmax=vmax, interpolation="nearest")
    axh.set_xlim(-0.5, n - 0.5)
    axh.set_ylim(n - 0.5, -0.5)
    fs = 5.5
    axh.set_xticks(range(n)); axh.set_xticklabels(names, rotation=90, fontsize=fs)
    axh.yaxis.tick_right()
    axh.set_yticks(range(n)); axh.set_yticklabels(names, fontsize=fs)
    axh.set_xticks(np.arange(-0.5, n, 1), minor=True)
    axh.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axh.grid(which="minor", color="white", linewidth=0.35)
    axh.tick_params(which="minor", length=0)
    axh.tick_params(length=0)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_lab, fontsize=7)
    cb.ax.tick_params(labelsize=6)

    if axb is not None:
        hsv = plt.get_cmap("hsv")
        cols = [hsv(0.02 + 0.90 * i / max(n - 1, 1)) for i in range(n)]
        cnt = np.array([cells.get(labels[i].lower(), 1) for i in order], float)
        axb.barh(np.arange(n), np.maximum(cnt, 1), color=cols, height=0.8)
        axb.set_xscale("log")
        axb.set_ylim(n - 0.5, -0.5)
        axb.set_yticks(range(n))
        axb.set_yticklabels(names, fontsize=fs)
        axb.tick_params(length=0)
        axb.set_xlabel("Number of cells")
        for s in ("top", "right", "left"):
            axb.spines[s].set_visible(False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[plot] wrote {out}")
    if write_tsv:
        write_ordered_tsv(out.with_suffix(".tsv"), P, labels, order)
        if color != "r":
            write_ordered_tsv(out.with_name(out.stem + "_pearson.tsv"), R, labels, order)
    return out


def main() -> int:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-tsv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell-meta", type=Path,
                    default=here.parents[2] / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--color", choices=["percentile", "r", "clip"], default="percentile")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--leaf-order", choices=["optimal", "linkage"], default="optimal",
                    help="optimal: rotate forks so neighbors are similar; "
                         "linkage: SciPy default leaf order")
    args = ap.parse_args()

    labels, R = load_similarity_tsv(args.sim_tsv)
    cells = load_cell_counts(args.cell_meta)
    title = args.title
    if not title:
        title = f"TF–TF Pearson\n{args.color}\n({args.sim_tsv.stem})"
    plot_correlation_matrix(
        R, labels, args.out, cells=cells, color=args.color,
        vmin=args.vmin, title=title, dpi=args.dpi,
        optimize_leaves=(args.leaf_order == "optimal"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
