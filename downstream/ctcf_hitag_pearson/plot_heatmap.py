#!/usr/bin/env python3
"""CTCF 10 kb correlation heatmap: YlOrRd, vmin=0, vmax=1."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/work/users/d/y/dyy12/XuLab/downstream/ctcf_hitag_pearson")


def plot_matrix(tsv: Path, png: Path, title: str, cbar_label: str) -> None:
    lines = tsv.read_text().splitlines()
    labels = lines[0].split("\t")[1:]
    R = np.array([[float(x) for x in ln.split("\t")[1:]] for ln in lines[1:]], dtype=float)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.05 * n + 2.2, 1.05 * n + 1.4))
    im = ax.imshow(R, cmap="YlOrRd", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            ink = "white" if v >= 0.65 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=ink, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    plot_matrix(
        OUT / "CTCF_spearman_10kb.tsv",
        OUT / "CTCF_spearman_10kb.png",
        "CTCF  Spearman r, 10 kb bins",
        "Spearman r",
    )
    plot_matrix(
        OUT / "CTCF_pearson_10kb.tsv",
        OUT / "CTCF_pearson_10kb.png",
        "CTCF  Pearson r, 10 kb bins",
        "Pearson r",
    )
