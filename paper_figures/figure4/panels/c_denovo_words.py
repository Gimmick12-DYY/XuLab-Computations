"""Figure 4c, de novo version: for each example pair (rows RBBP4-ZNF644, ADNP-MTF2, DR1-NFYA), the three most similar pairs of de novo words,
found separately in each TF's own sites with no source motif. Each motif slot shows TF A's de novo logo above TF B's, each labelled with its
own de novo consensus, and one line with the Tomtom similarity (q) between the two words.

Inputs (source_data/): c_denovo_words.tsv, c_denovo_words_matrices.tsv (selection = "all"; made by prepare_denovo_words.py)
Output: c_denovo_words.{pdf,svg,png}, 183 x 150 mm (print size), Arial 7 pt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

PAIRS = ["RBBP4-ZNF644", "ADNP-MTF2", "DR1-NFYA"]
BASE_COLOURS = {"A": "#2C7A6B", "C": "#2166AC", "G": "#D98C1F", "T": "#B2182B"}   # palette C hues
SELECTION = "all"


def draw_logo(ax, P, font, width):
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D
    P = P / P.sum(1, keepdims=True)
    ic = 2 + (P * np.log2(np.maximum(P, 1e-9))).sum(1)
    for i, (row, h) in enumerate(zip(P, ic)):
        y = 0.0
        for k in np.argsort(row):
            hh = row[k] * h
            if hh < 0.01:
                continue
            tp = TextPath((0, 0), "ACGT"[k], size=1, prop=font)
            bb = tp.get_extents()
            tr = Affine2D().translate(-bb.x0, -bb.y0).scale(0.9 / bb.width, hh / bb.height).translate(i + 0.05, y)
            ax.add_patch(PathPatch(tr.transform_path(tp), fc=BASE_COLOURS["ACGT"[k]], lw=0))
            y += hh
    ax.set_xlim(0, width)
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([0, 2])
    ax.spines["bottom"].set_visible(False)


def main(src, out):
    style.use(base_pt=7.0, small_pt=7.0)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    bold = font_manager.FontProperties(family="Arial", weight="bold")
    W = pd.read_csv(src / "c_denovo_words.tsv", sep="\t")
    M = pd.read_csv(src / "c_denovo_words_matrices.tsv", sep="\t")
    W, M = W[W.selection == SELECTION], M[M.selection == SELECTION]
    fig = plt.figure(figsize=(183 * style.MM, 150 * style.MM))
    gs = fig.add_gridspec(3, 3, left=0.13, right=0.99, top=0.95, bottom=0.05, hspace=0.6, wspace=0.38)
    for i, pair in enumerate(PAIRS):
        rows = W[W.pair == pair].sort_values("rank")
        tfs = pair.split("-")
        for j in range(3):
            sub = gs[i, j].subgridspec(2, 1, hspace=0.45)
            axes = [fig.add_subplot(sub[k]) for k in range(2)]
            if j == 0:
                axes[0].text(-0.42, -0.2, pair, transform=axes[0].transAxes, rotation=90, ha="center", va="center", fontweight="bold")
            if j >= len(rows):
                for ax in axes:
                    ax.axis("off")
                continue
            r = rows.iloc[j]
            mats = [M[(M.pair == pair) & (M["rank"] == r["rank"]) & (M.tf == who)].sort_values("position")[["A", "C", "G", "T"]].to_numpy(float)
                    for who in ("A", "B")]
            width = max(len(m) for m in mats)
            for k, (tf, word, mat) in enumerate(zip(tfs, (r.word_A, r.word_B), mats)):
                draw_logo(axes[k], mat, bold, width)
                axes[k].set_ylabel(tf, rotation=0, ha="right", va="center", labelpad=2)
                axes[k].set_title(word, pad=1)
            axes[1].text(0.5, -0.12, f"Tomtom q = {r.tomtom_q:.1g}", transform=axes[1].transAxes, ha="center", va="top")
    fig.text(0.005, 0.005, "Logo height in bits (0-2); de novo words found separately in each TF's sites", ha="left", va="bottom")
    style.save(fig, out, "c_denovo_words")
