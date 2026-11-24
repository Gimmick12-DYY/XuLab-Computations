"""Figure 4c: shared enriched motifs for the three example pairs (rows RBBP4-ZNF644, ADNP-MTF2, DR1-NFYA; three motifs each).
For every motif, two logos in parallel: one built from TF A's own motif instances and one from TF B's (source_data/c_logo_counts.tsv,
sites = A / B), labelled by the consensus of both TFs' instances; one line per motif gives the largest of the four p-values (both TFs x
discovery / held-out chromosomes). A dagger marks a generic motif (enriched in > 10 of 79 TFs).

Inputs (source_data/): c_shared_motifs.tsv, c_logo_counts.tsv (made by prepare_source_data.py)
Output: c_shared_motifs.{pdf,svg,png}, 183 x 150 mm (print size), Arial 7 pt.
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


def draw_logo(ax, counts, font):
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D
    P = (counts + 0.25) / (counts + 0.25).sum(1, keepdims=True)
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
    ax.set_xlim(0, len(P))
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([0, 2])
    ax.spines["bottom"].set_visible(False)


def main(src, out):
    style.use(base_pt=7.0, small_pt=7.0)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    bold = font_manager.FontProperties(family="Arial", weight="bold")
    S = pd.read_csv(src / "c_shared_motifs.tsv", sep="\t")
    C = pd.read_csv(src / "c_logo_counts.tsv", sep="\t")
    fig = plt.figure(figsize=(183 * style.MM, 150 * style.MM))
    gs = fig.add_gridspec(3, 3, left=0.13, right=0.99, top=0.95, bottom=0.05, hspace=0.55, wspace=0.38)
    for i, pair in enumerate(PAIRS):
        rows = S[S.pair == pair].sort_values("rank")
        tfs = pair.split("-")
        for j in range(3):
            sub = gs[i, j].subgridspec(2, 1, hspace=0.15)
            axes = [fig.add_subplot(sub[k]) for k in range(2)]
            if j == 0:
                axes[0].text(-0.42, -0.08, pair, transform=axes[0].transAxes, rotation=90, ha="center", va="center", fontweight="bold")
            if j >= len(rows):
                for ax in axes:
                    ax.axis("off")
                continue
            r = rows.iloc[j]
            for k, (who, tf) in enumerate(zip(("A", "B"), tfs)):
                cnt = C[(C.pair == pair) & (C["rank"] == r["rank"]) & (C.sites == who)].sort_values("position")[["A", "C", "G", "T"]].to_numpy(float)
                draw_logo(axes[k], cnt, bold)
                axes[k].set_ylabel(tf, rotation=0, ha="right", va="center", labelpad=2)
            axes[0].set_title(r.consensus + ("†" if r.generic else ""), pad=2)
            axes[1].text(0.5, -0.12, f"p ≤ {r.p_max_of_4:.2g}", transform=axes[1].transAxes, ha="center", va="top")
    fig.text(0.005, 0.005, "Logo height in bits (0-2)", ha="left", va="bottom")
    style.save(fig, out, "c_shared_motifs")
