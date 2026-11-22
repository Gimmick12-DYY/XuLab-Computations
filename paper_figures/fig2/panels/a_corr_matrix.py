"""Fig 2 corr_matrix_6x6: summit-group Pearson r, single-cell raw (rows) vs bulk ChIP (columns), colour clipped at +-0.2
(diverging, centred at 0; every cell annotated). Diagonal-minus-off-diagonal p from all 720 row-label permutations,
also without CTCF (120). Input source_data/corr_matrix_6x6.tsv."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
DIV = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"]


def perm_p(M):
    k = len(M); off = ~np.eye(k, dtype=bool); st = lambda A: A.diagonal().mean() - A[off].mean()
    obs = st(M); null = np.array([st(M[list(p)]) for p in itertools.permutations(range(k))])
    return obs, float(np.mean(null >= obs - 1e-12))


def main(src, out):
    P = style.use()
    D = pd.read_csv(Path(src) / "corr_matrix_6x6.tsv", sep="\t", index_col=0)
    tfs = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]   # row/column order of the source table (TFS)
    M = D.values.astype(float)
    obs, p = perm_p(M); _, p5 = perm_p(M[1:, 1:])
    lim = 0.2
    fig = style.figure_mm(89, 80)
    ax = fig.add_axes([0.25, 0.30, 0.56, 0.56])
    cm = LinearSegmentedColormap.from_list("div", DIV)
    im = ax.imshow(np.clip(M, -lim, lim), cmap=cm, vmin=-lim, vmax=lim)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.75 * lim else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in tfs], rotation=90)
    ax.set_yticks(range(6), [f"{t} sc" for t in tfs])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"diagonal − off-diagonal = {obs:+.3f}, p = {p:.3f}\n(without CTCF p = {p5:.3f}; 720 / 120 permutations)", fontsize=6)
    cax = fig.add_axes([0.84, 0.30, 0.03, 0.56])
    cb = fig.colorbar(im, cax=cax); cb.set_ticks([-0.2, -0.1, 0, 0.1, 0.2]); cb.outline.set_linewidth(0.4)
    cb.set_label("Pearson r (colour clipped at ±0.2)")
    style.save(fig, out, "fig2_corr_matrix_6x6")
    print("own r:", dict(zip(tfs, np.diag(M).round(3))), "p", round(p, 4), "noCTCF", round(p5, 4))
