"""Fig 2 corr_matrix_6x6 without the 79-TF pool (round 12; source downstream/fig2_qc/r12_nopool.py).
fig2_corr_matrix_6x6_nopool (+_colz): sc value = log2((centre +-100 bp + .5)/(flank 1-2 kb + .5)), no pool subtraction;
fig2_corr_matrix_6x6_nopool_cpm (+_colz): sc value = log2 centre CPM + 1 (no flank, no pool);
fig2_corr_matrix_6x6_nopool_vs_pooled: pooled (current) | no-pool centre/flank | no-pool CPM, plain r on one colour scale
(top) and column z-score on one scale (bottom)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

_spec = importlib.util.spec_from_file_location("a_corr", Path(__file__).with_name("a_corr_matrix.py"))
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)
TFS = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]


def main(src, out):
    if not (Path(src) / "corr_nopool_cf_6x6.tsv").exists():
        return
    A.deseq2(src, out, "corr_nopool_cf_6x6.tsv", "fig2_corr_matrix_6x6_nopool",
             "summit groups; sc log2 centre/flank, no 79-TF pool; ZNF777 depth")
    A.deseq2(src, out, "corr_nopool_cpm_6x6.tsv", "fig2_corr_matrix_6x6_nopool_cpm",
             "summit groups; sc log2 centre CPM, no flank, no pool; ZNF777 depth")
    P = style.use()
    sets = [("corr_matrix_6x6.tsv", "pooled (current)"), ("corr_nopool_cf_6x6.tsv", "no pool: centre/flank"),
            ("corr_nopool_cpm_6x6.tsv", "no pool: centre CPM")]
    Ms = [pd.read_csv(Path(src) / f, sep="\t", index_col=0).values.astype(float) for f, _ in sets]
    lim = float(np.ceil(max(np.abs(M).max() for M in Ms) * 10) / 10)
    cm = LinearSegmentedColormap.from_list("div", A.DIV)
    fig = style.figure_mm(180, 130)
    for k, (M, (_, ttl)) in enumerate(zip(Ms, sets)):
        for r, (X, v, lab) in enumerate(((M, lim, "Pearson r"), ((M - M.mean(0)) / M.std(0), 2.0, "column z"))):
            ax = fig.add_axes([0.09 + k * 0.31, 0.56 - r * 0.47, 0.22, 0.34])
            im = ax.imshow(np.clip(X, -v, v), cmap=cm, vmin=-v, vmax=v)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{X[i, j]:+.2f}", ha="center", va="center", fontsize=4.5,
                            color="white" if abs(X[i, j]) > 0.75 * v else P.neutral.ink)
                ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.7))
            obs, p = A.perm_p(M); _, p5 = A.perm_p(M[1:, 1:])
            ax.set_xticks(range(6), [f"{t}" for t in TFS], rotation=90, fontsize=5)
            ax.set_yticks(range(6), [f"{t} sc" for t in TFS], fontsize=5); ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if r == 0:
                ax.set_title(f"{ttl}\nown best {sum(np.argmax(M[i]) == i for i in range(6))}/6; p = {p:.3f} (no CTCF {p5:.3f})", fontsize=5.5)
            if k == 2:
                cax = ax.inset_axes([1.06, 0, 0.05, 1]); cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.3)
                cb.ax.tick_params(labelsize=4.5); cb.set_label(lab + (f" (±{v:g})" if r == 0 else " (±2)"), fontsize=5)
    fig.text(0.5, 0.985, "Summit-group 6x6: with vs without the 79-TF pool (columns = bulk ChIP; same groups, ZNF777 depth)",
             ha="center", va="top", fontsize=6.5)
    style.save(fig, out, "fig2_corr_matrix_6x6_nopool_vs_pooled")
