"""Supplementary to Figure 3b: RBBP4 pairs and module edges (> 25 kb) on a Peakachu 0.7 loop vs their 10,000-shuffle null
(same-chromosome shifts keeping each pair's spacing; perm_tf_long.py). Histogram of the null, dashed = null median (95 % interval in
the label), black line = observed; p = (1 + shuffles >= observed) / (1 + 10,000).

Input (source_data/): rbbp4_shuffle_null.npz.   Output: supplementary/S_rbbp4_permutation.{pdf,svg,png}, 116 x 56 mm (print size).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402


def main(src, out):
    P = style.use(base_pt=7, small_pt=6)
    import matplotlib.pyplot as plt
    N = P.neutral
    z = np.load(src / "rbbp4_shuffle_null.npz")
    null, obs, n = z["null"] * 100, z["obs"] * 100, z["n"]
    fig, axes = plt.subplots(1, 2, figsize=(116 * style.MM, 56 * style.MM), gridspec_kw=dict(wspace=0.32))
    for ax, j, what in zip(axes, (0, 1), ("pairs", "module edges")):
        v = null[:, j]
        ge = int((v >= obs[j]).sum())
        p = (1 + ge) / (1 + len(v))
        step = 100 / n[j]                                  # one bin per attainable value for module edges (n = 1,031)
        bins = 60 if j == 0 else np.arange(v.min() - step / 2, v.max() + step, step)
        ax.hist(v, bins=bins, color=N.mid, lw=0)
        md, lo, hi = np.median(v), *np.percentile(v, [2.5, 97.5])
        ax.axvline(md, color=N.dark, lw=0.8, ls=(0, (3, 2)))
        ax.axvline(obs[j], color=N.ink, lw=1.6)
        top = ax.get_ylim()[1]
        ax.text(md + 0.25, top * 0.98, f"null median {md:.1f}%\n(95%: {lo:.1f}–{hi:.1f}%)", va="top", fontsize=6, color=N.dark)
        ax.text(obs[j] - 0.2, top * 0.8, f"observed {obs[j]:.1f}%", ha="right", va="center", fontsize=6.5)
        ax.set_xlabel(f"% of {what} >25 kb on a 0.7 loop")
        ax.set_ylabel("shuffles")
        ax.set_xlim(right=obs[j] + 2)
        ptxt = f"p < 1e-4: {ge} of {len(v):,} shuffles ≥ observed" if p < 1e-4 * 1.0001 else f"p = {p:.2g}: {ge} of {len(v):,} shuffles ≥ observed"
        ax.set_title(f"RBBP4 {what} >25 kb (n = {int(n[j]):,})\n{ptxt}", loc="left", fontsize=7)
    style.save(fig, Path(out) / "supplementary", "S_rbbp4_permutation", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
