"""Fig 2 example: CTCF clustered peak heatmap (deepTools-style k-means clusters stacked; source r28_ctcf_clusters.py).
Clusters (k = 3, k-means on z-scored log1p ChIP and raw centre signal): "ChIP + ours", "ChIP only", "ours only",
stacked in that order with white separators and a left label with n sites. Within a cluster rows are sorted by ChIP
centre signal ("ours only" by raw centre). Display as the main heatmap: consecutive rows averaged in groups (group size =
ceil(total sites / 500), same for every cluster and panel), Gaussian sigma 100 bp along rows for ChIP and raw; imputed
(Mdrop5) on its own +-5 kb axis at 1 kb, no smoothing. Colour per panel 0 to the 95th percentile (turbo). Profiles on top:
one line per cluster (mean over the cluster's sites, unsmoothed)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

ORDER = ["ChIP + ours", "ChIP only", "ours only"]
COL = {"ChIP + ours": "#B03A2E", "ChIP only": "#55606E", "ours only": "#E8821E"}


def blocks(X, b, sigma):
    m = len(X) // b * b
    Y = X[:m].reshape(-1, b, X.shape[1]).mean(1)
    if m < len(X):
        Y = np.vstack([Y, X[m:].mean(0, keepdims=True)])
    return gaussian_filter1d(Y, sigma, axis=1, mode="nearest") if sigma else Y


def main(src, out):
    style.use()
    z = np.load(Path(src) / "ctcf_clusters.npz", allow_pickle=True)
    lab = z["label"].astype(str)
    idx = []
    for c in ORDER:
        m = np.flatnonzero(lab == c)
        key = z["raw_c"][m] if c == "ours only" else z["chip_c"][m]
        idx.append(m[np.argsort(-key, kind="stable")])
    n = sum(len(i) for i in idx); b = int(np.ceil(n / 500))
    fig = style.figure_mm(120, 150)
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.16, right=0.985, top=0.93, bottom=0.07)
    spec = [("chip", "Bulk ChIP (HEK293)", "CPM", np.arange(-975, 1000, 50), 1000, "±1 kb, 50 bp", 2),
            ("raw", "sc raw", "CPM", np.arange(-975, 1000, 50), 1000, "±1 kb, 50 bp", 2),
            ("imp", "sc imputed", "frac. cells", np.arange(-5, 6) * 1000, 5500, "±5 kb, 1 kb bins", 0)]
    for j, (key, ttl, unit, xs, half, xl, sig) in enumerate(spec):
        g = gs[j].subgridspec(4, 1, height_ratios=[3.2, 22, 0.9, 0.5], hspace=0.06)
        axp = fig.add_subplot(g[0])
        for c, ix in zip(ORDER, idx):
            axp.plot(xs, z[key][ix].mean(0), color=COL[c], lw=0.8, label=c)
        axp.set_xlim(-half, half); axp.set_ylim(0, None); axp.set_xticks([])
        axp.yaxis.set_major_locator(plt.MaxNLocator(2)); axp.tick_params(axis="y", labelsize=5, length=1.5, pad=1)
        axp.set_title(ttl, fontsize=6, pad=2)
        if j == 0:
            axp.legend(fontsize=5, loc="upper left", handlelength=1.0, borderaxespad=0.1, labelspacing=0.2)
        Ys = [blocks(z[key][ix], b, sig) for ix in idx]
        Y = np.vstack(Ys); vm = float(np.percentile(Y, 95)) or 1.0
        ax = fig.add_subplot(g[1])
        im = ax.imshow(Y, aspect="auto", cmap="turbo", vmin=0, vmax=vm, interpolation="nearest", extent=[-half, half, len(Y), 0], rasterized=True)
        y0 = 0
        for c, Yc, ix in zip(ORDER, Ys, idx):
            if y0:
                ax.axhline(y0, color="white", lw=1.2)
            if j == 0:
                ax.text(-half * 1.06, y0 + len(Yc) / 2, f"{c}\n{len(ix):,}", ha="right", va="center", fontsize=6, color=COL[c])
            y0 += len(Yc)
        kb = half // 1000
        ax.set_yticks([]); ax.set_xticks([-kb * 1000, 0, kb * 1000], [f"−{kb}", "0", f"+{kb}"])
        ax.tick_params(axis="x", pad=1, length=1.5, labelsize=5); ax.set_xlabel(xl, fontsize=5, labelpad=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(True)
        cax = fig.add_subplot(g[3]); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
        cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
        for lb, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
            lb.set_ha(ha)
        cb.set_label(f"{unit}\n{n:,} sites", fontsize=5, labelpad=1)
    fig.text(0.57, 0.985, "CTCF: ChIP summits + our peak summits, k-means clusters", ha="center", va="top", fontsize=7, fontweight="bold")
    style.save(fig, out, "fig2_example_CTCF_clustered_heatmap", raster_dpi=600)
