"""Fig 2 tornado plots at the round 11 ChIP-differential peaks (source downstream/fig2_qc/r11t_tornado.py).
fig2_tornado_chipdeseq2: rows = up to 1,000 differential peaks per ChIP (top by DESeq2 LFC), one block per ChIP, sorted
by the block's own ChIP signal; +-2 kb at 50 bp; columns = 6 bulk ChIPs then our 6 TFs raw (5' end CPM); each column on
its own scale (0 to the 98th percentile of nonzero values); Gaussian smoothing sigma 100 bp along x only; mean profile per
block above each column. fig2_tornado_chipdeseq2_imputed: same rows, our 6 TFs imputed (Mdrop5) at native 1 kb bins,
+-10 kb (grey = outside the mask)."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

CHIPC = LinearSegmentedColormap.from_list("chip", ["#FFFFFF", "#9E9E9E", "#1A1A1A"])
OURC = LinearSegmentedColormap.from_list("ours", ["#FFFFFF", "#FDBB84", "#E34A33", "#7F0000"])
IMPC = LinearSegmentedColormap.from_list("imp", ["#FFFFFF", "#9ECAE1", "#3182BD", "#08306B"])
BLK = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]


def tornado(z, cols, name, xt, title, src_n):
    style.use()
    blk = z["block"]; tfs = [str(t) for t in z["tfs"]]; n = len(cols)
    edges = np.r_[0, np.cumsum(z["block_n"])]
    fig = style.figure_mm(183, 200)
    L, Rr, w = 0.12, 0.99, None
    w = (Rr - L) / n
    for j, (X, ttl, cm) in enumerate(cols):
        nz = X[np.isfinite(X) & (X > 0)]; vmax = float(np.percentile(nz, 98)) if len(nz) else 1.0
        axp = fig.add_axes([L + j * w + 0.004, 0.80, w - 0.008, 0.09])
        x = np.arange(X.shape[1])
        for b in range(6):
            axp.plot(x, np.nanmean(X[blk == b], 0), color=BLK[b], lw=0.6)
        axp.set_xticks([]); axp.tick_params(labelsize=4.5, length=1, pad=1)
        axp.set_title(ttl, fontsize=6, pad=2)
        for s_ in ("top", "right"):
            axp.spines[s_].set_visible(False)
        ax = fig.add_axes([L + j * w + 0.004, 0.10, w - 0.008, 0.68])
        Xm = np.ma.masked_invalid(np.clip(X, 0, vmax)); cmc = cm.copy(); cmc.set_bad("#D9D9D9")
        im = ax.imshow(Xm, aspect="auto", cmap=cmc, vmin=0, vmax=vmax, interpolation="nearest", rasterized=True)
        for e in edges[1:-1]:
            ax.axhline(e - .5, color="#1A1A1A", lw=0.4)
        ax.set_yticks([]); ax.set_xticks([0, (X.shape[1] - 1) / 2, X.shape[1] - 1], xt if j == 0 else ["", "0", ""], fontsize=4.5); ax.tick_params(axis="x", length=1, pad=1)
        if j == 0:
            for b in range(6):
                ax.text(-0.08, 1 - (edges[b] + edges[b + 1]) / 2 / edges[-1], f"{tfs[b]}\ndiff. peaks\n(n = {edges[b + 1] - edges[b]:,})",
                        transform=ax.transAxes, ha="right", va="center", fontsize=5.5, color=BLK[b])
        cax = ax.inset_axes([0.1, -0.07, 0.8, 0.012]); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.3); cb.set_ticks([0, vmax]); cb.set_ticklabels(["0", f"{vmax:.2g}"]); cb.ax.tick_params(labelsize=4.5, length=1, pad=1)
    fig.text(0.55, 0.955, title, ha="center", va="top", fontsize=7)
    style.save(fig, src_n[0], name, raster_dpi=600)


def main(src, out):
    f = Path(src) / "tornado_chipdeseq2.npz"
    if not f.exists():
        return
    z = np.load(f); tfs = [str(t) for t in z["tfs"]]
    sm = lambda A: gaussian_filter1d(A.astype(float), sigma=2, axis=1)   # 100 bp at 50 bp bins
    cols = [(sm(z["chip"][i]), f"{t}\nChIP", CHIPC) for i, t in enumerate(tfs)] + [(sm(z["raw"][i]), f"{t}\nours (raw)", OURC) for i, t in enumerate(tfs)]
    tornado(z, cols, "fig2_tornado_chipdeseq2", ["-2 kb", "0", "+2 kb"],
            "ChIP-differential peaks (DESeq2 ~ChIP), summit ±2 kb: bulk ChIP (grey) and our raw single-cell signal (red), CPM", (out,))
    cols = [(z["imp"][i].astype(float), f"{t}\nours (imputed)", IMPC) for i, t in enumerate(tfs)]
    tornado(z, cols, "fig2_tornado_chipdeseq2_imputed", ["-10 kb", "0", "+10 kb"],
            "Same rows: our imputed signal (Mdrop5; fraction of cells, native 1 kb bins, ±10 kb; grey = outside mask)", (out,))
