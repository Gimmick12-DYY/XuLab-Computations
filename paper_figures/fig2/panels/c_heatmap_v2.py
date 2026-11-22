"""Fig 2 main heatmap, proposal v2 (closer to Kaya-Okur et al. 2019 Fig 2): one shared axis, +-1 kb at 50 bp.
  Bulk ChIP: CPM, 50 bp bins.  sc raw: 5' ends CPM, 50 bp bins, 3-bin running mean.
  sc imputed (MASK = "Mdrop5"): the 1 kb imputed value of the genomic bin covering each 50 bp position (from K_impM),
    (no extra smoothing beyond the shared noise reduction below).
  Rows: the reference's top 10,000 ChIP summits; rows whose summit bin is outside MASK are dropped from all three panels
    (share reported in heatmap_v2_rows.tsv and the legend); remaining out-of-mask flank positions are drawn as 0.
  Sorted by ChIP within +-200 bp.
  Noise reduction, identical for all three panels: consecutive sorted rows averaged in groups so ~500 rows are drawn
    (group size = ceil(n / 500), e.g. 20 for 10,000 sites), then a Gaussian smoothing along each row (sigma = 100 bp =
    2 bins). Colour per panel 0 to the 95th percentile of the row-averaged, smoothed values (turbo). The profile on top
    is the mean of the unsmoothed rows.
Outputs fig2_main_heatmap_v2 (top 10,000 sites), fig2_main_heatmap_v2_top5000 (the 5,000 rows with the highest ChIP
centre signal), and per-TF fig2_main_heatmap_v2_<TF>."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

MASK = "Mdrop5"
TFS = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
W, STEP = 1000, 50
GROUP = [1]
XS = np.arange(-W, W, STEP) + STEP // 2


def tracks(z):
    keep = z["K_maskM"][:, 10].astype(bool)
    s = z["K_summit"].astype(np.int64)
    k = (s[:, None] + XS[None, :]) // 1000 - (s // 1000)[:, None] + 10
    imp = np.take_along_axis(z["K_impM"].astype(np.float32), k, 1)
    imp = np.where(np.take_along_axis(z["K_maskM"], k, 1), imp, 0)
    T = {"ChIP": z["V3_ChIP"].astype(np.float32), "raw": z["V3_raw"].astype(np.float32), "imputed": imp}
    return {k_: v[keep] for k_, v in T.items()}, keep


TARGET_ROWS, SIGMA_BP, CAP_PCT = 500, 100, 95


def denoise(X):
    """rows averaged in consecutive groups (~TARGET_ROWS drawn), then Gaussian smoothing along each row."""
    b = max(1, int(np.ceil(len(X) / TARGET_ROWS)))
    n = len(X) // b * b
    Y = X[:n].reshape(-1, b, X.shape[1]).mean(1)
    if n < len(X):
        Y = np.vstack([Y, X[n:].mean(0, keepdims=True)])
    return gaussian_filter1d(Y, sigma=SIGMA_BP / STEP, axis=1, mode="nearest"), b


def panel(fig, cell, X, title, unit, n):
    import matplotlib.pyplot as plt
    g = cell.subgridspec(4, 1, height_ratios=[3.0, 22, 0.9, 0.5], hspace=0.06)
    axp = fig.add_subplot(g[0]); pr = X.mean(0)
    axp.plot(XS, pr, color="#1A1A1A", lw=0.7); axp.set_xlim(-W, W); axp.set_ylim(0, pr.max() * 1.15 or 1)
    axp.set_xticks([]); axp.yaxis.set_major_locator(plt.MaxNLocator(2)); axp.tick_params(axis="y", labelsize=5, length=1.5, pad=1)
    axp.set_title(title, fontsize=6, pad=2)
    ax = fig.add_subplot(g[1])
    Y, b = denoise(X)
    vm = float(np.percentile(Y, CAP_PCT)) or 1.0
    im = ax.imshow(Y, aspect="auto", cmap="turbo", vmin=0, vmax=vm, interpolation="nearest",
                   extent=[-W, W, len(X), 0], rasterized=True)
    GROUP[0] = b
    ax.set_yticks([]); ax.set_xticks([-W, 0, W], ["−1 kb", "0", "+1 kb"])
    for lb, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
        lb.set_ha(ha)
    ax.tick_params(axis="x", pad=1, length=1.5, labelsize=5)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(True)
    cax = fig.add_subplot(g[3]); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
    for lb, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        lb.set_ha(ha)
    cb.set_label(f"{unit}\n{n:,} sites", fontsize=5, labelpad=1)


def build(src, out, tfs, name, w_mm, h_mm, ncol, rows, top=None):
    fig = style.figure_mm(w_mm, h_mm)
    nrow = int(np.ceil(len(tfs) / ncol))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.14, hspace=0.14, left=0.03 if ncol > 1 else 0.08, right=0.985,
                          top=0.95 if nrow > 1 else 0.90, bottom=0.03 if nrow > 1 else 0.05)
    for k, t in enumerate(tfs):
        z = np.load(Path(src) / f"heatmaps_{t}.npz")
        T, keep = tracks(z)
        if top:
            T = {k_: v[:top] for k_, v in T.items()}
        rows[t] = dict(TF=t, top_chip_summits=len(keep), kept_inside_mask=int(keep.sum()), dropped_share=float(1 - keep.mean()),
                       rows_drawn_from=len(T["ChIP"]), group_size=int(np.ceil(len(T["ChIP"]) / TARGET_ROWS)))
        r, c = divmod(k, ncol)
        sub = gs[r, c].subgridspec(1, 3, wspace=0.32)
        for j, (ty, ttl, un) in enumerate((("ChIP", "Bulk ChIP", "CPM"), ("raw", "sc raw", "CPM"), ("imputed", "sc imputed", "frac. cells"))):
            panel(fig, sub[j], T[ty], ttl, un, len(T[ty]))
        pos = gs[r, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + (0.018 if nrow > 1 else 0.05), f"{t} ({CELL[t]} ChIP)", ha="center", fontsize=7, fontweight="bold")
    style.save(fig, out, name, raster_dpi=600)


def main(src, out):
    style.use()
    rows = {}
    build(src, out, TFS, "fig2_main_heatmap_v2", 183, 235, 3, rows)
    r5 = {}
    build(src, out, TFS, "fig2_main_heatmap_v2_top5000", 183, 235, 3, r5, top=5000)
    print(pd.DataFrame(r5.values()).round(3).to_string())
    for t in TFS:
        build(src, out, [t], f"fig2_main_heatmap_v2_{t}", 89, 120, 1, {})
    pd.DataFrame(rows.values()).to_csv(Path(src) / "heatmap_v2_rows.tsv", sep="\t", index=False, float_format="%.4f")
    print(pd.DataFrame(rows.values()).round(3).to_string())
