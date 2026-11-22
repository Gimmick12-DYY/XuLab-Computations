"""Fig 2 peak heatmaps in the style of Kaya-Okur et al. 2019 (Nat Commun, Fig 2): per TF, Bulk ChIP | sc raw |
sc imputed, +-1 kb at 50 bp bins, one colour scale per panel (0 to the 98th percentile of nonzero values, turbo),
colour bar with unit and number of sites underneath, rows sorted top to bottom by centre signal (+-200 bp).
  V3 (main): the reference's top 10,000 ChIP summits, the same rows in every panel, sorted by ChIP.
  V2 (supplement): ChIP summits + our raw peaks > 500 bp from any ChIP peak, same rows, sorted by ChIP.
  V1 (supplement): each dataset on its own peaks, sorted by its own signal.
Variants: _profile adds the mean signal per bin over all rows on top of each panel (Kaya-Okur Fig 2g; y from 0, own
axis per panel). V3_smoothed: raw from the unsmoothed 50 bp counts (V3_raw50), summed to 200 bp bins, one 3-bin running mean, and rows
averaged in groups of 20 consecutive sorted rows (smoothed once); ChIP and imputed as V3 (legend note, not on the panel).
Imputed (all V3 versions) is drawn on its native 1 kb grid: +-10 kb, 21 bins centred on the 1 kb bin holding the summit.
Imputed mask = MASK below (one named parameter). MASK "Mdrop5" = the accessibility mask: 1 kb bins with mean ATAC CPM
> 0.0781542 over the two pooled HEK293T ATAC bigwigs; imputed = raw-positive cells + model calls inside the mask
(source K_impM/K_maskM, r24_fig2_mask_source.py); bins outside the mask in light grey (NA).
K_M / K_B versions put ChIP, raw and imputed all on that 1 kb grid: K_M imputed under MASK (main), K_B genome-wide
pre-mask imputed (supplement). Imputed always includes the kept raw-positive cells (keep_raw), noted in the legend.
Raw = aggregated 5' ends, CPM, 3-bin running mean; imputed = fraction of cells positive in the 1 kb bin covering each
position (imputation is at 1 kb). Heatmap bodies are rasterised. Input source_data/heatmaps_<TF>.npz."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

MASK = "Mdrop5"   # imputed mask for every Fig 2 imputed panel; arrays K_impM / K_maskM are built for this mask
TFS = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
TTL = {"ChIP": "Bulk ChIP", "raw": "sc raw", "imputed": "sc imputed"}
UNIT = {"ChIP": "CPM", "raw": "CPM", "imputed": "frac. cells"}
SUB = {"V3": "top 10,000 ChIP summits; same rows in every panel, sorted by ChIP",
       "K_M": f"top 10,000 ChIP summits; all panels on the 1 kb grid; imputed under the {MASK} accessibility mask (grey = outside)",
       "K_B": "top 10,000 ChIP summits; all panels on the 1 kb grid; imputed = genome-wide pre-mask",
       "V3_smoothed": "top 10,000 ChIP summits; same rows, sorted by ChIP; raw at 200 bp bins",
       "V2": "ChIP summits + our peaks; same rows in every panel, sorted by ChIP",
       "V1": "each panel on its own peaks, sorted by its own signal"}
W = 1000


def smooth_raw(X):
    """50 bp -> 200 bp (sum of 4 bins), 3-bin running mean, then rows averaged in groups of 20."""
    X = X.astype(np.float32)
    X = X.reshape(len(X), -1, 4).sum(2)
    k = np.ones(3) / 3
    X = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, X)
    n = len(X) // 20 * 20
    return np.vstack([X[:n].reshape(-1, 20, X.shape[1]).mean(1), X[n:].mean(0, keepdims=True)]) if n < len(X) else X.reshape(-1, 20, X.shape[1]).mean(1)


def panel(fig, cell, X, title, unit, profile=False, nsites=None, W=1000, mask=None, xlab=None):
    X = X.astype(np.float32)
    if mask is not None:
        X = np.where(mask, X, np.nan)
    if profile:
        g = cell.subgridspec(4, 1, height_ratios=[3.2, 22, 0.9, 0.5], hspace=0.06)
        axp = fig.add_subplot(g[0])
        xs = np.linspace(-W, W, X.shape[1] + 1); xs = (xs[:-1] + xs[1:]) / 2
        pr = np.nanmean(X, 0)
        axp.plot(xs, pr, color="#1A1A1A", lw=0.7)
        axp.set_xlim(-W, W); axp.set_ylim(0, pr.max() * 1.15 if pr.max() > 0 else 1); axp.set_xticks([])
        axp.yaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(2))
        axp.tick_params(axis="y", labelsize=5, length=1.5, pad=1)
        axp.set_title(title, fontsize=6, pad=2)
        hi, ci = 1, 3
    else:
        g = cell.subgridspec(3, 1, height_ratios=[22, 0.9, 0.5], hspace=0.04)
        hi, ci = 0, 2
    ax = fig.add_subplot(g[hi])
    nz = X[np.isfinite(X) & (X > 0)]
    vm = float(np.percentile(nz, 98)) if nz.size else 1.0
    cm = plt.get_cmap("turbo").copy(); cm.set_bad("#E3E3E3")
    im = ax.imshow(np.ma.masked_invalid(X), aspect="auto", cmap=cm, vmin=0, vmax=vm, interpolation="nearest",
                   extent=[-W, W, len(X), 0], rasterized=True)
    kb = W // 1000
    ax.set_yticks([]); ax.set_xticks([-kb * 1000, 0, kb * 1000], [f"−{kb}", "0", f"+{kb}"])
    if xlab:
        ax.set_xlabel(xlab, fontsize=5, labelpad=0)
    for lb, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
        lb.set_ha(ha)
    ax.tick_params(axis="x", pad=1, length=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    if not profile:
        ax.set_title(title, fontsize=6, pad=2)
    cax = fig.add_subplot(g[ci])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
    for lb, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        lb.set_ha(ha)
    cb.set_label(f"{unit}\n{(nsites or len(X)):,} sites", fontsize=5, labelpad=1)


def build(src, out, ver, tfs, name, w_mm, h_mm, ncol, profile=False):
    fig = style.figure_mm(w_mm, h_mm)
    nrow = int(np.ceil(len(tfs) / ncol))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.12, hspace=0.16, left=0.015 if ncol > 1 else 0.07, right=0.985, top=0.93 if nrow > 1 else 0.86,
                          bottom=0.04 if nrow > 1 else 0.07)
    for k, t in enumerate(tfs):
        z = np.load(Path(src) / f"heatmaps_{t}.npz")
        r, c = divmod(k, ncol)
        sub = gs[r, c].subgridspec(1, 3, wspace=0.34 if profile else 0.14)
        base = "V3" if ver == "V3_smoothed" else ver
        for j, ty in enumerate(("ChIP", "raw", "imputed")):
            kw = {}
            if ver in ("K_M", "K_B"):
                X = z[{"ChIP": "K_ChIP", "raw": "K_raw", "imputed": "K_impM" if ver == "K_M" else "K_impB"}[ty]]
                kw = dict(W=10500, xlab="±10 kb, 1 kb bins")
                if ty == "imputed" and ver == "K_M":
                    kw["mask"] = z["K_maskM"]
            elif ty == "imputed" and base == "V3":
                X = z["K_impM"]; kw = dict(W=10500, mask=z["K_maskM"], xlab="±10 kb, 1 kb bins")
            else:
                X = z[f"{base}_{ty}"]
                if ty != "imputed":
                    kw = dict(xlab="±1 kb, 50 bp bins" if not (ver == "V3_smoothed" and ty == "raw") else "±1 kb, 200 bp bins")
            n = len(X)
            if ver == "V3_smoothed" and ty == "raw":
                X = smooth_raw(z["V3_raw50"])
            panel(fig, sub[j], X, TTL[ty], UNIT[ty], profile=profile, nsites=n, **kw)
        pos = gs[r, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + (0.02 if nrow > 1 else 0.05), f"{t} ({CELL[t]} ChIP)", ha="center",
                 fontsize=7, fontweight="bold")
    fig.text(0.5, 0.995, SUB[ver] if ncol > 1 else SUB[ver].replace("; ", "\n", 1), ha="center", va="top", fontsize=6)
    style.save(fig, out, name, raster_dpi=600)


def main(src, out):
    """Main: K_M (all panels on the 1 kb grid, +-10 kb, imputed under MASK, profile row), overview + per TF.
    Supplement: V3 (ChIP/raw at 50 bp, +-1 kb; the CTCF footprint) and K_B (genome-wide pre-mask imputed)."""
    style.use()
    build(src, out, "K_M", TFS, "fig2_main_heatmap", 183, 245, 3, profile=True)
    for t in TFS:
        build(src, out, "K_M", [t], f"fig2_main_heatmap_{t}", 89, 125, 1, profile=True)
    build(src, out, "V3", TFS, "fig2_supp_heatmap_50bp", 183, 245, 3, profile=True)
    build(src, out, "K_B", TFS, "fig2_supp_heatmap_premask", 183, 245, 3, profile=True)
