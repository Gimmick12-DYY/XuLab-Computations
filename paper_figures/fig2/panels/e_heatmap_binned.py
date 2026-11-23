"""Fig 2 heatmap alternatives, rank-grouped (not the main panel). Rows = the main heatmap's sites (top ChIP summits whose
summit bin is inside Mdrop5), sorted by ChIP centre signal (+-200 bp) and split into 50 equal groups by ChIP rank; each
drawn row = the mean over its group (~120-190 sites). Colour per panel 0 to the 95th percentile of the drawn rows
(turbo); profile = mean over all sites; colour bar + n sites under each panel.
  fig2_main_heatmap_v4_rankgroups : ChIP (50 bp, +-1 kb, no smoothing), raw (50 bp, +-1 kb, 3-bin running mean from the
      source plus Gaussian sigma 50 bp), imputed under Mdrop5 on its own +-5 kb axis at 1 kb (11 bins).
  fig2_main_heatmap_v4_continuous : the same, but imputed = the model's continuous pseudobulk score (mean predicted
      probability per 1 kb bin over cells) at the summit bin +-2 kb (5 bins), available only for the audit bins, so all
      three panels use only the sites covered by those bins (n shown under each panel). The model score has a high
      baseline (~0.15-0.25), so this panel's colour runs from its 5th to its 95th percentile (stated on the colour bar)."""
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

_s = importlib.util.spec_from_file_location("v2", Path(__file__).resolve().parent / "c_heatmap_v2.py")
V2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(V2)
TFS, CELL, XS, W, STEP = V2.TFS, V2.CELL, V2.XS, V2.W, V2.STEP
G = 50


def groups(X):
    return np.vstack([g.mean(0) for g in np.array_split(X, G)])


def panel(fig, cell, X, title, unit, n, xs, half, xlab, sigma_bins=0, vmin_pct=None):
    g = cell.subgridspec(4, 1, height_ratios=[3.0, 22, 0.9, 0.5], hspace=0.06)
    axp = fig.add_subplot(g[0]); pr = np.nanmean(X, 0)
    axp.plot(xs, pr, color="#1A1A1A", lw=0.7); axp.set_xlim(-half, half); axp.set_ylim(0, np.nanmax(pr) * 1.15 or 1)
    axp.set_xticks([]); axp.yaxis.set_major_locator(plt.MaxNLocator(2)); axp.tick_params(axis="y", labelsize=5, length=1.5, pad=1)
    axp.set_title(title, fontsize=6, pad=2)
    ax = fig.add_subplot(g[1])
    Y = groups(X)
    if sigma_bins:
        Y = gaussian_filter1d(Y, sigma=sigma_bins, axis=1, mode="nearest")
    vm = float(np.nanpercentile(Y, 95)) or 1.0
    v0 = float(np.nanpercentile(Y, vmin_pct)) if vmin_pct is not None else 0.0
    im = ax.imshow(Y, aspect="auto", cmap="turbo", vmin=v0, vmax=vm, interpolation="nearest",
                   extent=[-half, half, len(X), 0], rasterized=True)
    kb = int(round(xs.max() / 1000)) if xs.max() >= 1000 else 1
    ax.set_yticks([]); ax.set_xticks([-kb * 1000, 0, kb * 1000], [f"−{kb} kb", "0", f"+{kb} kb"])
    for lb, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
        lb.set_ha(ha)
    ax.tick_params(axis="x", pad=1, length=1.5, labelsize=5)
    if xlab:
        ax.set_xlabel(xlab, fontsize=5, labelpad=0)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(True)
    cax = fig.add_subplot(g[3]); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([v0, vm]); cb.set_ticklabels([f"{v0:.2g}" if v0 else "0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
    for lb, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        lb.set_ha(ha)
    cb.set_label(f"{unit}\n{n:,} sites", fontsize=5, labelpad=1)


def build(src, out, name, mode, tfs=TFS, w=183, h=235, ncol=3):
    fig = style.figure_mm(w, h)
    nrow = int(np.ceil(len(tfs) / ncol))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.14, hspace=0.16, left=0.03 if ncol > 1 else 0.08, right=0.985,
                          top=0.95 if nrow > 1 else 0.90, bottom=0.04 if nrow > 1 else 0.06)
    for k, t in enumerate(tfs):
        z = np.load(Path(src) / f"heatmaps_{t}.npz")
        T, keep = V2.tracks(z)
        if mode == "continuous":
            imp = z["K_cont"][keep].astype(np.float32); ok = z["K_cont_ok"][keep].astype(bool)
            T = {k_: v[ok] for k_, v in T.items()}; imp = imp[ok]
            ixs, ih, il, iu, it = np.arange(-2, 3) * 1000, 2500, "±2 kb, 1 kb bins", "mean predicted prob.", "sc imputed (model score)"
        else:
            imp = np.where(z["K_maskM"][keep][:, 5:16], z["K_impM"][keep][:, 5:16], 0).astype(np.float32)
            ixs, ih, il, iu, it = np.arange(-5, 6) * 1000, 5500, "±5 kb, 1 kb bins", "frac. cells", "sc imputed"
        n = len(T["ChIP"])
        r, c = divmod(k, ncol)
        sub = gs[r, c].subgridspec(1, 3, wspace=0.32)
        panel(fig, sub[0], T["ChIP"], "Bulk ChIP", "CPM", n, XS, W, "±1 kb, 50 bp")
        panel(fig, sub[1], T["raw"], "sc raw", "CPM", n, XS, W, "±1 kb, 50 bp", sigma_bins=1)
        panel(fig, sub[2], imp, it, iu, n, ixs, ih, il, vmin_pct=5 if mode == "continuous" else None)
        pos = gs[r, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + (0.02 if nrow > 1 else 0.05), f"{t} ({CELL[t]} ChIP)", ha="center", fontsize=7, fontweight="bold")
    style.save(fig, out, name, raster_dpi=600)


def main(src, out):
    style.use()
    build(src, out, "fig2_main_heatmap_v4_rankgroups", "rank")
    build(src, out, "fig2_main_heatmap_v4_continuous", "continuous")
