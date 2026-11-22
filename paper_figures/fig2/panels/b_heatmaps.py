"""Fig 2 peak heatmaps in the style of Kaya-Okur et al. 2019 (Nat Commun, Fig 2): per TF, Bulk ChIP | sc raw |
sc imputed, +-1 kb at 50 bp bins, one colour scale per panel (0 to the 98th percentile of nonzero values, turbo),
colour bar with unit and number of sites underneath, rows sorted top to bottom by centre signal (+-200 bp).
  V3 (main): the reference's top 10,000 ChIP summits, the same rows in every panel, sorted by ChIP.
  V2 (supplement): ChIP summits + our raw peaks > 500 bp from any ChIP peak, same rows, sorted by ChIP.
  V1 (supplement): each dataset on its own peaks, sorted by its own signal.
Raw = aggregated 5' ends, CPM, 3-bin running mean; imputed = fraction of cells positive in the 1 kb bin covering each
position (imputation is at 1 kb). Heatmap bodies are rasterised. Input source_data/heatmaps_<TF>.npz."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

TFS = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
TTL = {"ChIP": "Bulk ChIP", "raw": "sc raw", "imputed": "sc imputed"}
UNIT = {"ChIP": "CPM", "raw": "CPM", "imputed": "frac. cells"}
SUB = {"V3": "top 10,000 ChIP summits; same rows in every panel, sorted by ChIP",
       "V2": "ChIP summits + our peaks; same rows in every panel, sorted by ChIP",
       "V1": "each panel on its own peaks, sorted by its own signal"}
W = 1000


def panel(fig, cell, X, title, unit):
    g = cell.subgridspec(3, 1, height_ratios=[22, 0.9, 0.5], hspace=0.04)
    ax = fig.add_subplot(g[0])
    X = X.astype(np.float32)
    nz = X[X > 0]
    vm = float(np.percentile(nz, 98)) if nz.size else 1.0
    im = ax.imshow(X, aspect="auto", cmap="turbo", vmin=0, vmax=vm, interpolation="nearest",
                   extent=[-W, W, len(X), 0], rasterized=True)
    ax.set_yticks([]); ax.set_xticks([-W, 0, W], ["−1", "0", "+1"])
    for lb, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
        lb.set_ha(ha)
    ax.tick_params(axis="x", pad=1, length=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.set_title(title, fontsize=6, pad=2)
    cax = fig.add_subplot(g[2])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
    for lb, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        lb.set_ha(ha)
    cb.set_label(f"{unit}\n{len(X):,} sites", fontsize=5, labelpad=1)


def build(src, out, ver, tfs, name, w_mm, h_mm, ncol):
    fig = style.figure_mm(w_mm, h_mm)
    nrow = int(np.ceil(len(tfs) / ncol))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.12, hspace=0.16, left=0.015, right=0.985, top=0.93 if nrow > 1 else 0.86,
                          bottom=0.04 if nrow > 1 else 0.07)
    for k, t in enumerate(tfs):
        z = np.load(Path(src) / f"heatmaps_{t}.npz")
        r, c = divmod(k, ncol)
        sub = gs[r, c].subgridspec(1, 3, wspace=0.14)
        for j, ty in enumerate(("ChIP", "raw", "imputed")):
            panel(fig, sub[j], z[f"{ver}_{ty}"], TTL[ty], UNIT[ty])
        pos = gs[r, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + (0.02 if nrow > 1 else 0.05), f"{t} ({CELL[t]} ChIP)", ha="center",
                 fontsize=7, fontweight="bold")
    fig.text(0.5, 0.995, f"{SUB[ver]}  ·  distance from centre (kb), 50 bp bins", ha="center", va="top", fontsize=6)
    style.save(fig, out, name, raster_dpi=600)


def main(src, out):
    style.use()
    for ver in ("V3", "V2", "V1"):
        build(src, out, ver, TFS, f"fig2_heatmaps_{ver}_all", 183, 230, 3)
        for t in TFS:
            build(src, out, ver, [t], f"fig2_heatmaps_{ver}_{t}", 89, 115, 1)
