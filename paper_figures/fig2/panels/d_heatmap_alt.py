"""Fig 2 heatmap alternatives (not the main panel), same rows / order / row grouping / smoothing as the main v2:
  fig2_main_heatmap_v3_rownorm      : ChIP and raw as v2; imputed row-normalised = (x + pc) / (flank + pc), flank = mean
                                      of the 1 kb bins 2-5 kb away on both sides (from the +-10 kb 1 kb arrays),
                                      pc = 5th percentile of the nonzero per-site flank means; ratios averaged in the
                                      row groups, shown as log2 on a diverging map, colour symmetric at the 95th
                                      percentile of |log2| per panel (rounded up to 0.5) (chosen over a turbo
                                      ratio because enrichment and depletion read symmetrically around 0).
  fig2_main_heatmap_v3_rownorm_all  : the same row normalisation applied to all three panels.
  fig2_main_heatmap_v3_imp5kb       : ChIP and raw as v2 (+-1 kb, 50 bp); imputed on its own +-5 kb axis at 1 kb (11 bins).
Also writes source_data/imputed_noise.tsv: per TF, the median number of positive cells in the summit bin and the
median binomial coefficient of variation of the per-site fraction, sqrt((1 - p) / (n p)), n = cells imputed."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

_s = importlib.util.spec_from_file_location("v2", Path(__file__).resolve().parent / "c_heatmap_v2.py")
V2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(V2)
TFS, CELL, XS, W, STEP = V2.TFS, V2.CELL, V2.XS, V2.W, V2.STEP
DIV = LinearSegmentedColormap.from_list("div", ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"])
FLK = np.r_[5:9, 12:16]   # 1 kb bins 2-5 kb from the summit bin (index 10)


def flanks(z, keep):
    """per-site flank means in the units of the 50 bp tracks."""
    f = {"ChIP": z["K_ChIP"][keep][:, FLK].astype(float).mean(1),            # bigWig mean CPM per base, same as 50 bp
         "raw": z["K_raw"][keep][:, FLK].astype(float).mean(1) / (1000 / STEP),   # CPM per kb -> per 50 bp bin
         "imputed": np.where(z["K_maskM"][keep][:, FLK], z["K_impM"][keep][:, FLK], 0).astype(float).mean(1)}
    return f


def rownorm(X, fl):
    nz = fl[fl > 0]; pc = float(np.percentile(nz, 5)) if nz.size else 1e-9
    return (X + pc) / (fl[:, None] + pc)


def panel_div(fig, cell, R, title, n):
    import matplotlib.pyplot as plt
    g = cell.subgridspec(4, 1, height_ratios=[3.0, 22, 0.9, 0.5], hspace=0.06)
    axp = fig.add_subplot(g[0]); pr = np.log2(R.mean(0))
    axp.plot(XS, pr, color="#1A1A1A", lw=0.7); axp.axhline(0, color="#BDBDBD", lw=0.4); axp.set_xlim(-W, W); axp.set_xticks([])
    axp.yaxis.set_major_locator(plt.MaxNLocator(3)); axp.tick_params(axis="y", labelsize=5, length=1.5, pad=1)
    axp.set_title(title, fontsize=6, pad=2)
    ax = fig.add_subplot(g[1])
    b = max(1, int(np.ceil(len(R) / V2.TARGET_ROWS))); m = len(R) // b * b
    Y = R[:m].reshape(-1, b, R.shape[1]).mean(1)
    if m < len(R):
        Y = np.vstack([Y, R[m:].mean(0, keepdims=True)])
    Y = np.log2(gaussian_filter1d(Y, sigma=V2.SIGMA_BP / STEP, axis=1, mode="nearest"))
    lim = max(0.5, float(np.ceil(np.percentile(np.abs(Y), 95) * 2) / 2))   # symmetric, 95th pct of |log2|, 0.5 steps
    im = ax.imshow(np.clip(Y, -lim, lim), aspect="auto", cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest",
                   extent=[-W, W, len(R), 0], rasterized=True)
    ax.set_yticks([]); ax.set_xticks([-W, 0, W], ["−1 kb", "0", "+1 kb"])
    for lb, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
        lb.set_ha(ha)
    ax.tick_params(axis="x", pad=1, length=1.5, labelsize=5)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(True)
    cax = fig.add_subplot(g[3]); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels([f"{-lim:g}", "0", f"{lim:g}"]); cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
    cb.set_label(f"log2 / flank\n{n:,} sites", fontsize=5, labelpad=1)


def panel_imp5kb(fig, cell, X, title, n):
    import matplotlib.pyplot as plt
    g = cell.subgridspec(4, 1, height_ratios=[3.0, 22, 0.9, 0.5], hspace=0.06)
    xs = np.arange(-5, 6) * 1000
    axp = fig.add_subplot(g[0]); pr = X.mean(0)
    axp.plot(xs, pr, color="#1A1A1A", lw=0.7); axp.set_xlim(-5500, 5500); axp.set_ylim(0, pr.max() * 1.15 or 1); axp.set_xticks([])
    axp.yaxis.set_major_locator(plt.MaxNLocator(2)); axp.tick_params(axis="y", labelsize=5, length=1.5, pad=1)
    axp.set_title(title, fontsize=6, pad=2)
    ax = fig.add_subplot(g[1])
    b = max(1, int(np.ceil(len(X) / V2.TARGET_ROWS))); m = len(X) // b * b
    Y = X[:m].reshape(-1, b, X.shape[1]).mean(1)
    if m < len(X):
        Y = np.vstack([Y, X[m:].mean(0, keepdims=True)])
    vm = float(np.percentile(Y, V2.CAP_PCT)) or 1.0
    im = ax.imshow(Y, aspect="auto", cmap="turbo", vmin=0, vmax=vm, interpolation="nearest", extent=[-5500, 5500, len(X), 0], rasterized=True)
    ax.set_yticks([]); ax.set_xticks([-5000, 0, 5000], ["−5 kb", "0", "+5 kb"])
    for lb, ha in zip(ax.get_xticklabels(), ("left", "center", "right")):
        lb.set_ha(ha)
    ax.tick_params(axis="x", pad=1, length=1.5, labelsize=5)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(True)
    cax = fig.add_subplot(g[3]); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
    cb.set_label(f"frac. cells (1 kb bins)\n{n:,} sites", fontsize=5, labelpad=1)


def build(src, out, name, mode):
    fig = style.figure_mm(183, 235)
    gs = fig.add_gridspec(2, 3, wspace=0.14, hspace=0.14, left=0.03, right=0.985, top=0.95, bottom=0.03)
    for k, t in enumerate(TFS):
        z = np.load(Path(src) / f"heatmaps_{t}.npz")
        T, keep = V2.tracks(z)
        fl = flanks(z, keep)
        r, c = divmod(k, 3)
        sub = gs[r, c].subgridspec(1, 3, wspace=0.32)
        for j, (ty, ttl, un) in enumerate((("ChIP", "Bulk ChIP", "CPM"), ("raw", "sc raw", "CPM"), ("imputed", "sc imputed", "frac. cells"))):
            n = len(T[ty])
            if mode == "rownorm_all" or (mode == "rownorm" and ty == "imputed"):
                panel_div(fig, sub[j], rownorm(T[ty], fl[ty]), ttl, n)
            elif mode == "imp5kb" and ty == "imputed":
                X = np.where(z["K_maskM"][keep][:, 5:16], z["K_impM"][keep][:, 5:16], 0).astype(np.float32)
                panel_imp5kb(fig, sub[j], X, "sc imputed (±5 kb, 1 kb bins)", n)
            else:
                V2.panel(fig, sub[j], T[ty], ttl, un, n)
        pos = gs[r, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.018, f"{t} ({CELL[t]} ChIP)", ha="center", fontsize=7, fontweight="bold")
    style.save(fig, out, name, raster_dpi=600)


def noise_table(src):
    sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
    from fig2_common import INP  # noqa: E402
    fr = np.load(INP / "frame.npz", allow_pickle=True); tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
    rows = []
    for t in TFS:
        z = np.load(Path(src) / f"heatmaps_{t}.npz"); keep = z["K_maskM"][:, 10].astype(bool)
        p = z["K_impM"][keep][:, 10].astype(float); n = ncell[tf79.index(t)]
        k = p * n
        cv = np.sqrt((1 - p[p > 0]) / (n * p[p > 0]))
        rows.append(dict(TF=t, cells_imputed=int(n), median_positive_cells_at_summit_bin=float(np.median(k)),
                         share_sites_zero=float((k == 0).mean()), median_binomial_CV_per_site=float(np.median(cv))))
    D = pd.DataFrame(rows); D.to_csv(Path(src) / "imputed_noise.tsv", sep="\t", index=False, float_format="%.4f")
    print(D.round(3).to_string())


def main(src, out):
    style.use()
    build(src, out, "fig2_main_heatmap_v3_rownorm", "rownorm")
    build(src, out, "fig2_main_heatmap_v3_rownorm_all", "rownorm_all")
    build(src, out, "fig2_main_heatmap_v3_imp5kb", "imp5kb")
    noise_table(src)
