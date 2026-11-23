"""Fig 2 heatmap at ChIP-differential peaks (round 11; source downstream/fig2_qc/r11_f_corr.py, spec
round11_chipdeseq2/SPEC.md 7). Rows = each ChIP's DESeq2-differential regions (top 5,000 by LFC), one block per ChIP,
sorted by that ChIP's signal, averaged in groups of 25. Panels: ChIP (block's own ChIP, +-1 kb, scaled to the block's
99th percentile) | our raw, 6 TFs (+-1 kb, 50 bp; log2 of the TF's 5' end CPM over the 79-TF pool's) | imputed, 6 TFs
(Mdrop5, 1 kb bins +-5 kb; log2 of the TF's fraction of cells over the pool mean; grey = outside the mask).
Output fig2_heatmap_chipdeseq2."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

DIV = LinearSegmentedColormap.from_list("div", ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", "#1A1A1A"])


def main(src, out):
    f = Path(src) / "heatmap_chipdeseq2.npz"
    if not f.exists():
        return
    style.use()
    z = np.load(f)
    tfs = [str(t) for t in z["tfs"]]; gsz = [int(x) for x in z["block_rows"]]; n = [int(x) for x in z["block_n"]]
    raw, imp, chip = z["raw"], z["imp"], z["chip"]
    lr = float(np.nanpercentile(np.abs(raw), 95)); li = float(np.nanpercentile(np.abs(imp), 95))
    DIVg = DIV.copy(); DIVg.set_bad("#BDBDBD")
    fig = style.figure_mm(180, 150)
    gs = fig.add_gridspec(1, 13, wspace=0.08, left=0.10, right=0.99, top=0.86, bottom=0.16, width_ratios=[1.2] + [1] * 12)
    cols = [("chip", None)] + [("raw", i) for i in range(6)] + [("imp", i) for i in range(6)]
    for j, (kind, i) in enumerate(cols):
        ax = fig.add_subplot(gs[j])
        if kind == "chip":
            im = ax.imshow(np.clip(chip, 0, 1), aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest", rasterized=True)
            ax.set_title("ChIP\n(block's own)", fontsize=6, pad=3)
        elif kind == "raw":
            im = ax.imshow(np.clip(raw[i], -lr, lr), aspect="auto", cmap=DIV, vmin=-lr, vmax=lr, interpolation="nearest", rasterized=True)
            ax.set_title(f"{tfs[i]}\nraw", fontsize=6, pad=3)
        else:
            im = ax.imshow(np.ma.masked_invalid(np.clip(imp[i], -li, li)), aspect="auto", cmap=DIVg, vmin=-li, vmax=li,
                           interpolation="nearest", rasterized=True)
            ax.set_title(f"{tfs[i]}\nimputed", fontsize=6, pad=3)
        y = 0
        for b, k in enumerate(gsz):
            if y:
                ax.axhline(y - 0.5, color="#E8821E", lw=0.6)
            if j == 0:
                ax.text(-0.15, 1 - (y + k / 2) / sum(gsz), f"{tfs[b]}\ndifferential\n({n[b]:,})", transform=ax.transAxes,
                        ha="right", va="center", fontsize=5.5)
            y += k
        w = imp.shape[2] if kind == "imp" else raw.shape[2]
        ax.set_yticks([])
        ax.set_xticks([0, (w - 1) / 2, w - 1], ["-5", "0", "+5 kb"] if kind == "imp" else ["-1 kb", "0", "+1 kb"], fontsize=4.5)
        ax.tick_params(axis="x", length=1, pad=1)
        if j in (0, 1, 7):
            cax = ax.inset_axes([0.0, -0.09, 1.0 if j == 0 else 3.0, 0.015])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.3)
            cb.ax.tick_params(labelsize=4.5, length=1, pad=1)
            cb.set_label({0: "ChIP (block 99th pct = 1)", 1: "raw: log2 TF / 79-TF pool",
                          7: "imputed: log2 TF / pool mean"}[j], fontsize=5, labelpad=1)
    fig.text(0.55, 0.97, "Our data at ChIP-differential peaks (DESeq2 ~ChIP, padj < 0.05, LFC > 1); rows averaged in groups of 25",
             ha="center", va="top", fontsize=6.5)
    style.save(fig, out, "fig2_heatmap_chipdeseq2", raster_dpi=600)
