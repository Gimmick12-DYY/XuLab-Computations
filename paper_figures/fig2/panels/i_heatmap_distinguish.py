"""Fig 2 'can our TFs be told apart' heatmap (round 11 follow-up; source downstream/fig2_qc/r11g_distinguish.py).
Rows = ChIP-differential sites (DESeq2 ~ChIP), one block per ChIP, ordered by ChIP LFC, 50 groups per block.
Columns: block's own ChIP (log2 CPM, grey) | our 6 TFs raw (summit-group value of the 6x6, ZNF777 depth) | our 6 TFs
imputed (Mdrop5); each of our columns z-scored across rows. Boxes = expected own-block cells. Below: mean column z per
(our TF, ChIP block), raw and imputed, with own-block-is-max count and the 720-permutation p. Output
fig2_heatmap_chipdeseq2_distinguish."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

DIV = LinearSegmentedColormap.from_list("div", ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", "#1A1A1A"])
V = 2.0


def main(src, out):
    f = Path(src) / "heatmap_chipdeseq2_distinguish.npz"
    if not f.exists():
        return
    P = style.use()
    z = np.load(f); tfs = [str(t) for t in z["tfs"]]; gb = z["gblk"]; n = z["block_n"]; stt = z["stats"]
    fig = style.figure_mm(150, 210)
    hx, hy, hw, hh = 0.17, 0.42, 0.78, 0.50
    ax0 = fig.add_axes([hx, hy, 0.055, hh])
    ch = z["chip"].copy()
    for b in range(6):   # scale the ChIP column within each block (different ChIPs)
        m = gb == b; ch[m] = (ch[m] - ch[m].min()) / max(np.ptp(ch[m]), 1e-9)
    ax0.imshow(ch[:, None], aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax0.set_xticks([0], ["own\nChIP"], fontsize=5.5); ax0.set_yticks([])
    nr = len(gb); edges = np.r_[0, np.cumsum([int((gb == b).sum()) for b in range(6)])]
    for b in range(6):
        ax0.text(-0.9, (edges[b] + edges[b + 1]) / 2 - 0.5, f"{tfs[b]}\nChIP-differential\n({int(n[b]):,} sites)",
                 ha="right", va="center", fontsize=5.5)
    for k, (key, lab) in enumerate((("rawz", "raw"), ("impz", "imputed"))):
        ax = fig.add_axes([hx + 0.075 + k * 0.36, hy, 0.33, hh])
        im = ax.imshow(np.clip(z[key], -V, V), aspect="auto", cmap=DIV, vmin=-V, vmax=V, interpolation="nearest")
        for b in range(6):
            ax.add_patch(Rectangle((b - .5, edges[b] - .5), 1, edges[b + 1] - edges[b], fill=False, ec=P.neutral.ink, lw=0.9))
            if b:
                ax.axhline(edges[b] - .5, color=P.neutral.ink, lw=0.3)
        ax.set_xticks(range(6), tfs, rotation=90, fontsize=5.5); ax.set_yticks([]); ax.xaxis.tick_top()
        ax.set_title(f"our {lab} (column z across rows)", fontsize=6, pad=22)
        for a in (ax0,):
            for b in range(1, 6):
                a.axhline(edges[b] - .5, color=P.neutral.ink, lw=0.3)
    cax = fig.add_axes([hx + 0.075, hy - 0.03, 0.69, 0.008]); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5); cb.set_label("column z-score (each of our TFs across all rows; clipped ±2)", fontsize=5.5)
    for k, (key, lab) in enumerate((("M_raw", "raw"), ("M_imp", "imputed"))):
        M = z[key]; ax = fig.add_axes([0.13 + k * 0.45, 0.06, 0.30, 0.23])
        lim = max(0.5, float(np.ceil(np.abs(M).max() * 10) / 10))
        ax.imshow(np.clip(M, -lim, lim), cmap=DIV, vmin=-lim, vmax=lim)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=4.8, color="white" if abs(M[i, j]) > .75 * lim else P.neutral.ink)
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
        ax.set_xticks(range(6), [f"{t} block" for t in tfs], rotation=90, fontsize=5); ax.set_yticks(range(6), [f"{t} (ours)" for t in tfs], fontsize=5)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        d, p, p5, own = stt[k]
        ax.set_title(f"{lab}: mean column z per block (±{lim:g})\nown block highest {int(own)}/6; diag−off {d:+.2f}, p = {p:.3f} (no CTCF {p5:.3f})", fontsize=5.5)
    fig.text(0.5, 0.995, "Can our TFs be told apart at ChIP-differential peaks? (DESeq2 ~ChIP; 50 site groups per block)", ha="center", va="top", fontsize=6.5)
    style.save(fig, out, "fig2_heatmap_chipdeseq2_distinguish")
