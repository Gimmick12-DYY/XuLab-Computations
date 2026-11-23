"""Fig 2 peak-presence heatmap (round 10; source downstream/fig2_qc/r10_presence.py, spec round10_presence/SPEC.md).
Rows = union of 4 bulk ChIPs' summit +-250 bp regions minus those present in all 4, grouped by presence pattern
(single-ChIP blocks, then pairs, then triples), ordered within a block by summed ChIP signal. Columns:
  Bulk ChIP presence (4) | our peak presence (A, lab MACS2 q<0.05) | our signal (B, log2((CPM_tf + 1) / (CPM_pool + 1)) over the leave-one-out
  79-TF pool) | imputed presence (C, Mdrop5, >= 95th percentile)
Rows averaged in consecutive groups within each block (group size = ceil(total / 600); at least one drawn row per block),
so presence columns show the fraction present (white 0 to black 1); (B) diverging, centred at 0, capped at the 95th
percentile of |value|. Outputs fig2_presence_heatmap_4ChIP (main set) and fig2_presence_heatmap_4ChIP_ZNF777variant."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import style  # noqa: E402

DIV = LinearSegmentedColormap.from_list("div", ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B"])
GREY = LinearSegmentedColormap.from_list("bw", ["#FFFFFF", "#1A1A1A"])
TARGET = 600


def grouped(X, sizes, g):
    out, edges, y = [], [], 0
    for n in sizes:
        B = X[y:y + n].astype(float); y += n
        k = max(1, int(np.ceil(n / g)))
        out.append(np.vstack([b.mean(0) for b in np.array_split(B, k)]))
        edges.append(k)
    return np.vstack(out), edges


def build(src, out, setname, name):
    P = style.use()
    z = np.load(Path(src) / f"presence_{setname}.npz", allow_pickle=True)
    tfs = [str(t) for t in z["tfs"]]; names = [str(b) for b in z["block_names"]]; sizes = [int(n) for n in z["block_n"]]
    g = int(np.ceil(sum(sizes) / TARGET))
    panels = [("chip", "Bulk ChIP\npeak present", GREY, 0, 1), ("labp", "Our peak\npresent", GREY, 0, 1),
              ("enr", "Our signal\nlog2 vs 79-TF pool", DIV, None, None), ("impp", "Imputed\npresent (≥95th pct)", GREY, 0, 1)]
    fig = style.figure_mm(120, 170)
    gs = fig.add_gridspec(1, 4, wspace=0.12, left=0.24, right=0.99, top=0.90, bottom=0.10)
    for j, (key, ttl, cm, v0, v1) in enumerate(panels):
        Y, edges = grouped(z[key], sizes, g)
        if key == "enr":
            lim = float(np.percentile(np.abs(Y), 95)); v0, v1 = -lim, lim
        ax = fig.add_subplot(gs[j])
        im = ax.imshow(np.clip(Y, v0, v1), aspect="auto", cmap=cm, vmin=v0, vmax=v1, interpolation="nearest", rasterized=True)
        y = 0
        for k_, (nm, n) in enumerate(zip(names, sizes)):
            if y:
                ax.axhline(y - 0.5, color="#E8821E" if key != "enr" else "#1A1A1A", lw=0.5)
            if j == 0:
                ax.text(-0.9, y + edges[k_] / 2 - 0.5, f"{nm} ({n:,})", ha="right", va="center", fontsize=5)
            y += edges[k_]
        ax.set_yticks([]); ax.set_xticks(range(4), tfs, rotation=90, fontsize=6); ax.tick_params(axis="x", length=0, pad=1)
        ax.set_title(ttl, fontsize=6, pad=3)
        cax = ax.inset_axes([0.05, -0.16 if j else -0.16, 0.9, 0.018])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.3)
        cb.ax.tick_params(labelsize=5, length=1.2, pad=1)
        cb.set_ticks([v0, 0, v1] if key == "enr" else [0, 1])
        cb.set_ticklabels([f"{v0:.1f}", "0", f"{v1:.1f}"] if key == "enr" else ["0", "1"])
        cb.set_label("log2 enrichment" if key == "enr" else "fraction present", fontsize=5, labelpad=1)
    fig.text(0.6, 0.985, f"{sum(sizes):,} regions; rows averaged in groups of {g}", ha="center", va="top", fontsize=6)
    style.save(fig, out, name, raster_dpi=600)


def main(src, out):
    for setname, name in (("main", "fig2_presence_heatmap_4ChIP"), ("znf777", "fig2_presence_heatmap_4ChIP_ZNF777variant")):
        if (Path(src) / f"presence_{setname}.npz").exists():
            build(src, out, setname, name)
