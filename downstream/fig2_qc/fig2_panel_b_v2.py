#!/usr/bin/env python
"""Panel b v2, declared recipe (as the Paired-Tag reference panel b): regions = union of reference peaks, 500 bp around
summits, merged; values = log2(CPM + 1) of reads per region (sc raw = aggregated 5' ends at FULL depth; bulk = ChIP reads);
Pearson across regions; average-linkage clustering (1 - r).
  4x4: CTCF HEK293T ChIP-nexus (GSE103651) + ZNF282 HEK293T ChIP-exo (GSM2466515), new counts here.
  12x12: all 6 references (same recipe; counts already computed by 09b_Pb_versions.py, Pb3_corr_V1_plain_raw.tsv;
         CTCF bulk there is the ENCODE HEK293 BAM, so it is labelled HEK293).
Outputs OUT/fig2_draft/panel_b_v2_{4x4,12x12}.{png,pdf,svg,tsv}."""
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from fig2_common import D, INP, OUT, ROOT, read_bed

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
OD = OUT / "fig2_draft"
tmp = OD / "tmp_b2"
tmp.mkdir(parents=True, exist_ok=True)
CH = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


# ---- 4x4 counts ----
PK = {"CTCF": D / "CTCF_HEK293T_GSE103651_peaks.bed", "ZNF282": D / "ZNF282_HEK293_peaks.bed"}
BAM = {"CTCF": ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam", "ZNF282": D / "ZNF282_HEK293.bam"}
parts = []
for t, p in PK.items():
    pk = read_bed(p, ncol=10)
    pk = pk[pk[0].isin(CH)]
    mid = pk[1] + pk[9].astype(int)
    parts.append(pd.DataFrame({0: pk[0], 1: np.maximum(mid - 250, 0), 2: mid + 250}))
pd.concat(parts).sort_values([0, 1]).to_csv(tmp / "s.bed", sep="\t", header=False, index=False)
(tmp / "r.bed").write_text(sh("bedtools", "merge", "-i", str(tmp / "s.bed")))
reg = read_bed(tmp / "r.bed").reset_index(drop=True)
out = sh("bedtools", "multicov", "-bams", str(BAM["CTCF"]), str(BAM["ZNF282"]), "-bed", str(tmp / "r.bed"))
B = np.array([[float(x) for x in l.split("\t")[3:]] for l in out.splitlines()])
S = np.zeros((len(reg), 2))
for k, t in enumerate(["CTCF", "ZNF282"]):
    z = np.load(INP / "five" / f"{t}.npz")
    for c in CH:
        g = np.flatnonzero(reg[0].values == c)
        if len(g):
            S[g, k] = np.searchsorted(z[c], reg[2].values[g]) - np.searchsorted(z[c], reg[1].values[g])
X = np.column_stack([B, S])
L = np.log2(X / X.sum(0, keepdims=True) * 1e6 + 1)
R4 = np.corrcoef(L.T)
labs4 = ["CTCF ChIP (HEK293T)", "ZNF282 ChIP (HEK293T)", "CTCF sc", "ZNF282 sc"]
pd.DataFrame(R4, labs4, labs4).to_csv(OD / "panel_b_v2_4x4.tsv", sep="\t", float_format="%.4f")
print("regions", len(reg)); print(pd.DataFrame(R4, labs4, labs4).round(3).to_string(), flush=True)

# ---- 12x12 (existing counts, same recipe) ----
R12df = pd.read_csv(OUT / "panels/Pb3_corr_V1_plain_raw.tsv", sep="\t", index_col=0)
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
labs12 = [(f"{l.split('_')[1]} ChIP ({CELL[l.split('_')[1]]})" if l.startswith("bulk") else f"{l.split('_')[1]} sc") for l in R12df.index]
R12 = R12df.values
pd.DataFrame(R12, labs12, labs12).to_csv(OD / "panel_b_v2_12x12.tsv", sep="\t", float_format="%.4f")
print(pd.DataFrame(R12, labs12, labs12).round(2).to_string(), flush=True)


def report(R, labs):
    k = len(labs) // 2
    Z = linkage(squareform(np.clip(1 - R, 0, None), checks=False), "average")
    from scipy.cluster.hierarchy import to_tree
    root = to_tree(Z)
    for i in range(k):
        blk = R[k + i, :k]
        want, node = {i, k + i}, root
        while True:
            for ch in (node.left, node.right):
                if ch is not None and want <= set(ch.pre_order()):
                    node = ch
                    break
            else:
                break
        nxt = sum(x < k for x in node.pre_order()) == 1
        print(f"{labs[k + i]}: own r {blk[i]:.3f}, best other {np.delete(blk, i).max():.3f} -> own>other {blk[i] > np.delete(blk, i).max()}; clusters with own ChIP {nxt}", flush=True)
    return Z


def draw(R, labs, Z, name, size):
    n = len(labs)
    fig = plt.figure(figsize=size)
    x0, y0, w = 0.2, 0.1, 0.52
    axt = fig.add_axes([x0, y0 + w, w, 0.13]); axl = fig.add_axes([x0 - 0.14, y0, 0.13, w])
    dt = dendrogram(Z, ax=axt, no_labels=True, color_threshold=0, above_threshold_color=P.neutral.ink)
    dendrogram(Z, ax=axl, orientation="left", no_labels=True, color_threshold=0, above_threshold_color=P.neutral.ink)
    axl.invert_yaxis()
    for a in (axt, axl):
        a.axis("off")
        for l in a.collections:
            l.set_linewidth(0.6)
    o = dt["leaves"]
    lo = np.floor(R.min() * 10) / 10
    ax = fig.add_axes([x0, y0, w, w])
    im = ax.imshow(R[np.ix_(o, o)], cmap=plt.get_cmap("RdYlBu_r"), vmin=max(lo, -1), vmax=1, aspect="auto")
    nm = [labs[i] for i in o]
    ax.set_xticks(range(n), nm, rotation=90)
    ax.yaxis.tick_right(); ax.set_yticks(range(n), nm)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        if t.get_text().endswith(" sc"):
            t.set_fontweight("bold")
    ax.tick_params(length=0, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([x0 + w + (0.30 if n == 4 else 0.33), y0 + 0.05, 0.025, w - 0.05])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Pearson correlation", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OD / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


print("4x4:"); Z4 = report(R4, labs4)
print("12x12:"); Z12 = report(R12, labs12)
draw(R4, labs4, Z4, "panel_b_v2_4x4", (2.8, 2.3))
draw(R12, labs12, Z12, "panel_b_v2_12x12", (4.2, 3.6))
