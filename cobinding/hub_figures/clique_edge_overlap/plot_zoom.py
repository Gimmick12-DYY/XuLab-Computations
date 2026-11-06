"""Zoom panels for hub anchors from raw read 5' ends: CPM per bin, light smoothing (3-bin running mean),
per-track y-cap, and TF rows chosen by local enrichment at the anchor elements, plus a pooled all-TF track."""
import argparse, gzip
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, pandas as pd
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
ap = argparse.ArgumentParser()
ap.add_argument("--reads", required=True); ap.add_argument("--genes", required=True); ap.add_argument("--chromhmm", required=True)
ap.add_argument("--edges", required=True); ap.add_argument("--win", nargs="+", required=True, help="chrom:start-end")
ap.add_argument("--anchor", nargs="+", required=True, help="chrom:start-end of the hub element in each window (same order)")
ap.add_argument("--bin", type=int, default=500); ap.add_argument("--smooth", type=int, default=3)
ap.add_argument("--ntf", type=int, default=8); ap.add_argument("--select", choices=["enrichment", "edges"], default="enrichment")
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = pd.read_csv(a.reads, sep="\t", header=None, names=["tf", "chrom", "pos"])
tot = R[R.chrom == "TOTAL"].set_index("tf").pos.astype(float); R = R[R.chrom != "TOTAL"]; R["pos"] = R.pos.astype(int)
E = pd.read_csv(a.edges, sep="\t"); edge_tfs = pd.Series([t for l in E.tfs.str.split(",") for t in l]).value_counts()
parse = lambda w: (w.split(":")[0], *map(int, w.split(":")[1].split("-")))
W = [parse(w) for w in a.win]; A = [parse(x) for x in a.anchor]
def track(tfs, c, s, e):
    edges = np.arange(s, e + a.bin, a.bin)
    p = R[(R.tf.isin(tfs)) & (R.chrom == c) & (R.pos >= s) & (R.pos < e)].pos.values
    h = np.histogram(p, bins=edges)[0] / tot[list(tfs)].sum() * 1e6
    if a.smooth > 1: h = np.convolve(h, np.ones(a.smooth) / a.smooth, mode="same")
    return edges, h
def enrich(tf, c, s, e, anc):   # raw-count enrichment with pseudocounts: (anchor reads + 1) / (expected from window + 1)
    p = R[(R.tf == tf) & (R.chrom == c) & (R.pos >= s) & (R.pos < e)].pos.values
    lo_, hi_ = anc[1] - 250, anc[2] + 250
    k = ((p >= lo_) & (p < hi_)).sum(); bg = ((p < lo_) | (p >= hi_)).sum()
    exp = bg * (hi_ - lo_) / ((e - s) - (hi_ - lo_))
    return (k + 1) / (exp + 1)
cand = [t for t in tot.index if tot[t] >= 1e6]
if a.select == "enrichment":   # min enrichment over the anchors (TF must stand out at every anchor), rank by it
    sc = {t: min(enrich(t, c, s, e, an) for (c, s, e), an in zip(W, A)) for t in cand}
    tfs = [t for t, _ in sorted(sc.items(), key=lambda z: -z[1])[:a.ntf]]
else:   # same rows as the full hub view: hub TFs ranked by shared edges, the rest pooled
    tfs = list(edge_tfs.index[:a.ntf]); sc = {}
REST = [t for t in edge_tfs.index[a.ntf:] if t in tot.index]
rows = tfs + ([f"other {len(REST)} TFs"] if REST else [])
G = pd.read_csv(a.genes, sep="\t")
HS = []
with gzip.open(a.chromhmm, "rt") as fh:
    for ln in fh:
        f = ln.split("\t"); HS.append((f[0], int(f[1]), int(f[2]), f[3]))
GC = lambda s: "#E67E22" if s.startswith("Enh") else "#C0392B" if s.startswith("Tss") else "#5D8AA8" if s.startswith("Tx") else None
fig, axs = plt.subplots(len(rows) + 2, len(W), figsize=(6.2 * len(W), 0.55 * len(rows) + 2.6), squeeze=False,
                        gridspec_kw={"height_ratios": [1] * len(rows) + [0.5, 1.0], "hspace": 0.12, "wspace": 0.12})
cols = plt.get_cmap("tab10")
for j, ((c, s, e), an) in enumerate(zip(W, A)):
    col = axs[:, j]
    for k, r in enumerate(rows):
        ax = col[k]; ts = REST if r.startswith("other") else [r]
        ed, h = track(ts, c, s, e)
        ctr = (ed[:-1] + ed[1:]) / 2; on = (ctr >= an[1] - a.bin) & (ctr <= an[2] + a.bin)
        ylim = max(h[on].max() if on.any() else 0, np.percentile(h, 99)) * 1.1 + 1e-9
        ax.fill_between(ed[:-1], 0, np.minimum(h, ylim), step="post", color="#7F7F7F" if r.startswith("other") else cols(k % 10), lw=0)
        ax.axvspan(an[1], an[2], color="#FAD7A0" if j == 0 else "#F5B7B1", zorder=0, lw=0)
        ax.set_xlim(s, e); ax.set_ylim(0, ylim); ax.set_yticks([]); ax.set_xticks([])
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#BBB")
        if j == 0:
            lab = r if r.startswith("other") else f"{r}" + (f" ({sc[r]:.1f}×)" if sc else f" ({edge_tfs[r]})")
            ax.text(-0.01, 0.4, lab, transform=ax.transAxes, ha="right", va="center", fontsize=10)
    ax = col[-2]
    for (hc, hs_, he, st) in HS:
        if hc == c and he > s and hs_ < e and GC(st):
            ax.add_patch(Rectangle((max(hs_, s), 0.15), min(he, e) - max(hs_, s), 0.7, color=GC(st), lw=0))
    ax.set_xlim(s, e); ax.set_ylim(0, 1); ax.axis("off")
    if j == 0: ax.text(-0.01, 0.5, "ChromHMM", transform=ax.transAxes, ha="right", va="center", fontsize=10)
    ax = col[-1]; lane = 0
    for g in G[(G.txEnd > s) & (G.txStart < e)].sort_values("txStart").itertuples():
        y = -lane * 1.2; lane = (lane + 1) % 2
        ax.plot([max(g.txStart, s), min(g.txEnd, e)], [y, y], color="#333", lw=1)
        for xs, xe in zip(map(int, g.exonStarts.strip(",").split(",")), map(int, g.exonEnds.strip(",").split(","))):
            if xe > s and xs < e: ax.add_patch(Rectangle((xs, y - 0.2), xe - xs, 0.4, color="#333", lw=0))
        ax.text((max(g.txStart, s) + min(g.txEnd, e)) / 2, y - 0.35, g.name, ha="center", va="top", fontsize=9.5, style="italic")
    ax.set_xlim(s, e); ax.set_ylim(-2.4, 0.8); ax.set_yticks([])
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.3f}")); ax.set_xlabel(f"{c} (Mb, hg38)")
    col[0].set_title(f"{c}:{s:,}-{e:,}", loc="left", fontsize=11.5)
fig.text(0.99, 0.005, f"read 5′ ends, CPM per {a.bin} bp, {a.smooth}-bin running mean; each track scaled to its anchor peak; "
         + ("rows = the 8 TFs (≥1 M reads) most enriched at both anchors vs the rest of the window (×, weaker anchor)" if sc else "rows = the hub's TFs as in the full view; (n) = shared edges the TF is in"), ha="right", fontsize=9, color="#555")
fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight", facecolor="white"); fig.savefig(a.out + ".svg", bbox_inches="tight", facecolor="white")
print("rows:", [(t, round(sc.get(t, 0), 2)) for t in tfs])
