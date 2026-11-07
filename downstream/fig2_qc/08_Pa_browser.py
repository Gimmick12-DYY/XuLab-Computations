#!/usr/bin/env python
"""P-a: browser panels, bulk ChIP vs aggregated raw sc vs imputed (+ pooled 79-TF panel in grey).

Loci: 2 each for CTCF, ZNF282 and MAZ. CTCF and MAZ: centred on T5 anchors (reference-unique peak
with own motif). ZNF282 (no usable motif): summits of reference-unique ZNF282 peaks.
Ranked by own-TF raw reads within +-250 bp, requiring own CPM >= 2x pooled-panel CPM there and
>= 1 Mb between the two loci. Window +-5 kb.
Tracks: bulk = BAM coverage per base, mean per 50 bp, per million mapped reads (samtools idxstats);
raw = sc read 5' ends per 50 bp, per million TF reads; imputed = % of cells positive per 1 kb bin;
pooled = all 79 TFs, 5' ends per 50 bp per million reads. Same y-scale for a track across a TF's 2 loci.
Outputs OUT/panels/Pa_loci.tsv, Pa_tracks.npz, Pa_browser.png/.pdf.
"""
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig2_common import NT, BAM, BIN, INP, OUT, REF, ROOT, bin_index, load_imp7, read_bed, TFS

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
op = OUT / "panels"
HALF, STEP = 5000, 50
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = list(fr["tf79"])
ncell = fr["ncells_imp79"]
imp7 = load_imp7()


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


five = {t: dict(np.load(INP / "five" / f"{t}.npz")) for t in tf79}
tot = {t: sum(len(v) for v in five[t].values()) for t in tf79}
ptot = sum(tot.values())


def count(t, c, a, b):
    p = five[t][c]
    return np.searchsorted(p, b) - np.searchsorted(p, a)


def pooled_count(c, a, b):
    return sum(count(t, c, a, b) for t in tf79)


loci = []
PA_TFS = ("CTCF", "ZNF282", "MAZ")


def anchor_table(tf):
    f = OUT / "spec" / f"T5_anchors_{tf}.bed"
    if f.exists():
        return pd.read_csv(f, sep="\t", header=None)
    uq = read_bed(INP / f"unique_{tf}.bed")
    full = read_bed(REF[tf], ncol=10)
    key = set(zip(uq[0], uq[1], uq[2]))
    full = full[[k in key for k in zip(full[0], full[1], full[2])]]
    return pd.DataFrame({0: full[0].values, 1: (full[1] + full[9].astype(int)).values})


for tf in PA_TFS:
    A = anchor_table(tf)
    A = A[A[0].isin(list(off))].reset_index(drop=True)
    own = np.array([count(tf, c, p - 250, p + 250) for c, p in zip(A[0], A[1])])
    top = np.argsort(-own)[:200]
    chosen = []
    for k in top:
        c, p = A.iloc[k, 0], int(A.iloc[k, 1])
        oc = own[k] / tot[tf] * 1e6
        pc = pooled_count(c, p - 250, p + 250) / ptot * 1e6
        if pc > 0 and oc / pc < 2:
            continue
        if any(c == x[1] and abs(p - x[2]) < 1_000_000 for x in chosen):
            continue
        chosen.append((tf, c, p, int(own[k]), oc, pc))
        if len(chosen) == 2:
            break
    loci += chosen
L = pd.DataFrame(loci, columns=["tf", "chrom", "anchor", "own_reads_500bp", "own_cpm", "pooled_cpm"])
L.to_csv(op / "Pa_loci.tsv", sep="\t", index=False, float_format="%.3g")
print(L.to_string(), flush=True)

mapped = {t: sum(int(l.split("\t")[2]) for l in sh("samtools", "idxstats", str(BAM[t])).splitlines()) for t in PA_TFS}
edges = np.arange(-HALF, HALF + 1, STEP)
data = {}
for r in L.itertuples():
    c, a0, a1 = r.chrom, r.anchor - HALF, r.anchor + HALF
    dep = sh("samtools", "depth", "-a", "-r", f"{c}:{a0 + 1}-{a1}", str(BAM[r.tf]))
    d = np.array([int(x.split("\t")[2]) for x in dep.splitlines()], float)
    d = np.pad(d, (0, 2 * HALF - len(d)))[:2 * HALF]
    bulk = d.reshape(-1, STEP).mean(1) / mapped[r.tf] * 1e6
    p = five[r.tf][c]
    raw = np.histogram(p[(p >= a0) & (p < a1)] - r.anchor, edges)[0] / tot[r.tf] * 1e6
    pooled = np.zeros(len(edges) - 1)
    for t in tf79:
        q = five[t][c]
        pooled += np.histogram(q[(q >= a0) & (q < a1)] - r.anchor, edges)[0]
    pooled = pooled / ptot * 1e6
    o = off[c][0]
    b0, b1 = a0 // BIN, (a1 - 1) // BIN
    i = TFS.index(r.tf)
    impv = imp7[i, o + b0:o + b1 + 1] / ncell[tf79.index(r.tf)] * 100
    impx = np.arange(b0, b1 + 2) * BIN - r.anchor
    data[(r.tf, c, r.anchor)] = dict(bulk=bulk, raw=raw, pooled=pooled, imp=impv, impx=impx)
np.savez(op / "Pa_tracks.npz", **{f"{k[0]}_{k[1]}_{k[2]}_{n}": v for k, d in data.items() for n, v in d.items()})

peaks = {t: read_bed(REF[t]) for t in PA_TFS}
rows = [("bulk", "Bulk ChIP\n(CPM)", P.neutral.ink), ("raw", "sc raw\n(CPM)", P.arms.raw),
        ("imp", "sc imputed\n(% cells)", P.arms.imputed), ("pooled", "79-TF pool\n(CPM)", P.neutral.mid)]
fig, axes = plt.subplots(len(rows), 2 * len(PA_TFS), figsize=(7.2, 3.6), sharex=True)
cx = (edges[:-1] + edges[1:]) / 2
keys = list(data)
for col, k in enumerate(keys):
    tf, c, anc = k
    pair = [kk for kk in keys if kk[0] == tf]
    for ri, (name, lab, colr) in enumerate(rows):
        ax = axes[ri, col]
        ymax = max((data[kk][name].max() for kk in pair)) * 1.1 or 1
        if name == "imp":
            ax.stairs(data[k]["imp"], data[k]["impx"], fill=True, color=colr, lw=0)
        else:
            ax.fill_between(cx, data[k][name], step="mid", color=colr, lw=0)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, float(f"{ymax / 1.1:.2g}")])
        if col % 2 == 0:
            ax.set_ylabel(lab, fontsize=6)
        if ri == 0:
            pk = peaks[tf]
            pk = pk[(pk[0] == c) & (pk[2] > anc - HALF) & (pk[1] < anc + HALF)]
            for s, e in zip(pk[1], pk[2]):
                ax.axvspan(s - anc, e - anc, color=P.highlight, zorder=0, lw=0)
            ax.set_title(f"{tf}-unique site\n{c}:{anc:,}", fontsize=6)
        ax.set_xlim(-HALF, HALF)
for ax in axes[-1]:
    ax.set_xticks([-HALF, 0, HALF])
    ax.set_xticklabels([f"-{HALF // 1000} kb", "site", f"+{HALF // 1000} kb"])
fig.tight_layout(h_pad=0.3, w_pad=0.6)
for ext in ("png", "pdf"):
    fig.savefig(op / f"Pa_browser.{ext}", dpi=300)
print("saved")
