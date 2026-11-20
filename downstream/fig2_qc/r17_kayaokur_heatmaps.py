#!/usr/bin/env python
"""Peak heatmaps in the style of Kaya-Okur et al. 2019 (Nat Commun) Fig 1: tall single-column heatmaps, +-1 kb, rows
sorted by centre signal (descending), turbo colormap, one colour scale per panel (0 to its 98th percentile), colour bar,
unit and site count under each panel, no profiles. Per TF: Bulk ChIP | sc raw | sc imputed.
Signals (50 bp bins over +-1 kb, 40 bins):
  ChIP    = bamCoverage CPM bigWig (r13_bamcoverage.sbatch: dups ignored, reads extended to 200 bp)
  sc raw  = aggregated single-cell 5' ends, CPM, 3-bin running mean
  imputed = fraction of cells positive in the 1 kb genomic bin covering each 50 bp position (1 kb resolution, so +-1 kb
            spans 2-3 bins). The imputed matrix holds model predictions inside the ATAC union and, outside it, the raw
            observed bins (02_impute.py keep_raw=True), so outside open chromatin "imputed" = binarised raw.
Peak sets:
  ChIP peaks    = reference summits (CTCF ENCFF314ZAL; MAZ/ZBTB7A/ZIC2/ZNF777 ENCODE IDR; ZNF282 GSM2466515 MACS2 re-call)
  our raw peaks = lab MACS2 peaks on the TF's aggregated reads, q < 0.05, summit
  imputed peaks = the TF's 1 kb bins with the highest aggregated imputed fraction, as many as our raw peaks, bin centre
V1 "paper style": each panel = that dataset's own peaks, sorted by its own centre signal (+-200 bp).
V2 "same regions": one set per TF = ChIP summits plus our raw summits not within 500 bp of a ChIP peak; same rows in all
   panels; sorted by mean of rank-normalised ChIP and raw centre signal. V2b: same rows sorted by ChIP centre signal.
Outputs OH/kayaokur_style/: <TF>_{V1,V2,V2b}.{png,pdf,svg}, overview_{V1,V2,V2b}.*, kayaokur_counts.tsv."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from matplotlib import font_manager

from fig2_common import INP, OUT, REF, ROOT, TFS, bin_index, load_imp7

OH = OUT / "round6_diffpeaks" / "heatmaps"
OK = OH / "kayaokur_style"
OK.mkdir(parents=True, exist_ok=True)
W, STEP = 1000, 50
NB = 2 * W // STEP
xs = np.arange(-W, W, STEP) + STEP // 2
CTR = slice(NB // 2 - 4, NB // 2 + 4)  # +-200 bp
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
LAB = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
PEAKS = dict(REF)
PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}

bi, off, N = bin_index()
csize = dict(zip(bi.chrom, bi["size"]))
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]
ncell = fr["ncells_imp79"]
imp = load_imp7()
chrom_of_bin = np.empty(N, dtype=object)
for c, (a, nb) in off.items():   # off[c] = (first bin, number of bins)
    chrom_of_bin[a:a + nb] = c


def ok_rows(chrom, pos):
    m = np.array([c in csize and p - W >= 0 and p + W <= csize[c] for c, p in zip(chrom, pos)])
    return m


def chip_peaks(t):
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10))
    P = P[P[0].isin(CH)]
    return P[0].values, (P[1] + P[9].clip(lower=0)).values.astype(int), P[1].values, P[2].values


def raw_peaks(t):
    P = pd.read_csv(LAB.format(t=t), sep="\t", header=None, usecols=range(10))
    P = P[P[0].isin(CH) & (P[8] >= 1.30103)]
    return P[0].values, (P[1] + P[9]).values.astype(int)


def imp_peaks(i, t, n):
    v = imp[i] / ncell[tf79.index(t)]
    top = np.argsort(-v, kind="stable")[:n]
    top = top[v[top] > 0]
    c = chrom_of_bin[top]
    keep = np.array([x in CH for x in c])
    top, c = top[keep], c[keep]
    pos = (top - np.array([off[x][0] for x in c])) * 1000 + 500
    return c, pos.astype(int)


def m_chip(t, chrom, pos):
    bw = pyBigWig.open(str(OH / "bw" / f"{t}.bw"))
    M = np.zeros((len(pos), NB))
    for r, (c, x) in enumerate(zip(chrom, pos)):
        M[r] = np.nan_to_num(np.array(bw.stats(c, int(x - W), int(x + W), nBins=NB, type="mean"), dtype=float))
    return M


k3 = np.ones(3) / 3
_five = {}


def m_raw(t, chrom, pos):
    if t not in _five:
        z = np.load(INP / "five" / f"{t}.npz")
        _five[t] = ({c: z[c] for c in z.files}, sum(len(z[c]) for c in z.files))
    z, tot = _five[t]
    edges = np.arange(-W, W + 1, STEP)
    M = np.zeros((len(pos), NB))
    for c in np.unique(chrom):
        rr = np.flatnonzero(chrom == c)
        p = z.get(c)
        if p is None:
            continue
        lo = np.searchsorted(p, pos[rr] - W); hi = np.searchsorted(p, pos[rr] + W)
        for r, a, b in zip(rr, lo, hi):
            M[r] = np.histogram(p[a:b] - pos[r], edges)[0]
    M = M / tot * 1e6
    return np.apply_along_axis(lambda r: np.convolve(r, k3, mode="same"), 1, M)


def m_imp(i, t, chrom, pos):
    base = np.array([off[c][0] for c in chrom])
    gb = np.clip(base[:, None] + (pos[:, None] + xs[None, :]) // 1000, 0, N - 1)
    return imp[i][gb] / ncell[tf79.index(t)]


def rank01(v):
    return pd.Series(v).rank(pct=True).values


import pickle
MODE_TF = sys.argv[1] if len(sys.argv) > 1 else None
data, counts = {}, []
for i, t in enumerate(TFS):
    cache = OK / f"cache_{t}.pkl"
    if MODE_TF is None:
        d, c = pickle.load(open(cache, "rb"))
        data[t] = d; counts.append(c); continue
    if t != MODE_TF:
        continue
    cc, cp, cs, ce = chip_peaks(t)
    m = ok_rows(cc, cp); cc, cp, cs, ce = cc[m], cp[m], cs[m], ce[m]
    rc, rp = raw_peaks(t)
    m = ok_rows(rc, rp); rc, rp = rc[m], rp[m]
    ic, ip = imp_peaks(i, t, len(rp))
    m = ok_rows(ic, ip); ic, ip = ic[m], ip[m]
    V1 = {"ChIP": m_chip(t, cc, cp), "raw": m_raw(t, rc, rp), "imputed": m_imp(i, t, ic, ip)}
    V1o = {k: np.argsort(-M[:, CTR].mean(1), kind="stable") for k, M in V1.items()}
    # V2 union: ChIP summits + our summits not within 500 bp of a ChIP peak interval
    keep = np.ones(len(rp), bool)
    for c in np.unique(rc):
        rr = np.flatnonzero(rc == c); q = np.flatnonzero(cc == c)
        if not len(q):
            continue
        o = np.argsort(cs[q]); s_, e_ = cs[q][o], ce[q][o]
        emax = np.maximum.accumulate(e_)
        j = np.searchsorted(s_, rp[rr] + 500, side="right") - 1
        keep[rr] = ~((j >= 0) & (emax[np.clip(j, 0, None)] >= rp[rr] - 500))
    uc = np.r_[cc, rc[keep]]; up = np.r_[cp, rp[keep]]
    src = np.r_[np.zeros(len(cp), int), np.ones(keep.sum(), int)]
    V2 = {"ChIP": m_chip(t, uc, up), "raw": m_raw(t, uc, up), "imputed": m_imp(i, t, uc, up)}
    sc = (rank01(V2["ChIP"][:, CTR].mean(1)) + rank01(V2["raw"][:, CTR].mean(1))) / 2
    o2 = np.argsort(-sc, kind="stable")
    o2b = np.argsort(-V2["ChIP"][:, CTR].mean(1), kind="stable")
    data[t] = dict(V1=V1, V1o=V1o, V2=V2, o2=o2, o2b=o2b)
    counts.append(dict(TF=t, chip_peaks=len(cp), raw_peaks_q05=len(rp), imputed_peak_bins=len(ip),
                       union_V2=len(up), union_from_chip=int((src == 0).sum()), union_raw_only=int((src == 1).sum()),
                       raw_peaks_within500_of_chip=int((~keep).sum())))
    print(counts[-1], flush=True)
    pickle.dump((data[t], counts[-1]), open(cache, "wb"), protocol=4)
if MODE_TF is not None:
    sys.exit()
pd.DataFrame(counts).to_csv(OK / "kayaokur_counts.tsv", sep="\t", index=False)

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "xtick.major.size": 2, "xtick.labelsize": 6})
CM = plt.get_cmap("turbo")
MM = 1 / 25.4
UNIT = {"ChIP": "CPM", "raw": "CPM", "imputed": "frac. cells"}
TTL = {"ChIP": "Bulk ChIP", "raw": "sc raw", "imputed": "sc imputed"}


def hpanel(fig, cell, M, order, title, unit):
    g = cell.subgridspec(3, 1, height_ratios=[20, 0.6, 0.45], hspace=0.05)
    ax = fig.add_subplot(g[0])
    X = M[order]
    nz = X[X > 0]
    vm = float(np.percentile(nz, 98)) if nz.size else 1.0
    ax.imshow(X, aspect="auto", cmap=CM, vmin=0, vmax=vm, interpolation="nearest",
              extent=[-W, W, len(X), 0], rasterized=True)
    ax.set_yticks([]); ax.set_xticks([-W, W], ["−1 kb", "+1 kb"])
    for lb, ha in zip(ax.get_xticklabels(), ("left", "right")):
        lb.set_ha(ha)
    ax.set_title(title, fontsize=7, pad=2)
    cax = fig.add_subplot(g[2])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vm), cmap=CM), cax=cax, orientation="horizontal")
    cb.set_ticks([0, vm]); cb.set_ticklabels(["0", f"{vm:.2g}"]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    for lb, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
        lb.set_ha(ha)
    cb.set_label(f"{unit}\n{len(X):,} sites", fontsize=6, labelpad=1)


def panels(t, ver):
    d = data[t]
    out = []
    for ty in ("ChIP", "raw", "imputed"):
        if ver == "V1":
            out.append((d["V1"][ty], d["V1o"][ty], f"{TTL[ty]}", UNIT[ty]))
        else:
            out.append((d["V2"][ty], d["o2" if ver == "V2" else "o2b"], TTL[ty], UNIT[ty]))
    return out


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OK / f"{name}.{ext}", dpi=600)
    plt.close(fig)


SUB = {"V1": "each panel: its own peaks", "V2": "same regions (ChIP ∪ our peaks), sorted by ChIP + raw",
       "V2b": "same regions (ChIP ∪ our peaks), sorted by ChIP"}
for ver in ("V1", "V2", "V2b"):
    for t in TFS:
        fig = plt.figure(figsize=(75 * MM, 110 * MM))
        gs = fig.add_gridspec(1, 3, wspace=0.18, left=0.04, right=0.96, top=0.88, bottom=0.03)
        for j, (M, o, ttl, un) in enumerate(panels(t, ver)):
            hpanel(fig, gs[j], M, o, ttl, un)
        fig.suptitle(f"{t} ({CELL[t]} ChIP)\n{SUB[ver]}", fontsize=7, y=0.985)
        save(fig, f"{t}_{ver}")
    fig = plt.figure(figsize=(183 * MM, 235 * MM))
    gs = fig.add_gridspec(2, 3, wspace=0.14, hspace=0.16, left=0.02, right=0.99, top=0.95, bottom=0.03)
    for k, t in enumerate(TFS):
        r, c = divmod(k, 3)
        sub = gs[r, c].subgridspec(1, 3, wspace=0.15)
        for j, (M, o, ttl, un) in enumerate(panels(t, ver)):
            hpanel(fig, sub[j], M, o, ttl, un)
        pos = gs[r, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.022, f"{t} ({CELL[t]} ChIP)", ha="center", fontsize=7, fontweight="bold")
    fig.suptitle(SUB[ver], fontsize=7, y=0.995)
    save(fig, f"overview_{ver}")
print("done")
