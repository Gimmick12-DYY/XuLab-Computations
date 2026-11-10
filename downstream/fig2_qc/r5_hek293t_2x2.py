#!/usr/bin/env python
"""HEK293T-only 2x2 for Fig 2: sc CTCF and sc ZNF282 vs HEK293T bulk ChIP. Pre-specified, no version search.

References (both HEK293T): CTCF = GSE103651 ChIP-nexus (GSM2777335/6, Millipore 07-729; MACS2 peaks
data/CTCF_HEK293T_GSE103651_peaks.bed, pooled BAM chip_gse103651/work/bam/gse103651.pooled.bam);
ZNF282 = GSM2466515 HA-tag ChIP-exo (MACS2 peaks data/ZNF282_HEK293_peaks.bed; file name is historical,
the cells are HEK293T).
Spec (same as the published summit-group Pearson): summits with bulk own - other log CPM > 1 (2x), top 5000 by
signalValue, 30 groups per reference. bulk = log2(CPM+1) +-250 bp; sc = log2((C+.5)/(F+.5)) - same for
leave-one-out 79-TF pool (C +-100 bp, F 1-2 kb); ZNF777 depth (420,339 reads), 3 seeds (r averaged).
Reports the 2x2, own > other in rows and columns, per seed, and each chromosome half (odd autosomes; even + X)
computed separately. Figures: 2x2 block, clustered 4x4, browser (2 loci per TF: unique to that reference, peak
isolated within +-5 kb from other peaks of that TF, >= 90% k50-mappable in +-500 bp, own/pool CPM >= 2).
Outputs OUT/hek293t_2x2/."""
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from fig2_common import BIN, D, DM_DEPTH, DM_SEEDS, INP, OUT, REF, ROOT, TFS, bin_index, load_imp7, read_bed

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
O5 = OUT / "hek293t_2x2"
O5.mkdir(parents=True, exist_ok=True)
tmp = O5 / "tmp"
tmp.mkdir(exist_ok=True)
T2 = ["CTCF", "ZNF282"]
PK = {"CTCF": D / "CTCF_HEK293T_GSE103651_peaks.bed", "ZNF282": D / "ZNF282_HEK293_peaks.bed"}
BAM2 = {"CTCF": ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam", "ZNF282": D / "ZNF282_HEK293.bam"}
CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
TOPN, G, CEN, FL0, FL1, HALF = 5000, 30, 100, 1000, 2000, 5000


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


rows = []
for t in T2:
    pk = read_bed(PK[t], ncol=10)
    pk = pk[pk[0].isin(CHROMS)]
    s = pd.DataFrame({"ref": t, "chrom": pk[0].values, "pos": (pk[1] + pk[9].astype(int)).values,
                      "signal": pk[6].astype(float).values, "pk_start": pk[1].values, "pk_end": pk[2].values})
    rows.append(s.sort_values("signal", ascending=False).drop_duplicates(["chrom", "pos"]))
S = pd.concat(rows, ignore_index=True)

# overlap with the old CTCF 2-of-3 set
a = tmp / "ctcf_293t.bed"
read_bed(PK["CTCF"])[[0, 1, 2]].to_csv(a, sep="\t", header=False, index=False)
n_new = len(read_bed(a))
ov1 = len(sh("bedtools", "intersect", "-u", "-a", str(a), "-b", str(REF["CTCF"])).splitlines())
n_old = len(read_bed(REF["CTCF"]))
ov2 = len(sh("bedtools", "intersect", "-u", "-a", str(REF["CTCF"]), "-b", str(a)).splitlines())
print(f"CTCF HEK293T nexus peaks {n_new}; overlapping old 2-of-3 {ov1} ({ov1/n_new:.1%}); old 2-of-3 {n_old}, "
      f"overlapping HEK293T {ov2} ({ov2/n_old:.1%})", flush=True)

# bulk counts
bb = tmp / "w250.bed"
o = np.lexsort((S.pos.values, S.chrom.values))
pd.DataFrame({0: S.chrom.values[o], 1: S.pos.values[o] - 250, 2: S.pos.values[o] + 250, 3: o}).to_csv(
    bb, sep="\t", header=False, index=False)
out = sh("bedtools", "multicov", "-bams", *[str(BAM2[t]) for t in T2], "-bed", str(bb))
Bk = np.zeros((len(S), 2))
for l in out.splitlines():
    p = l.split("\t")
    Bk[int(p[3])] = [float(x) for x in p[4:]]
L = np.log2(Bk / Bk.sum(0, keepdims=True) * 1e6 + 1)
own = np.array([L[i, T2.index(t)] for i, t in enumerate(S.ref.values)])
oth = np.array([L[i, 1 - T2.index(t)] for i, t in enumerate(S.ref.values)])
S["spec"] = own - oth
S["preferred"] = S.spec > 1
print(S.groupby("ref").preferred.agg(["size", "sum"]), flush=True)

# sc counts
tf79 = list(np.load(INP / "frame.npz", allow_pickle=True)["tf79"])
C79 = np.zeros((len(tf79), len(S)), np.int64)
F79 = np.zeros_like(C79)
tot79 = np.zeros(len(tf79))
for k, t in enumerate(tf79):
    z = np.load(INP / "five" / f"{t}.npz")
    tot79[k] = sum(len(z[c]) for c in z.files)
    for c in CHROMS:
        m = np.flatnonzero(S.chrom.values == c)
        if len(m):
            p, x = z[c], S.pos.values[m]
            cnt = lambda a_, b_: np.searchsorted(p, x + b_, side="right") - np.searchsorted(p, x + a_, side="left")
            C79[k, m] = cnt(-CEN, CEN)
            F79[k, m] = cnt(-FL1, -FL0 - 1) + cnt(FL0 + 1, FL1)
pi = [tf79.index(t) for t in T2]
Cpool, Fpool = C79.sum(0), F79.sum(0)
S.to_csv(O5 / "summits.tsv", sep="\t", index=False)
cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
HALVES = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}
pdm = np.minimum(1, DM_DEPTH / tot79[pi])[:, None]
THIN = [(np.random.default_rng(s).binomial(C79[pi], pdm), np.random.default_rng(s + 1).binomial(F79[pi], pdm))
        for s in DM_SEEDS]


def groups(half):
    gr = []
    for t in T2:
        ix = np.flatnonzero((S.ref.values == t) & S.preferred.values & HALVES[half])
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:TOPN]
        gr += [g for g in np.array_split(ix, G) if len(g)]
    return gr


def R_of(gr, Cs, Fs):
    gB = np.array([Bk[g].sum(0) for g in gr])
    bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    sc = np.zeros((len(gr), 2))
    for i in range(2):
        for a, g in enumerate(gr):
            cp = Cpool[g].sum() - C79[pi[i], g].sum()
            fp = Fpool[g].sum() - F79[pi[i], g].sum()
            sc[a, i] = np.log2((Cs[i, g].sum() + .5) / (Fs[i, g].sum() + .5)) - np.log2((cp + .5) / (fp + .5))
    return np.corrcoef(np.column_stack([bulk, sc]).T)


labs = ["HEK293T ChIP CTCF", "HEK293T ChIP ZNF282", "sc CTCF", "sc ZNF282"]
res = []
Rall = None
for half in ("all", "odd", "even_X"):
    gr = groups(half)
    Rs = [R_of(gr, *th) for th in THIN]
    R = np.mean(Rs, 0)
    if half == "all":
        Rall = R
    for sd, X in [(f"seed{s}", r) for s, r in zip(DM_SEEDS, Rs)] + [("mean", R)]:
        B = X[2:, :2]
        res.append(dict(half=half, seed=sd, n_groups=len(gr), n_summits_CTCF=sum(len(g) for g in gr[:G]),
                        n_summits_ZNF282=sum(len(g) for g in gr[G:]), r_scCTCF_bulkCTCF=B[0, 0],
                        r_scCTCF_bulkZNF282=B[0, 1], r_scZNF282_bulkCTCF=B[1, 0], r_scZNF282_bulkZNF282=B[1, 1],
                        own_gt_other_rows=int(B[0, 0] > B[0, 1]) + int(B[1, 1] > B[1, 0]),
                        own_gt_other_cols=int(B[0, 0] > B[1, 0]) + int(B[1, 1] > B[0, 1])))
    pd.DataFrame(R, labs, labs).to_csv(O5 / f"corr4x4_{half}.tsv", sep="\t", float_format="%.4f")
rd = pd.DataFrame(res)
rd.to_csv(O5 / "r_2x2.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(rd.round(3).to_string(), flush=True)

# 2x2 block
B = Rall[2:, :2]
fig, ax = plt.subplots(figsize=(2.3, 1.9))
lim = max(0.2, np.abs(B).max())
im = ax.imshow(B, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
for a in range(2):
    for b in range(2):
        ax.text(b, a, f"{B[a, b]:+.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(B[a, b]) > 0.6 * lim else INK)
    ax.add_patch(Rectangle((a - .5, a - .5), 1, 1, fill=False, ec=INK, lw=0.6))
ax.set_xticks([0, 1], ["HEK293T ChIP\nCTCF", "HEK293T ChIP\nZNF282"])
ax.set_yticks([0, 1], ["sc CTCF", "sc ZNF282"])
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson r")
ax.set_title("sc vs HEK293T ChIP, summit groups, ZNF777 depth", fontsize=6)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(O5 / f"block_2x2.{ext}", dpi=600 if ext == "png" else None)
plt.close(fig)

# clustered 4x4
Z = linkage(squareform(np.clip(1 - Rall, 0, None), checks=False), "average")
fig = plt.figure(figsize=(2.9, 2.9))
axd = fig.add_axes([0.36, 0.76, 0.42, 0.13])
d = dendrogram(Z, ax=axd, no_labels=True, color_threshold=0, above_threshold_color=INK)
axd.axis("off")
o = d["leaves"]
axh = fig.add_axes([0.36, 0.30, 0.42, 0.46])
im = axh.imshow(Rall[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-1, vmax=1, aspect="auto")
axh.set_xticks(range(4), [labs[k] for k in o], rotation=90)
axh.set_yticks(range(4), [labs[k] for k in o])
for tl in axh.get_xticklabels() + axh.get_yticklabels():
    tl.set_color(INK if tl.get_text().startswith("HEK") else P.arms.raw)
axh.tick_params(length=0)
fig.colorbar(im, cax=fig.add_axes([0.83, 0.30, 0.03, 0.25]), label="Pearson r")
fig.suptitle("HEK293T ChIP and sc tracks, Pearson over summit groups, ZNF777 depth", fontsize=5.5, y=0.97)
for ext in ("png", "pdf"):
    fig.savefig(O5 / f"cluster_4x4.{ext}", dpi=600 if ext == "png" else None)
plt.close(fig)

# ---- browser ----
five = {t: dict(np.load(INP / "five" / f"{t}.npz")) for t in tf79}
ptot = tot79.sum()
tot = dict(zip(tf79, tot79))
bi, off, N = bin_index()
imp = load_imp7()
ncell = np.load(INP / "frame.npz", allow_pickle=True)["ncells_imp79"]
pks = {t: read_bed(PK[t]) for t in T2}
umap = D / "mappability/k50.umap.bed.gz"
loci = []
for t in T2:
    o2 = T2[1 - T2.index(t)]
    m = S.ref.values == t
    idx = np.flatnonzero(m & S.preferred.values)
    keep = []
    pk = pks[t].sort_values([0, 1])
    pko = pks[o2]
    cand = idx[np.argsort(-C79[tf79.index(t), idx])][:400]
    for i in cand:
        c, x = S.chrom.values[i], S.pos.values[i]
        pkc = pk[(pk[0] == c)]
        nearby = ((pkc[2] > x - HALF) & (pkc[1] < x + HALF)).sum()
        if nearby > 1:
            continue
        ot = pko[pko[0] == c]
        if ((ot[2] > x - HALF) & (ot[1] < x + HALF)).any():
            continue
        oc = C79[tf79.index(t), i] / tot[t] * 1e6
        pc = Cpool[i] / ptot * 1e6
        if pc <= 0 or oc / pc < 2:
            continue
        keep.append(i)
    w = tmp / f"cand_{t}.bed"
    pd.DataFrame({0: S.chrom.values[keep], 1: S.pos.values[keep] - 500, 2: S.pos.values[keep] + 500, 3: keep}).sort_values(
        [0, 1]).to_csv(w, sep="\t", header=False, index=False)
    cov = sh("bedtools", "coverage", "-a", str(w), "-b", str(umap)).splitlines()
    good = {int(l.split("\t")[3]) for l in cov if float(l.split("\t")[-1]) >= 0.9}
    chosen = []
    for i in keep:
        if i not in good:
            continue
        c, x = S.chrom.values[i], S.pos.values[i]
        if any(c == S.chrom.values[j] and abs(x - S.pos.values[j]) < 1_000_000 for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) == 2:
            break
    loci += [(t, i) for i in chosen]
L2 = pd.DataFrame([dict(tf=t, chrom=S.chrom.values[i], pos=int(S.pos.values[i]),
                        own_reads_200bp=int(C79[tf79.index(t), i]), pool_reads_200bp=int(Cpool[i])) for t, i in loci])
L2.to_csv(O5 / "browser_loci.tsv", sep="\t", index=False)
print(L2.to_string(), flush=True)

mapped = {t: sum(int(l.split("\t")[2]) for l in sh("samtools", "idxstats", str(BAM2[t])).splitlines()) for t in T2}
STEP = 50
edges = np.arange(-HALF, HALF + 1, STEP)
cx = (edges[:-1] + edges[1:]) / 2
data = []
for t, i in loci:
    c, x = S.chrom.values[i], int(S.pos.values[i])
    a0, a1 = x - HALF, x + HALF
    dep = sh("samtools", "depth", "-a", "-r", f"{c}:{a0 + 1}-{a1}", str(BAM2[t]))
    dv = np.array([int(l.split("\t")[2]) for l in dep.splitlines()], float)
    dv = np.pad(dv, (0, 2 * HALF - len(dv)))[:2 * HALF]
    bulk = dv.reshape(-1, STEP).mean(1) / mapped[t] * 1e6
    q = five[t][c]
    raw = np.histogram(q[(q >= a0) & (q < a1)] - x, edges)[0] / tot[t] * 1e6
    pool = np.zeros(len(cx))
    for u in tf79:
        q2 = five[u][c]
        pool += np.histogram(q2[(q2 >= a0) & (q2 < a1)] - x, edges)[0]
    pool = pool / ptot * 1e6
    o_ = off[c][0]
    b0, b1 = a0 // BIN, (a1 - 1) // BIN
    iv = imp[TFS.index(t), o_ + b0:o_ + b1 + 1] / ncell[tf79.index(t)] * 100
    ix = np.arange(b0, b1 + 2) * BIN - x
    data.append(dict(t=t, c=c, x=x, bulk=bulk, raw=raw, pool=pool, imp=iv, impx=ix))
tracks = [("bulk", "HEK293T ChIP\n(CPM)", INK), ("raw", "sc raw\n(CPM)", P.arms.raw),
          ("imp", "sc imputed\n(% cells)", P.arms.imputed), ("pool", "79-TF pool\n(CPM)", P.neutral.mid)]
fig, axes = plt.subplots(4, len(data), figsize=(1.75 * len(data), 3.4), sharex=True)
for col, d_ in enumerate(data):
    same = [e for e in data if e["t"] == d_["t"]]
    for ri, (nm, lab, cl) in enumerate(tracks):
        ax = axes[ri, col]
        ymax = max(e[nm].max() for e in same) * 1.1 or 1
        if nm == "imp":
            ax.stairs(d_["imp"], d_["impx"], fill=True, color=cl, lw=0)
        else:
            ax.fill_between(cx, d_[nm], step="mid", color=cl, lw=0)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, float(f"{ymax / 1.1:.2g}")])
        if col % 2 == 0:
            ax.set_ylabel(lab, fontsize=6)
        if ri == 0:
            pk = pks[d_["t"]]
            pk = pk[(pk[0] == d_["c"]) & (pk[2] > d_["x"] - HALF) & (pk[1] < d_["x"] + HALF)]
            for s_, e_ in zip(pk[1], pk[2]):
                ax.axvspan(s_ - d_["x"], e_ - d_["x"], color=P.highlight, zorder=0, lw=0)
            ax.set_title(f"{d_['t']}\n{d_['c']}:{d_['x']:,}", fontsize=6)
        ax.set_xlim(-HALF, HALF)
for ax in axes[-1]:
    ax.set_xticks([-HALF, 0, HALF], [f"-{HALF // 1000} kb", "summit", f"+{HALF // 1000} kb"])
fig.tight_layout(h_pad=0.3, w_pad=0.6)
for ext in ("png", "pdf"):
    fig.savefig(O5 / f"browser_CTCF_ZNF282.{ext}", dpi=300)
print("done")
