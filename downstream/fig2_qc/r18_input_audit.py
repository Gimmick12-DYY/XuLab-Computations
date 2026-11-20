#!/usr/bin/env python
"""Fig 2 input audit (user 2026-11-20: "your input is strange, why negative correlation?"). Outputs OUT/input_audit/.
A1 raw identity: fragment file per TF, read count, md5 of the first 1,000 lines; duplicate-file check over all 79;
   5' cache total (inputs/five) vs tf_stats reads; raw.npz row order vs tf_stats (asserted in 00_build_inputs).
A2 imputed labels: for every TF, Pearson of its imputed track (cells positive / n cells) with each of the 79 raw tracks
   (log2 CPM+1 per 1 kb bin) over the ATAC-union bins; rank of its own raw track (1 = best).
A3 bulk identity: BAM header chr1 length (hg38 = 248,956,422), chr naming, reads on chr1 (idxstats), ENCODE target label.
B1 raw 5' end profiles (CPM, +-1 kb, 20 bp bins) at each reference's top 5,000 summits (signalValue) for CTCF, MAZ, ZNF282,
   own TF vs 79-TF pool (pool scaled to the same total).
B2 genome-wide 10 kb bins (chr1-22, X; blacklist bins dropped): raw pseudobulk log2 CPM+1 vs ChIP log2 CPM+1, 6x6 Pearson.
B3 raw CTCF vs CTCF ChIP on U1 union peaks (r15), inside vs outside the ATAC union.
C  negative cells: imputed CTCF vs CTCF ChIP on U1 (6c method a), and raw CTCF for comparison; scatter by promoter (peak
   within 1 kb of a GENCODE v46 TSS) vs distal; r per stratum."""
import gzip
import hashlib
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp
from matplotlib import font_manager

from fig2_common import BAM, BLACKLIST, COUNTS, FRAG_DIR, GTF, INP, OUT, REF, ROOT, TFS, bin_index, read_bed, interval_mask

OA = OUT / "input_audit"
OA.mkdir(parents=True, exist_ok=True)
OH = OUT / "round6_diffpeaks" / "heatmaps"
PEAKS = dict(REF)
PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
log = open(OA / "audit_log.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()


def sh(c):
    return subprocess.run(c, shell=True, check=True, capture_output=True, text=True, executable="/bin/bash").stdout


bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]
atac = fr["atac_union"].astype(bool)
ncell = fr["ncells_imp79"]
stats = pd.read_csv(COUNTS / "tf_stats.tsv", sep="\t")
raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
assert list(stats.tf) == tf79, "tf_stats order != frame tf79"

# ---------------- A1 ----------------
say("== A1 raw read files")
md5s, recs = {}, []
for t in tf79:
    f = FRAG_DIR / f"TF.{t}.bed.gz"
    h = hashlib.md5(sh(f"zcat {f} | head -1000").encode()).hexdigest()
    md5s.setdefault(h, []).append(t)
    if t in TFS:
        z = np.load(INP / "five" / f"{t}.npz")
        n5 = sum(len(z[c]) for c in z.files)
        recs.append(dict(TF=t, file=str(f), md5_first1000=h, reads_in_5p_cache=n5,
                         reads_tf_stats=int(stats.reads[stats.tf == t].iloc[0]),
                         raw_npz_row_sum=int(raw[tf79.index(t)].sum())))
        say(recs[-1])
dups = {h: v for h, v in md5s.items() if len(v) > 1}
say("files sharing identical first 1,000 lines:", dups if dups else "none (all 79 distinct)")
A1 = pd.DataFrame(recs)

# ---------------- A2 ----------------
say("== A2 imputed track vs 79 raw tracks at ATAC bins")
imp = sp.load_npz(INP / "imp79.npz").tocsr()
ia = np.flatnonzero(atac)
R = raw[:, ia].toarray().astype(np.float32)
R = np.log2(R / R.sum(1, keepdims=True) * 1e6 + 1)
Rz = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-12)
rank_rows, own_best = [], 0
for i, t in enumerate(tf79):
    v = imp[i, ia].toarray().ravel() / ncell[i]
    vz = (v - v.mean()) / (v.std() + 1e-12)
    r = Rz @ vz / len(vz)
    rk = int(1 + (r > r[i]).sum())
    own_best += rk == 1
    if t in TFS:
        top = np.argsort(-r)[:3]
        rank_rows.append(dict(TF=t, r_own=float(r[i]), own_rank_of79=rk, best_raw=tf79[top[0]], r_best=float(r[top[0]]),
                              second=tf79[top[1]], third=tf79[top[2]]))
        say(rank_rows[-1])
say(f"imputed tracks whose own raw track is the best of 79: {own_best}/79")
A2 = pd.DataFrame(rank_rows)
del R, Rz

# ---------------- A3 ----------------
say("== A3 bulk BAMs")
enc_acc = {"CTCF": "ENCSR000DTW", "MAZ": "ENCSR290SSQ", "ZBTB7A": "ENCSR773REP", "ZIC2": "ENCSR728MWW", "ZNF777": "ENCSR295BIP"}
a3 = []
for t in TFS:
    hdr = sh(f"samtools view -H {BAM[t]} | grep '^@SQ' | head -30")
    chr1 = [l for l in hdr.splitlines() if "SN:chr1\t" in l + "\t"]
    L = chr1[0].split("LN:")[1].split()[0] if chr1 else "no chr1"
    idx = sh(f"samtools idxstats {BAM[t]} | awk '$1==\"chr1\"{{print $3}}'").strip()
    tgt = "GEO GSM2466515 (HA-ZNF282, not ENCODE)"
    if t in enc_acc:
        d = json.load(open(OUT / "round6_diffpeaks/chip_selfcheck/encode_json" / f"{enc_acc[t]}.json"))
        tgt = f"{enc_acc[t]} target={d.get('target', {}).get('label')} biosample={d.get('biosample_summary', '')[:40]}"
    a3.append(dict(TF=t, bam=str(BAM[t]), chr1_length=L, hg38=(L == "248956422"), chr1_reads=int(idx or 0), source=tgt))
    say(a3[-1])
A3 = pd.DataFrame(a3)

# ---------------- B1 ----------------
say("== B1 raw profiles at own top 5,000 summits")
five = {t: dict(np.load(INP / "five" / f"{t}.npz")) for t in TFS}
tot = {t: sum(len(v) for v in five[t].values()) for t in TFS}
pool5, ptot = {}, 0
for t in tf79:
    z = np.load(INP / "five" / f"{t}.npz")
    for c in z.files:
        pool5.setdefault(c, []).append(z[c])
    ptot += sum(len(z[c]) for c in z.files)
pool5 = {c: np.sort(np.concatenate(v)) for c, v in pool5.items()}
W, ST = 1000, 20
edges = np.arange(-W, W + 1, ST); xs = (edges[:-1] + edges[1:]) / 2


def prof(arr, total, chrom, pos):
    h = np.zeros(len(edges) - 1)
    for c in np.unique(chrom):
        p = arr.get(c)
        if p is None:
            continue
        q = pos[chrom == c]
        lo = np.searchsorted(p, q - W); hi = np.searchsorted(p, q + W)
        for a, b, x in zip(lo, hi, q):
            h += np.histogram(p[a:b] - x, edges)[0]
    return h / len(pos) / total * 1e6


def top_summits(t, n=5000):
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    P = P.sort_values(6, ascending=False).head(n)
    return P[0].values, (P[1] + P[9].clip(lower=0)).values.astype(int)


B1 = {}
for t in ("CTCF", "MAZ", "ZNF282"):
    c, p = top_summits(t)
    B1[t] = dict(own=prof(five[t], tot[t], c, p), pool=prof(pool5, ptot, c, p))
    ctr = slice(len(xs) // 2 - 5, len(xs) // 2 + 5); fl = np.r_[0:10, len(xs) - 10:len(xs)]
    say(t, "own c/f", round(B1[t]["own"][ctr].mean() / B1[t]["own"][fl].mean(), 2),
        "pool c/f", round(B1[t]["pool"][ctr].mean() / B1[t]["pool"][fl].mean(), 2),
        "own/pool at centre", round(B1[t]["own"][ctr].mean() / B1[t]["pool"][ctr].mean(), 2))

# ---------------- B2 ----------------
say("== B2 genome-wide 10 kb bins")
bl = read_bed(BLACKLIST)
rows_b, rows_s = [], []
keep_all = []
for c in CH:
    o, nb = off[c]
    n10 = (nb + 9) // 10
    m1 = interval_mask(bl, {c: (0, nb)}, nb)          # 1 kb blacklist bins on this chrom
    bad = np.add.reduceat(m1.astype(int), np.arange(0, nb, 10)) > 0
    Sraw = np.vstack([np.add.reduceat(raw[tf79.index(t), o:o + nb].toarray().ravel(), np.arange(0, nb, 10)) for t in TFS]).T
    Bc = np.zeros((n10, 6))
    for k, t in enumerate(TFS):
        bw = pyBigWig.open(str(OH / "bw" / f"{t}.bw"))
        L = bw.chroms()[c]
        v = np.array(bw.stats(c, 0, L, nBins=n10, type="mean"), dtype=float)
        Bc[:, k] = np.nan_to_num(v)
    rows_s.append(Sraw[~bad]); rows_b.append(Bc[~bad])
S10 = np.vstack(rows_s).astype(float); B10 = np.vstack(rows_b)
ls_ = np.log2(S10 / S10.sum(0) * 1e6 + 1); lb_ = np.log2(B10 / B10.sum(0) * 1e6 + 1)
M10 = np.corrcoef(np.column_stack([lb_, ls_]).T)[6:, :6]
say("bins", len(S10)); say(pd.DataFrame(M10, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).round(3).to_string())

# ---------------- B3 / C ----------------
say("== B3/C union peaks U1")
U = pd.read_csv(OUT / "round6_diffpeaks/union_6c/tmp/U1.bed", sep="\t", header=None)
uc, us, ue = U[0].values, U[1].values, U[2].values
o = np.array([off[c][0] for c in uc]); a = o + us // 1000; b = o + (ue - 1) // 1000
acs = np.r_[0, np.cumsum(atac.astype(int))]
inA = (acs[b + 1] - acs[a]) > 0
k = TFS.index("CTCF")
Craw = np.zeros(len(U))
for c in np.unique(uc):
    m = np.flatnonzero(uc == c); p = five["CTCF"][c]
    Craw[m] = np.searchsorted(p, ue[m]) - np.searchsorted(p, us[m])
bw = pyBigWig.open(str(OH / "bw" / "CTCF.bw"))
Cchip = np.array([np.nan_to_num(bw.stats(c, int(s), int(e), type="mean")[0] or 0) * (e - s) for c, s, e in zip(uc, us, ue)])
ics = np.r_[0, np.cumsum(imp[tf79.index("CTCF")].toarray().ravel().astype(float))]
Cimp = ics[b + 1] - ics[a]
tss = {}
with gzip.open(GTF, "rt") as fh:
    for ln in fh:
        if ln[0] == "#":
            continue
        f = ln.split("\t", 8)
        if f[2] == "transcript":
            tss.setdefault(f[0], []).append(int(f[3]) if f[6] == "+" else int(f[4]))
tss = {c: np.unique(v) for c, v in tss.items()}
prom = np.zeros(len(U), bool)
for c in np.unique(uc):
    m = np.flatnonzero(uc == c); v = tss.get(c)
    if v is None:
        continue
    i = np.searchsorted(v, us[m] - 1000)
    prom[m] = (i < len(v)) & (v[np.clip(i, 0, len(v) - 1)] <= ue[m] + 1000)
lcpm = lambda x: np.log2(x / x.sum() * 1e6 + 1)
X_chip, X_raw, X_imp = lcpm(Cchip), lcpm(Craw), lcpm(Cimp)
res = []
for lab, m in (("all", np.ones(len(U), bool)), ("inside ATAC", inA), ("outside ATAC", ~inA), ("promoter", prom), ("distal", ~prom),
               ("inside ATAC & promoter", inA & prom), ("inside ATAC & distal", inA & ~prom)):
    r_raw = np.corrcoef(X_chip[m], X_raw[m])[0, 1]
    mi = m & inA
    r_imp = np.corrcoef(X_chip[mi], X_imp[mi])[0, 1] if mi.sum() > 10 else np.nan
    res.append(dict(stratum=lab, n=int(m.sum()), r_rawCTCF_vs_CTCFChIP=r_raw, n_imp=int(mi.sum()), r_impCTCF_vs_CTCFChIP=r_imp,
                    mean_CTCFChIP=X_chip[m].mean(), mean_rawCTCF=X_raw[m].mean(), mean_impCTCF=X_imp[mi].mean() if mi.any() else np.nan))
B3 = pd.DataFrame(res); say(B3.round(3).to_string())

# ---------------- save ----------------
with pd.ExcelWriter(OA / "input_audit.xlsx") as xw:
    A1.to_excel(xw, sheet_name="A1_raw_files", index=False)
    pd.DataFrame([{"duplicate_first1000": json.dumps(dups) if dups else "none"}]).to_excel(xw, sheet_name="A1_duplicates", index=False)
    A2.to_excel(xw, sheet_name="A2_imputed_labels", index=False)
    A3.to_excel(xw, sheet_name="A3_bulk", index=False)
    pd.DataFrame(M10, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).to_excel(xw, sheet_name="B2_10kb_pearson")
    B3.to_excel(xw, sheet_name="B3_C_union_strata", index=False)

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
plt.rcParams.update({"font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.5})
MM = 1 / 25.4
# B1 profiles
fig, axs = plt.subplots(1, 3, figsize=(150 * MM, 45 * MM), layout="constrained")
for ax, t in zip(axs, ("CTCF", "MAZ", "ZNF282")):
    ax.plot(xs, B1[t]["own"], color=P.neutral.ink, lw=0.8, label=f"{t} sc raw")
    ax.plot(xs, B1[t]["pool"], color=P.neutral.mid, lw=0.8, label="79-TF pool")
    ax.set_title(f"at top 5,000 {t} ChIP summits", fontsize=7); ax.set_xlabel("distance (bp)"); ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=6)
axs[0].set_ylabel("5' ends, CPM per site")
for e in ("png", "pdf", "svg"):
    fig.savefig(OA / f"B1_raw_profiles.{e}", dpi=600)
plt.close(fig)
# B2 matrix
fig, ax = plt.subplots(figsize=(70 * MM, 62 * MM), layout="constrained")
im = ax.imshow(M10, cmap=P.div_cmap(), vmin=-1, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{M10[i, j]:.2f}", ha="center", va="center", fontsize=6)
ax.set_xticks(range(6), [f"{t} ChIP" for t in TFS], rotation=90); ax.set_yticks(range(6), [f"{t} sc raw" for t in TFS])
ax.set_title("Genome-wide 10 kb bins, Pearson of log2 CPM", fontsize=7); fig.colorbar(im, ax=ax, fraction=0.046)
for e in ("png", "pdf", "svg"):
    fig.savefig(OA / f"B2_10kb_pearson.{e}", dpi=600)
plt.close(fig)
# C scatter
fig, axs = plt.subplots(1, 2, figsize=(130 * MM, 60 * MM), layout="constrained")
for ax, (Y, lab, m) in zip(axs, ((X_raw, "CTCF sc raw", np.ones(len(U), bool)), (X_imp, "CTCF sc imputed (ATAC-mask peaks)", inA))):
    for mm, col, nm in ((m & ~prom, P.neutral.mid, "distal"), (m & prom, "#C0392B", "promoter")):
        r = np.corrcoef(X_chip[mm], Y[mm])[0, 1]
        ax.scatter(X_chip[mm], Y[mm], s=0.3, alpha=0.25, color=col, rasterized=True, label=f"{nm} (n={mm.sum():,}, r={r:.2f})")
    r = np.corrcoef(X_chip[m], Y[m])[0, 1]
    ax.set_title(f"{lab}; all r = {r:.2f}", fontsize=7); ax.set_xlabel("CTCF ChIP log2(CPM+1)"); ax.set_ylabel(f"{lab} log2(CPM+1)")
    ax.legend(frameon=False, fontsize=6, markerscale=10); ax.spines[["top", "right"]].set_visible(False)
for e in ("png", "pdf", "svg"):
    fig.savefig(OA / f"C_ctcf_scatter_U1.{e}", dpi=600)
plt.close(fig)
say("done")
