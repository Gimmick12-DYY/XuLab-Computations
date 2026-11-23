#!/usr/bin/env python
"""Source data for the Fig 2 CTCF clustered heatmap example (paper_figures/fig2/panels/c_ctcf_clusters.py).
Sites: CTCF ChIP summits (ENCODE HEK293 ENCFF314ZAL, the Fig 2 reference) and our CTCF lab MACS2 summits (q < 0.05),
chr1-22 + X. An our-summit within 200 bp of a ChIP summit is "shared" (the ChIP summit coordinate is kept); the rest
are ChIP-only and ours-only.
Signals: bulk ChIP CPM (bigWig mean, 50 bp, +-1 kb); sc raw 5' ends CPM (50 bp, +-1 kb, 3-bin running mean); sc imputed
= Mdrop5-masked imputed fraction of cells per 1 kb bin (summit bin +-5, out-of-mask bins 0).
Clustering: k-means on two per-site features, log1p ChIP centre (mean CPM within +-200 bp) and log1p raw centre (CPM sum
within +-200 bp), each z-scored; k = 2-5 compared by inertia and silhouette (silhouette on a 20,000-site sample);
seed 0, n_init 50. (deepTools clusters on full profiles; our raw signal is too sparse per site for that.)
Sanity checks per cluster: CTCF motif within summit +-100 bp (FIMO p < 1e-4; HOCOMOCO v11 CTCF_HUMAN.H11MO.0.A, the
local motif database; JASPAR MA0139 is not on Longleaf), and the share of sites in the HEK293T ATAC union.
Writes paper_figures/fig2/source_data/ctcf_clusters.npz and ctcf_clusters_summary.tsv, ctcf_clusters_k.tsv."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import ATAC_UNION, FASTA, INP, MEME_DB, OUT, ROOT, bin_index, interval_mask, read_bed  # noqa: E402

SD = ROOT / "paper_figures/fig2/source_data"; TMP = OUT / "fig2_ctcf_clusters"; TMP.mkdir(parents=True, exist_ok=True)
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
W, STEP = 1000, 50; NB = 2 * W // STEP; CTR = slice(NB // 2 - 4, NB // 2 + 4)
bi, off, N = bin_index(); csize = dict(zip(bi.chrom, bi["size"]))
P = pd.read_csv(ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed", sep="\t", header=None, usecols=range(10))
P = P[P[0].isin(CH)]; chip = pd.DataFrame({"c": P[0].values, "s": (P[1] + P[9].clip(lower=0)).values.astype(int)})
L = pd.read_csv("/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.CTCF/TF.CTCF_peaks.narrowPeak", sep="\t", header=None, usecols=range(10))
L = L[L[0].isin(CH) & (L[8] >= 1.30103)]; ours = pd.DataFrame({"c": L[0].values, "s": (L[1] + L[9]).values.astype(int)})
chip_hit = np.zeros(len(chip), bool); ours_hit = np.zeros(len(ours), bool)
for c in CH:
    a = np.flatnonzero(chip.c.values == c); b = np.flatnonzero(ours.c.values == c)
    if not len(a) or not len(b):
        continue
    cs = np.sort(chip.s.values[a]); o = np.argsort(chip.s.values[a])
    x = ours.s.values[b]; j = np.searchsorted(cs, x)
    d = np.minimum(np.abs(x - cs[np.clip(j, 0, len(cs) - 1)]), np.abs(x - cs[np.clip(j - 1, 0, len(cs) - 1)]))
    ours_hit[b] = d <= 200
    y = chip.s.values[a]; os_ = np.sort(x); k = np.searchsorted(os_, y)
    d2 = np.minimum(np.abs(y - os_[np.clip(k, 0, len(os_) - 1)]), np.abs(y - os_[np.clip(k - 1, 0, len(os_) - 1)]))
    chip_hit[a] = d2 <= 200
S = pd.concat([chip.assign(origin=np.where(chip_hit, "shared", "ChIP only")), ours[~ours_hit].assign(origin="ours only")], ignore_index=True)
S = S[[x - W >= 0 and x + W <= csize[c] for c, x in zip(S.c, S.s)]].reset_index(drop=True)
print(S.origin.value_counts().to_dict(), flush=True)

bw = pyBigWig.open(str(OUT / "round6_diffpeaks/heatmaps/bw/CTCF.bw"))
CH50 = np.array([np.nan_to_num(np.array(bw.stats(c, int(x - W), int(x + W), nBins=NB, type="mean"), dtype=float)) for c, x in zip(S.c, S.s)])
z = np.load(INP / "five/CTCF.npz"); tot = sum(len(z[k]) for k in z.files); edges = np.arange(-W, W + 1, STEP)
R50 = np.zeros((len(S), NB))
for c in np.unique(S.c):
    m = np.flatnonzero(S.c.values == c); p = z[c]
    lo = np.searchsorted(p, S.s.values[m] - W); hi = np.searchsorted(p, S.s.values[m] + W)
    for r, a, b in zip(m, lo, hi):
        R50[r] = np.histogram(p[a:b] - S.s.values[r], edges)[0]
R50 = R50 / tot * 1e6
RAW = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, R50)
fr = np.load(INP / "frame.npz", allow_pickle=True); tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
imp = sp.load_npz(INP / "imp79_mdrop5.npz").tocsr()[tf79.index("CTCF")].toarray().ravel() / ncell[tf79.index("CTCF")]
mk = np.load(ROOT / "regulatome/m46_stage1_imputed/work/masks_extra.npz")["Mdrop5"]
reg = pd.read_csv(ROOT / "unified/work/nfya/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
kp = reg[0].isin(off).values; idx = np.array([off[c][0] if c in off else 0 for c in reg[0]]) + reg[1].astype(np.int64).values // 1000
fmask = np.zeros(N, bool); fmask[idx[kp]] = mk[kp]
o = np.array([off[c][0] for c in S.c]); nb = np.array([off[c][1] for c in S.c])
gb = np.clip(o[:, None] + (S.s.values // 1000)[:, None] + np.arange(-5, 6)[None, :], o[:, None], (o + nb - 1)[:, None])
IMP = np.where(fmask[gb], imp[gb], 0)

F = np.column_stack([np.log1p(CH50[:, CTR].mean(1)), np.log1p(R50[:, CTR].sum(1))])
Fz = (F - F.mean(0)) / F.std(0)
rng = np.random.default_rng(0); samp = rng.choice(len(Fz), min(20000, len(Fz)), replace=False)
krows = []
for k in range(2, 6):
    km = KMeans(k, random_state=0, n_init=50).fit(Fz)
    krows.append(dict(k=k, inertia=float(km.inertia_), silhouette=float(silhouette_score(Fz[samp], km.labels_[samp]))))
    print(krows[-1], flush=True)
K = pd.DataFrame(krows); K.to_csv(SD / "ctcf_clusters_k.tsv", sep="\t", index=False, float_format="%.4f")
km = KMeans(3, random_state=0, n_init=50).fit(Fz); lab = km.labels_
cen = pd.DataFrame({"lab": lab, "chip": Fz[:, 0], "raw": Fz[:, 1]}).groupby("lab").mean()
both = int((cen.chip + cen.raw).idxmax()); rest = [i for i in range(3) if i != both]
chip_only = max(rest, key=lambda i: cen.chip[i] - cen.raw[i]); ours_only = [i for i in rest if i != chip_only][0]
name = {both: "ChIP + ours", chip_only: "ChIP only", ours_only: "ours only"}

# motif and ATAC sanity checks
bed = TMP / "sites.bed"
pd.DataFrame({0: S.c, 1: S.s - 100, 2: S.s + 100, 3: [f"r{i}" for i in range(len(S))]}).to_csv(bed, sep="\t", header=False, index=False)
fa = subprocess.run(["bedtools", "getfasta", "-fi", str(FASTA), "-bed", str(bed), "-name"], check=True, capture_output=True, text=True).stdout
(TMP / "sites.fa").write_text("".join((l.split("::")[0] if l.startswith(">") else l) + "\n" for l in fa.splitlines()))
fo = subprocess.run(["fimo", "--text", "--thresh", "1e-4", "--motif", "CTCF_HUMAN.H11MO.0.A", str(MEME_DB), str(TMP / "sites.fa")],
                    check=True, capture_output=True, text=True).stdout
hit = set(l.split("\t")[2] for l in fo.splitlines()[1:] if l.strip())
motif = np.array([f"r{i}" in hit for i in range(len(S))])
am = interval_mask(read_bed(ATAC_UNION), off, N)
atac = am[o + S.s.values // 1000]
rows = []
for i in (both, chip_only, ours_only):
    m = lab == i
    rows.append(dict(cluster=name[i], n=int(m.sum()), mean_chip_centre_CPM=float(CH50[m][:, CTR].mean()), mean_raw_centre_CPM=float(R50[m][:, CTR].sum(1).mean()),
                     CTCF_motif_share=float(motif[m].mean()), ATAC_union_share=float(atac[m].mean()),
                     origin_shared=int((S.origin.values[m] == "shared").sum()), origin_ChIP_only=int((S.origin.values[m] == "ChIP only").sum()),
                     origin_ours_only=int((S.origin.values[m] == "ours only").sum())))
T = pd.DataFrame(rows); T.to_csv(SD / "ctcf_clusters_summary.tsv", sep="\t", index=False, float_format="%.4f")
print(T.round(3).to_string())
order = np.array([list(name).index(i) for i in lab])
np.savez_compressed(SD / "ctcf_clusters.npz", chip=CH50.astype(np.float32), raw=RAW.astype(np.float32), imp=IMP.astype(np.float32),
                    label=np.array([name[i] for i in lab]), chip_c=CH50[:, CTR].mean(1).astype(np.float32), raw_c=R50[:, CTR].sum(1).astype(np.float32),
                    origin=S.origin.values.astype(str))
print("done")
