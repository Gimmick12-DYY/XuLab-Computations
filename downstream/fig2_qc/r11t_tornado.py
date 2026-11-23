#!/usr/bin/env python
"""Round 11 tornado source (user 2026-11-23 23:08Z). Rows = round 11 ChIP-differential regions (DESeq2 ~ChIP, padj<0.05,
LFC>1), per ChIP the top 1,000 by LFC that carry that ChIP's summit (all if fewer), centred on that summit, sorted within
the block by the block's own ChIP signal (mean of +-250 bp, high first). Profiles +-2 kb, 50 bp bins:
  ChIP  6 bulk ChIPs from the round 6 bigWigs (round6_diffpeaks/heatmaps/bw)
  raw   our 6 TFs, 5' end CPM per bin (full depth, per-TF total reads)
Imputed: same rows, +-10 kb at the native 1 kb bins, fraction of cells, Mdrop5 (outside the mask = NaN).
Peak/flank per (our TF, block): mean profile at +-100 bp over mean at 1.5-2 kb both sides (raw), centre bin over
bins 8-10 kb away (imputed). Writes paper_figures/fig2/source_data/tornado_chipdeseq2.npz, OUT/round11_chipdeseq2/tornado_peakflank.tsv."""
import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp

from r11_common import O11
from fig2_common import INP, OUT, ROOT, TFS, bin_index

SD = ROOT / "paper_figures/fig2/source_data"; TOPN, W, BW = 1000, 2000, 50; NB = 2 * W // BW
Rg = pd.read_csv(O11 / "regions.tsv", sep="\t")
D = {t: pd.read_csv(O11 / f"deseq2_{t}.tsv.gz", sep="\t") for t in TFS}
bws = {t: pyBigWig.open(str(OUT / "round6_diffpeaks/heatmaps/bw" / f"{t}.bw")) for t in TFS}
rows = []
for b, t in enumerate(TFS):
    d = np.flatnonzero((np.nan_to_num(D[t].padj.values, nan=1) < 0.05) & (D[t].lfc.values > 1))
    d = d[Rg[f"summit_{t}"].values[d] >= 0]
    d = d[np.argsort(-D[t].lfc.values[d], kind="stable")][:TOPN]
    c, s = Rg.chrom.values[d], Rg[f"summit_{t}"].values[d].astype(int)
    own = np.array([np.nan_to_num(bws[t].stats(ci, int(x - 250), int(x + 250), type="mean")[0] or 0) for ci, x in zip(c, s)])
    o = np.argsort(-own, kind="stable")
    rows.append(pd.DataFrame({"block": b, "chrom": c[o], "summit": s[o], "lfc": D[t].lfc.values[d][o]}))
R = pd.concat(rows, ignore_index=True)
c, s, blk = R.chrom.values, R.summit.values, R.block.values
chip = np.zeros((6, len(R), NB), np.float32)
for i, t in enumerate(TFS):
    for r in range(len(R)):
        chip[i, r] = np.nan_to_num(np.array(bws[t].stats(c[r], int(s[r] - W), int(s[r] + W), nBins=NB, type="mean"), float))
edges = np.arange(-W, W + 1, BW); raw = np.zeros((6, len(R), NB), np.float32)
for i, t in enumerate(TFS):
    z = np.load(INP / "five" / f"{t}.npz"); tot = sum(len(z[k]) for k in z.files)
    for ch in np.unique(c):
        if ch not in z.files:
            continue
        q = z[ch]; m = np.flatnonzero(c == ch)
        lo = np.searchsorted(q, s[m] - W); hi = np.searchsorted(q, s[m] + W)
        for r, a, e in zip(m, lo, hi):
            raw[i, r] = np.histogram(q[a:e] - s[r], edges)[0]
    raw[i] *= 1e6 / tot
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True); tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
mk = np.load(ROOT / "regulatome/m46_stage1_imputed/work/masks_extra.npz")["Mdrop5"]
reg = pd.read_csv(ROOT / "unified/work/nfya/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
kp = reg[0].isin(off).values; idx = np.array([off[x][0] if x in off else 0 for x in reg[0]]) + reg[1].astype(np.int64).values // 1000
fmask = np.zeros(N, bool); fmask[idx[kp]] = mk[kp]
b0 = np.array([off[x][0] for x in c]) + s // 1000; lo0 = np.array([off[x][0] for x in c]); nb = np.array([off[x][1] for x in c])
hb = b0[:, None] + np.arange(-10, 11)[None, :]; ok = (hb >= lo0[:, None]) & (hb < (lo0 + nb)[:, None]); hb = np.clip(hb, 0, N - 1)
IMP = sp.load_npz(INP / "imp79_mdrop5.npz").tocsr()
imp = np.zeros((6, len(R), 21), np.float32)
for i, t in enumerate(TFS):
    v = IMP[tf79.index(t)].toarray().ravel() / ncell[tf79.index(t)]
    imp[i] = np.where(ok & fmask[hb], v[hb], np.nan)
ctr, fl = slice(NB // 2 - 2, NB // 2 + 2), np.r_[0:10, NB - 10:NB]
pf = []
for i, t in enumerate(TFS):
    for b, u in enumerate(TFS):
        m = blk == b; pr = raw[i, m].mean(0); ip = np.nanmean(imp[i, m], 0)
        pf.append(dict(ours=t, block=u, n=int(m.sum()), raw_peak_over_flank=float(pr[ctr].mean() / max(pr[fl].mean(), 1e-9)),
                       imp_centre_over_flank=float(ip[10] / max(np.nanmean(ip[np.r_[0:3, 18:21]]), 1e-9))))
P = pd.DataFrame(pf); P.to_csv(O11 / "tornado_peakflank.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 200)
print(P.pivot(index="ours", columns="block", values="raw_peak_over_flank").loc[TFS, TFS].round(2).to_string())
print(P.pivot(index="ours", columns="block", values="imp_centre_over_flank").loc[TFS, TFS].round(2).to_string())
np.savez_compressed(SD / "tornado_chipdeseq2.npz", tfs=np.array(TFS), block=blk, chip=chip, raw=raw, imp=imp,
                    block_n=np.array([int((blk == b).sum()) for b in range(6)]))
print("rows", len(R), "per block", np.bincount(blk)); print("done")
