#!/usr/bin/env python
"""Fig 2 heatmaps on the native 1 kb grid (adds arrays to paper_figures/fig2/source_data/heatmaps_<TF>.npz).
Rows = the V3 rows (the reference's top 10,000 ChIP summits, sorted by ChIP centre signal, same order as r21).
Window = +-10 kb: the 1 kb genomic bin containing the summit plus 10 bins on each side (21 bins).
  K_ChIP     bulk ChIP CPM (bamCoverage bigWig mean over each 1 kb tile)
  K_raw      sc raw 5' ends per 1 kb tile, CPM (ctcf_promoter_fp/counts/raw.npz)
  K_impA     current imputed (impute/matrix_csr.npz: ATAC-union mask on model calls, raw-positive cells kept),
             fraction of cells positive; K_maskA = bin inside the ATAC union (model calls allowed there)
  K_impB     genome-wide pre-mask imputed (impute/matrix_csr.npz.pre_openmask), fraction of cells positive
Both imputed versions include the kept raw-positive cells (02_impute.py keep_raw=True).
Also writes heatmap_mask_fraction.tsv: share of rows whose summit bin is outside the ATAC union."""
import sys

import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import COUNTS, INP, OUT, REF, ROOT, TFS, bin_index  # noqa: E402

SD = ROOT / "paper_figures/fig2/source_data"
U = ROOT / "unified/work"
W, NB = 1000, 40
CTR = slice(NB // 2 - 4, NB // 2 + 4)
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
bi, off, N = bin_index(); csize = dict(zip(bi.chrom, bi["size"]))
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]; atac = fr["atac_union"].astype(bool); ncell = fr["ncells_imp79"]
stats = pd.read_csv(COUNTS / "tf_stats.tsv", sep="\t"); raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
impA = sp.load_npz(INP / "imp79.npz").tocsr()
K = np.arange(-10, 11)
OH = OUT / "round6_diffpeaks/heatmaps"
rows = []
for t in TFS:
    z = dict(np.load(SD / f"heatmaps_{t}.npz"))
    # rebuild the V3 row order exactly as r21 (same filter, same sort)
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    sm = (P[1] + P[9].clip(lower=0)).values
    ok = np.array([ch in csize and x - W >= 0 and x + W <= csize[ch] for ch, x in zip(P[0].values, sm)])
    pc, ps, sig = P[0].values[ok], sm[ok].astype(int), P[6].values[ok]
    top = np.argsort(-sig, kind="stable")[:10000]
    bw = pyBigWig.open(str(OH / "bw" / f"{t}.bw"))
    cen = np.array([np.nan_to_num(np.array(bw.stats(c, int(x - W), int(x + W), nBins=NB, type="mean"), dtype=float))[CTR].mean()
                    for c, x in zip(pc[top], ps[top])])
    o3 = top[np.argsort(-cen, kind="stable")]
    chk = np.array([np.nan_to_num(np.array(bw.stats(c, int(x - W), int(x + W), nBins=NB, type="mean"), dtype=float))
                    for c, x in zip(pc[o3[:200]], ps[o3[:200]])])
    assert np.allclose(chk, z["V3_ChIP"][:200].astype(float), atol=0.05), f"{t}: V3 row order mismatch"
    c3, s3 = pc[o3], ps[o3]
    gb = np.array([off[c][0] for c in c3])[:, None] + (s3 // 1000)[:, None] + K[None, :]
    lo = np.array([off[c][0] for c in c3])[:, None]; hi = lo + np.array([off[c][1] for c in c3])[:, None]
    inside = (gb >= lo) & (gb < hi); gbc = np.clip(gb, 0, N - 1)
    # ChIP CPM per 1 kb tile
    KC = np.zeros(gb.shape)
    for r, (c, x) in enumerate(zip(c3, s3)):
        b0 = (x // 1000 - 10) * 1000; b1 = (x // 1000 + 11) * 1000
        a, e = max(0, b0), min(csize[c], b1)
        v = np.nan_to_num(np.array(bw.stats(c, int(a), int(e), nBins=(e - a) // 1000, type="mean"), dtype=float)) if e > a else np.zeros(0)
        KC[r, (a - b0) // 1000:(a - b0) // 1000 + len(v)] = v
    k = tf79.index(t)
    rv = raw[list(stats.tf).index(t)].toarray().ravel().astype(float); rv = rv / rv.sum() * 1e6
    KR = np.where(inside, rv[gbc], 0)
    av = impA[k].toarray().ravel() / ncell[k]
    KA = np.where(inside, av[gbc], 0); MA = np.where(inside, atac[gbc], False)
    # pre-mask imputed, mapped to the frame
    reg = pd.read_csv(U / t.lower() / "impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
    keep = reg[0].isin(off).values
    idx = np.array([off[c][0] if c in off else 0 for c in reg[0]]) + reg[1].astype(np.int64).values // 1000
    m = sp.load_npz(U / t.lower() / "impute/matrix_csr.npz.pre_openmask").tocsr(); m.data[:] = 1
    if m.shape[0] != len(reg):
        m = m.T.tocsr()
    pre = np.zeros(N); pre[idx[keep]] = np.asarray(m.sum(1)).ravel()[keep] / m.shape[1]
    KB = np.where(inside, pre[gbc], 0)
    z.update(K_ChIP=KC.astype(np.float32), K_raw=KR.astype(np.float32), K_impA=KA.astype(np.float32),
             K_maskA=MA, K_impB=KB.astype(np.float32), K_chrom=c3.astype(str), K_summit=s3.astype(np.int64))
    np.savez_compressed(SD / f"heatmaps_{t}.npz", **z)
    rows.append(dict(TF=t, rows=len(o3), summit_bin_outside_ATAC_union=float(1 - MA[:, 10].mean()),
                     all_21_bins_outside=float((~MA).all(1).mean()), impA_summit_bin_zero=float((KA[:, 10] == 0).mean()),
                     impB_summit_bin_zero=float((KB[:, 10] == 0).mean())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(SD / "heatmap_mask_fraction.tsv", sep="\t", index=False, float_format="%.4f")
print("done")
