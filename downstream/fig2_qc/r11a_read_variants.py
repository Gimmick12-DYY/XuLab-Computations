#!/usr/bin/env python
"""Round 5: per-TF summit counts under read-level filters (declared before results).
  L1 dedup: one read per (cell, chrom, 5' position, strand)
  L2 cell filter: keep cells whose fraction of reads in ATAC-union 1 kb bins is >= that TF's median (top half).
     The lab's sgRNA/barcode assignment confidence is not available (TF1000cells.meta.csv and the combined h5ad hold
     only TF labels), so that part is untested.
  L3 drop reads whose 5' end falls in a HOT or blacklisted 1 kb bin (r11_prep_hot.py)
Variants = all 8 on/off combinations (index bit0 = L1, bit1 = L2, bit2 = L3; 0 = no filter).
Counts at round-2 summits: C = 5' ends within +-100 bp, F = 1-2 kb flanks.
Usage: r11a_read_variants.py <array index 1..79>. Output OUT/round5_reads/per_tf/<TF>.npz."""
import sys

import numpy as np
import pandas as pd

from fig2_common import FRAG_DIR, INP, OUT, bin_index

O5 = OUT / "round5_reads"
(O5 / "per_tf").mkdir(parents=True, exist_ok=True)
tf79 = [str(t) for t in np.load(INP / "frame.npz", allow_pickle=True)["tf79"]]
tf = tf79[int(sys.argv[1]) - 1]
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
atac = fr["atac_union"]
hb = np.load(O5 / "hot_or_blacklist.npy")
S = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", usecols=["chrom", "pos"])
keep_ch = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]
d = pd.read_csv(FRAG_DIR / f"TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 3, 5],
                names=["chrom", "start", "end", "cell", "strand"], dtype={"chrom": "category", "cell": "category", "strand": "category"})
d = d[d.chrom.isin(keep_ch)].reset_index(drop=True)
five = np.where(d.strand.values == "+", d.start.values, d.end.values - 1).astype(np.int64)
gb = d.chrom.astype(str).map({c: off[c][0] for c in keep_ch}).values + five // 1000
dup = pd.DataFrame({"c": d.cell.cat.codes, "ch": d.chrom.cat.codes, "p": five, "s": d.strand.cat.codes}).duplicated().values
inatac = atac[gb]
frip = pd.Series(inatac).groupby(d.cell.cat.codes.values).mean()
cell_ok_codes = frip.index[frip.values >= frip.median()]
cellpass = np.isin(d.cell.cat.codes.values, cell_ok_codes)
hotbl = hb[gb]
masks = []
for v in range(8):
    m = np.ones(len(d), bool)
    if v & 1:
        m &= ~dup
    if v & 2:
        m &= cellpass
    if v & 4:
        m &= ~hotbl
    masks.append(m)
C = np.zeros((8, len(S)), np.int32)
F = np.zeros_like(C)
tot = np.zeros(8, np.int64)
chrom_arr = d.chrom.astype(str).values
for v, m in enumerate(masks):
    tot[v] = int(m.sum())
    for c in keep_ch[:-1]:
        idx = np.flatnonzero(S.chrom.values == c)
        if not len(idx):
            continue
        p = np.sort(five[m & (chrom_arr == c)])
        x = S.pos.values[idx]
        cnt = lambda a, b: np.searchsorted(p, x + b, side="right") - np.searchsorted(p, x + a, side="left")
        C[v, idx] = cnt(-100, 100)
        F[v, idx] = cnt(-2000, -1001) + cnt(1001, 2000)
np.savez(O5 / "per_tf" / f"{tf}.npz", C=C, F=F, tot=tot, n_reads=len(d), dup_rate=float(dup.mean()),
         n_cells=int(d.cell.nunique()), n_cells_kept=int(len(cell_ok_codes)), frip_median=float(frip.median()),
         frac_reads_hot_bl=float(hotbl.mean()))
print(tf, len(d), "dup", round(float(dup.mean()), 4), "cells", d.cell.nunique(), "kept", len(cell_ok_codes), "tot", tot.tolist())
