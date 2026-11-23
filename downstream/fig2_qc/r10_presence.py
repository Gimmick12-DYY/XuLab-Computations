#!/usr/bin/env python
"""Round 10 (spec round10_presence/SPEC.md): peak-presence rows from 4 bulk ChIPs and our data at the same rows.
Writes paper_figures/fig2/source_data/presence_<set>.npz and presence_<set>_blocks.tsv for set in main (CTCF, ZBTB7A,
ZIC2, ZNF282) and znf777 (CTCF, ZBTB7A, ZIC2, ZNF777)."""
import itertools
import subprocess
import sys

import numpy as np
import pandas as pd
import pyBigWig
import scipy.sparse as sp

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BLACKLIST, INP, OUT, REF, ROOT, bin_index  # noqa: E402

SD = ROOT / "paper_figures/fig2/source_data"; TMP = OUT / "round10_presence"; TMP.mkdir(parents=True, exist_ok=True)
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
LAB = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True); tf79 = [str(t) for t in fr["tf79"]]; ncell = fr["ncells_imp79"]
mk = np.load(ROOT / "regulatome/m46_stage1_imputed/work/masks_extra.npz")["Mdrop5"]
reg = pd.read_csv(ROOT / "unified/work/nfya/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
kp = reg[0].isin(off).values; idx = np.array([off[c][0] if c in off else 0 for c in reg[0]]) + reg[1].astype(np.int64).values // 1000
fmask = np.zeros(N, bool); fmask[idx[kp]] = mk[kp]
IMP = sp.load_npz(INP / "imp79_mdrop5.npz").tocsr()
five = {}
def load5(t):
    if t not in five:
        z = np.load(INP / "five" / f"{t}.npz"); five[t] = {c: z[c] for c in z.files}
    return five[t]
TOT = {t: sum(len(v) for v in load5(t).values()) for t in tf79}
POOL = {}
for t in tf79:
    for c, v in load5(t).items():
        POOL.setdefault(c, []).append(v)
POOL = {c: np.sort(np.concatenate(v)) for c, v in POOL.items()}; PTOT = sum(len(v) for v in POOL.values())
five.clear()


def sh(c):
    return subprocess.run(c, shell=True, check=True, capture_output=True, text=True, executable="/bin/bash").stdout


def summit_bed(t, path):
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    s = P[1] + P[9].clip(lower=0)
    pd.DataFrame({0: P[0], 1: (s - 250).clip(lower=0), 2: s + 250}).sort_values([0, 1]).to_csv(path, sep="\t", header=False, index=False)


def overlap(rows_bed, other_bed):
    out = sh(f"bedtools intersect -c -a {rows_bed} -b {other_bed}")
    return np.array([int(l.split("\t")[-1]) > 0 for l in out.splitlines()])


def counts(d, c, s, e):
    x = np.zeros(len(c))
    for ch in np.unique(c):
        m = np.flatnonzero(c == ch); p = d.get(ch)
        if p is None:
            continue
        x[m] = np.searchsorted(p, e[m]) - np.searchsorted(p, s[m])
    return x


for setname, tfs in (("main", ["CTCF", "ZBTB7A", "ZIC2", "ZNF282"]), ("znf777", ["CTCF", "ZBTB7A", "ZIC2", "ZNF777"])):
    beds = {}
    for t in tfs:
        beds[t] = TMP / f"chip_{t}.bed"; summit_bed(t, beds[t])
    sh(f"cat {' '.join(str(beds[t]) for t in tfs)} | sort -k1,1 -k2,2n | bedtools merge -i - | bedtools intersect -v -a - -b {BLACKLIST} > {TMP / f'rows_{setname}.bed'}")
    R = pd.read_csv(TMP / f"rows_{setname}.bed", sep="\t", header=None)
    chip = np.column_stack([overlap(TMP / f"rows_{setname}.bed", beds[t]) for t in tfs])
    keep = ~chip.all(1); R = R[keep].reset_index(drop=True); chip = chip[keep]
    R.to_csv(TMP / f"rows_{setname}_kept.bed", sep="\t", header=False, index=False)
    c, s, e = R[0].values, R[1].values.astype(int), R[2].values.astype(int)
    sig = np.zeros(len(R))
    for t in tfs:
        bw = pyBigWig.open(str(OUT / "round6_diffpeaks/heatmaps/bw" / f"{t}.bw"))
        sig += np.array([np.nan_to_num(bw.stats(ci, int(a), int(b), type="mean")[0] or 0) * (b - a) for ci, a, b in zip(c, s, e)])
    labp = []
    for t in tfs:
        L = pd.read_csv(LAB.format(t=t), sep="\t", header=None, usecols=range(9)); L = L[L[0].isin(CH) & (L[8] >= 1.30103)]
        L[[0, 1, 2]].sort_values([0, 1]).to_csv(TMP / f"lab_{t}.bed", sep="\t", header=False, index=False)
        labp.append(overlap(TMP / f"rows_{setname}_kept.bed", TMP / f"lab_{t}.bed"))
    labp = np.column_stack(labp)
    enr = []
    pool_x = counts(POOL, c, s, e)
    for t in tfs:
        x = counts(load5(t), c, s, e); five.clear()
        p = pool_x - x
        enr.append(np.log2((x / TOT[t] * 1e6 + 1) / (p / (PTOT - TOT[t]) * 1e6 + 1)))   # CPM + 1 (spec correction)
    enr = np.column_stack(enr)
    mid = np.array([off[ci][0] for ci in c]) + ((s + e) // 2) // 1000
    impp = []
    for t in tfs:
        v = IMP[tf79.index(t)].toarray().ravel() / ncell[tf79.index(t)]
        thr = np.percentile(v[fmask], 95)
        impp.append(fmask[mid] & (v[mid] >= thr))
    impp = np.column_stack(impp)
    # pattern blocks
    pats = []
    for k in (1, 2, 3):
        for comb in itertools.combinations(range(4), k):
            pats.append(comb)
    code = np.array(["+".join(tfs[i] for i in range(4) if chip[r, i]) for r in range(len(R))])
    order, blocks, rows = [], [], []
    for comb in pats:
        name = "+".join(tfs[i] for i in comb) + (" only" if len(comb) == 1 else "")
        m = np.flatnonzero(code == "+".join(tfs[i] for i in comb))
        if not len(m):
            continue
        m = m[np.argsort(-sig[m], kind="stable")]
        order.append(m); blocks.append((name, len(m)))
        rows.append(dict(block=name, n=len(m), **{f"signal_{t}": float(enr[m, j].mean()) for j, t in enumerate(tfs)},
                         **{f"ourpeak_{t}": float(labp[m, j].mean()) for j, t in enumerate(tfs)},
                         **{f"imputed_{t}": float(impp[m, j].mean()) for j, t in enumerate(tfs)}))
    o = np.concatenate(order)
    T = pd.DataFrame(rows); T.to_csv(SD / f"presence_{setname}_blocks.tsv", sep="\t", index=False, float_format="%.4f")
    np.savez_compressed(SD / f"presence_{setname}.npz", tfs=np.array(tfs), chip=chip[o], labp=labp[o], enr=enr[o].astype(np.float32),
                        impp=impp[o], block_names=np.array([b[0] for b in blocks]), block_n=np.array([b[1] for b in blocks]))
    pd.set_option("display.width", 250)
    print(setname, "rows before removing all-4:", int(keep.size), "removed (all 4):", int((~keep).sum()), "kept:", len(R))
    print(T.round(3).to_string(), flush=True)
print("done")
