# m46 Amendment 3: as Amendment 1 (m46_macs_call.py) but the imputed-only calls restricted to a mask (raw kept). usage: m46_macs_mask.py <mask> <TF index>
# per TF, imputed pseudobulk -> MACS3 peaks (P1 no control, P2 ATAC lambda + qpois, P3 pseudo-fragment callpeak with/without ATAC control) and local contrast at lab summits. usage: m46_macs_call.py <TF index 0..62>
import os, sys, json, subprocess, gzip
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; U = f"{X}/unified/work"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; M = f"{X}/regulatome/m46_stage1_imputed"; W = f"{M}/work"; PK = f"{M}/peaks"
ATAC_TA = f"{X}/regulatome/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"; MACS = "macs3"
MK = sys.argv[1]; PK = f"{M}/peaks_{MK}"; TFS = [l.strip() for l in open(f"{W}/tfs_mask_ok.txt") if l.strip()]; t = TFS[int(sys.argv[2])]; tl = t.lower(); wd = f"{W}/{MK}_{t}"; os.makedirs(wd, exist_ok=True); os.makedirs(f"{PK}/{t}", exist_ok=True)
run = lambda c: subprocess.run(c, shell=True, check=True, executable="/bin/bash")
reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]; R = pd.DataFrame({"c": [r.split(":")[0] for r in reg], "s": [int(r.split(":")[1].split("-")[0]) for r in reg]}); R["e"] = R.s + 1000
OFF = {c: (g.index.values[0], g.s.values[0], len(g)) for c, g in R.groupby("c", sort=False)}
for c, g in R.groupby("c", sort=False): assert (np.diff(g.s.values) == 1000).all() and (np.diff(g.index.values) == 1).all(), c
def bidx(c, pos):  # region index of genomic positions (vectorised; -1 if outside the tiling)
    out = np.full(len(pos), -1); c = np.asarray(c); pos = np.asarray(pos)
    for ch in np.unique(c):
        if ch not in OFF: continue
        i0, s0, n = OFF[ch]; mm = c == ch; k = (pos[mm] - s0) // 1000; ok = (k >= 0) & (k < n); r = np.full(mm.sum(), -1); r[ok] = i0 + k[ok]; out[mm] = r
    return out
own = [l.strip() for l in open(f"{U}/{tl}/impute/regions.tsv")]; assert own == reg, "region order differs"
import scipy.io as sio
mask = np.load(f"{W}/masks.npz")[MK]
raw = sio.mmread(f"{U}/{tl}/mm/matrix.mtx.gz").tocsr(); raw.data[:] = 1; pre = sp.load_npz(f"{U}/{tl}/impute/matrix_csr.npz.pre_openmask").tocsr(); pre.data[:] = 1
only = (pre - pre.multiply(raw)).tocsr(); only.eliminate_zeros(); m = (only.multiply(mask[:, None]) + raw).tocsr(); m.data[:] = 1; del raw, pre, only
src = f"pre_openmask, mask {MK}"
v = np.asarray(m.sum(1)).ravel().astype(float); info = {"TF": t, "mask": MK, "matrix used": src, "cells": m.shape[1], "positive cell x bin calls": int(m.nnz), "bins with >= 1 cell": int((v > 0).sum())}
# ATAC per bin (shared cache)
AC = f"{W}/atac_per_bin.npy"
if False:
    a = np.zeros(len(R))
    for ch in pd.read_csv(ATAC_TA, sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"], chunksize=5_000_000):
        pos = np.where(ch.st == "+", ch.s, ch.e - 1); idx = bidx(ch.c.values, pos); np.add.at(a, idx[idx >= 0], 1)
    np.save(AC + ".tmp.npy", a); os.replace(AC + ".tmp.npy", AC)
A = np.load(AC)
def bdg(path, val):
    nz = val > 0; D = R[nz].copy(); D["v"] = val[nz]; D = D.sort_values(["c", "s"]); D.to_csv(path, sep="\t", header=False, index=False, float_format="%.5g")
# P1: CPM, cutoff = 20,000th highest bin
cpm = v / v.sum() * 1e6; bdg(f"{wd}/treat_cpm.bdg", cpm); cut = np.sort(cpm[cpm > 0])[::-1][min(19999, (cpm > 0).sum() - 1)]
run(f"{MACS} bdgpeakcall -i {wd}/treat_cpm.bdg -c {cut:.6g} -l 1000 -g 1000 --outdir {PK}/{t} -o P1.narrowPeak > /dev/null 2>&1")
# P2: qpois vs ATAC lambda (bin, +/-5 kb mean, genome mean), ATAC scaled to the TF total
sc = A * v.sum() / A.sum(); lam = sc.copy(); gm = sc.mean()
for ch, g in R.groupby("c"):
    ii = g.index.values; x = sc[ii]; k = np.convolve(x, np.ones(11) / 11, mode="same"); lam[ii] = np.maximum(np.maximum(x, k), gm)
bdg(f"{wd}/treat_cnt.bdg", v); D = R.copy(); D["v"] = lam; D.sort_values(["c", "s"]).to_csv(f"{wd}/ctrl_lambda.bdg", sep="\t", header=False, index=False, float_format="%.5g")
run(f"{MACS} bdgcmp -t {wd}/treat_cnt.bdg -c {wd}/ctrl_lambda.bdg -m qpois --outdir {wd} -o qpois.bdg > /dev/null 2>&1")
run(f"{MACS} bdgpeakcall -i {wd}/qpois.bdg -c 2 -l 1000 -g 1000 --outdir {PK}/{t} -o P2.narrowPeak > /dev/null 2>&1")
# P3: pseudo-fragments at bin centres
r_, _ = m.nonzero(); fr = R.iloc[r_]
with gzip.open(f"{wd}/frag.bed.gz", "wt", compresslevel=1) as fh:
    np.savetxt(fh, np.column_stack([fr.c.values, (fr.s.values + 499).astype(str), (fr.s.values + 501).astype(str)]), fmt="%s", delimiter="\t")
del m, r_, fr
base = f"{MACS} callpeak -t {wd}/frag.bed.gz -f BED --nomodel --shift -500 --extsize 1000 --keep-dup all -g hs -q 0.05 --outdir {PK}/{t}"
run(f"{base} -n P3a > {wd}/p3a.log 2>&1"); run(f"{base} -c {ATAC_TA} -n P3b > {wd}/p3b.log 2>&1")
# local contrast at the top-5,000 lab summits
P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; sm = (P.s + P.off).values
idx = lambda d: bidx(P.c.values.astype(str), sm + d)
c0, cl, cr = idx(0), idx(-2000), idx(2000); ok = (c0 >= 0) & (cl >= 0) & (cr >= 0); ctr, fl = cpm[c0[ok]], (cpm[cl[ok]] + cpm[cr[ok]]) / 2; ratio = (ctr + 0.1) / (fl + 0.1)
info.update({"local contrast: median summit / 2 kb flank": float(np.median(ratio)), "local contrast: share of summits > flank": float((ctr > fl).mean()), "lab summits with signal": float((ctr > 0).mean())})
for k in ("P1", "P2", "P3a", "P3b"):
    f = f"{PK}/{t}/{k}.narrowPeak" if k in ("P1", "P2") else f"{PK}/{t}/{k}_peaks.narrowPeak"; info[f"{k} peaks"] = sum(1 for _ in open(f)) if os.path.exists(f) else 0
json.dump(info, open(f"{PK}/{t}/info.json", "w")); print(info)
for f in ("treat_cpm.bdg", "treat_cnt.bdg", "ctrl_lambda.bdg", "qpois.bdg", "frag.bed.gz"): os.remove(f"{wd}/{f}")
