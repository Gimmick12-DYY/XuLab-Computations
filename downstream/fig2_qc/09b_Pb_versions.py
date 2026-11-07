#!/usr/bin/env python
"""P-b versions that keep all 6 TFs despite shared CpG-island promoter peaks.

Regions: summit +-250 bp of every reference peak (CTCF majority set has no summit: peak midpoint),
merged across the 6 references (chr1-22, X). Values: log2(CPM + 1), CPM over the region set.
Bulk = ChIP BAM reads (bedtools multicov; no input/control BAMs are available, so not input-normalised).
sc: raw (full depth), raw_dm (thinned to ZNF777 depth, 3 seeds, statistics averaged), imputed
(cells positive summed over the 1 kb bins a region touches).
  V1  plain Pearson over all regions
  V2  within-assay centring: each region's mean across the 6 tracks removed separately in bulk and in sc
  V3  specificity-ranked equal-size sets: per TF j, score = lc(bulk_j) - mean lc(other 5 bulk);
      top N regions per TF (N = 500, 1000), union of the 6 sets; plain (V3) and V2-centred (V3c)
  V4  partial correlation: every track residualised (OLS) on lc(OmniATAC fold-enrichment over the
      region) and lc(pooled 79-TF sc reads); Pearson of residuals
For each version: 12x12 Pearson (bulk + sc), average-linkage clustering (1 - r), the 6x6 sc-vs-bulk
block (own-best count, diag - offdiag, exact permutation p over 720 row shuffles), clusters-next-to-own,
and the 6x6 bulk-vs-bulk block (reference ceiling: mean off-diagonal r).
Outputs OUT/panels/Pb3_*; summary rows.
"""
import subprocess

import numpy as np
import pandas as pd
import pyBigWig
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from fig2_common import (ATAC_BW, BAM, BIN, DM_DEPTH, DM_SEEDS, INP, NT, OUT, REF, TFS, append_summary, bin_index,
                         diag_stats, load_imp7, read_bed)

op = OUT / "panels"
tmp = op / "tmp"
tmp.mkdir(parents=True, exist_ok=True)
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = list(fr["tf79"])
imp = load_imp7()
HALF = 250


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def lc(C):
    return np.log2(C / C.sum(0, keepdims=True) * 1e6 + 1)


# ---- regions ----
parts = []
for t in TFS:
    pk = read_bed(REF[t], ncol=10)
    if pk.shape[1] >= 10 and t != "CTCF":
        mid = pk[1] + pk[9].astype(int)
    else:
        mid = (pk[1] + pk[2]) // 2
    parts.append(pd.DataFrame({0: pk[0], 1: np.maximum(mid - HALF, 0), 2: mid + HALF}))
allr = pd.concat(parts)
allr = allr[allr[0].str.match(r"^chr(\d+|X)$")].sort_values([0, 1])
allr.to_csv(tmp / "Pb3_summits.bed", sep="\t", header=False, index=False)
(tmp / "Pb3_regions.bed").write_text(sh("bedtools", "merge", "-i", str(tmp / "Pb3_summits.bed")))
reg = read_bed(tmp / "Pb3_regions.bed").reset_index(drop=True)
print("regions", len(reg), flush=True)

# ---- counts ----
out = sh("bedtools", "multicov", "-bams", *[str(BAM[t]) for t in TFS], "-bed", str(tmp / "Pb3_regions.bed"))
B = np.array([[int(x) for x in l.split("\t")[3:]] for l in out.splitlines()], float)
S = np.zeros((len(reg), NT)); pool = np.zeros(len(reg)); tot = {}
for t in tf79:
    z = np.load(INP / "five" / f"{t}.npz")
    c = np.zeros(len(reg)); n = 0
    for ch in z.files:
        pos = z[ch]; n += len(pos)
        g = np.flatnonzero(reg[0].values == ch)
        if len(g):
            c[g] = np.searchsorted(pos, reg[2].values[g]) - np.searchsorted(pos, reg[1].values[g])
    tot[t] = n; pool += c
    if t in TFS:
        S[:, TFS.index(t)] = c
o = np.array([off[c][0] for c in reg[0]])
a, b = o + reg[1].values // BIN, o + (reg[2].values - 1) // BIN
cs = np.concatenate([np.zeros((NT, 1)), np.cumsum(imp, 1)], 1)
I = (cs[:, b + 1] - cs[:, a]).T
bw = pyBigWig.open(str(ATAC_BW))
atac = np.array([np.nan_to_num(bw.stats(c, int(s), int(e), type="mean")[0] or 0) for c, s, e in zip(reg[0], reg[1], reg[2])])
np.savez(op / "Pb3_counts.npz", B=B, S=S, pool=pool, I=I, atac=atac, chrom=reg[0].values, start=reg[1].values,
         end=reg[2].values)

Lb = lc(B)
spec = np.column_stack([Lb[:, i] - np.delete(Lb, i, 1).mean(1) for i in range(NT)])
dm_p = np.array([min(1, DM_DEPTH / tot[t]) for t in TFS])
arms = {"raw": [S], "raw_dm": [np.random.default_rng(s).binomial(S.astype(np.int64), dm_p).astype(float) for s in DM_SEEDS],
        "imputed": [I]}
cov = np.column_stack([np.ones(len(reg)), np.log2(atac + 1), lc(pool[:, None])[:, 0]])


def rowc(X):
    return X - X.mean(1, keepdims=True)


def resid(X, idx):
    C = cov[idx]
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta


def topn(n):
    m = np.zeros(len(reg), bool)
    for i in range(NT):
        m[np.argsort(-spec[:, i])[:n]] = True
    return np.flatnonzero(m)


allidx = np.arange(len(reg))
versions = [("V1_plain", allidx, "plain"), ("V2_centred", allidx, "centre"),
            ("V3_top500", topn(500), "plain"), ("V3c_top500", topn(500), "centre"),
            ("V3_top1000", topn(1000), "plain"), ("V3c_top1000", topn(1000), "centre"),
            ("V4_partial", allidx, "partial")]


def transform(Xb, Xs, idx, how):
    if how == "centre":
        return rowc(Xb), rowc(Xs)
    if how == "partial":
        return resid(Xb, idx), resid(Xs, idx)
    return Xb, Xs


def cluster_next(R, k):
    Z = linkage(squareform(np.clip(1 - R, 0, None), checks=False), "average")
    root = to_tree(Z)
    nxt = []
    for i in range(k):
        want, node = {i, k + i}, root
        while True:
            for ch in (node.left, node.right):
                if ch is not None and want <= set(ch.pre_order()):
                    node = ch
                    break
            else:
                break
        nxt.append(int(sum(x < k for x in node.pre_order()) == 1))
    return Z, nxt


labs = [f"bulk_{t}" for t in TFS] + [f"sc_{t}" for t in TFS]
rows, summ = [], []
for vname, idx, how in versions:
    for arm, reps in arms.items():
        Rs = []
        for Sx in reps:
            # CPM over the version's region set
            Xb, Xs = transform(lc(B[idx]), lc(Sx[idx]), idx, how)
            Rs.append(np.corrcoef(np.column_stack([Xb, Xs]).T))
        stats = [diag_stats(R[NT:, :NT]) for R in Rs]
        R = np.mean(Rs, 0)
        Z, nxt = cluster_next(R, NT)
        tag = f"{vname}_{arm}"
        pd.DataFrame(R, labs, labs).to_csv(op / f"Pb3_corr_{tag}.tsv", sep="\t", float_format="%.4f")
        np.savez(op / f"Pb3_link_{tag}.npz", Z=Z, labs=np.array(labs), R=R)
        block = R[NT:, :NT]
        bb = R[:NT, :NT]
        own_best = [int(np.argmax(block[i]) == i) for i in range(NT)]
        for i, t in enumerate(TFS):
            rows.append(dict(version=vname, arm=arm, n_regions=len(idx), sc_track=t, r_own=block[i, i],
                             r_best_other=float(np.max(np.delete(block[i], i))),
                             best_bulk=TFS[int(np.argmax(block[i]))], own_is_best_bulk=own_best[i],
                             clusters_next_to_own=nxt[i]))
        summ.append(dict(test=f"Pb3_{vname}", arm=arm, variant="sc_vs_bulk_r", n_regions=len(idx),
                         own_best_row=sum(own_best), clusters_next_to_own=sum(nxt),
                         own_best_col=int(sum(np.argmax(block[:, j]) == j for j in range(NT))),
                         mean_diag=float(np.diag(block).mean()),
                         diag_minus_offdiag=float(np.mean([s["diag_minus_offdiag"] for s in stats])),
                         perm_p=float(np.mean([s["perm_p"] for s in stats])),
                         perm_z=float(np.mean([s["perm_z"] for s in stats])),
                         bulk_bulk_mean_offdiag=float(bb[~np.eye(NT, dtype=bool)].mean()),
                         sc_sc_mean_offdiag=float(R[NT:, NT:][~np.eye(NT, dtype=bool)].mean())))
        print(tag, len(idx), summ[-1], flush=True)
pd.DataFrame(rows).to_csv(op / "Pb3_own_reference.tsv", sep="\t", index=False, float_format="%.4f")
append_summary(summ)
pd.set_option("display.width", 250)
print(pd.DataFrame(summ).to_string())
