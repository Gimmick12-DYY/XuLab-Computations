#!/usr/bin/env python
"""Diagnose the 6c union-peak correlations (user: "your input is strange, why negative correlation?").
On U1 (merged union of the 6 reference peak sets, r15 tmp/U1.bed): region width; per-track correlation with log width;
CTCF raw vs CTCF ChIP within CTCF reference peaks; and the 6x6 recomputed four ways:
 (1) as r15: log2 CPM of read counts per merged region (ChIP from bigWig CPM mean x width, raw 5' ends)
 (2) density: log2 of CPM per kb (removes the width effect)
 (3) fixed windows: each reference summit +-250 bp (not merged regions), union of the 6 summit sets
 (4) fixed windows, density, restricted to summits of one reference at a time (own-peak check)."""
import numpy as np, pandas as pd, pyBigWig
from fig2_common import INP, OUT, REF, ROOT, TFS
OU = OUT / "round6_diffpeaks" / "union_6c"; OH = OUT / "round6_diffpeaks" / "heatmaps"
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
five = {t: dict(np.load(INP / "five" / f"{t}.npz")) for t in TFS}
tot = {t: sum(len(v) for v in five[t].values()) for t in TFS}
bws = {t: pyBigWig.open(str(OH / "bw" / f"{t}.bw")) for t in TFS}
def counts(c, s, e):
    B = np.zeros((len(c), 6)); S = np.zeros((len(c), 6))
    for k, t in enumerate(TFS):
        bw = bws[t]
        B[:, k] = [np.nan_to_num(bw.stats(ci, int(a), int(b), type="mean")[0] or 0) for ci, a, b in zip(c, s, e)]
        for ch in np.unique(c):
            m = np.flatnonzero(c == ch); p = five[t].get(ch)
            if p is None: continue
            S[m, k] = np.searchsorted(p, e[m]) - np.searchsorted(p, s[m])
        S[:, k] = S[:, k] / tot[t] * 1e6
    return B, S          # B = mean CPM per bp-bin (density), S = raw CPM count
def M(Xb, Xs):
    return np.corrcoef(np.column_stack([Xb, Xs]).T)[6:, :6]
def show(name, Bm):
    own = np.diag(Bm); best = sum(np.argmax(Bm[i]) == i for i in range(6))
    print(f"\n== {name}: own-best rows {best}/6; own r", dict(zip(TFS, own.round(3))))
    print(pd.DataFrame(Bm, [f"{t} sc" for t in TFS], [f"{t} ChIP" for t in TFS]).round(2).to_string())
U = pd.read_csv(OU / "tmp" / "U1.bed", sep="\t", header=None)
c, s, e = U[0].values, U[1].values, U[2].values; w = (e - s).astype(float)
print("U1 n", len(U), "width quantiles", np.percentile(w, [5, 25, 50, 75, 95, 99]).astype(int))
B, S = counts(c, s, e)
Bcount = B * w[:, None]   # ChIP CPM summed over the region
lw = np.log2(w)
print("corr with log2 width: ChIP counts", [round(np.corrcoef(lw, np.log2(Bcount[:, k] + 1))[0, 1], 2) for k in range(6)],
      " raw counts", [round(np.corrcoef(lw, np.log2(S[:, k] + 1))[0, 1], 2) for k in range(6)])
show("(1) as r15: log2(CPM count + 1) per merged region", M(np.log2(Bcount + 1), np.log2(S + 1)))
show("(2) density: log2(CPM per kb + 1) per merged region", M(np.log2(B * 1000 + 1), np.log2(S / w[:, None] * 1000 + 1)))
# (3) fixed windows at reference summits
rows = []
for t in TFS:
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    sm = (P[1] + P[9].clip(lower=0)).values
    rows.append(pd.DataFrame({"c": P[0].values, "s": sm - 250, "e": sm + 250, "ref": t}))
F = pd.concat(rows, ignore_index=True); F = F[F.s > 0]
Bf, Sf = counts(F.c.values, F.s.values, F.e.values)
show("(3) fixed +-250 bp at all reference summits (union of 6 sets)", M(np.log2(Bf * 500 + 1), np.log2(Sf + 1)))
for t in TFS:
    m = (F.ref == t).values
    r = np.corrcoef(np.log2(Bf[m, TFS.index(t)] * 500 + 1), np.log2(Sf[m, TFS.index(t)] + 1))[0, 1]
    print(f"(4) within {t} reference summits (n={m.sum()}): r(own ChIP, own raw) = {r:.3f};",
          "raw CPM mean at own summits vs at other refs' summits:", round(Sf[m, TFS.index(t)].mean(), 4), "vs", round(Sf[~m, TFS.index(t)].mean(), 4))
