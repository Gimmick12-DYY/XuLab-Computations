"""TF x TF co-occupancy of co-binding cliques: stratified (Mantel-Haenszel) odds ratio in a shared universe.

Universe: union of every TF's Cicero input peaks (overlapping peaks merged; ENCODE blacklist, chrM/Y removed).
For each TF and universe region: in that TF's clique regions (1/0).
Cell (A,B): Mantel-Haenszel odds ratio of "in A's cliques" vs "in B's cliques" across strata of
  promoter (TSS +/-1 kb) or distal  x  peak breadth (number of TFs whose peak set contains the region)
  x  region width tertile,
Robins-Breslow-Greenland SE, CMH chi-square p, BH FDR, shrinkage of log OR by its SE (normal prior, sd 1).
"""
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

X = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
inputs = [l.split("\t") for l in Path(sys.argv[3]).read_text().splitlines()]
TSS = Path(sys.argv[4]); BL = Path(sys.argv[5])
KEEP = {f"chr{i}" for i in range(1, 23)} | {"chrX"}


def parse(r):
    r = r.strip()
    if ":" in r:
        c, se = r.split(":"); s, e = se.split("-")
    else:
        c, s, e = r.split("_")
    return c, int(s), int(e)


peaks, cliq = {}, {}
for tf, cpath, rpath in inputs:
    rp = X / rpath.lstrip("/")
    with gzip.open(rp, "rt") as fh:
        peaks[tf] = [p for p in (parse(x) for x in fh if x.strip()) if p[0] in KEEP]
    d = pd.read_csv(X / cpath.strip().lstrip("/"), sep="\t", usecols=["regions"])
    cliq[tf] = {parse(x) for rr in d.regions for x in str(rr).split(";")}
tfs = sorted(peaks)

# universe = merged union of all peaks
allp = sorted({p for v in peaks.values() for p in v})
uni = []
for c, s, e in allp:
    if uni and uni[-1][0] == c and s <= uni[-1][2]:
        uni[-1][2] = max(uni[-1][2], e)
    else:
        uni.append([c, s, e])
bl = defaultdict(list)
for ln in gzip.open(BL, "rt"):
    f = ln.split(); bl[f[0]].append((int(f[1]), int(f[2])))
for c in bl:
    bl[c].sort()


def in_bl(c, s, e):
    for a, b in bl.get(c, []):
        if a < e and b > s:
            return True
    return False


uni = [u for u in uni if not in_bl(*u)]
U = pd.DataFrame(uni, columns=["chrom", "start", "end"])
n = len(U)
print(f"universe regions: {n:,}")
starts = {c: g.start.values for c, g in U.groupby("chrom")}
ends = {c: g.end.values for c, g in U.groupby("chrom")}
offs = {c: g.index.values for c, g in U.groupby("chrom")}


def to_idx(regs):
    out = set()
    for c, s, e in regs:
        if c not in starts:
            continue
        i = np.searchsorted(starts[c], e, side="left") - 1
        while i >= 0 and ends[c][i] > s:
            out.add(offs[c][i]); i -= 1
    return out


P = np.zeros((n, len(tfs)), dtype=bool); C = np.zeros((n, len(tfs)), dtype=bool)
for j, tf in enumerate(tfs):
    P[list(to_idx(peaks[tf])), j] = True
    ci = to_idx(cliq[tf])
    if ci:
        C[list(ci), j] = True
size = C.sum(0)
print("clique regions per TF in universe: min", size.min(), "max", size.max())

# strata
tss = defaultdict(list)
for ln in gzip.open(TSS, "rt"):
    f = ln.split(); tss[f[0]].append(int(f[1]))
prom = np.zeros(n, dtype=bool)
for c in starts:
    t = np.sort(np.array(tss.get(c, [])))
    if not len(t):
        continue
    s, e, ix = starts[c], ends[c], offs[c]
    lo = np.searchsorted(t, s - 1000, side="left"); hi = np.searchsorted(t, e + 1000, side="right")
    prom[ix] = hi > lo
breadth = P.sum(1)
bb = np.digitize(breadth, [2, 4, 8, 16, 32])
w = (U.end - U.start).values
wb = np.digitize(w, np.quantile(w, [1 / 3, 2 / 3]))
stratum = prom.astype(int) * 100 + bb * 10 + wb
labs, sid = np.unique(stratum, return_inverse=True)
print(f"strata: {len(labs)}; promoter regions {prom.mean():.1%}")

T = len(tfs)
sumR = np.zeros((T, T)); sumS = np.zeros((T, T)); sPR = np.zeros((T, T)); sPSQR = np.zeros((T, T)); sQS = np.zeros((T, T))
sumA = np.zeros((T, T)); sumE = np.zeros((T, T)); sumV = np.zeros((T, T)); sumExp = np.zeros((T, T))
Cf = C.astype(np.float64)
for k in range(len(labs)):
    m = sid == k
    ns = m.sum()
    if ns < 2:
        continue
    Xk = Cf[m]
    a = Xk.T @ Xk
    r = Xk.sum(0)
    b = r[:, None] - a; c = r[None, :] - a; d = ns - a - b - c
    R = a * d / ns; Ss = b * c / ns; Pp = (a + d) / ns; Q = (b + c) / ns
    sumR += R; sumS += Ss; sPR += Pp * R; sPSQR += Pp * Ss + Q * R; sQS += Q * Ss
    E = np.outer(r, r) / ns
    V = np.outer(r * (ns - r), r * (ns - r)) / (ns ** 2 * (ns - 1)) ** 1.0
    V = np.outer(r, r) * np.outer(ns - r, ns - r) / (ns ** 2 * (ns - 1))
    sumA += a; sumE += E; sumV += V
with np.errstate(divide="ignore", invalid="ignore"):
    logOR = np.log(sumR / sumS)
    var = sPR / (2 * sumR ** 2) + sPSQR / (2 * sumR * sumS) + sQS / (2 * sumS ** 2)
    se = np.sqrt(var)
    stat = (np.abs(sumA - sumE) - 0.5).clip(min=0) ** 2 / sumV
    p = chi2.sf(stat, 1)
iu = np.triu_indices(T, 1)
ok = np.isfinite(logOR[iu]) & np.isfinite(se[iu]) & (se[iu] > 0)
lo_, se_ = logOR[iu][ok], se[iu][ok]
# normal prior on log OR with sd 1 (weakly informative): estimates with large SE shrink toward 0
tau2 = 1.0
shr = np.full((T, T), np.nan)
shrunk = lo_ * tau2 / (tau2 + se_ ** 2)
tmp = np.full(len(iu[0]), np.nan); tmp[ok] = shrunk
shr[iu] = tmp; shr.T[iu] = tmp
pv = p[iu]
q = np.full(len(pv), np.nan)
okp = np.isfinite(pv)
o = np.argsort(pv[okp]); m_ = okp.sum()
qq = pv[okp][o] * m_ / np.arange(1, m_ + 1)
qq = np.minimum.accumulate(qq[::-1])[::-1]
tq = np.empty(m_); tq[o] = np.minimum(qq, 1); q[okp] = tq
Q = np.full((T, T), np.nan); Q[iu] = q; Q.T[iu] = q
rows = []
for k, (i, j) in enumerate(zip(*iu)):
    rows.append({"tf1": tfs[i], "tf2": tfs[j], "shared": int(sumA[i, j]), "expected_within_strata": sumE[i, j],
                 "log_OR_MH": logOR[i, j], "se": se[i, j], "log_OR_shrunk": shr[i, j], "p": p[i, j], "q": Q[i, j]})
df = pd.DataFrame(rows).sort_values("log_OR_shrunk", ascending=False)
df.to_csv(OUT / "tf_pair_stratified_OR.tsv", sep="\t", index=False)
pd.Series(size, index=tfs, name="clique_regions").to_csv(OUT / "tf_clique_region_counts.tsv", sep="\t")
np.save(OUT / "shrunk.npy", shr); np.save(OUT / "q.npy", Q); np.save(OUT / "expected.npy", sumE)
Path(OUT / "tfs.txt").write_text("\n".join(tfs))
print(f"tau2={tau2:.3f}; pairs FDR<0.05: {(df.q < 0.05).sum()} of {len(df)}; positive: {((df.q < 0.05) & (df.log_OR_MH > 0)).sum()}")
for a, b in [("EZH2", "MTF2"), ("MAZ", "ZNF143"), ("WIZ", "ZNF644"), ("RBBP4", "MBD3"), ("RBBP7", "MBD3"), ("ETS2", "ZNF282"), ("FOXK1", "MLX")]:
    x, y = sorted((a, b)); r = df[(df.tf1 == x) & (df.tf2 == y)]
    print(a, b, r[["shared", "expected_within_strata", "log_OR_MH", "log_OR_shrunk", "q"]].round(3).values.tolist())
print(df.head(15).round(3).to_string(index=False))
