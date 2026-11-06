"""Cicero co-binding pairs (1st step, p<=0.05, span >25 kb) shared by >=2 TFs.
Peaks of all TFs' pairs are merged where they overlap into one region set; a pair is shared when two TFs have
a pair joining the same two merged regions. ENCODE blacklist v2 regions and chrM/chrY dropped."""
import glob, gzip
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np, pandas as pd
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 0; SUF = f"_top{TOPN}" if TOPN else ""
KEEP = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
def iv(r):
    c, se = r.split(":"); s, e = se.split("-"); return c, int(s), int(e)
P = {}
for f in sorted(glob.glob("pairs/**/*.peak_edges.tsv", recursive=True)):
    tf = Path(f).name.split(".")[0]
    if tf in P: continue
    d = pd.read_csv(f, sep="\t", usecols=["peak1", "peak2", "pval"])
    d = d[d.pval <= 0.05]
    P[tf] = d
allp = sorted({iv(x) for d in P.values() for col in ("peak1", "peak2") for x in d[col]})
allp = [p for p in allp if p[0] in KEEP]
uni = []
for c, s, e in allp:
    if uni and uni[-1][0] == c and s <= uni[-1][2]: uni[-1][2] = max(uni[-1][2], e)
    else: uni.append([c, s, e])
bl = defaultdict(list)
for ln in gzip.open("../strat/hg38-blacklist.v2.bed.gz", "rt"):
    x = ln.split(); bl[x[0]].append((int(x[1]), int(x[2])))
blk = {i for i, (c, s, e) in enumerate(uni) if any(a < e and b > s for a, b in bl.get(c, []))}
starts = defaultdict(list); ids = defaultdict(list)
for i, (c, s, e) in enumerate(uni): starts[c].append(s); ids[c].append(i)
starts = {c: np.array(v) for c, v in starts.items()}
def uid(r):
    c, s, e = iv(r)
    return ids[c][np.searchsorted(starts[c], s, side="right") - 1] if c in starts else None
name = lambda i: f"{uni[i][0]}:{uni[i][1]}-{uni[i][2]}"
mid = lambda i: (uni[i][1] + uni[i][2]) // 2
pair_tfs = defaultdict(set); per_tf = {}
for tf, d in P.items():
    S = set()
    for a, b in zip(d.peak1, d.peak2):
        i, j = uid(a), uid(b)
        if i is None or j is None or i == j or i in blk or j in blk: continue
        i, j = min(i, j), max(i, j)
        if mid(j) - mid(i) <= 25000: continue
        S.add((i, j))
    if TOPN:
        if len(S) < TOPN:
            continue
        d2 = P[tf]
        pv = {}
        for a, b, p_ in zip(d2.peak1, d2.peak2, d2.pval):
            i, j = uid(a), uid(b)
            if i is None or j is None: continue
            k = (min(i, j), max(i, j))
            if k in S: pv[k] = min(pv.get(k, 1), p_)
        S = set(sorted(S, key=lambda k: pv[k])[:TOPN])     # depth-matched: each TF's TOPN strongest pairs
    per_tf[tf] = len(S)
    for e in S: pair_tfs[e].add(tf)
shared = {e: t for e, t in pair_tfs.items() if len(t) >= 2}
rows = [{"region1": name(i), "region2": name(j), "distance_kb": round((mid(j) - mid(i)) / 1e3, 1), "n_tfs": len(t), "tfs": ",".join(sorted(t))}
        for (i, j), t in sorted(shared.items(), key=lambda z: -len(z[1]))]
pd.DataFrame(rows).to_csv(f"shared_pairs{SUF}.tsv", sep="\t", index=False)
pd.Series(per_tf).to_csv(f"pairs_per_tf{SUF}.tsv", sep="\t", header=False)
from collections import Counter
print(len(P), "TFs;", len(pair_tfs), "distinct pairs >25 kb;", len(shared), "shared by >=2 TFs;", dict(sorted(Counter(len(t) for t in shared.values()).items())))
