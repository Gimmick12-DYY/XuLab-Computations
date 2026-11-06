#!/usr/bin/env python3
"""Reads (5' ends) of every TF in each element (±250 bp), plus each TF's total reads, for CPM."""
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
els = [l.strip() for l in open(sys.argv[1]) if l.strip()]
FR, OUT, PAD = Path(sys.argv[2]), Path(sys.argv[3]), 250
iv = {}
for k, r in enumerate(els):
    c, se = r.split(":"); s, e = map(int, se.split("-"))
    iv.setdefault(c, []).append((s - PAD, e + PAD, k))
idx = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), np.array([k for _, _, k in sorted(v)])) for c, v in iv.items()}
cnt, tot = {}, {}
for f in sorted(FR.glob("TF.*.bed.gz")):
    tf = f.name[3:-7]; x = np.zeros(len(els), int); n = 0
    with gzip.open(f, "rt") as fh:
        for ln in fh:
            n += 1
            p = ln.split("\t", 6)
            if p[0] not in idx: continue
            q = int(p[1]) if p[5][0] == "+" else int(p[2]) - 1
            S, E, K = idx[p[0]]
            i = np.searchsorted(S, q, side="right") - 1
            while i >= 0 and S[i] > q - 20000:
                if S[i] <= q < E[i]: x[K[i]] += 1
                i -= 1
    cnt[tf], tot[tf] = x, n
    print(tf, n, flush=True)
pd.DataFrame(cnt, index=els).to_csv(OUT / "element_tf_counts.tsv", sep="\t")
pd.Series(tot).to_csv(OUT / "tf_total_reads.tsv", sep="\t", header=False)
