#!/usr/bin/env python
"""Round 2, R2-4 data: aggregate +-1 kb profiles (10 bp bins) at each reference's top-2000 summits
(all / unique / preferred), for all 79 TFs. Pool-normalised profile of panel TF i =
(reads_i per bin / reads_i total) / (pool-minus-i per bin / pool-minus-i total).
Output OUT/round2/r2_profiles.npz: H (79 x sets x 6 refs x 200 bins), tf79, tot79, sets, edges.
"""
import numpy as np
import pandas as pd

from fig2_common import INP, OUT, TFS

R2 = OUT / "round2"
S = pd.read_csv(R2 / "summits.tsv", sep="\t")
W, STEP = 1000, 10
edges = np.arange(-W, W + 1, STEP)
SETS = ("all", "unique", "preferred")
sel = {}
for typ in SETS:
    for t in TFS:
        m = S.ref.values == t
        if typ != "all":
            m &= S[typ].values.astype(bool)
        ix = np.flatnonzero(m)
        sel[(typ, t)] = ix[np.argsort(-S.signal.values[ix], kind="stable")][:2000]
tf79 = list(np.load(INP / "frame.npz", allow_pickle=True)["tf79"])
H = np.zeros((len(tf79), len(SETS), len(TFS), len(edges) - 1))
tot = np.zeros(len(tf79))
for k, t in enumerate(tf79):
    z = np.load(INP / "five" / f"{t}.npz")
    pos = {c: z[c] for c in z.files}
    tot[k] = sum(len(v) for v in pos.values())
    for a, typ in enumerate(SETS):
        for b, r in enumerate(TFS):
            ix = sel[(typ, r)]
            rel = []
            for c in np.unique(S.chrom.values[ix]):
                x = S.pos.values[ix][S.chrom.values[ix] == c]
                p = pos[c]
                lo, hi = np.searchsorted(p, x - W), np.searchsorted(p, x + W)
                rel += [p[l:h] - xx for xx, l, h in zip(x, lo, hi)]
            if rel:
                H[k, a, b] = np.histogram(np.concatenate(rel), edges)[0]
    print(t, flush=True)
np.savez(R2 / "r2_profiles.npz", H=H, tf79=np.array(tf79), tot79=tot, sets=np.array(SETS), edges=edges)
