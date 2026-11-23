#!/usr/bin/env python
"""Round 9 step b (one TF per task): read 5' ends per consensus peak per replicate (SPEC.md 2).
Usage: r9_deseq2_b_counts.py <task 0-78> -> OUT/round9_deseq2/counts/<TF>.npz (counts: peaks x reps, rep names, kind)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import OUT  # noqa: E402

O9 = OUT / "round9_deseq2"; (O9 / "counts").mkdir(exist_ok=True)
FR = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment")   # read-only
tfs = sorted(p.name[3:-7] for p in FR.glob("TF.*.bed.gz"))
t = tfs[int(sys.argv[1])]
pk = pd.read_csv(O9 / "peaks.bed", sep="\t", header=None)
d = pd.read_csv(FR / f"TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 3, 5], names=["c", "s", "e", "bc", "st"])
d["p"] = np.where(d.st.values == "+", d.s.values, d.e.values - 1)
d["lib"] = d.bc.str.replace(r"_[^_]+$", "", regex=True)
share = d.lib.value_counts(normalize=True)
libs = list(share[share >= 0.10].index)
if len(libs) >= 2:
    kind = "library"; rep = d.lib.map({l: i for i, l in enumerate(libs)}).fillna(-1).astype(int).values; names = libs
else:
    kind = "random cell split"; cells = d.bc.unique(); rng = np.random.default_rng(2026)
    a = dict(zip(cells, rng.integers(0, 3, len(cells)))); rep = d.bc.map(a).values; names = [f"rand{i}" for i in range(3)]
C = np.zeros((len(pk), len(names)), np.int32)
for c, g in pk.groupby(0):
    m = d.c.values == c
    if not m.any():
        continue
    pos, rr = d.p.values[m], rep[m]
    o = np.argsort(pos); pos, rr = pos[o], rr[o]
    lo = np.searchsorted(pos, g[1].values); hi = np.searchsorted(pos, g[2].values)
    for j in range(len(names)):
        cs = np.r_[0, np.cumsum(rr == j)]
        C[g.index.values, j] = cs[hi] - cs[lo]
np.savez_compressed(O9 / "counts" / f"{t}.npz", counts=C, reps=np.array(names), kind=kind,
                    lib_shares=np.array([f"{k}:{v:.3f}" for k, v in share.head(8).items()]))
print(t, kind, names, C.sum(0), flush=True)
