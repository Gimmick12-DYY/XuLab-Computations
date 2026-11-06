#!/usr/bin/env python3
"""Pseudobulk imputed signal per TF in given windows: fraction of cells with a (raw or imputed) 1 kb bin."""
import sys, json
from pathlib import Path
import numpy as np, scipy.sparse as sp, pandas as pd
U = Path("/work/users/d/y/dyy12/XuLab/unified/work"); OUT = Path(sys.argv[1]); tfs = sys.argv[2].split(","); wins = sys.argv[3:]
reg = pd.read_csv(U / "nfya/impute/regions.tsv", header=None)[0]
c = reg.str.split(":").str[0]; s = reg.str.split(":").str[1].str.split("-").str[0].astype(int)
rows = []
for tf in tfs:
    d = U / tf.lower() / "impute"
    if not (d / "matrix_csr.npz").exists():
        print("missing", tf, flush=True); continue
    M = sp.load_npz(d / "matrix_csr.npz").tocsr()
    n = M.shape[1]
    for w in wins:
        ch, se = w.split(":"); a, b = map(int, se.split("-"))
        idx = np.where((c.values == ch) & (s.values >= a) & (s.values < b))[0]
        v = np.asarray(M[idx].sum(1)).ravel() / n
        for i, x in zip(idx, v):
            rows.append((tf, ch, int(s.values[i]), float(x)))
    print(tf, n, flush=True)
pd.DataFrame(rows, columns=["tf", "chrom", "start", "frac_cells"]).to_csv(OUT, sep="\t", index=False)
