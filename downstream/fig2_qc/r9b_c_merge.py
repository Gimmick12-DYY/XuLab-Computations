#!/usr/bin/env python
"""Round 9b step c (secondary arm): assemble our count matrix + sample table (as round 9 step c); bulk ChIP counts per
region, one BAM per task (bedtools multicov -q 30), then 'combine' -> chip_counts.tsv.gz, chip_log2fc.tsv.gz.
Usage: r9b_c_merge.py matrix | <0-5> | combine"""
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BAM, OUT, TFS  # noqa: E402

O = OUT / "round9b_deseq2_multigroup" / "chip1kb"; (O / "bulk").mkdir(exist_ok=True)
a = sys.argv[1]
if a == "matrix":
    cols, samples = [], []
    for f in sorted((O / "counts").glob("*.npz")):
        z = np.load(f); t = f.stem
        for j, r in enumerate(z["reps"]):
            cols.append(sp.csc_matrix(z["counts"][:, j:j + 1]))
            samples.append(dict(sample=f"{t}|{r}", tf=t, rep=str(r), kind=str(z["kind"]), reads=int(z["counts"][:, j].sum())))
    M = sp.hstack(cols).tocsc(); sio.mmwrite(str(O / "counts.mtx"), M, field="integer")
    S = pd.DataFrame(samples); S.to_csv(O / "samples.tsv", sep="\t", index=False)
    print("matrix", M.shape, "TFs", S.tf.nunique(), flush=True)
elif a == "combine":
    B = np.column_stack([np.load(O / "bulk" / f"{t}.npy") for t in TFS]).astype(float)
    L = np.log2(B / B.sum(0, keepdims=True) * 1e6 + 1)
    FC = np.column_stack([L[:, j] - np.delete(L, j, 1).mean(1) for j in range(len(TFS))])
    pd.DataFrame(FC, columns=TFS).to_csv(O / "chip_log2fc.tsv.gz", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(B, columns=TFS).astype(int).to_csv(O / "chip_counts.tsv.gz", sep="\t", index=False)
    print("bulk counts per ChIP", dict(zip(TFS, B.sum(0).astype(int))))
else:
    t = TFS[int(a)]
    out = subprocess.run(["bedtools", "multicov", "-q", "30", "-bams", str(BAM[t]), "-bed", str(O / "peaks.bed")],
                         check=True, capture_output=True, text=True).stdout
    np.save(O / "bulk" / f"{t}.npy", np.array([int(l.split("\t")[4]) for l in out.splitlines()], np.int64))
    print(t, "done")
