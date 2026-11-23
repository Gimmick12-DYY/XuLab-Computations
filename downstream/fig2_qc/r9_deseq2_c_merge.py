#!/usr/bin/env python
"""Round 9 step c: assemble our count matrix (peaks x replicates, Matrix Market) + sample table; bulk ChIP counts per
peak (bedtools multicov, duplicates excluded by default, MAPQ >= 30) and the ChIP log2FC table (SPEC.md 2, 4)."""
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BAM, OUT, TFS  # noqa: E402

O9 = OUT / "round9_deseq2"
cols, samples, fall = [], [], []
for f in sorted((O9 / "counts").glob("*.npz")):
    z = np.load(f); t = f.stem
    for j, r in enumerate(z["reps"]):
        cols.append(sp.csc_matrix(z["counts"][:, j:j + 1])); samples.append(dict(sample=f"{t}|{r}", tf=t, rep=str(r), kind=str(z["kind"]), reads=int(z["counts"][:, j].sum())))
    if str(z["kind"]) != "library":
        fall.append(t)
M = sp.hstack(cols).tocsc()
sio.mmwrite(str(O9 / "counts.mtx"), M, field="integer")
S = pd.DataFrame(samples); S.to_csv(O9 / "samples.tsv", sep="\t", index=False)
print("matrix", M.shape, "TFs", S.tf.nunique(), "random-split TFs:", fall, flush=True)
out = subprocess.run(["bedtools", "multicov", "-q", "30", "-bams", *[str(BAM[t]) for t in TFS], "-bed", str(O9 / "peaks.bed")],
                     check=True, capture_output=True, text=True).stdout
B = np.array([[float(x) for x in l.split("\t")[4:]] for l in out.splitlines()])
L = np.log2(B / B.sum(0, keepdims=True) * 1e6 + 1)
FC = np.column_stack([L[:, j] - np.delete(L, j, 1).mean(1) for j in range(len(TFS))])
pd.DataFrame(FC, columns=TFS).to_csv(O9 / "chip_log2fc.tsv.gz", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(B, columns=TFS).astype(int).to_csv(O9 / "chip_counts.tsv.gz", sep="\t", index=False)
print("bulk counts per ChIP", dict(zip(TFS, B.sum(0).astype(int))), flush=True)
