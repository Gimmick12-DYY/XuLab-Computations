#!/usr/bin/env python
"""Round 9 step c2 (execution change, same method as step c): bulk ChIP counts per consensus peak, one BAM per task
(bedtools multicov -q 30, duplicates excluded), then 'combine' writes chip_counts.tsv.gz and chip_log2fc.tsv.gz exactly
as r9_deseq2_c_merge.py would.  Usage: r9_deseq2_c2_bulk.py <0-5> | combine"""
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BAM, OUT, TFS  # noqa: E402

O9 = OUT / "round9_deseq2"; (O9 / "bulk").mkdir(exist_ok=True)
if sys.argv[1] == "combine":
    B = np.column_stack([np.load(O9 / "bulk" / f"{t}.npy") for t in TFS]).astype(float)
    L = np.log2(B / B.sum(0, keepdims=True) * 1e6 + 1)
    FC = np.column_stack([L[:, j] - np.delete(L, j, 1).mean(1) for j in range(len(TFS))])
    pd.DataFrame(FC, columns=TFS).to_csv(O9 / "chip_log2fc.tsv.gz", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(B, columns=TFS).astype(int).to_csv(O9 / "chip_counts.tsv.gz", sep="\t", index=False)
    print("bulk counts per ChIP", dict(zip(TFS, B.sum(0).astype(int))))
else:
    t = TFS[int(sys.argv[1])]
    out = subprocess.run(["bedtools", "multicov", "-q", "30", "-bams", str(BAM[t]), "-bed", str(O9 / "peaks.bed")],
                         check=True, capture_output=True, text=True).stdout
    np.save(O9 / "bulk" / f"{t}.npy", np.array([int(l.split("\t")[4]) for l in out.splitlines()], np.int64))
    print(t, "done")
