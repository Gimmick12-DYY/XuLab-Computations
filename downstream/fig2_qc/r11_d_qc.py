#!/usr/bin/env python
"""Round 11 step d (SPEC.md 4): count matrix (regions x 12) + read-type table + replicate correlation QC."""
import numpy as np
import pandas as pd

from r11_common import O11, SAMPLES

names = [f"{t}_rep{r}" for t, r, _ in SAMPLES]
C = np.column_stack([np.load(O11 / "counts" / f"{n}.npy") for n in names])
pd.DataFrame(C, columns=names).to_csv(O11 / "counts.tsv.gz", sep="\t", index=False)
info = pd.concat([pd.read_csv(O11 / "counts" / f"readinfo_{n}.tsv", sep="\t") for n in names])
info["frac_in_regions"] = info.in_regions / info.reads_counted
info.to_csv(O11 / "readinfo.tsv", sep="\t", index=False)
L = np.log2(C / C.sum(0, keepdims=True) * 1e6 + 1)
R = pd.DataFrame(np.corrcoef(L.T), names, names); R.to_csv(O11 / "qc_corr_12x12.tsv", sep="\t", float_format="%.4f")
rows = []
for i in range(0, 12, 2):
    t = names[i].split("_rep")[0]; others = [j for j in range(12) if j not in (i, i + 1)]
    rows.append(dict(ChIP=t, rep1_vs_rep2=R.iat[i, i + 1], max_vs_other_ChIP=float(R.values[[i, i + 1]][:, others].max())))
Q = pd.DataFrame(rows); Q.to_csv(O11 / "qc_rep_corr.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 200)
print(info.to_string()); print(Q.round(3).to_string()); print(R.round(2).to_string())
