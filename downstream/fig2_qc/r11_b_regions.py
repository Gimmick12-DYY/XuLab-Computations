#!/usr/bin/env python
"""Round 11 step b (SPEC.md 3): union of the 6 ChIP summits +-250 bp, merged, blacklist removed; per region and ChIP the
strongest summit and its signalValue -> O11/regions.tsv (+ regions.bed)."""
import subprocess

import numpy as np
import pandas as pd

from r11_common import O11
from fig2_common import BLACKLIST, REF, ROOT, TFS

CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
parts = []
for t in TFS:
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    s = (P[1] + P[9].clip(lower=0)).astype(int)
    parts.append(pd.DataFrame({"c": P[0].values, "s": (s - 250).clip(lower=0).values, "e": (s + 250).values,
                               "tf": t, "summit": s.values, "sig": P[6].values}))
A = pd.concat(parts).sort_values(["c", "s"]).reset_index(drop=True)
A[["c", "s", "e"]].to_csv(O11 / "summits.bed", sep="\t", header=False, index=False)
m = subprocess.run(f"bedtools merge -i {O11 / 'summits.bed'} | bedtools intersect -v -a - -b {BLACKLIST}", shell=True,
                   check=True, capture_output=True, text=True).stdout
M = pd.read_csv(pd.io.common.StringIO(m), sep="\t", header=None, names=["chrom", "start", "end"])
M["id"] = [f"r{i}" for i in range(len(M))]
for t in TFS:
    M[f"summit_{t}"] = -1; M[f"sig_{t}"] = np.nan
for c, g in M.groupby("chrom"):
    a = A[A.c == c]
    k = np.searchsorted(g.start.values, a.s.values, side="right") - 1
    ok = (k >= 0) & (a.e.values <= g.end.values[np.clip(k, 0, None)])
    a = a[ok].assign(row=g.index.values[k[ok]]).sort_values("sig", ascending=False).drop_duplicates(["row", "tf"])
    for t, b in a.groupby("tf"):
        M.loc[b.row.values, f"summit_{t}"] = b.summit.values; M.loc[b.row.values, f"sig_{t}"] = b.sig.values
M.to_csv(O11 / "regions.tsv", sep="\t", index=False)
M[["chrom", "start", "end", "id"]].to_csv(O11 / "regions.bed", sep="\t", header=False, index=False)
print("summits", len(A), "regions", len(M), {t: int((M[f"summit_{t}"] >= 0).sum()) for t in TFS}, flush=True)
