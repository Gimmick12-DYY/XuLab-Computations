#!/usr/bin/env python
"""Round 9 step a: consensus peaks (SPEC.md 1) -> OUT/round9_deseq2/peaks.bed (chrom, start, end, id)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BLACKLIST, OUT, REF, ROOT, TFS  # noqa: E402

O9 = OUT / "round9_deseq2"; O9.mkdir(parents=True, exist_ok=True)
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
MAC = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2")
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
parts, n = [], {}
for d in sorted(MAC.glob("TF.*")):
    t = d.name[3:]
    P = pd.read_csv(d / f"TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=range(10))
    P = P[P[0].isin(CH) & (P[8] >= 1.30103)]
    s = P[1] + P[9]; parts.append(pd.DataFrame({0: P[0], 1: (s - 250).clip(lower=0), 2: s + 250})); n[f"lab_{t}"] = len(P)
for t in TFS:
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    s = P[1] + P[9].clip(lower=0); parts.append(pd.DataFrame({0: P[0], 1: (s - 250).clip(lower=0), 2: s + 250})); n[f"chip_{t}"] = len(P)
A = pd.concat(parts).sort_values([0, 1]); A.to_csv(O9 / "all_summits.bed", sep="\t", header=False, index=False)
m = subprocess.run(f"bedtools merge -i {O9 / 'all_summits.bed'} | bedtools intersect -v -a - -b {BLACKLIST}", shell=True,
                   check=True, capture_output=True, text=True).stdout
M = pd.read_csv(pd.io.common.StringIO(m), sep="\t", header=None); M[3] = [f"p{i}" for i in range(len(M))]
M.to_csv(O9 / "peaks.bed", sep="\t", header=False, index=False)
print("lab TFs", sum(k.startswith("lab_") for k in n), "summits", len(A), "consensus peaks", len(M), flush=True)
pd.Series(n).to_csv(O9 / "peak_inputs.tsv", sep="\t", header=False)
