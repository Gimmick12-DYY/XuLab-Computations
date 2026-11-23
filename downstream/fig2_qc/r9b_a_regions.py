#!/usr/bin/env python
"""Round 9b step a (secondary arm, SPEC.md 7): union of the 6 ChIP summits +-1 kb, merged, blacklist removed
-> OUT/round9b_deseq2_multigroup/chip1kb/peaks.bed (chrom, start, end, id)."""
import subprocess
import sys

import pandas as pd

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import BLACKLIST, OUT, REF, ROOT, TFS  # noqa: E402

O = OUT / "round9b_deseq2_multigroup" / "chip1kb"; O.mkdir(parents=True, exist_ok=True)
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
parts = []
for t in TFS:
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    s = P[1] + P[9].clip(lower=0); parts.append(pd.DataFrame({0: P[0], 1: (s - 1000).clip(lower=0), 2: s + 1000}))
A = pd.concat(parts).sort_values([0, 1]); A.to_csv(O / "summits_1kb.bed", sep="\t", header=False, index=False)
m = subprocess.run(f"bedtools merge -i {O / 'summits_1kb.bed'} | bedtools intersect -v -a - -b {BLACKLIST}", shell=True,
                   check=True, capture_output=True, text=True).stdout
M = pd.read_csv(pd.io.common.StringIO(m), sep="\t", header=None); M[3] = [f"r{i}" for i in range(len(M))]
M.to_csv(O / "peaks.bed", sep="\t", header=False, index=False)
print("ChIP summits", len(A), "merged regions", len(M), flush=True)
