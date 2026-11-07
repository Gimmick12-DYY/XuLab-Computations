#!/usr/bin/env python
"""Cache per-TF single-cell read 5' positions (bp, hg38) as sorted int32 arrays per chromosome.

Input: /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.<TF>.bed.gz (read-only; single-end reads,
not Tn5-shifted, not deduplicated). 5' end = start on '+', end-1 on '-' (same rule as
ctcf_promoter_fp/01_build_counts.py, so totals match raw.npz).
Output: OUT/inputs/five/<TF>.npz with one array per chromosome (chr1-22, X, Y).
Usage: 00b_five_prime_cache.py [TF ...]   (default: all 79)
"""
import sys

import numpy as np
import pandas as pd

from fig2_common import FRAG_DIR, INP

keep = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]
out = INP / "five"
out.mkdir(parents=True, exist_ok=True)
tfs = sys.argv[1:] or sorted(p.name[3:-7] for p in FRAG_DIR.glob("TF.*.bed.gz"))
for tf in tfs:
    if (out / f"{tf}.npz").exists():
        continue
    d = pd.read_csv(FRAG_DIR / f"TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5],
                    names=["chrom", "start", "end", "strand"], dtype={"chrom": "category", "strand": "category"})
    d = d[d.chrom.isin(keep)]
    five = np.where(d.strand.values == "+", d.start.values, d.end.values - 1).astype(np.int32)
    arrs = {}
    for c in keep:
        arrs[c] = np.sort(five[(d.chrom == c).values])
    np.savez(out / f"{tf}.npz", **arrs)
    print(tf, len(d), flush=True)
