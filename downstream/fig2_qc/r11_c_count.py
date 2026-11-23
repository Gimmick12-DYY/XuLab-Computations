#!/usr/bin/env python
"""Round 11 step c (SPEC.md 2; one BAM per task): filtered 5' ends per region. Usage: r11_c_count.py <0-11>
-> O11/counts/<ChIP>_rep<k>.npy, readinfo_<ChIP>_rep<k>.tsv (paired?, read length, reads counted)."""
import subprocess
import sys

import numpy as np
import pandas as pd

from r11_common import O11, SAMPLES

t, rep, bam = SAMPLES[int(sys.argv[1])]
(O11 / "counts").mkdir(exist_ok=True)
head = subprocess.run(f"samtools view {bam} | head -20000", shell=True, capture_output=True, text=True, executable="/bin/bash").stdout.splitlines()
flags = np.array([int(l.split("\t")[1]) for l in head]); rl = int(np.median([len(l.split("\t")[9]) for l in head]))
paired = bool((flags & 1).any())
sel = "-f 64" if paired else ""
cmd = f"samtools view -u -q 30 -F 0xF04 {sel} {bam} | bedtools bamtobed -i - | cut -f1,2,3,6"
pos = {}
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, executable="/bin/bash")
for ch in pd.read_csv(p.stdout, sep="\t", header=None, names=["c", "s", "e", "st"], chunksize=5_000_000,
                      dtype={"c": str, "s": np.int64, "e": np.int64, "st": str}):
    five = np.where(ch.st.values == "+", ch.s.values, ch.e.values - 1)
    for c, ix in ch.groupby("c").indices.items():
        pos.setdefault(c, []).append(five[ix])
assert p.wait() == 0
pos = {c: np.sort(np.concatenate(v)) for c, v in pos.items()}
n = sum(len(v) for v in pos.values())
Rg = pd.read_csv(O11 / "regions.tsv", sep="\t", usecols=["chrom", "start", "end"])
x = np.zeros(len(Rg), np.int64)
for c, g in Rg.groupby("chrom"):
    q = pos.get(c)
    if q is not None:
        x[g.index.values] = np.searchsorted(q, g.end.values) - np.searchsorted(q, g.start.values)
np.save(O11 / "counts" / f"{t}_rep{rep}.npy", x)
pd.DataFrame([dict(ChIP=t, rep=rep, bam=str(bam), paired=paired, read_length=rl, reads_counted=n, in_regions=int(x.sum()))]).to_csv(
    O11 / "counts" / f"readinfo_{t}_rep{rep}.tsv", sep="\t", index=False)
print(t, rep, "paired" if paired else "single", rl, n, int(x.sum()), flush=True)
