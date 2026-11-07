#!/usr/bin/env python
"""Diagnostic 2a: how much do the panel bulk references overlap each other?

Outputs (OUT/diag/):
  ref_jaccard_bp.tsv        bp-level Jaccard (bedtools jaccard on merged peaks)
  ref_frac_overlap.tsv      row i, col j: fraction of i's peaks overlapping >=1 peak of j
  ref_jaccard_bins.tsv      Jaccard of 1 kb positive bins
  ref_annotation.tsv        per reference: n peaks, median width, frac at promoter (TSS+-1kb),
                            frac in ATAC union3, frac shared with >=1 / >=3 other references, n unique
"""
import subprocess

import numpy as np
import pandas as pd

from fig2_common import NT, ATAC_UNION, GTF, INP, OUT, REF, REF_SOURCE, TFS, load_inputs, read_bed, write_mat

od = OUT / "diag"
od.mkdir(parents=True, exist_ok=True)
tmp = od / "tmp"
tmp.mkdir(exist_ok=True)


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


for tf in TFS:
    read_bed(REF[tf])[[0, 1, 2]].sort_values([0, 1]).to_csv(tmp / f"{tf}.raw.bed", sep="\t", header=False, index=False)
    (tmp / f"{tf}.bed").write_text(sh("bedtools", "merge", "-i", str(tmp / f"{tf}.raw.bed")))

g = pd.read_csv(GTF, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 6])
g = g[g[2] == "transcript"]
t = np.where(g[6] == "+", g[3] - 1, g[4] - 1)
pd.DataFrame({0: g[0].values, 1: np.maximum(t - 1000, 0), 2: t + 1001}).sort_values([0, 1]).to_csv(
    tmp / "prom.unsorted.bed", sep="\t", header=False, index=False)
(tmp / "prom.bed").write_text(sh("bedtools", "merge", "-i", str(tmp / "prom.unsorted.bed")))
(tmp / "atac.bed").write_text(sh("bash", "-c", f"zcat {ATAC_UNION} | cut -f1-3 | sort -k1,1 -k2,2n | bedtools merge"))

J = np.eye(NT)
F = np.eye(NT)
for i, a in enumerate(TFS):
    na = len(read_bed(tmp / f"{a}.raw.bed"))
    for j, b in enumerate(TFS):
        if i == j:
            continue
        if j > i:
            out = sh("bedtools", "jaccard", "-a", str(tmp / f"{a}.bed"), "-b", str(tmp / f"{b}.bed")).splitlines()
            J[i, j] = J[j, i] = float(out[1].split("\t")[2])
        nov = len(sh("bedtools", "intersect", "-u", "-a", str(tmp / f"{a}.raw.bed"), "-b", str(tmp / f"{b}.bed")).splitlines())
        F[i, j] = nov / na
write_mat(J, od / "ref_jaccard_bp.tsv", rlab="reference", clab="reference")
write_mat(F, od / "ref_frac_overlap.tsv", rlab="reference_peaks", clab="overlapping_reference")

fr = load_inputs()
ref, uni = fr["ref"], fr["universe"]
JB = np.eye(NT)
for i in range(NT):
    for j in range(NT):
        a, b = ref[i] & uni, ref[j] & uni
        JB[i, j] = (a & b).sum() / (a | b).sum()
write_mat(JB, od / "ref_jaccard_bins.tsv", rlab="reference", clab="reference")

rows = []
for i, tf in enumerate(TFS):
    raw = tmp / f"{tf}.raw.bed"
    pk = read_bed(raw)
    n = len(pk)
    others = [str(tmp / f"{o}.bed") for o in TFS if o != tf]
    nref = np.zeros(n, int)
    for o in others:
        hit = sh("bedtools", "intersect", "-c", "-a", str(raw), "-b", o).splitlines()
        nref += np.array([int(x.split("\t")[-1]) > 0 for x in hit])
    rows.append(dict(
        reference=tf, source=REF_SOURCE[tf], n_peaks=n, median_width=int((pk[2] - pk[1]).median()),
        frac_promoter=len(sh("bedtools", "intersect", "-u", "-a", str(raw), "-b", str(tmp / "prom.bed")).splitlines()) / n,
        frac_atac_union=len(sh("bedtools", "intersect", "-u", "-a", str(raw), "-b", str(tmp / "atac.bed")).splitlines()) / n,
        frac_shared_ge1=float((nref >= 1).mean()), frac_shared_ge3=float((nref >= 2).mean()),  # self + >=2 others = >=3 refs
        n_unique=int((nref == 0).sum()), frac_unique=float((nref == 0).mean()),
        n_unique_bins=int((fr["unique"][i] & uni).sum()), n_ref_bins=int((ref[i] & uni).sum()),
        frac_unique_promoter=float(fr["promoter"][fr["unique"][i] & uni].mean()),
        frac_unique_atac=float(fr["atac_union"][fr["unique"][i] & uni].mean()),
    ))
pd.DataFrame(rows).to_csv(od / "ref_annotation.tsv", sep="\t", index=False, float_format="%.4g")
print(pd.DataFrame(rows).drop(columns="source").to_string())
print(pd.DataFrame(J, TFS, TFS).round(3).to_string())
print(pd.DataFrame(F, TFS, TFS).round(3).to_string())
print("shared>=3 bins:", int((fr["shared3"] & uni).sum()))
