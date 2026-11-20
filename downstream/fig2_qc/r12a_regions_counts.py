#!/usr/bin/env python
"""Round 6 (differential peaks), step 1: regions, counts and annotations (see round6_SPEC.md).
Rows = D-B candidate regions (ATAC union3 + lab MACS2 q<0.05 summits +-250 bp of the 6 panel TFs, merged, chr1-22/X,
minus blacklist) plus each panel TF's D-A regions (MACS2 re-call vs the other-78 pool, q<0.05, top 5,000 by p, summit +-250).
Per row: sc 5'-end counts for all 79 TFs, bulk reads for 7 BAMs (6 references + CTCF HEK293T nexus), overlap with
7 reference peak sets, OmniATAC fold-enrichment mean, GC fraction. Output OUT/round6_diffpeaks/regions.npz + rows.tsv."""
import subprocess

import numpy as np
import pandas as pd
import pyBigWig

from fig2_common import ATAC_BW, ATAC_UNION, BAM, BLACKLIST, D, FASTA, INP, OUT, REF, ROOT, TFS

O6 = OUT / "round6_diffpeaks"
tmp = O6 / "tmp"
tmp.mkdir(parents=True, exist_ok=True)
CH = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
LAB = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
RECALL = str(ROOT / "complex_screen/pair_motif_composite/recall/peaks_tf") + "/TF.{t}_peaks.narrowPeak"
NEXUS_BAM = ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam"
REFS = dict(REF); REFS["CTCF"] = D / "CTCF_HEK293_peaks.bed"; REFS["CTCF_nexus"] = D / "CTCF_HEK293T_GSE103651_peaks.bed"
BAMS = dict(BAM); BAMS["CTCF_nexus"] = NEXUS_BAM
COLS = TFS + ["CTCF_nexus"]


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def np_read(p):
    return pd.read_csv(p, sep="\t", header=None, usecols=range(10))


# candidates
parts = [pd.read_csv(ATAC_UNION, sep="\t", header=None, usecols=[0, 1, 2])]
for t in TFS:
    pk = np_read(LAB.format(t=t))
    pk = pk[pk[8] >= 1.30103]
    s = pk[1] + pk[9]
    parts.append(pd.DataFrame({0: pk[0], 1: np.maximum(s - 250, 0), 2: s + 250}))
c = pd.concat(parts)
c = c[c[0].isin(CH)].sort_values([0, 1])
c.to_csv(tmp / "cand_raw.bed", sep="\t", header=False, index=False)
(tmp / "cand_m.bed").write_text(sh("bedtools", "merge", "-i", str(tmp / "cand_raw.bed")))
(tmp / "cand.bed").write_text(sh("bedtools", "subtract", "-A", "-a", str(tmp / "cand_m.bed"), "-b", str(BLACKLIST)))
cand = pd.read_csv(tmp / "cand.bed", sep="\t", header=None)
rows = [pd.DataFrame({"kind": "cand", "chrom": cand[0], "start": cand[1], "end": cand[2], "p": np.nan})]
for t in TFS:
    pk = np_read(RECALL.format(t=t))
    pk = pk[(pk[8] >= 1.30103) & pk[0].isin(CH)].sort_values(7, ascending=False).drop_duplicates([0, 1, 9]).head(5000)
    s = pk[1] + pk[9]
    rows.append(pd.DataFrame({"kind": f"DA_{t}", "chrom": pk[0].values, "start": np.maximum(s - 250, 0).values,
                              "end": (s + 250).values, "p": pk[7].values}))
    print(t, "D-A q<0.05 available", int(((np_read(RECALL.format(t=t))[8] >= 1.30103)).sum()), "kept", len(pk), flush=True)
R = pd.concat(rows, ignore_index=True)
R.to_csv(O6 / "rows.tsv", sep="\t", index=False)
print("rows", len(R), "candidates", int((R.kind == "cand").sum()), flush=True)

# sc counts
tf79 = [str(t) for t in np.load(INP / "frame.npz", allow_pickle=True)["tf79"]]
SC = np.zeros((79, len(R)), np.int32); TOT = np.zeros(79)
for k, t in enumerate(tf79):
    z = np.load(INP / "five" / f"{t}.npz")
    TOT[k] = sum(len(z[c_]) for c_ in z.files)
    for ch in CH:
        m = np.flatnonzero(R.chrom.values == ch)
        if len(m):
            p = z[ch]
            SC[k, m] = np.searchsorted(p, R.end.values[m]) - np.searchsorted(p, R.start.values[m])
print("sc counts done", flush=True)

# bed of rows with index
bed = tmp / "rows.bed"
o = np.lexsort((R.start.values, R.chrom.values))
pd.DataFrame({0: R.chrom.values[o], 1: R.start.values[o], 2: R.end.values[o], 3: o}).to_csv(bed, sep="\t", header=False, index=False)
out = sh("bedtools", "multicov", "-bams", *[str(BAMS[k]) for k in COLS], "-bed", str(bed))
BK = np.zeros((len(R), len(COLS)))
for l in out.splitlines():
    q = l.split("\t")
    BK[int(q[3])] = [float(x) for x in q[4:]]
MAPPED = np.array([sum(int(l.split("\t")[2]) for l in sh("samtools", "idxstats", str(BAMS[k])).splitlines()) for k in COLS], float)
print("bulk counts done", flush=True)
OV = np.zeros((len(R), len(COLS)), bool)
for j, k in enumerate(COLS):
    refb = tmp / f"ref_{k}.bed"
    pd.read_csv(REFS[k], sep="\t", header=None, usecols=[0, 1, 2]).sort_values([0, 1]).to_csv(refb, sep="\t", header=False, index=False)
    hit = sh("bedtools", "intersect", "-u", "-a", str(bed), "-b", str(refb))
    OV[[int(l.split("\t")[3]) for l in hit.splitlines()], j] = True
bw = pyBigWig.open(str(ATAC_BW))
AT = np.array([np.nan_to_num(bw.stats(c_, int(s), int(e), type="mean")[0] or 0) for c_, s, e in zip(R.chrom, R.start, R.end)])
nuc = sh("bedtools", "nuc", "-fi", str(FASTA), "-bed", str(bed))
GC = np.zeros(len(R))
for l in nuc.splitlines()[1:]:
    q = l.split("\t")
    GC[int(q[3])] = float(q[5])
np.savez(O6 / "regions.npz", SC=SC, TOT=TOT, tf79=np.array(tf79), BK=BK, MAPPED=MAPPED, OV=OV, AT=AT, GC=GC, cols=np.array(COLS))
print("done")
