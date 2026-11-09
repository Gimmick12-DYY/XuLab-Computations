#!/usr/bin/env python
"""Round 2, step 1: bulk summits per reference and per-summit sc read counts (centre and flanks).

Summits: narrowPeak column 10 (peak start + offset); duplicate summits removed; chr1-22, X.
  CTCF: summits of ENCODE ENCFF314ZAL (data/CTCF_HEK293_peaks.bed) that fall inside the
        majority-2/3 CTCF set; others: their own reference file. ZNF282: MACS2 summits of the ChIP-exo.
  Bulk rank signal = narrowPeak signalValue (column 7).
Annotations per summit: promoter (within TSS +-1 kb, GENCODE v46 basic), unique (summit not inside any
peak of the other 5 references), motif (CTCF: CTCF_HUMAN.H11MO.0.A p<1e-5; MAZ: MAZ_HUMAN.H11MO.0.A p<1e-4;
best FIMO hit in summit +-100 bp), bulk read counts of all 6 ChIP BAMs in summit +-250 bp (for the
'preferred' set: own log CPM minus mean of the other 5 > 1).
sc counts per summit for all 79 TFs (read 5' ends): centre = |d| <= 100 bp, flank = 1000 < |d| <= 2000 bp.
Outputs OUT/round2/: summits.tsv, summit_counts.npz (C, F: 79 x S int32; tf79; tot79).
"""
import subprocess

import numpy as np
import pandas as pd

from fig2_common import BAM, D, FASTA, GTF, INP, MEME_DB, OUT, REF, TFS, read_bed

R2 = OUT / "round2"
R2.mkdir(parents=True, exist_ok=True)
tmp = R2 / "tmp"
tmp.mkdir(exist_ok=True)
CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
CEN, FL0, FL1 = 100, 1000, 2000


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def merged(df):
    out = {}
    for c, g in df.groupby(0):
        s, e = g[1].values.astype(np.int64), g[2].values.astype(np.int64)
        o = np.argsort(s)
        s, e = s[o], e[o]
        ms, me = [s[0]], [e[0]]
        for a, b in zip(s[1:], e[1:]):
            if a <= me[-1]:
                me[-1] = max(me[-1], b)
            else:
                ms.append(a); me.append(b)
        out[c] = (np.array(ms), np.array(me))
    return out


def inside(iv, chrom, pos):
    res = np.zeros(len(pos), bool)
    for c in np.unique(chrom):
        if c not in iv:
            continue
        s, e = iv[c]
        m = chrom == c
        k = np.searchsorted(s, pos[m], side="right") - 1
        res[m] = (k >= 0) & (pos[m] < e[np.maximum(k, 0)])
    return res


# ---- summits ----
rows = []
for t in TFS:
    src = D / "CTCF_HEK293_peaks.bed" if t == "CTCF" else REF[t]
    pk = read_bed(src, ncol=10)
    pk = pk[pk[0].isin(CHROMS)]
    s = pd.DataFrame({"ref": t, "chrom": pk[0].values, "pos": (pk[1] + pk[9].astype(int)).values,
                      "signal": pk[6].astype(float).values})
    s = s.sort_values("signal", ascending=False).drop_duplicates(["chrom", "pos"])
    if t == "CTCF":
        s = s[inside(merged(read_bed(REF["CTCF"])), s.chrom.values, s.pos.values)]
    rows.append(s)
S = pd.concat(rows, ignore_index=True)
ivs = {t: merged(read_bed(REF[t])) for t in TFS}
S["unique"] = True
for t in TFS:
    m = S.ref.values == t
    for o in TFS:
        if o != t:
            S.loc[m, "unique"] &= ~inside(ivs[o], S.chrom.values[m], S.pos.values[m])
g = pd.read_csv(GTF, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 6])
g = g[g[2] == "transcript"]
tss = np.where(g[6] == "+", g[3] - 1, g[4] - 1)
prom = merged(pd.DataFrame({0: g[0].values, 1: np.maximum(tss - 1000, 0), 2: tss + 1001}))
S["promoter"] = inside(prom, S.chrom.values, S.pos.values)
print(S.groupby("ref").agg(n=("pos", "size"), unique=("unique", "mean"), promoter=("promoter", "mean")), flush=True)

# ---- motifs (CTCF, MAZ summits) ----
S["motif"] = np.nan
for t, motif, thr in (("CTCF", "CTCF_HUMAN.H11MO.0.A", 1e-5), ("MAZ", "MAZ_HUMAN.H11MO.0.A", 1e-4)):
    idx = np.flatnonzero(S.ref.values == t)
    bed = tmp / f"motif_{t}.bed"
    pd.DataFrame({0: S.chrom.values[idx], 1: S.pos.values[idx] - CEN, 2: S.pos.values[idx] + CEN,
                  3: [f"r{k}" for k in idx]}).to_csv(bed, sep="\t", header=False, index=False)
    fa = sh("bedtools", "getfasta", "-fi", str(FASTA), "-bed", str(bed), "-name")
    (tmp / f"motif_{t}.fa").write_text("".join((l.split("::")[0] if l.startswith(">") else l) + "\n"
                                               for l in fa.splitlines()))
    txt = sh("fimo", "--text", "--thresh", str(thr), "--motif", motif, str(MEME_DB), str(tmp / f"motif_{t}.fa"))
    hit = set(pd.read_csv(pd.io.common.StringIO(txt), sep="\t").sequence_name.astype(str))
    S.loc[idx, "motif"] = [f"r{k}" in hit for k in idx]
    print(t, "motif frac", S.loc[idx, "motif"].mean(), flush=True)

# ---- bulk counts at summit +-250 ----
bb = tmp / "summits_250.bed"
o = np.lexsort((S.pos.values, S.chrom.values))
pd.DataFrame({0: S.chrom.values[o], 1: S.pos.values[o] - 250, 2: S.pos.values[o] + 250, 3: o}).to_csv(
    bb, sep="\t", header=False, index=False)
out = sh("bedtools", "multicov", "-bams", *[str(BAM[t]) for t in TFS], "-bed", str(bb))
Bk = np.zeros((len(S), len(TFS)))
for l in out.splitlines():
    p = l.split("\t")
    Bk[int(p[3])] = [float(x) for x in p[4:]]
lc = np.log2(Bk / Bk.sum(0, keepdims=True) * 1e6 + 1)
for k, t in enumerate(TFS):
    S[f"bulk_{t}"] = Bk[:, k]
own = np.array([lc[r, TFS.index(t)] for r, t in enumerate(S.ref.values)])
others = np.array([np.delete(lc[r], TFS.index(t)).mean() for r, t in enumerate(S.ref.values)])
S["bulk_spec"] = own - others
S["preferred"] = S.bulk_spec > 1
print(S.groupby("ref")[["preferred"]].mean(), flush=True)
S.to_csv(R2 / "summits.tsv", sep="\t", index=False)

# ---- sc counts ----
tf79 = list(np.load(INP / "frame.npz", allow_pickle=True)["tf79"])
C = np.zeros((len(tf79), len(S)), np.int32)
F = np.zeros_like(C)
tot = np.zeros(len(tf79), np.int64)
for k, t in enumerate(tf79):
    z = np.load(INP / "five" / f"{t}.npz")
    tot[k] = sum(len(z[c]) for c in z.files)
    for c in CHROMS:
        m = np.flatnonzero(S.chrom.values == c)
        if not len(m):
            continue
        p = z[c]
        x = S.pos.values[m]
        cnt = lambda a, b: np.searchsorted(p, x + b, side="right") - np.searchsorted(p, x + a, side="left")
        C[k, m] = cnt(-CEN, CEN)
        F[k, m] = cnt(-FL1, -FL0 - 1) + cnt(FL0 + 1, FL1)
    print("counts", t, flush=True)
np.savez(R2 / "summit_counts.npz", C=C, F=F, tf79=np.array(tf79), tot79=tot)
print("done", len(S))
