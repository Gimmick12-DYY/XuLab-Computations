#!/usr/bin/env python
"""T5: motif-anchored aggregate raw-read profiles at reference-unique sites.

CTCF: unique CTCF majority peaks, best FIMO hit of CTCF_HUMAN.H11MO.0.A (p < 1e-5) per peak.
MAZ: unique MAZ peaks, summit +-100 bp, best FIMO hit of MAZ_HUMAN.H11MO.0.A (p < 1e-4).
Anchor = motif centre, oriented by motif strand. Profiles: read 5' ends +-1 kb in 20 bp bins for the panel
TFs and the pooled 79-TF panel, as reads per million TF reads per site (CPM / site).
Centre enrichment = mean density within +-100 bp / mean density at 600-1000 bp (both flanks).
Outputs OUT/spec/T5_anchors_{TF}.bed, T5_profile_{TF}.tsv, T5_centre_enrichment.tsv.
"""
import subprocess

import numpy as np
import pandas as pd

from fig2_common import NT, FASTA, INP, MEME_DB, OUT, REF, TFS, read_bed

od = OUT / "spec"
od.mkdir(parents=True, exist_ok=True)
W, STEP = 1000, 20
edges = np.arange(-W, W + 1, STEP)
centres = (edges[:-1] + edges[1:]) / 2
MOTIF = {"CTCF": ("CTCF_HUMAN.H11MO.0.A", 1e-5, None), "MAZ": ("MAZ_HUMAN.H11MO.0.A", 1e-4, 100)}
tf79 = list(np.load(INP / "frame.npz", allow_pickle=True)["tf79"])


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def anchors(tf):
    motif, thr, half = MOTIF[tf]
    uq = read_bed(INP / f"unique_{tf}.bed")
    if half is not None:  # use summits of the original narrowPeak rows that are unique
        full = read_bed(REF[tf], ncol=10)
        key = set(zip(uq[0], uq[1], uq[2]))
        full = full[[k in key for k in zip(full[0], full[1], full[2])]]
        s = full[1] + full[9].astype(int)
        reg = pd.DataFrame({0: full[0].values, 1: (s - half).values, 2: (s + half).values})
    else:
        reg = uq[[0, 1, 2]].copy()
    reg = reg.reset_index(drop=True)
    reg[3] = [f"r{k}" for k in range(len(reg))]   # plain names: FIMO would parse chr:a-b names itself
    bed = od / f"T5_regions_{tf}.bed"
    reg.to_csv(bed, sep="\t", header=False, index=False)
    fa = od / f"T5_regions_{tf}.fa"
    fasta = sh("bedtools", "getfasta", "-fi", str(FASTA), "-bed", str(bed), "-name")
    # keep only the plain rN name: FIMO parses chr:start-end headers into genomic coordinates
    fa.write_text("".join(l.split("::")[0] + "
" if l.startswith(">") else l + "
" for l in fasta.splitlines()))
    txt = sh("fimo", "--text", "--thresh", str(thr), "--motif", motif, str(MEME_DB), str(fa))
    h = pd.read_csv(pd.io.common.StringIO(txt), sep="\t")
    h = h.sort_values("p-value").drop_duplicates("sequence_name")
    k = h.sequence_name.astype(str).str.replace(r"::.*$", "", regex=True).str[1:].astype(int).values
    cen = reg[1].values[k] + (h.start.values + h.stop.values) // 2 - 1
    a = pd.DataFrame({"chrom": reg[0].values[k], "pos": cen, "strand": h.strand.values, "p": h["p-value"].values})
    a = a.sort_values(["chrom", "pos"]).reset_index(drop=True)
    a.assign(end=a.pos + 1)[["chrom", "pos", "end", "p", "p", "strand"]].to_csv(
        od / f"T5_anchors_{tf}.bed", sep="\t", header=False, index=False)
    print(tf, "regions", len(reg), "with motif", len(a), flush=True)
    return a, len(reg)


def profile(five, A):
    h = np.zeros(len(centres))
    for c, g in A.groupby("chrom"):
        pos = five[c]
        lo = np.searchsorted(pos, g.pos.values - W)
        hi = np.searchsorted(pos, g.pos.values + W)
        for a, s, l, u in zip(g.pos.values, g.strand.values, lo, hi):
            rel = pos[l:u] - a
            if s == "-":
                rel = -rel
            h += np.histogram(rel, edges)[0]
    return h


rows = []
for tf in ("CTCF", "MAZ"):
    A, nreg = anchors(tf)
    prof = {}
    pooled = np.zeros(len(centres)); ptot = 0
    for t in tf79:
        z = np.load(INP / "five" / f"{t}.npz")
        five = {c: z[c] for c in z.files}
        tot = sum(len(v) for v in five.values())
        h = profile(five, A)
        pooled += h; ptot += tot
        if t in TFS:
            prof[t] = h / tot * 1e6 / len(A)
    prof["pooled79"] = pooled / ptot * 1e6 / len(A)
    df = pd.DataFrame(prof, index=centres).T
    df.columns = [f"{int(c)}" for c in centres]
    df.to_csv(od / f"T5_profile_{tf}.tsv", sep="\t", float_format="%.5g")
    inner = np.abs(centres) <= 100
    outer = np.abs(centres) >= 600
    for k, v in prof.items():
        rows.append(dict(anchor=tf, n_regions=nreg, n_anchors=len(A), track=k,
                         centre_cpm_per_site=float(v[inner].sum()), centre_over_flank=float(v[inner].mean() / v[outer].mean())))
out = pd.DataFrame(rows)
out.to_csv(od / "T5_centre_enrichment.tsv", sep="\t", index=False, float_format="%.4g")
print(out.to_string())
