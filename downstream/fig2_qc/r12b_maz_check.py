#!/usr/bin/env python
"""Round 12 follow-up (user 2026-11-23 21:38-21:43Z: why MAZ is weak, why MAZ sc tracks the ZNF282 ChIP column).
Read-only on existing inputs; same 180 summit groups as r12_nopool.py (no-pool centre/flank variant, ZNF777 depth,
3 seeds). Per ChIP block of 30 groups: promoter share (TSS +-1 kb, summits.tsv), CpG-island-like share (summit +-250 bp
GC > 0.5 and CpG obs/exp > 0.6; no CGI track on Longleaf), mean ATAC fold-enrichment +-250 bp, top HEK293T ChromHMM
18-state labels, share within 250 bp of a MAZ ChIP peak, MAZ core motif (GGGAGGG / CCCTCCC) within +-100 bp;
per-block mean of MAZ sc and ZNF282 sc; leave-top-3-out r; MAZ-block ranges; depth and FRiP of the 6 TFs.
Writes OUT/round12_nopool/maz_check_*.tsv."""
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pyBigWig

sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream/fig2_qc")
from fig2_common import ATAC_BW, ATAC_UNION, D, DM_DEPTH, DM_SEEDS, FASTA, INP, OUT, REF, TFS  # noqa: E402

O = OUT / "round12_nopool"; T = O / "tmp_maz"; T.mkdir(exist_ok=True)
G = 30
S = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
z = np.load(OUT / "round2/summit_counts.npz", allow_pickle=True)
C79, F79, tf79, tot79 = z["C"].astype(np.int64), z["F"].astype(np.int64), list(z["tf79"]), z["tot79"]
pi = [tf79.index(t) for t in TFS]
Bk = S[[f"bulk_{t}" for t in TFS]].values
gr, gref = [], []
for t in TFS:
    ix = np.flatnonzero((S.ref.values == t) & S.preferred.values.astype(bool))
    ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:5000]
    for g in np.array_split(ix, G):
        gr.append(g); gref.append(t)
gref = np.array(gref)
gB = np.array([Bk[g].sum(0) for g in gr]); bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
pf = np.minimum(1, DM_DEPTH / tot79[pi])
SC = []
for s in DM_SEEDS:
    Cs = np.random.default_rng(s).binomial(C79[pi], pf[:, None]); Fs = np.random.default_rng(s + 1).binomial(F79[pi], pf[:, None])
    SC.append(np.array([[np.log2((Cs[i, g].sum() + .5) / (Fs[i, g].sum() + .5)) for i in range(6)] for g in gr]))
SC = np.mean(SC, 0)   # groups x 6, no-pool centre/flank
iM, iZ = TFS.index("MAZ"), TFS.index("ZNF282")
pd.set_option("display.width", 250)

# 1) columns of the no-pool matrix
M = pd.read_csv(O / "m_nopool_cf_all.tsv", sep="\t", index_col=0)
c1 = pd.DataFrame({"ZNF282 ChIP col": M["ZNF282 ChIP"], "MAZ ChIP col": M["MAZ ChIP"]})
print("1) no-pool centre/flank r\n", c1.round(3).to_string())
print("column mean / SD: ZNF282", round(M["ZNF282 ChIP"].mean(), 3), round(M["ZNF282 ChIP"].std(ddof=0), 3),
      "| MAZ", round(M["MAZ ChIP"].mean(), 3), round(M["MAZ ChIP"].std(ddof=0), 3))

# 2) annotation per summit, then per block (all 5,000 summits of each ChIP)
allix = np.concatenate(gr); sub = S.iloc[allix].reset_index(drop=True); blk = np.repeat(gref, [len(g) for g in gr])
bed = pd.DataFrame({0: sub.chrom, 1: (sub.pos - 250).clip(lower=0), 2: sub.pos + 250, 3: np.arange(len(sub))})
bed.to_csv(T / "s250.bed", sep="\t", header=False, index=False)
seqs = subprocess.run(f"bedtools getfasta -fi {FASTA} -bed {T / 's250.bed'} -name -tab", shell=True, check=True,
                      capture_output=True, text=True).stdout.splitlines()
seq = {int(l.split("\t")[0].split("::")[0]): l.split("\t")[1].upper() for l in seqs}
gc = np.zeros(len(sub)); oe = np.zeros(len(sub)); mot = np.zeros(len(sub), bool)
pat = re.compile("GGGAGGG|CCCTCCC")
for i in range(len(sub)):
    q = seq.get(i, ""); n = max(len(q), 1); c, g = q.count("C"), q.count("G")
    gc[i] = (c + g) / n; oe[i] = q.count("CG") * n / max(c * g, 1); mot[i] = bool(pat.search(q[150:351]))
cgi = (gc > 0.5) & (oe > 0.6)
bw = pyBigWig.open(str(ATAC_BW))
atac = np.array([np.nan_to_num(bw.stats(c, int(max(p - 250, 0)), int(p + 250), type="mean")[0] or 0) for c, p in zip(sub.chrom, sub.pos)])
def near(bedfile, d):
    out = subprocess.run(f"bedtools window -u -w {d} -a {T / 's250c.bed'} -b {bedfile}", shell=True, check=True, capture_output=True, text=True).stdout
    hit = np.zeros(len(sub), bool); hit[[int(l.split("\t")[3]) for l in out.splitlines()]] = True; return hit
pd.DataFrame({0: sub.chrom, 1: sub.pos, 2: sub.pos + 1, 3: np.arange(len(sub))}).to_csv(T / "s250c.bed", sep="\t", header=False, index=False)
mazpk = near(REF["MAZ"], 250)
hmm = subprocess.run(f"bedtools intersect -wa -wb -a {T / 's250c.bed'} -b {D / 'HEK293T_chromHMM18.bed.gz'}", shell=True, check=True,
                     capture_output=True, text=True).stdout
st = pd.Series({int(l.split("\t")[3]): l.split("\t")[7] for l in hmm.splitlines()}).reindex(range(len(sub))).fillna("none").values
rows = []
for t in TFS:
    m = blk == t
    vc = pd.Series(st[m]).value_counts(normalize=True)
    rows.append(dict(block=t, promoter=sub.promoter.values[m].astype(bool).mean(), cgi_like=cgi[m].mean(), GC=gc[m].mean(),
                     atac_FE=atac[m].mean(), near_MAZ_peak_250=mazpk[m].mean(), MAZ_core_motif=mot[m].mean(),
                     chromhmm_top3=", ".join(f"{k} {v:.2f}" for k, v in vc.head(3).items())))
A2 = pd.DataFrame(rows); A2.to_csv(O / "maz_check_blocks.tsv", sep="\t", index=False, float_format="%.3f")
print("2) per ChIP block (5,000 summits each)\n", A2.round(3).to_string())

# 3) MAZ sc and ZNF282 sc per block; leave-top-3-out
B3 = pd.DataFrame([dict(block=t, MAZ_sc=SC[gref == t, iM].mean(), ZNF282_sc=SC[gref == t, iZ].mean(),
                        ZNF282_ChIP=bulk[gref == t, iZ].mean(), MAZ_ChIP=bulk[gref == t, iM].mean()) for t in TFS])
B3.to_csv(O / "maz_check_sc_blocks.tsv", sep="\t", index=False, float_format="%.3f"); print("3) per-block means (no-pool centre/flank)\n", B3.round(3).to_string())
def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])
x = bulk[:, iZ]; top = np.argsort(-x)[:3]; keep = np.setdiff1d(np.arange(len(x)), top)
print("MAZ sc vs ZNF282 ChIP r all", round(r(SC[:, iM], x), 3), "| without top-3 ZNF282-ChIP groups", round(r(SC[keep, iM], x[keep]), 3),
      "| top-3 groups are from", list(gref[top]))
for i, t in ((iM, "MAZ sc"), (iZ, "ZNF282 sc")):
    rr = [r(SC[gref != b, i], x[gref != b]) for b in TFS]
    print(t, "vs ZNF282 ChIP, dropping one block at a time:", dict(zip(TFS, np.round(rr, 3))))

# 4) MAZ block ranges, depth and FRiP
m = gref == "MAZ"
print("4) MAZ block (30 groups): MAZ ChIP log2CPM range", np.round([bulk[m, iM].min(), bulk[m, iM].max()], 2), "SD", round(bulk[m, iM].std(), 3),
      "| MAZ sc range", np.round([SC[m, iM].min(), SC[m, iM].max()], 2), "SD", round(SC[m, iM].std(), 3),
      "| r within MAZ block", round(r(SC[m, iM], bulk[m, iM]), 3))
for t in TFS:
    mm = gref == t; i = TFS.index(t)
    print(f"   {t} own block: ChIP SD {bulk[mm, i].std():.3f}, own sc SD {SC[mm, i].std():.3f}, r within block {r(SC[mm, i], bulk[mm, i]):+.3f}")
au = pd.read_csv(ATAC_UNION, sep="\t", header=None, usecols=[0, 1, 2])
LAB = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
dr = []
for t in TFS:
    f5 = np.load(INP / "five" / f"{t}.npz"); tot = sum(len(f5[c]) for c in f5.files)
    def frac(P):
        n = 0
        for c, g in P.groupby(0):
            if c in f5.files:
                q = f5[c]; n += int((np.searchsorted(q, g[2].values) - np.searchsorted(q, g[1].values)).sum())
        return n / tot
    L = pd.read_csv(LAB.format(t=t), sep="\t", header=None, usecols=range(9)); L = L[L[8] >= 1.30103]
    dr.append(dict(TF=t, reads=tot, cells=int(np.load(INP / "frame.npz", allow_pickle=True)["ncells_imp79"][tf79.index(t)]),
                   own_peaks=len(L), FRiP_own_peaks=frac(L), frac_in_ATAC_peaks=frac(au)))
D4 = pd.DataFrame(dr); D4.to_csv(O / "maz_check_depth.tsv", sep="\t", index=False, float_format="%.4f"); print(D4.round(3).to_string())
