#!/usr/bin/env python
"""Coordinate audit for the round 6 differential regions (D-B top 1,000, shared removed) used by r13/r13b/r16.
Checks: region width; overlap with the HEK293T ATAC union; overlap with the TF's own lab MACS2 q<0.05 peaks; within 500 bp
of the reference peak file, observed vs (a) 20 random shuffles within the genome (bedtools shuffle, blacklist excluded,
chr1-22,X) and (b) 20 random draws of candidate rows (rows.tsv) matched on nothing; ChIP centre/flank per block from the
r13 matrices (ChIP column of each block) to reconcile L3 (pooled) with the self-check; and the hg38 check (raw 5' end
density peaks at region centres; chr1 max coordinate)."""
import subprocess, tempfile
import numpy as np, pandas as pd
from fig2_common import OUT, REF, ROOT, TFS, ATAC_UNION, BLACKLIST
O6 = OUT / "round6_diffpeaks"; OS = O6 / "chip_selfcheck"
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
LAB = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
R = pd.read_csv(O6 / "rows.tsv", sep="\t")
print("rows.tsv columns", list(R.columns)[:12], "n", len(R), "chr1 max end", R[R.chrom == "chr1"].end.max())
S = np.load(O6 / "diff_sets.npz")
sets = {t: S[f"D-B_1000_{t}"] for t in TFS}
u, cnt = np.unique(np.concatenate(list(sets.values())), return_counts=True)
shared = set(u[cnt > 1]); sets = {t: np.array([i for i in v if i not in shared]) for t, v in sets.items()}
tmp = tempfile.mkdtemp(dir="/work/users/d/y/dyy12/tmp")
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
gfile = f"{tmp}/g.txt"
fai = pd.read_csv(str(ROOT / "downstream/cache/hg38.fa") + ".fai", sep="\t", header=None)
fai[fai[0].isin(CH)][[0, 1]].to_csv(gfile, sep="\t", header=False, index=False)
def sh(c): return subprocess.run(c, shell=True, check=True, capture_output=True, text=True, executable="/bin/bash").stdout
def wbed(df, p): df.to_csv(p, sep="\t", header=False, index=False); return p
def frac_hit(bed, ref, w=500):
    n = int(sh(f"wc -l < {bed}"))
    h = int(sh(f"bedtools window -u -w {w} -a {bed} -b {ref} | wc -l"))
    return h / n
atac = f"{tmp}/atac.bed"; sh(f"zcat -f {ATAC_UNION} | cut -f1-3 | sort -k1,1 -k2,2n > {atac}")
rec = []
for t in TFS:
    ix = sets[t]
    c = R.chrom.values[ix]; m = ((R.start.values[ix] + R.end.values[ix]) // 2).astype(int)
    pt = wbed(pd.DataFrame({0: c, 1: m, 2: m + 1}), f"{tmp}/{t}_mid.bed")
    reg = wbed(pd.DataFrame({0: c, 1: R.start.values[ix], 2: R.end.values[ix]}), f"{tmp}/{t}_reg.bed")
    ref = f"{tmp}/{t}_ref.bed"; sh(f"grep -P '^chr([0-9]+|X)\\t' {PEAKS[t]} | cut -f1-3 > {ref}")
    lab = f"{tmp}/{t}_lab.bed"; sh(f"awk '$9>=1.30103' {LAB.format(t=t)} | cut -f1-3 > {lab}")
    obs = frac_hit(pt, ref)
    shuf = [frac_hit(wbed(pd.read_csv(sh(f"bedtools shuffle -i {pt} -g {gfile} -excl {BLACKLIST} -chrom -seed {s} > {tmp}/s.bed; echo {tmp}/s.bed").strip(), sep='\t', header=None), f"{tmp}/s2.bed"), ref) for s in range(20)]
    shuf_atac = [frac_hit(wbed(pd.read_csv(sh(f"bedtools shuffle -i {pt} -g {gfile} -incl {atac} -seed {s} > {tmp}/sa.bed; echo {tmp}/sa.bed").strip(), sep='\t', header=None), f"{tmp}/sa2.bed"), ref) for s in range(20)]
    rec.append(dict(TF=t, n=len(ix), median_width=int(np.median(R.end.values[ix] - R.start.values[ix])),
                    mid_in_ATAC_union=frac_hit(pt, atac, 0), region_overlaps_own_lab_MACS2_q05=frac_hit(reg, lab, 0),
                    within500_ref_observed=obs, within500_ref_genome_shuffle=np.mean(shuf),
                    within500_ref_ATAC_shuffle=np.mean(shuf_atac)))
    print(rec[-1], flush=True)
T = pd.DataFrame(rec)
# ChIP centre/flank per block from the r13 heatmap matrices (+-2 kb, 50 bp; centre +-250 bp, flank outer 500 bp)
H = np.load(O6 / "heatmaps" / "heatmap_matrices.npz", allow_pickle=True)
blk = H["blocks"]; NB = H["bulk"].shape[2]; CTR = slice(NB // 2 - 5, NB // 2 + 5); FL = np.r_[0:10, NB - 10:NB]
cfb = {}
for k, t in enumerate(TFS):
    p = H["bulk"][k][blk == t].mean(0); cfb[t] = p[CTR].mean() / p[FL].mean()
T["chip_cf_r13_own_block"] = T.TF.map(cfb)
mat = np.zeros_like(H["bulk"][0])
for k, t in enumerate(TFS):
    mat[blk == t] = H["bulk"][k][blk == t]
p = mat.mean(0); print("L3 pooled ChIP c/f", p[CTR].mean() / p[FL].mean())
pn = mat[blk != "CTCF"].mean(0); print("L3 pooled ChIP c/f without CTCF block", pn[CTR].mean() / pn[FL].mean())
T.to_csv(OS / "coord_audit.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); print(T.round(4).to_string())
