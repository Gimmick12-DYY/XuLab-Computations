# Union-ATAC elements (data/HEK293T_ATAC_union3.bed.gz): covariates as in Gate 1 v2 (GC, k50 mappability, width, OmniATAC Tn5 cuts), sequences, and FIMO p < 1e-4
# for CTCF MA0139.2 plus 3 column-shuffled copies (same composition, destroyed motif).
import gzip, os, re, subprocess
import numpy as np, pandas as pd
from cm_common import *
os.makedirs(U, exist_ok=True); os.makedirs(f"{O}/calibration", exist_ok=True)
sh = lambda c: subprocess.run(c, shell=True, check=True, executable="/bin/bash")
FA = f"{X}/downstream/cache/hg38.fa"; K50 = f"{X}/regulatome/gate1/v2/k50.merged.bed"; TAG = f"{X}/regulatome/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"
E = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
E = E[E.chrom.str.fullmatch(r"chr([0-9]+|X|Y)")].sort_values(["chrom", "start"]).reset_index(drop=True)
for c, d in E.groupby("chrom"): assert (d.start.values[1:] >= d.end.values[:-1]).all(), f"overlapping elements on {c}"
E["eid"] = [f"U{i:06d}" for i in range(len(E))]; E.to_csv(f"{U}/union.bed", sep="\t", header=False, index=False)
sh(f"bedtools nuc -fi {FA} -bed {U}/union.bed | cut -f4,6 | tail -n +2 > {U}/gc.tsv")
sh(f"bedtools coverage -a {U}/union.bed -b {K50} | cut -f4,8 > {U}/map.tsv")
sh(f"bedtools getfasta -fi {FA} -bed {U}/union.bed -nameOnly -fo {U}/union.fa")
T = pd.read_csv(TAG, sep="\t", header=None, usecols=[0, 1], names=["chrom", "pos"], dtype={"chrom": str, "pos": np.int64}); at = np.zeros(len(E))
for c, d in E.groupby("chrom"):
    p = np.sort(T.pos.values[T.chrom.values == c]); at[d.index.values] = np.searchsorted(p, d.end.values) - np.searchsorted(p, d.start.values)
gc = pd.read_csv(f"{U}/gc.tsv", sep="\t", header=None, index_col=0)[1].reindex(E.eid).values; mp = pd.read_csv(f"{U}/map.tsv", sep="\t", header=None, index_col=0)[1].reindex(E.eid).values
assert not np.isnan(gc).any() and not np.isnan(mp).any()
pd.DataFrame(dict(eid=E.eid, width=E.end - E.start, gc=gc, map=mp, atac=at)).to_csv(f"{U}/covariates.tsv", sep="\t", index=False)
# real and shuffled CTCF motifs
txt = open(f"{X}/motif_analysis/cache/motifdb/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt").read()
hdr = txt.split("MOTIF ")[0]; blk = [b for b in txt.split("MOTIF ")[1:] if b.startswith("MA0139.2")][0]
ls = blk.strip().split("\n"); i = [k for k, l in enumerate(ls) if l.startswith("letter-probability")][0]
w = int(re.search(r"w= *(\d+)", ls[i]).group(1)); M = ls[i + 1:i + 1 + w]; out = hdr + "MOTIF " + blk.rstrip() + "\n\n"; r = np.random.default_rng(20261101)
for s in range(3): out += f"MOTIF shuffle{s + 1} CTCF_columns_shuffled\n{ls[i]}\n" + "\n".join(M[j] for j in r.permutation(w)) + "\n\n"
open(f"{O}/calibration/ctcf_real_shuf.meme", "w").write(out)
mb = "/nas/longleaf/rhel9/apps/meme/5.5.7"
sh(f"{mb}/libexec/meme-5.5.7/fasta-get-markov -m 0 {U}/union.fa {U}/union.bg >/dev/null 2>&1; {mb}/bin/fimo --thresh 1e-4 --max-stored-scores 5000000 --text --bfile {U}/union.bg {O}/calibration/ctcf_real_shuf.meme {U}/union.fa > {O}/calibration/fimo.tsv 2>/dev/null")
print(len(E), "elements", int(E.eval("end-start").sum() / 1e6), "Mb; atac in elements", int(at.sum()))
