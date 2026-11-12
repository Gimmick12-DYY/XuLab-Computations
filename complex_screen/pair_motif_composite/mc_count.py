# Motif-centred read profiles, one library: for every motif, read 5' ends (TF.<TF>.bed.gz, single-end; + strand start, - strand 1-bp end) within
# +/- 50 bp of the motif centres (centre) and 200-500 bp on both sides (flank), summed over all FIMO instances (p <= 1e-5, eligible non-HOT
# union-ATAC bins), separately for odd (disc) and even + X (val) chromosomes. Also the total library reads in those bins (depth).
# Output motif_centred/counts/<TF>.npz: motifs, centre[part], flank[part], n_inst[part], reads_in_bins
import sys, glob, gzip
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"; G = f"{O}/motif_centred"; t = sys.argv[1]; FR = "/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment"
R = pd.read_csv(f"{FR}/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"]); R["p"] = np.where(R.st == "-", R.e - 1, R.s)
pos = {c: np.sort(g.p.values) for c, g in R.groupby("c")}
part = lambda c: "disc" if c in {f"chr{i}" for i in range(1, 23, 2)} else ("val" if c in {f"chr{i}" for i in range(2, 23, 2)} | {"chrX"} else None)
B = pd.read_csv(f"{G}/bins.bed", sep="\t", header=None, names=["c", "s", "e", "n"]); nb = 0
for c, g in B.groupby("c"):
    if c in pos: p = pos[c]; nb += int((np.searchsorted(p, g.e.values) - np.searchsorted(p, g.s.values)).sum())
def cnt(p, a, b): return np.searchsorted(p, b) - np.searchsorted(p, a)
motifs, cen, fla, ninst = [], {"disc": [], "val": []}, {"disc": [], "val": []}, {"disc": [], "val": []}
for f in sorted(glob.glob(f"{G}/fimo/c*.tsv.gz")):
    D = pd.read_csv(f, sep="\t", header=None, names=["m", "c", "x", "st"]); D["pt"] = D.c.map(part); D = D[D.pt.notna()]
    for m, g in D.groupby("m", sort=False):
        motifs.append(m); acc = {pt: [0, 0, 0] for pt in ["disc", "val"]}
        for (c, pt), h in g.groupby(["c", "pt"]):
            if c not in pos: continue
            p = pos[c]; x = h.x.values; acc[pt][0] += int(cnt(p, x - 50, x + 51).sum()); acc[pt][1] += int((cnt(p, x - 500, x - 199) + cnt(p, x + 200, x + 501)).sum()); acc[pt][2] += len(x)
        for pt in acc: cen[pt].append(acc[pt][0]); fla[pt].append(acc[pt][1]); ninst[pt].append(acc[pt][2])
np.savez_compressed(f"{G}/counts/{t}.npz", motifs=np.array(motifs), reads_in_bins=nb, **{f"centre_{pt}": np.array(cen[pt]) for pt in cen}, **{f"flank_{pt}": np.array(fla[pt]) for pt in fla}, **{f"ninst_{pt}": np.array(ninst[pt]) for pt in ninst})
print(t, len(motifs), "motifs; reads in bins", nb)
