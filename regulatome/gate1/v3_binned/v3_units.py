# Gate 1 v3 (binned) units. Genome grid of 1 kb bins (chr1-22, X, Y); per-bin GC (hg38) and k50 Umap mappable fraction.
# Units (kept if mappable fraction >= 0.5):
#   bin10k, bin50k  fixed genome bins
#   reprpc          ChromHMM-18 ReprPC + ReprPCWk segments merged within 5 kb, blacklist removed
#   genebody        protein-coding gene body +/- 10 kb (GENCODE v46; ABC gene bounds), one unit per gene (units may overlap)
# Each unit type gets a sparse 1 kb-bin x unit weight matrix (fraction of the bin inside the unit), unit GC / mappability
# (bp-weighted) and width. Reads are counted per 1 kb bin and pushed through these maps.
import os, subprocess
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; V3 = f"{X}/regulatome/gate1/v3_binned"; U = f"{V3}/units"; os.makedirs(U, exist_ok=True)
FA = f"{X}/downstream/cache/hg38.fa"; K50 = f"{X}/regulatome/gate1/v2/k50.merged.bed"
BL = f"{X}/downstream/cache/hg38-blacklist.v2.bed.gz"
CS = pd.read_csv(f"{X}/regulatome/m2_abc/inputs/chrom_sizes.tsv", sep="\t", header=None, names=["chrom", "size"])
CS = CS[CS.chrom.str.fullmatch(r"chr([0-9]+|X|Y)")].reset_index(drop=True)
nb = (CS["size"] // 1000 + 1).values; off = np.concatenate([[0], np.cumsum(nb)]); NB = int(off[-1])
np.savez(f"{U}/grid.npz", chrom=CS.chrom.values, size=CS["size"].values, nbins=nb, offset=off)

def sh(cmd): return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, executable="/bin/bash").stdout
# per-1 kb-bin GC and mappable fraction
if not os.path.exists(f"{U}/bin1k_cov.npz"):
    with open(f"{U}/bins1k.bed", "w") as f:
        for c, s, n in zip(CS.chrom, CS["size"], nb):
            st = np.arange(n) * 1000; en = np.minimum(st + 1000, s)
            f.write("".join(f"{c}\t{a}\t{b}\n" for a, b in zip(st, en) if b > a))
    gc = sh(f"bedtools nuc -fi {FA} -bed {U}/bins1k.bed | tail -n +2 | cut -f5").split()
    mp = sh(f"bedtools coverage -a {U}/bins1k.bed -b {K50} | cut -f7").split()  # unsorted mode keeps the order of -a
    np.savez_compressed(f"{U}/bin1k_cov.npz", gc=np.array(gc, float), map=np.array(mp, float))
    os.remove(f"{U}/bins1k.bed")
cov = np.load(f"{U}/bin1k_cov.npz"); GC1, MP1 = cov["gc"], cov["map"]
assert len(GC1) == NB and len(MP1) == NB, (len(GC1), len(MP1), NB)

def build(name, ivals):
    """ivals: DataFrame chrom, start, end, name -> sparse bins x units weights, unit covariates; keep mappable >= 0.5"""
    rows, cols, w = [], [], []
    ci = {c: i for i, c in enumerate(CS.chrom)}
    for u, (c, s, e) in enumerate(zip(ivals.chrom, ivals.start, ivals.end)):
        if c not in ci: continue
        b0, b1 = s // 1000, (e - 1) // 1000
        for b in range(b0, b1 + 1):
            ov = min(e, (b + 1) * 1000) - max(s, b * 1000)
            if ov > 0: rows.append(off[ci[c]] + b); cols.append(u); w.append(ov / 1000.0)
    M = sp.csr_matrix((w, (rows, cols)), shape=(NB, len(ivals)))
    bp = np.asarray(M.sum(0)).ravel() * 1000
    gcu = (M.T @ GC1) * 1000 / np.maximum(bp, 1); mpu = (M.T @ MP1) * 1000 / np.maximum(bp, 1)
    keep = (mpu >= 0.5) & (bp > 0)
    M = M[:, keep].tocsr(); iv = ivals[keep].reset_index(drop=True)
    sp.save_npz(f"{U}/{name}_map.npz", M)
    np.savez(f"{U}/{name}_cov.npz", gc=gcu[keep], map=mpu[keep], width=bp[keep])
    iv.to_csv(f"{U}/{name}.bed", sep="\t", header=False, index=False)
    print(f"{name}: {keep.sum():,} of {len(keep):,} units kept (mappable >= 0.5); median width {np.median(bp[keep]):,.0f} bp", flush=True)

for size, name in ((10000, "bin10k"), (50000, "bin50k")):
    rows = []
    for c, s in zip(CS.chrom, CS["size"]):
        st = np.arange(0, s, size); rows.append(pd.DataFrame({"chrom": c, "start": st, "end": np.minimum(st + size, s)}))
    iv = pd.concat(rows, ignore_index=True); iv["name"] = [f"{name}_{i}" for i in range(len(iv))]
    build(name, iv)
rp = sh(f"zcat {X}/data/HEK293T_chromHMM18.bed.gz | awk -F'\\t' '$4==\"ReprPC\"||$4==\"ReprPCWk\"' | cut -f1-3 | sort -k1,1 -k2,2n | "
        f"bedtools merge -d 5000 -i - | bedtools intersect -v -a - -b {BL}")
iv = pd.read_csv(pd.io.common.StringIO(rp), sep="\t", header=None, names=["chrom", "start", "end"]); iv["name"] = [f"reprpc_{i}" for i in range(len(iv))]
build("reprpc", iv)
G = pd.read_csv(f"{X}/regulatome/m2_abc/inputs/genes_pc_v46.bed", sep="\t", header=None,
                names=["chrom", "start", "end", "name", "score", "strand", "gid", "type"])
G["start"] = (G.start - 10000).clip(lower=0); G["end"] = G.end + 10000
build("genebody", G[["chrom", "start", "end", "name"]].reset_index(drop=True))
