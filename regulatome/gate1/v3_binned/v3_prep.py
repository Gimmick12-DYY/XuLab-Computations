# Gate 1 v3 prep, per TF: read 5' ends (fragments on /vast, read-only) -> per-cell 1 kb genome bins -> cell x unit matrices for
# the four unit types (bin10k, bin50k, reprpc, genebody). Diagnostics: reads per cell, reads per covered 1 kb bin, and the
# fraction of reads in active elements (elements_v2 promoter / distal_active / bivalent), ReprPC domains (reprpc units),
# and elsewhere.
import json, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; V3 = f"{X}/regulatome/gate1/v3_binned"; U = f"{V3}/units"
tf = sys.argv[1]
g = np.load(f"{U}/grid.npz", allow_pickle=True); ci = {c: i for i, c in enumerate(g["chrom"])}; off = g["offset"]; NB = int(off[-1])
r = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1, 3],
                names=["chrom", "start", "bc"], dtype={"chrom": str, "start": np.int64, "bc": str})
r = r[r.chrom.isin(ci)]
bcs, cell = np.unique(r.bc.values, return_inverse=True)
gb = off[r.chrom.map(ci).values] + r.start.values // 1000
C1 = sp.csr_matrix((np.ones(len(gb), np.float32), (cell, gb)), shape=(len(bcs), NB)); C1.sum_duplicates()
reads_cell = np.asarray(C1.sum(1)).ravel()
for name in ("bin10k", "bin50k", "reprpc", "genebody"):
    M = sp.load_npz(f"{U}/{name}_map.npz").tocsc()
    sp.save_npz(f"{V3}/cellmats/{tf}.{name}.npz", (C1 @ M).tocsr().astype(np.float32))
np.save(f"{V3}/cellmats/{tf}.reads.npy", reads_cell)
# diagnostics
binsum = np.asarray(C1.sum(0)).ravel()
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, usecols=[0, 1, 2, 7], names=["chrom", "start", "end", "cls"])
E = E[E.cls.isin(["promoter", "distal_active", "bivalent"])]
def in_ivals(df):
    hit = np.zeros(len(r), bool)
    for c, d in df.groupby("chrom"):
        m = (r.chrom == c).values; p = r.start.values[m]; s, e = np.sort(d.start.values), d.end.values[np.argsort(d.start.values)]
        k = np.searchsorted(s, p, side="right") - 1; ok = (k >= 0) & (p < e[np.clip(k, 0, None)]); hit[np.flatnonzero(m)[ok]] = True
    return hit
act = in_ivals(E)
RP = pd.read_csv(f"{U}/reprpc.bed", sep="\t", header=None, names=["chrom", "start", "end", "name"])
rep = in_ivals(RP) & ~act
d = dict(tf=tf, cells=int(len(bcs)), reads=int(len(r)), reads_per_cell=float(len(r) / len(bcs)),
         covered_1kb_bins=int((binsum > 0).sum()), reads_per_covered_bin=float(len(r) / max((binsum > 0).sum(), 1)),
         frac_active_elements=float(act.mean()), frac_reprpc_domains=float(rep.mean()), frac_other=float(1 - act.mean() - rep.mean()))
json.dump(d, open(f"{V3}/cellmats/{tf}.diag.json", "w")); print(d)
