# Repressor TF-TF correlation matrices (same construction as the ChromHMM-element matrices in tf_complex: signal = raw reads
# per unit / TF cell count, Pearson across units with any read, SpQN on the full 79-TF matrix ordered by cell count).
# Units: (a) active elements (elements_v2 promoter + distal_active + bivalent; M1 v2 counts), (b) ReprPC domains and (c) 50 kb
# bins (Gate 1 v3 units, mappable >= 0.5), (d) ENCODE H3K9me3 HEK293 replicated peaks (ENCFF037SXA) merged within 5 kb.
# Block statistic: mean pairwise percentile within the block (off-diagonal distribution of that matrix); null = 5,000
# depth-matched random sets (each member swapped for a non-member TF within +/-4 cell-count ranks).
#   usage: repressor_matrix.py count <TF>   |   repressor_matrix.py build
import json, subprocess, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/regulatome/repressor_matrix"
K9 = f"{X}/regulatome/encode_repressive_check/data/ENCFF037SXA.bed.gz"
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]

def k9_peaks():
    txt = subprocess.run(f"zcat {K9} | cut -f1-3 | sort -k1,1 -k2,2n | bedtools merge -d 5000 -i -", shell=True, capture_output=True,
                         text=True, check=True).stdout
    return pd.read_csv(pd.io.common.StringIO(txt), sep="\t", header=None, names=["chrom", "start", "end"])

if sys.argv[1] == "count":
    tf = sys.argv[2]; P = k9_peaks()
    r = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1],
                    names=["chrom", "start"], dtype={"chrom": str, "start": np.int64})
    out = np.zeros(len(P), np.int64)
    for c, d in P.groupby("chrom"):
        p = r.start.values[(r.chrom == c).values]; s, e, ix = d.start.values, d.end.values, d.index.values
        k = np.searchsorted(s, p, side="right") - 1; ok = (k >= 0) & (p < e[np.clip(k, 0, None)])
        np.add.at(out, ix[k[ok]], 1)
    np.save(f"{O}/k9counts/{tf}.npy", out); print(tf, len(P), "peaks", int(out.sum()), "reads in peaks")
    sys.exit()

sys.path.insert(0, f"{X}/tf_complex/scripts"); sys.path.insert(0, f"{X}/downstream")
from compartment_rpkm_correlation import spqn
cells = np.array([json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json"))["cells"] for t in tfs], float)
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
lib = (F.all_elements > 0.22).map({True: "high", False: "low"}).reindex(tfs)
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, usecols=[7], names=["cls"])
act = E.cls.isin(["promoter", "distal_active", "bivalent"]).values
V3 = f"{X}/regulatome/gate1/v3_binned/cellmats"
UNITS = {
    "active_elements": lambda t: np.load(f"{X}/regulatome/m1_binding/v2/counts/{t}.npz")["raw"][act].astype(float),
    "reprpc_domains": lambda t: np.asarray(sp.load_npz(f"{V3}/{t}.reprpc.npz").sum(0)).ravel(),
    "bins_50kb": lambda t: np.asarray(sp.load_npz(f"{V3}/{t}.bin50k.npz").sum(0)).ravel(),
    "h3k9me3_peaks": lambda t: np.load(f"{O}/k9counts/{t}.npy").astype(float),
}
BLOCKS = {"PRC2": ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7"], "NuRD": ["MBD3", "RBBP4", "RBBP7"], "ChAHP": ["ADNP", "HP1", "CHAMP1"],
          "broad-domain (WIZ/ZNF644)": ["DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"],
          "low-depth set": ["NEUROD4", "SOX12", "ZNF79", "ZNF606", "ZNF777", "ZNF581"]}
rank = pd.Series(cells, index=tfs).rank().values; rng = np.random.default_rng(20261030)
def pct_matrix(R):
    off = ~np.eye(len(R), dtype=bool); v = np.sort(R[off]); P_ = np.searchsorted(v, R, side="right") / len(v); np.fill_diagonal(P_, np.nan); return P_
def block_stat(P_, ix): return np.mean([P_[i, j] for a, i in enumerate(ix) for j in ix[a + 1:]])
rows = []
for u, f in UNITS.items():
    M = np.column_stack([f(t) / c for t, c in zip(tfs, cells)]); M = M[M.sum(1) > 0]
    R = np.nan_to_num(np.corrcoef(M, rowvar=False)); Rn = spqn(R, cells)
    for kind, Rm in (("pearson", R), ("spqn", Rn)):
        pd.DataFrame(Rm, index=tfs, columns=tfs).to_csv(f"{O}/{kind}_{u}.tsv", sep="\t", float_format="%.4f")
        P_ = pct_matrix(Rm)
        for b, mem in BLOCKS.items():
            ix = [tfs.index(t) for t in mem]; obs = block_stat(P_, ix); others = [i for i in range(len(tfs)) if tfs[i] not in mem]
            nd = []
            for _ in range(5000):
                pick = []
                for i in ix:
                    cand = [j for j in others if abs(rank[j] - rank[i]) <= 4 and j not in pick]; pick.append(rng.choice(cand))
                nd.append(block_stat(P_, pick))
            nd = np.array(nd)
            rows.append(dict(unit=u, matrix=kind, block=b, n_units=int(M.shape[0]), mean_r=float(np.mean([Rm[i, j] for a, i in enumerate(ix) for j in ix[a + 1:]])),
                             mean_pair_pct=obs, depth_matched_pct=float((nd < obs).mean())))
    print(u, M.shape[0], "units", flush=True)
B = pd.DataFrame(rows); B.to_csv(f"{O}/block_stats.tsv", sep="\t", index=False, float_format="%.3f")
pd.DataFrame({"tf": tfs, "cells": cells.astype(int), "library": lib.values}).to_csv(f"{O}/tf_meta.tsv", sep="\t", index=False)
pd.set_option("display.width", 200)
for kind in ("spqn", "pearson"):
    b = B[B.matrix == kind]
    print(f"\n{kind}: mean within-block r")
    print(b.pivot(index="block", columns="unit", values="mean_r").round(3).to_string())
    print(f"{kind}: depth-matched percentile")
    print(b.pivot(index="block", columns="unit", values="depth_matched_pct").round(3).to_string())
