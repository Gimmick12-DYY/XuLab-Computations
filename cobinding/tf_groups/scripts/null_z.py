"""Degree-preserving null (curveball) for TF x TF sharing. Keeps each TF's item count (depth) and each item's number of
TFs (popularity, e.g. HOT promoters). z = (obs - null mean)/null sd of the co-occurrence count A A^T.
With strata (library-quality bins) trades happen only within a stratum, so each item's TF count per stratum is kept too
(absorbs library-quality batch effects).
args: indir level(pair|region) nsamp subset(all|odd|even|top:<n>) seed [strata.tsv: tf<TAB>stratum]"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
IN = Path(sys.argv[1]); LEVEL = sys.argv[2]; NS = int(sys.argv[3]); SUB = sys.argv[4]; SEED = int(sys.argv[5])
rng = np.random.default_rng(SEED); STRATA = sys.argv[6] if len(sys.argv) > 6 else None
A = sp.load_npz(IN / f"inc_{LEVEL}.npz").tocsc().astype(np.int32)
tfs = (IN / "tfs.txt").read_text().split()
items = pd.read_csv(IN / ("pairs.tsv" if LEVEL == "pair" else "regions.tsv"), sep="\t")
if SUB in ("odd", "even"):
    num = items.chrom.str.replace("chr", "").replace("X", "23").astype(int)
    keep = np.where((num % 2 == 1) if SUB == "odd" else (num % 2 == 0))[0]; A = A[:, keep]
A = A.tocsr()
if SUB.startswith("top"):            # depth-matched: each TF's N strongest pairs (pair level only); TFs with fewer pairs dropped
    N = int(SUB.split(":")[1]); pv = sp.load_npz(IN / "pair_pval.npz").tocsr(); rows, cols = [], []
    for t in range(A.shape[0]):
        s, e = pv.indptr[t], pv.indptr[t + 1]
        if e - s < N: continue
        o = np.argsort(pv.data[s:e], kind="stable")[:N]; cols += list(pv.indices[s:e][o]); rows += [t] * N
    A = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=A.shape)
cs = np.asarray(A.sum(0)).ravel(); A = A[:, cs > 0]
R = [A.indices[A.indptr[t]:A.indptr[t + 1]].copy() for t in range(A.shape[0])]
nT, nI = A.shape
def cooc(R):
    r = np.concatenate([np.full(len(x), t) for t, x in enumerate(R)]); c = np.concatenate(R)
    M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nT, nI)); return (M @ M.T).toarray()
obs = cooc(R)
nonempty = [t for t in range(nT) if len(R[t])]
if STRATA:
    smap = pd.read_csv(STRATA, sep="\t", header=None, index_col=0)[1]; lab = np.array([smap.get(t, "NA") for t in tfs])
    groups = [np.array([t for t in nonempty if lab[t] == s]) for s in sorted(set(lab[nonempty]))]
    groups = [g for g in groups if len(g) >= 2]; gw = np.array([len(g) for g in groups], float); gw /= gw.sum()
    print("strata sizes", [len(g) for g in groups])
def trade():
    if STRATA:
        g = groups[rng.choice(len(groups), p=gw)]; a, b = rng.choice(g, 2, replace=False)
    else:
        a, b = rng.choice(nonempty, 2, replace=False)
    A_, B_ = R[a], R[b]
    sh = np.intersect1d(A_, B_, assume_unique=True)
    ao = np.setdiff1d(A_, sh, assume_unique=True); bo = np.setdiff1d(B_, sh, assume_unique=True)
    if not len(ao) or not len(bo): return
    pool = np.concatenate([ao, bo]); rng.shuffle(pool)
    R[a] = np.concatenate([sh, pool[:len(ao)]]); R[b] = np.concatenate([sh, pool[len(ao):]])
for _ in range(20 * nT * 10): trade()                 # burn-in
S1 = np.zeros_like(obs); S2 = np.zeros_like(obs); GE = np.zeros_like(obs)
for k in range(NS):
    for _ in range(10 * nT): trade()
    C = cooc(R); S1 += C; S2 += C * C; GE += (C >= obs)
    if (k + 1) % 50 == 0: print(k + 1, flush=True)
mu = S1 / NS; sd = np.sqrt(np.maximum(S2 / NS - mu ** 2, 0))
z = (obs - mu) / np.maximum(sd, 0.5); p = (GE + 1) / (NS + 1)
tag = f"{LEVEL}_{SUB.replace(':', '')}" + ("_strat" if STRATA else "")
for nm, M in [("obs", obs), ("exp", mu), ("sd", sd), ("z", z), ("p", p)]:
    pd.DataFrame(M, index=tfs, columns=tfs).to_csv(IN / f"null_{tag}_{nm}.tsv.gz", sep="\t")
pd.Series(np.asarray(A.sum(1)).ravel(), index=tfs).to_csv(IN / f"null_{tag}_depth.tsv", sep="\t", header=False)
print("done", tag, nT, nI)
