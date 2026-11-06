"""TF x Cicero-pair and TF x anchor-region incidence (1st-step Our Run k50, p<=0.05, span >25 kb, blacklist dropped).
Same merged-region universe as hub_figures/shared_pairs/shared_pairs.py. Writes inc_pair.npz, inc_region.npz, tfs.txt,
regions.tsv, pairs.tsv into the output dir.
With 2nd arg "cliques": keep only pairs whose two regions sit in the same clique of that TF (Our Run k50 cliques,
FDR<=0.05 edges, size>=3), i.e. clique edges >25 kb."""
import glob, gzip, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
ROOT = Path("/work/users/d/y/dyy12/XuLab")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); CLQ = len(sys.argv) > 2 and sys.argv[2] == "cliques"
KEEP = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
def iv(r):
    c, se = r.split(":"); s, e = se.split("-"); return c, int(s), int(e)
P = {}
EXTRA = {"RBBP4": "RBBP4_sweep_k50_i10000_s555"}   # RBBP4's Our Run settings (k50, cap 10k, seed 555) live outside ourrun_k50
CDIR = {}
files = sorted(glob.glob(str(ROOT / "cobinding/work/ourrun_k50/*/*.peak_edges.tsv")))
for tf, d in EXTRA.items(): files.append(str(ROOT / f"cobinding/work/{d}/{tf}.peak_edges.tsv")); CDIR[tf] = ROOT / f"cobinding/results/{d}"
for f in files:
    tf = Path(f).name.split(".")[0]
    d = pd.read_csv(f, sep="\t", usecols=["peak1", "peak2", "pval"]); P[tf] = d[d.pval <= 0.05]
allp = sorted({iv(x) for d in P.values() for col in ("peak1", "peak2") for x in d[col]})
allp = [p for p in allp if p[0] in KEEP]
uni = []
for c, s, e in allp:
    if uni and uni[-1][0] == c and s <= uni[-1][2]: uni[-1][2] = max(uni[-1][2], e)
    else: uni.append([c, s, e])
bl = defaultdict(list)
for ln in gzip.open(ROOT / "downstream/cache/hg38-blacklist.v2.bed.gz", "rt"):
    x = ln.split(); bl[x[0]].append((int(x[1]), int(x[2])))
blk = {i for i, (c, s, e) in enumerate(uni) if any(a < e and b > s for a, b in bl.get(c, []))}
starts = defaultdict(list); ids = defaultdict(list)
for i, (c, s, e) in enumerate(uni): starts[c].append(s); ids[c].append(i)
starts = {c: np.array(v) for c, v in starts.items()}
def uid(r):
    c, s, e = iv(r)
    return ids[c][np.searchsorted(starts[c], s, side="right") - 1] if c in starts else None
mid = lambda i: (uni[i][1] + uni[i][2]) // 2
memb = {}
if CLQ:
    for tf in list(P):
        f = CDIR.get(tf, ROOT / f"cobinding/results/ourrun_k50/{tf}") / "cliques.tsv"
        try: d = pd.read_csv(f, sep="\t", usecols=["clique_id", "regions"])
        except Exception: d = pd.DataFrame(columns=["clique_id", "regions"])
        m = defaultdict(set)
        for cid, rr in zip(d.clique_id, d.regions):
            for r in str(rr).split(";"):
                if iv(r)[0] in KEEP:
                    u = uid(r)
                    if u is not None: m[u].add(cid)
        memb[tf] = m
tfs = sorted(P); pairs = {}; best_p = defaultdict(dict); rowsP, colsP = [], []
for t, tf in enumerate(tfs):
    S = {}
    d = P[tf]
    for a, b, pv in zip(d.peak1, d.peak2, d.pval):
        i, j = uid(a), uid(b)
        if i is None or j is None or i == j or i in blk or j in blk: continue
        i, j = min(i, j), max(i, j)
        if mid(j) - mid(i) <= 25000: continue
        if CLQ and not (memb[tf].get(i, set()) & memb[tf].get(j, set())): continue
        S[(i, j)] = min(S.get((i, j), 1.0), pv)
    for k, pv in S.items():
        if k not in pairs: pairs[k] = len(pairs)
        rowsP.append(t); colsP.append(pairs[k]); best_p[t][pairs[k]] = pv
pk = sorted(pairs, key=pairs.get)
Ap = sp.csr_matrix((np.ones(len(rowsP), np.int8), (rowsP, colsP)), shape=(len(tfs), len(pk)))
pv = sp.csr_matrix(([best_p[t][c] for t, c in zip(rowsP, colsP)], (rowsP, colsP)), shape=Ap.shape)
used = sorted({i for k in pk for i in k}); rmap = {r: n for n, r in enumerate(used)}
pairs_df = pd.DataFrame({"r1": [rmap[i] for i, j in pk], "r2": [rmap[j] for i, j in pk],
                         "chrom": [uni[i][0] for i, j in pk], "dist_kb": [(mid(j) - mid(i)) / 1e3 for i, j in pk]})
reg_df = pd.DataFrame([uni[i] for i in used], columns=["chrom", "start", "end"])
R = Ap.tocoo(); rr = np.concatenate([R.row, R.row]); cc = np.concatenate([pairs_df.r1.values[R.col], pairs_df.r2.values[R.col]])
Ar = sp.csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(len(tfs), len(used))); Ar.data[:] = 1
sp.save_npz(OUT / "inc_pair.npz", Ap); sp.save_npz(OUT / "pair_pval.npz", pv); sp.save_npz(OUT / "inc_region.npz", Ar)
pairs_df.to_csv(OUT / "pairs.tsv", sep="\t", index=False); reg_df.to_csv(OUT / "regions.tsv", sep="\t", index=False)
Path(OUT / "tfs.txt").write_text("\n".join(tfs) + "\n")
cs = np.asarray(Ap.sum(0)).ravel()
print(len(tfs), "TFs;", Ap.shape[1], "pairs;", (cs >= 2).sum(), "shared>=2;", Ar.shape[1], "anchor regions;", (np.asarray(Ar.sum(0)).ravel() >= 2).sum(), "regions in >=2 TFs")
