#!/usr/bin/env python
"""P-b: clustered Pearson correlation of bulk ChIP vs aggregated sc tracks over peak regions.

Regions: (union) merged union of all reference peaks of the TF set; (unique) merged union of the
reference-unique peaks (T1 definition, unique among all panel references).
Values per region: bulk = reads from the ChIP BAM (bedtools multicov, MAPQ>=10 not enforced: ENCODE
filtered BAMs); raw = sc read 5' ends; raw_dm = raw thinned to ZNF777 depth (3 seeds, r averaged);
imputed = sum of cells-positive over the 1 kb bins the region touches.
Each track -> log2(CPM + 1) with CPM over the region set; Pearson across regions;
average-linkage clustering on 1 - r.
TF sets: all6; 'distinct4' = CTCF + ZNF282 + the 2 other references giving the lowest mean pairwise
bp-Jaccard (diag/ref_jaccard_bp.tsv, exhaustive); 'distinct3' = CTCF + ZNF282 + the best single one.
Per version: sc-vs-bulk block diag stats, own reference = highest-r bulk for each sc track, and
'clusters next to own' = smallest dendrogram cluster holding sc_i and bulk_i has no other bulk track.
Outputs OUT/panels/Pb_*.tsv (+ .npz with linkage for plotting).
"""
import itertools
import subprocess

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from fig2_common import (NT, BAM, BIN, append_summary, DM_DEPTH, DM_SEEDS, INP, OUT, REF, TFS, bin_index, diag_stats, load_imp7,
                         read_bed)

op = OUT / "panels"
op.mkdir(parents=True, exist_ok=True)
tmp = op / "tmp"
tmp.mkdir(exist_ok=True)
bi, off, N = bin_index()
imp7 = load_imp7()
J = pd.read_csv(OUT / "diag/ref_jaccard_bp.tsv", sep="\t", index_col=0)


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def regions(tfs, kind):
    files = [str(REF[t]) if kind == "union" else str(INP / f"unique_{t}.bed") for t in tfs]
    txt = sh("bash", "-c", "cat " + " ".join(files) + " | cut -f1-3 | grep -E '^chr([0-9]+|X)\\b' | "
             "sort -k1,1 -k2,2n | bedtools merge")
    p = tmp / f"reg_{kind}_{'_'.join(tfs)}.bed"
    p.write_text(txt)
    return p, read_bed(p)


bulk_cache = {}


def bulk_counts(path, tfs):
    key = str(path)
    if key not in bulk_cache:
        out = sh("bedtools", "multicov", "-bams", *[str(BAM[t]) for t in TFS], "-bed", str(path))
        bulk_cache[key] = np.array([[int(x) for x in l.split("\t")[3:]] for l in out.splitlines()], float)
    return bulk_cache[key][:, [TFS.index(t) for t in tfs]]


five = {t: dict(np.load(INP / "five" / f"{t}.npz")) for t in TFS}


def sc_counts(reg, tfs):
    C = np.zeros((len(reg), len(tfs)))
    for k, t in enumerate(tfs):
        for c, g in reg.groupby(0):
            pos = five[t][c]
            C[g.index.values, k] = np.searchsorted(pos, g[2].values) - np.searchsorted(pos, g[1].values)
    return C


def imp_counts(reg, tfs):
    C = np.zeros((len(reg), len(tfs)))
    o = np.array([off[c][0] for c in reg[0]])
    a = o + reg[1].values // BIN
    b = o + (reg[2].values - 1) // BIN
    cs = np.concatenate([np.zeros((NT, 1)), np.cumsum(imp7, 1)], 1)
    for k, t in enumerate(tfs):
        i = TFS.index(t)
        C[:, k] = cs[i, b + 1] - cs[i, a]
    return C


def logcpm(C):
    return np.log2(C / C.sum(0, keepdims=True) * 1e6 + 1)


def corr(B, S):
    X = np.column_stack([logcpm(B), logcpm(S)])
    return np.corrcoef(X.T)


def cluster_report(R, tfs):
    labs = [f"bulk_{t}" for t in tfs] + [f"sc_{t}" for t in tfs]
    Z = linkage(squareform(1 - R, checks=False), "average")
    root = to_tree(Z)
    k = len(tfs)
    nxt = []
    for i in range(k):
        want = {i, k + i}
        node = root
        while True:
            for ch in (node.left, node.right):
                if ch is not None and want <= set(ch.pre_order()):
                    node = ch
                    break
            else:
                break
        nxt.append(int(sum(x < k for x in node.pre_order()) == 1))
    return Z, labs, nxt


combos = []
fixed = ("CTCF", "ZNF282")
rest = [t for t in TFS if t not in fixed]
for m in (1, 2):
    best = min(itertools.combinations(rest, m),
               key=lambda s: np.mean([J.loc[x, y] for x, y in itertools.combinations(fixed + s, 2)]))
    combos.append((f"distinct{m + 2}", [*fixed, *best]))
sets = [(f"all{NT}", TFS)] + combos

rows, summ = [], []
for setname, tfs in sets:
    tfs = [t for t in TFS if t in tfs]
    for kind in ("union", "unique"):
        path, reg = regions(tfs, kind)
        B = bulk_counts(path, tfs)
        raw = sc_counts(reg, tfs)
        arms = {"raw": [raw],
                "raw_dm": [np.random.default_rng(s).binomial(raw.astype(np.int64),
                                                             np.minimum(1, DM_DEPTH / np.array(
                                                                 [sum(len(v) for v in five[t].values()) for t in tfs])))
                           for s in DM_SEEDS],
                "imputed": [imp_counts(reg, tfs)]}
        for arm, Ss in arms.items():
            R = np.mean([corr(B, S) for S in Ss], 0)
            k = len(tfs)
            block = R[k:, :k]   # rows sc, cols bulk
            Z, labs, nxt = cluster_report(R, tfs)
            tag = f"{setname}_{kind}_{arm}"
            pd.DataFrame(R, labs, labs).to_csv(op / f"Pb_corr_{tag}.tsv", sep="\t", float_format="%.4f")
            np.savez(op / f"Pb_link_{tag}.npz", Z=Z, labs=np.array(labs), R=R)
            own_best = [int(np.argmax(block[i]) == i) for i in range(k)]
            for i, t in enumerate(tfs):
                rows.append(dict(set=setname, regions=kind, arm=arm, n_regions=len(reg), sc_track=t,
                                 r_own=block[i, i], r_best_other=float(np.max(np.delete(block[i], i))),
                                 own_is_best_bulk=own_best[i], clusters_next_to_own=nxt[i]))
            d = {}
            if k == NT:
                d = diag_stats(block)
            summ.append(dict(test=f"Pb_pearson_{setname}_{kind}", arm=arm, variant="sc_vs_bulk_r", n_tracks=k,
                             own_best_bulk=sum(own_best), clusters_next_to_own=sum(nxt),
                             mean_diag=float(np.diag(block).mean()),
                             diag_minus_offdiag_plain=float(np.diag(block).mean() - block[~np.eye(k, dtype=bool)].mean()),
                             **d))
            print(setname, kind, arm, len(reg), "own-best", sum(own_best), "next", sum(nxt), flush=True)
pd.DataFrame(rows).to_csv(op / "Pb_own_reference.tsv", sep="\t", index=False, float_format="%.4f")
append_summary(summ)
print(pd.DataFrame(summ).to_string())
