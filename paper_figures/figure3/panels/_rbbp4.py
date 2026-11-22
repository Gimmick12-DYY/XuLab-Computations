"""RBBP4 pair / module tables shared by rbbp4_characterization.py and rbbp4_square.py.

Pairs = RBBP4 first-step Cicero pairs with p <= 0.05 whose two region midpoints are > 25 kb apart (same chromosome).
Modules = higher-order coordinated binding modules (rbbp4_modules.tsv); module edges = pairs of regions inside one module > 25 kb.
"""
from itertools import combinations

import numpy as np
import pandas as pd

import _genomics as G

ORDER = ["promoter", "enhancer", "repressive", "other"]


def load(src):
    LA = G.LoopArrays(src / "peakachu_5000.loops.0.7.bedpe.gz", src / "peakachu_10000.loops.0.7.bedpe.gz")
    EC = G.ElementClass(src / "refseq_select_tss.tsv.gz", src / "hek293t_chromhmm18.bed.gz")
    E = pd.read_csv(src / "rbbp4_peak_edges.tsv.gz", sep="\t")
    E = E[E.pval <= 0.05]
    pairs, keys = [], []
    for a, b in zip(E.peak1, E.peak2):
        p, q = G.parse_peak(a), G.parse_peak(b)
        if p[0] == q[0] and abs(G.mid(p) - G.mid(q)) > G.MIN_SPAN:
            pairs.append((p, q))
            keys.append(tuple(sorted([a, b])))
    ploop = np.array([LA.on_loop(p, q) for p, q in pairs])
    C = pd.read_csv(src / "rbbp4_modules.tsv", sep="\t", usecols=["clique_id", "n_regions", "regions"])
    mods, medges = [], {}
    for rr in C.regions.astype(str):
        regs = [x.strip() for x in rr.split(";")]
        pk = [G.parse_peak(x) for x in regs]
        es = [(p, q) for p, q in combinations(pk, 2) if abs(G.mid(p) - G.mid(q)) > G.MIN_SPAN]
        if es:
            mods.append(dict(size=len(pk), edges=len(es), loop=sum(LA.on_loop(p, q) for p, q in es)))
        for x, y in combinations(regs, 2):
            p, q = G.parse_peak(x), G.parse_peak(y)
            if abs(G.mid(p) - G.mid(q)) > G.MIN_SPAN:
                medges[tuple(sorted([x, y]))] = LA.on_loop(p, q)
    M = pd.DataFrame(mods)
    M["sz"] = np.where(M["size"] >= 5, "≥5", M["size"].astype(str))
    pc = []
    for p, q in pairs:
        a, b = sorted([EC(*p), EC(*q)], key=ORDER.index)
        pc.append(f"{a}-{b}")
    return dict(pairs=pairs, keys=keys, pair_on_loop=ploop, modules=M, module_edges=medges, pair_class=pd.Series(pc))
