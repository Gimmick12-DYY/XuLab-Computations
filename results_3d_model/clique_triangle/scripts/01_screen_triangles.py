"""Clique triangles: 3 anchors in one TF's Cicero clique (Our Run k50, 1st step) where all 3 pairs are loop-edges: > 25 kb and both ends within +-30 kb of the anchors of one WT
Peakachu 0.7 loop (5 kb or 10 kb), the rule of the chr16 hub. Triangles are pooled across TFs by their 10 kb bins. Writes triangles_all.tsv.
  python 01_screen_triangles.py <in dir with cl/<TF>/cliques.tsv and loops/peakachu_*.bedpe>"""
import glob, itertools, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
IN = sys.argv[1]; O = Path(__file__).resolve().parents[1]; PAD = 30000
L = defaultdict(list)
for r in (5000, 10000):
    for ln in open(f"{IN}/loops/peakachu_{r}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] != x[3]: continue
        a, b = sorted(((int(x[1]), int(x[2])), (int(x[4]), int(x[5]))))
        L[x[0]].append((*a, *b))
L = {c: np.array(v) for c, v in L.items()}; print("loops", sum(map(len, L.values())))
def p(r): c, s = r.split(":"); s, e = map(int, s.split("-")); return c, s, e
cache = {}
def on_loop(c, a, b):   # a, b = (s, e), a before b
    k = (c, a, b)
    if k not in cache:
        l = L.get(c)
        cache[k] = l is not None and bool(((l[:, 0] - PAD < a[1]) & (l[:, 1] + PAD > a[0]) & (l[:, 2] - PAD < b[1]) & (l[:, 3] + PAD > b[0])).any())
    return cache[k]
tri = defaultdict(lambda: dict(tfs=set(), regs=None)); nq = {}; ncl3 = 0; clq = set()
for f in sorted(glob.glob(f"{IN}/cl/*/cliques.tsv")):
    tf = Path(f).parent.name; d = pd.read_csv(f, sep="\t", usecols=["clique_id", "n_regions", "chromosome", "regions"]); n = 0
    for r in d[d.n_regions >= 3].itertuples():
        ncl3 += 1; rg = sorted(((s, e) for _, s, e in map(p, r.regions.split(";"))), key=lambda t: t[0]); c = r.chromosome; m = len(rg)
        mids = [(s + e) // 2 for s, e in rg]; adj = defaultdict(set)
        for i, j in itertools.combinations(range(m), 2):
            if mids[j] - mids[i] > 25000 and on_loop(c, rg[i], rg[j]): adj[i].add(j)
        hit = False
        for i in list(adj):
            for j in adj[i]:
                for k in adj[i] & adj.get(j, set()):
                    key = (c, mids[i] // 10000, mids[j] // 10000, mids[k] // 10000); t = tri[key]; t["tfs"].add(tf); n += 1; hit = True
                    if t["regs"] is None: t["regs"] = (rg[i], rg[j], rg[k])
        if hit: clq.add((tf, r.clique_id))
    nq[tf] = n
print("TFs with clique tables", len(nq), "; cliques with >= 3 regions", ncl3, "; with a triangle", len(clq))
rows = [dict(chrom=k[0], A=f"{k[0]}:{t['regs'][0][0]}-{t['regs'][0][1]}", B=f"{k[0]}:{t['regs'][1][0]}-{t['regs'][1][1]}", C=f"{k[0]}:{t['regs'][2][0]}-{t['regs'][2][1]}",
             n_tfs=len(t["tfs"]), tfs=",".join(sorted(t["tfs"]))) for k, t in tri.items()]
D = pd.DataFrame(rows); m = lambda r: (p(r)[1] + p(r)[2]) // 2
D["mA"], D["mB"], D["mC"] = D.A.map(m), D.B.map(m), D.C.map(m); D["AB_kb"] = (D.mB - D.mA) / 1000; D["BC_kb"] = (D.mC - D.mB) / 1000; D["AC_kb"] = (D.mC - D.mA) / 1000
D.to_csv(O / "triangles_all.tsv", sep="\t", index=False)
print("distinct triangles", len(D), "; per TF top", sorted(nq.items(), key=lambda t: -t[1])[:8]); print("TFs with any", sum(v > 0 for v in nq.values()), "of", len(nq))
P = D[(D.AB_kb >= 50) & (D.BC_kb >= 50) & (D.AC_kb <= 1500)]; print("spacing (each >= 50 kb, span <= 1.5 Mb)", len(P)); print(P.n_tfs.value_counts().sort_index().to_dict()); print("CTCF only", (P.tfs == "CTCF").sum(), "; chroms", P.chrom.value_counts().head(6).to_dict())
