"""GO enrichment (hypergeometric, BH) for gene lists vs a background list. GO Consortium go-basic.obo + goa_human.gaf
(XuLab/data/go, downloaded 2026-11-06), NOT-qualified annotations dropped, propagated up is_a/part_of.
Terms with 10-500 background genes. args: background.txt out.tsv list1.txt [list2.txt ...] (aspect via GO_ASPECT, default P)"""
import sys, os, gzip, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
D = "/work/users/d/y/dyy12/XuLab/data/go"; ASP = os.environ.get("GO_ASPECT", "P")
par, name, ns, cur = defaultdict(set), {}, {}, None
for ln in open(f"{D}/go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("name: "): name[cur] = ln[6:]
    elif cur and ln.startswith("namespace: "): ns[cur] = ln[11:]
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
anc = {}
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
sys.setrecursionlimit(10000)
g2t = defaultdict(set)
for ln in gzip.open(f"{D}/goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != ASP: continue
    g2t[x[2]] |= up(x[4])
bg = [g for g in open(sys.argv[1]).read().split() if g in g2t]; bgs = set(bg)
t2g = defaultdict(set)
for g in bg:
    for t in g2t[g]: t2g[t].add(g)
terms = [t for t, s in t2g.items() if 10 <= len(s) <= 500]
rows = []
for f in sys.argv[3:]:
    L = set(open(f).read().split()) & bgs; n = len(L); N = len(bg)
    if n < 5: continue
    for t in terms:
        k = len(t2g[t] & L)
        if k < 2: continue
        K = len(t2g[t]); rows.append({"list": os.path.basename(f).replace(".txt", ""), "term": t, "name": name.get(t, ""), "k": k, "n_list": n, "K": K, "N": N,
                                      "fold": (k / n) / (K / N), "p": hypergeom.sf(k - 1, N, K, n), "genes": ",".join(sorted(t2g[t] & L)[:30])})
R = pd.DataFrame(rows)
if len(R):
    R["q"] = R.groupby("list").p.transform(lambda p: multipletests(p, method="fdr_bh")[1]); R = R.sort_values(["list", "p"])
R.to_csv(sys.argv[2], sep="\t", index=False)
if len(R): print(R[R.q < 0.05].groupby("list").head(8)[["list", "name", "k", "n_list", "fold", "q"]].to_string())
