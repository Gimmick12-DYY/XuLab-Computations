"""Gene-set overlap (Jaccard, background genes) between the top pooled GO terms of each group in checks/go_per_tf, for merging
near-duplicate terms in the panel j draft (same GO files and background as go_per_tf.py).
  python scripts/go_term_jaccard.py <go_per_tf dir> [top_n, default 40]   (Longleaf, hic env)"""
import gzip, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import pandas as pd
X = Path("/work/users/d/y/dyy12/XuLab"); D = X / "data/go"; G = Path(sys.argv[1]); TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
par, cur = defaultdict(set), None
for ln in open(D / "go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
anc = {}; sys.setrecursionlimit(10000)
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
want = set()
for grp in ("specific", "conserved"): want |= set(pd.read_csv(G / f"pooled_{grp}.tsv", sep="\t", index_col=0).sort_values("p").index[:TOP])
N = pd.read_csv(X / "cobinding/tf_groups/results_location/nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
U = set(N[N.gene != ""].gene.str.split(",").explode())
t2g = defaultdict(set)
for ln in gzip.open(D / "goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P" or x[2] not in U: continue
    for t in up(x[4]) & want: t2g[t].add(x[2])
rows = [(a, b, len(t2g[a] & t2g[b]) / len(t2g[a] | t2g[b])) for a, b in combinations(sorted(want), 2) if t2g[a] | t2g[b]]
pd.DataFrame(rows, columns=["term_a", "term_b", "jaccard"]).to_csv(G / "top_term_jaccard.tsv", sep="\t", index=False); print(len(want), "terms")
