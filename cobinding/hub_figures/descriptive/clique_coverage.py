"""Per clique: >25 kb edges (pairs of regions in the clique) and how many sit on a 0.7 loop (both ends within ±30 kb
of the two anchors of one 5 kb or 10 kb Peakachu loop)."""
import glob
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
X = Path("../strat/x/work/users/d/y/dyy12/XuLab/cobinding/results")
L = defaultdict(list)
for f in ("loops/peakachu_5000.loops.0.7.bedpe", "loops/peakachu_10000.loops.0.7.bedpe"):
    for ln in open(f):
        x = ln.split()
        if x[0] == x[3]:
            a = (int(x[1]), int(x[2])); b = (int(x[4]), int(x[5]))
            if a > b: a, b = b, a
            L[x[0]].append((*a, *b))
LA = {c: np.array(v) for c, v in L.items()}
def iv(r):
    c, se = r.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
def on_loop(p, q, slop=30000):
    if (p[1] + p[2]) // 2 > (q[1] + q[2]) // 2: p, q = q, p
    A = LA.get(p[0])
    return A is not None and bool(((p[1] <= A[:, 1] + slop) & (p[2] >= A[:, 0] - slop) & (q[1] <= A[:, 3] + slop) & (q[2] >= A[:, 2] - slop)).any())
rows = []
files = {"RBBP4": X / "RBBP4_sweep_k50_i10000_s555/cliques.tsv"}
files.update({Path(f).parent.name: Path(f) for f in glob.glob(str(X / "ourrun_k50/*/cliques.tsv"))})
for tf, f in sorted(files.items()):
    d = pd.read_csv(f, sep="\t", usecols=["clique_id", "regions"])
    for cid, rr in zip(d.clique_id, d.regions.astype(str)):
        pk = [iv(x) for x in rr.split(";")]
        es = [(p, q) for p, q in combinations(pk, 2) if abs((p[1] + p[2]) // 2 - (q[1] + q[2]) // 2) > 25000]
        if not es: continue
        k = sum(on_loop(p, q) for p, q in es)
        rows.append((tf, cid, len(pk), len(es), k))
D = pd.DataFrame(rows, columns=["tf", "clique_id", "n_regions", "edges_gt25kb", "loop_edges"])
D["coverage"] = D.loop_edges / D.edges_gt25kb
D.to_csv("clique_loop_coverage.tsv", sep="\t", index=False)
r = D[D.tf == "RBBP4"]; print("RBBP4 cliques", len(r), "edges", r.edges_gt25kb.sum(), "loop edges", r.loop_edges.sum(), ">=1", (r.loop_edges >= 1).sum(), "all", (r.coverage == 1).sum())
