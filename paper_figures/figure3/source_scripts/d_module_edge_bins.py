"""Fig 3 panel d heatmap input: ALL module edges >25 kb of each TF (the edge set of the panel d bars), counted in genomic bins.
Edges and loop calls exactly as cobinding/hub_figures/descriptive/clique_coverage.py (the bars' clique_loop_coverage.tsv): every pair of
regions inside one module (cliques.tsv; ourrun_k50, RBBP4 from RBBP4_sweep_k50_i10000_s555) with midpoints >25 kb apart; on a loop =
both ends within +-30 kb of the two anchors of one Peakachu score >= 0.7 loop (5 kb or 10 kb, wild-type HEK293T). An edge is placed in
the bin of its midpoint (mean of the two region midpoints).
Writes d_module_edge_bins_<bin>.tsv.gz: tf, chrom, bin, n_edges, n_loop.
  python scripts/d_module_edge_bins.py <outdir> [bin_bp, default 1000000]   (Longleaf, hic env)"""
import glob, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
X = Path("/work/users/d/y/dyy12/XuLab"); R = X / "cobinding/results"; LD = X / "hic/work_wt293t/loops"
OUT = Path(sys.argv[1]); BIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
L = defaultdict(list)
for f in ("peakachu_5000.loops.0.7.bedpe", "peakachu_10000.loops.0.7.bedpe"):
    for ln in open(LD / f):
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
files = {"RBBP4": R / "RBBP4_sweep_k50_i10000_s555/cliques.tsv"}
files.update({Path(f).parent.name: Path(f) for f in glob.glob(str(R / "ourrun_k50/*/cliques.tsv"))})
cnt = defaultdict(lambda: [0, 0]); tot = {}
for tf, f in sorted(files.items()):
    d = pd.read_csv(f, sep="\t", usecols=["clique_id", "regions"]); n = 0
    for rr in d.regions.astype(str):
        pk = [iv(x) for x in rr.split(";")]
        for p, q in combinations(pk, 2):
            mp, mq = (p[1] + p[2]) // 2, (q[1] + q[2]) // 2
            if abs(mp - mq) <= 25000: continue
            k = (tf, p[0], (mp + mq) // 2 // BIN); c = cnt[k]; c[0] += 1; c[1] += on_loop(p, q); n += 1
    tot[tf] = n
D = pd.DataFrame([(*k, *v) for k, v in cnt.items()], columns=["tf", "chrom", "bin", "n_edges", "n_loop"])
D.to_csv(OUT / f"d_module_edge_bins_{BIN // 1000}kb.tsv.gz", sep="\t", index=False)
print("edges per TF:", {t: tot[t] for t in sorted(tot, key=lambda t: -tot[t])[:6]}, "total", sum(tot.values()), "; occupied bins", D.groupby(["chrom", "bin"]).ngroups)
