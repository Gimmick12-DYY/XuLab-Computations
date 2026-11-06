#!/usr/bin/env python3
"""Per-TF same-chromosome shuffle null for the S1D loop-overlap rates (pairs and clique edges with span >25 kb).
Same shuffle as fill_slide_loop_table.py (pairs shifted independently keeping spacing; cliques shifted as one block),
same loops (5 kb ∪ 10 kb Peakachu ≥0.7) and ±30 kb anchor slop. Writes the null rates of one chunk of shuffles."""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/hic/scripts")
import fill_slide_loop_table as F  # noqa: E402
ap = argparse.ArgumentParser()
for k in ("--tf", "--edges", "--cliques", "--fine", "--coarse", "--out"): ap.add_argument(k, required=True)
ap.add_argument("--n-perm", type=int, default=1000); ap.add_argument("--chunk", type=int, default=0)
a = ap.parse_args()
fine = F.U.load_bedpe(Path(a.fine), "5000"); coarse = F.U.load_bedpe(Path(a.coarse), "10000")
ix5 = F.U.index_anchors(fine, 5000); ix10 = F.U.index_anchors(coarse, 10000)
pairs = [p for p in F.U.load_edges(Path(a.edges)) if F.span_bp(p) > F.MIN_SPAN]
edges = [e for e in F.C.load_clique_edges(Path(a.cliques)) if F.span_bp(e) > F.MIN_SPAN] if Path(a.cliques).exists() else []
obs = F.rates(pairs, edges, fine, coarse, ix5, ix10)
rng = np.random.default_rng(10_000 * (a.chunk + 1) + 7)
out = np.full((a.n_perm, 2), np.nan)
for i in range(a.n_perm):
    r = F.rates(F.permute_pairs(pairs, F.HG38, rng), F.permute_clique_edges(edges, F.HG38, rng) if edges else [], fine, coarse, ix5, ix10)
    out[i] = (r["pair_rate"], r["edge_rate"])
    if (i + 1) % 100 == 0: print(a.tf, a.chunk, i + 1, flush=True)
np.savez(a.out, null=out, obs=np.array([obs["pair_rate"], obs["edge_rate"]]), n=np.array([obs["pair_d"], obs["edge_d"]]))
print("done", a.tf, a.chunk, obs["pair_rate"], obs["edge_rate"], flush=True)
