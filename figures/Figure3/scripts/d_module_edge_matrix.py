"""Fig 3 panel d heatmap input: module edges >25 kb shared by >=MIN_TF TFs (default 2), with Hi-C loop status per edge.
Module edges = pairs inside one TF's higher-order coordinated binding module (ourrun_k50 cliques, FDR <= 0.05 edges, size >= 3; RBBP4 from
its sweep run), >25 kb, from cobinding/tf_groups/results_cliques (build_incidence.py ... cliques): peaks of all TFs merged by overlap, an edge
is the same edge across TFs when both ends fall in the same two merged regions. Loop status: the panel d rule (Peakachu score >= 0.7 loops,
wild-type HEK293T, 5 kb then 10 kb; both edge ends within +-30 kb of the two loop anchors; compare_union_pairs_and_clique_edges.assign_loop),
applied to the merged-region coordinates of the edge.
Writes d_module_edges_shared.tsv.gz: one row per edge shared by >=MIN_TF TFs (chrom, r1/r2 coords, n_tfs, tfs, on_loop).
  python scripts/d_module_edge_matrix.py <outdir> [min_tfs, default 2]   (Longleaf, hic env)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
X = Path("/work/users/d/y/dyy12/XuLab"); sys.path.insert(0, str(X / "hic/scripts"))
import fill_slide_loop_table as F  # noqa: E402
TG = X / "cobinding/tf_groups/results_cliques"; L = X / "hic/work_wt293t/loops"; OUT = Path(sys.argv[1])
inc = sp.load_npz(TG / "inc_pair.npz").tocsc(); tfs = [t.strip() for t in open(TG / "tfs.txt") if t.strip()]
pr = pd.read_csv(TG / "pairs.tsv", sep="\t"); rg = pd.read_csv(TG / "regions.tsv", sep="\t")
MIN_TF = int(sys.argv[2]) if len(sys.argv) > 2 else 2; ntf = np.diff(inc.indptr); keep = np.where(ntf >= MIN_TF)[0]
fine = F.U.load_bedpe(L / "peakachu_5000.loops.0.7.bedpe", "5000"); coarse = F.U.load_bedpe(L / "peakachu_10000.loops.0.7.bedpe", "10000")
ix5 = F.U.index_anchors(fine, 5000); ix10 = F.U.index_anchors(coarse, 10000)
rows = []
for j in keep:
    a, b = rg.iloc[pr.r1[j]], rg.iloc[pr.r2[j]]
    e = {"p1": (a.chrom, int(a.start), int(a.end)), "p2": (b.chrom, int(b.start), int(b.end))}
    lp = F.C.assign_loop(e, fine, ix5, coarse, ix10, F.SLOP)
    rows.append(dict(chrom=a.chrom, r1_start=int(a.start), r1_end=int(a.end), r2_start=int(b.start), r2_end=int(b.end), dist_kb=pr.dist_kb[j],
                     n_tfs=int(ntf[j]), tfs=",".join(tfs[i] for i in inc.indices[inc.indptr[j]:inc.indptr[j + 1]]), on_loop=int(lp is not None)))
D = pd.DataFrame(rows); D.to_csv(OUT / "d_module_edges_shared.tsv.gz", sep="\t", index=False)
print(f"module edges {inc.shape[1]:,}; shared by >={MIN_TF} TFs {len(D):,}; on a loop {D.on_loop.sum():,} ({100 * D.on_loop.mean():.1f}%)")
