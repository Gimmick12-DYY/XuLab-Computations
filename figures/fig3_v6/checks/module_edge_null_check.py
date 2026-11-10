#!/usr/bin/env python3
"""Fig 3 v6 panel c check (2026-11-09): module-edge loop overlap vs two shuffle nulls.
The S1D null (perm_tf_long.py) shifts each module (clique) as ONE block; this script adds the per-edge null used for pairs
(each edge >25 kb shifted independently on its chromosome, keeping its spacing; fill_slide_loop_table.shift_pair), with the same
loops (Peakachu 5 kb + 10 kb, score >=0.7), same +-30 kb slop (assign_loop) and the same >25 kb edge set (load_clique_edges).
  task mode:    module_edge_null_check.py --tf TF --edges E --cliques C --fine F --coarse K --n-perm 1000 --chunk k --out x.npz
  summary mode: module_edge_null_check.py --summary <npz dir> --s1d <s1d_per_tf_null.tsv> --out <tsv>"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
for k in ("--tf", "--edges", "--cliques", "--fine", "--coarse", "--out", "--summary", "--s1d"): ap.add_argument(k)
ap.add_argument("--n-perm", type=int, default=1000); ap.add_argument("--chunk", type=int, default=0)
a = ap.parse_args()
if a.summary:
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
    S = pd.read_csv(a.s1d, sep="\t").set_index("tf"); rows = []
    for tf in sorted({Path(f).name.rsplit("_", 1)[0] for f in glob.glob(f"{a.summary}/*.npz")}):
        z = [np.load(f) for f in sorted(glob.glob(f"{a.summary}/{tf}_*.npz"))]
        nu = np.concatenate([q["null"] for q in z]); obs = float(z[0]["obs"]); n = int(z[0]["n"][0]); nm = int(z[0]["n"][1])
        rows.append(dict(tf=tf, module_edges=n, modules=nm, obs_pct=100 * obs, s1d_obs_pct=100 * S.edge_obs[tf],
                         block_null_med=100 * S.edge_null_med[tf], block_null_lo=100 * S.edge_null_lo[tf], block_null_hi=100 * S.edge_null_hi[tf], block_p=S.edge_p[tf],
                         edge_null_med=100 * np.median(nu), edge_null_lo=100 * np.percentile(nu, 2.5), edge_null_hi=100 * np.percentile(nu, 97.5),
                         edge_p=(1 + (nu >= obs).sum()) / (1 + len(nu)), n_perm=len(nu)))
    R = pd.DataFrame(rows).set_index("tf")
    R["block_q_bh40"] = multipletests(R.block_p, method="fdr_bh")[1]; R["edge_q_bh40"] = multipletests(R.edge_p, method="fdr_bh")[1]
    R.round(5).to_csv(a.out, sep="\t")
    print(f"{len(R)} TFs; obs matches S1D: {np.allclose(R.obs_pct, R.s1d_obs_pct)}; block null q<0.05: {(R.block_q_bh40 < .05).sum()}; per-edge null q<0.05: {(R.edge_q_bh40 < .05).sum()}")
    sys.exit()
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/hic/scripts")
import fill_slide_loop_table as F  # noqa: E402
fine = F.U.load_bedpe(Path(a.fine), "5000"); coarse = F.U.load_bedpe(Path(a.coarse), "10000")
ix5 = F.U.index_anchors(fine, 5000); ix10 = F.U.index_anchors(coarse, 10000)
edges = [e for e in F.C.load_clique_edges(Path(a.cliques)) if F.span_bp(e) > F.MIN_SPAN]
obs = F.score_pairs(edges, fine, coarse, ix5, ix10)[0] / len(edges)
chk = F.rates([], edges, fine, coarse, ix5, ix10)["edge_rate"]          # the S1D observed rate, same edge set
assert abs(obs - chk) < 1e-12, (obs, chk)
rng = np.random.default_rng(20_000 * (a.chunk + 1) + 11); out = np.empty(a.n_perm)
for i in range(a.n_perm):
    out[i] = F.score_pairs(F.permute_pairs(edges, F.HG38, rng), fine, coarse, ix5, ix10)[0] / len(edges)
np.savez(a.out, null=out, obs=np.array(obs), n=np.array([len(edges), len({e["clique_id"] for e in edges})]))
print("done", a.tf, a.chunk, obs, flush=True)
