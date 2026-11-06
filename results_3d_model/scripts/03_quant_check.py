"""Step 3: are Cicero shared clique edges / Cicero step-1 pairs / Peakachu loops closer in 3D than distance-matched bin pairs?

For each distinct 10 kb bin pair in a set (both bins fitted, not interpolated; same-bin pairs skipped), draw N_DRAW
random fitted bin pairs whose genomic distance is within +-5 kb of the pair's own span (pairs in the tested set excluded).
ratio = model 3D distance / median 3D distance of its matched draws.
Aggregate statistic: median ratio over pairs. One-sided empirical p: replicate k takes the k-th draw of every pair and
computes the same median ratio; p = (1 + #{null median <= observed}) / (1 + N_DRAW).
Done on the best model and on each of the top 10 models.
Outputs (out/): quant_check.tsv (per set x model), quant_pairs_best.tsv (per pair, best model), quant_null_best.npz
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--model", required=True, help="model sub-folder (top10_models.npy); outputs go there")
ap.add_argument("--n-draw", type=int, default=1000)
ap.add_argument("--tol", type=int, default=5000)
ap.add_argument("--res", type=int, default=10000)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
P0 = Path(a.prep); P = Path(a.model)
bins = pd.read_csv(P0 / "bins.tsv", sep="\t"); keep = bins.keep.values.astype(bool)
models = np.load(P / "top10_models.npy")
E = pd.read_csv(P0 / "shared_edges.tsv", sep="\t"); E["span"] = E.distance_kb * 1000
C = pd.read_csv(P0 / "cicero_pairs.tsv", sep="\t"); C["span"] = C.span_kb * 1000
Lp = pd.read_csv(P0 / "loops.tsv", sep="\t"); Lp["span"] = (Lp.bin2 - Lp.bin1) * a.res
sets = {
    "shared_clique_edges": E.groupby(["bin1", "bin2"], as_index=False).agg(span=("span", "median"), n=("tfs", "size")),
    "cicero_step1_pairs": C.groupby(["bin1", "bin2"], as_index=False).agg(span=("span", "median"), n=("tf", "nunique")),
    "peakachu_0.7_loops": Lp[["bin1", "bin2", "span"]].assign(n=1),
}
fit = np.where(keep)[0]
I, J = np.triu_indices(len(bins), 1)
ok = keep[I] & keep[J]; I, J = I[ok], J[ok]
gdist = (J - I) * a.res
rs = np.random.RandomState(a.seed)
rows, prow, nulls = [], [], {}
for name, S in sets.items():
    S = S[(S.bin1 != S.bin2) & keep[S.bin1] & keep[S.bin2]].reset_index(drop=True)
    own = set(zip(S.bin1, S.bin2))
    pool_ok = np.array([(i, j) not in own for i, j in zip(I, J)])
    draws = []
    for r in S.itertuples():
        cand = np.where(pool_ok & (np.abs(gdist - r.span) <= a.tol))[0]
        draws.append(rs.choice(cand, a.n_draw, replace=True))
    draws = np.array(draws)  # pairs x draws, index into I/J
    for m, X in enumerate(models):
        D = squareform(pdist(X))
        d_obs = D[S.bin1, S.bin2]
        d_null = D[I[draws], J[draws]]
        med = np.median(d_null, axis=1)
        ratio = d_obs / med
        null_med = np.median(d_null / med[:, None], axis=0)
        obs = np.median(ratio)
        p = (1 + (null_med <= obs).sum()) / (1 + a.n_draw)
        frac_closer = np.mean(d_obs < med)
        rows.append(dict(set=name, model_rank=m + 1, n_pairs=len(S), median_ratio=obs, p_one_sided=p,
                         frac_pairs_closer_than_matched_median=frac_closer,
                         median_dist=np.median(d_obs), median_matched_dist=np.median(d_null)))
        if m == 0:
            pct = (d_null < d_obs[:, None]).mean(1)
            prow.append(S.assign(set=name, dist=d_obs, matched_median=med, ratio=ratio, matched_percentile=pct))
            nulls[name] = (d_null / med[:, None]).ravel()
Q = pd.DataFrame(rows)
Q.to_csv(P / "quant_check.tsv", sep="\t", index=False)
pd.concat(prow).to_csv(P / "quant_pairs_best.tsv", sep="\t", index=False)
np.savez_compressed(P / "quant_null_best.npz", **nulls)
best = Q[Q.model_rank == 1].set_index("set")
top = Q.groupby("set").agg(median_ratio_top10=("median_ratio", "median"), p_top10_median=("p_one_sided", "median"),
                           p_top10_max=("p_one_sided", "max"), n_models_p05=("p_one_sided", lambda v: int((v < 0.05).sum())))
print(best[["n_pairs", "median_ratio", "p_one_sided", "frac_pairs_closer_than_matched_median"]].join(top).to_string())
