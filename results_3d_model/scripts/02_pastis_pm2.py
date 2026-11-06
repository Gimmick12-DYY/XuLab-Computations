"""Step 2: PASTIS PM2 (Varoquaux et al. 2014 Poisson model) on raw 10 kb counts with the ICE bias vector.

Per random seed: random start -> PASTIS MDS2 initialisation -> alternate (alpha, beta) estimation and structure
estimation (pastis.optimization.poisson_model.estimate_alpha_beta / poisson_structure.estimate_X) for N_OUTER rounds.
The pastis PM2.fit loop does not carry X between rounds, so the loop is written out here with the same functions.
The gradient is a vectorised copy of pastis' _poisson_gradient_sparse (same formula, np.add.at instead of a per-bin loop).
--zeros include (default) fits the same PM2 likelihood over ALL fitted bin pairs, zero-count pairs included
(Varoquaux et al. 2014 likelihood); pastis' sparse objective silently skips zero-count pairs, which is kept as
--zeros pastis for comparison. In 'include' mode X is fitted with L-BFGS on that full objective and alpha is fitted
with beta profiled out (closed form, as in pastis' _estimate_beta); initialisation is still pastis MDS2.
Dropped bins are left out of the fit; their coordinates are linearly interpolated afterwards (flag 'interpolated').

Outputs (out/): models/X_seedNN.npy (all bins incl. interpolated), runs.tsv (seed, alpha, beta, loglik),
best_model.tsv, top10_models.npy, model_qc.tsv
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore")
from pastis.optimization import poisson_structure as ps, poisson_model as pmod, mds

def grad_vec(X, counts, alpha=-3, beta=1, bias=None):
    bias = np.ones(X.shape[0]) if bias is None else bias.flatten()
    diff = X[counts.row] - X[counts.col]
    dis = np.sqrt((diff ** 2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha
    g = -((counts.data / fdis - 1) * fdis * alpha / dis ** 2)[:, None] * diff
    out = np.zeros(X.shape)
    np.add.at(out, counts.row, g); np.add.at(out, counts.col, -g)
    return out
ps._poisson_gradient_sparse = grad_vec

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--n-init", type=int, default=50)
ap.add_argument("--n-outer", type=int, default=5)
ap.add_argument("--max-iter", type=int, default=10000)
ap.add_argument("--zeros", choices=["include", "pastis"], default="include")
ap.add_argument("--out-sub", default=None, help="sub-folder of --prep for this run (default: model_<zeros>)")
a = ap.parse_args()
from scipy import optimize
P0 = Path(a.prep); P = P0 / (a.out_sub or f"model_{a.zeros}"); M = P / "models"; M.mkdir(parents=True, exist_ok=True)

bins = pd.read_csv(P0 / "bins.tsv", sep="\t")
raw = np.load(P0 / "counts_raw.npz")["counts"]
keep = bins.keep.values.astype(bool); idx = np.where(keep)[0]
C = raw[np.ix_(idx, idx)]
w = bins.weight.values[idx]
bias = 1.0 / w; bias /= bias.mean()
counts = sparse.coo_matrix(np.triu(C, 1))
counts.eliminate_zeros()
npair = len(idx) * (len(idx) - 1) // 2
print(f"fit bins {len(idx)}; nonzero pairs {counts.nnz}/{npair} ({100*counts.nnz/npair:.1f}%); total contacts {counts.sum():.0f}")

pi, pj = np.triu_indices(len(idx), 1); pc = C[pi, pj]; pbb = bias[pi] * bias[pj]
def full_obj(x, al, be):
    X = x.reshape(-1, 3); diff = X[pi] - X[pj]; d = np.sqrt((diff ** 2).sum(1))
    f = be * pbb * d ** al
    obj = f.sum() - (pc * np.log(f)).sum()
    g = ((f - pc) * al / d ** 2)[:, None] * diff
    G = np.zeros_like(X); np.add.at(G, pi, g); np.add.at(G, pj, -g)
    return obj, G.ravel()
def full_beta(X, al):
    d = np.sqrt(((X[pi] - X[pj]) ** 2).sum(1)); return pc.sum() / (pbb * d ** al).sum()
def full_alpha_beta(X):
    r = optimize.minimize_scalar(lambda al: full_obj(X.ravel(), al, full_beta(X, al))[0], bounds=(-6, -0.1), method="bounded")
    return r.x, full_beta(X, r.x)
def full_X(X, al, be):
    r = optimize.minimize(full_obj, X.ravel(), args=(al, be), jac=True, method="L-BFGS-B", options=dict(maxiter=a.max_iter))
    return r.x.reshape(-1, 3)

def interpolate(Xk):
    X = np.full((len(bins), 3), np.nan)
    X[idx] = Xk
    allb = np.arange(len(bins))
    for d in range(3):
        X[:, d] = np.interp(allb, idx, Xk[:, d])
    return X

rows = []
for seed in range(a.n_init):
    rs = np.random.RandomState(seed)
    X = mds.estimate_X(counts, alpha=-3., beta=1., ini="random", bias=bias, random_state=rs, maxiter=a.max_iter)
    alpha, beta = -3., 1.
    for it in range(a.n_outer):
        if a.zeros == "pastis":
            alpha, beta = pmod.estimate_alpha_beta(counts, X, bias=bias, ini=np.array([alpha, beta]), random_state=rs)
            alpha = float(np.atleast_1d(alpha)[0])
            X = ps.estimate_X(counts, alpha=alpha, beta=beta, ini=X, bias=bias, random_state=rs, maxiter=a.max_iter)
        else:
            alpha, beta = full_alpha_beta(X)
            X = full_X(X, alpha, beta)
    # Poisson log-likelihood without the log(c!) constant: all pairs ('include') or nonzero pairs ('pastis')
    if a.zeros == "pastis":
        ll = -ps.poisson_obj(X, counts.tocoo(), alpha=alpha, beta=beta, bias=bias)
    else:
        ll = -full_obj(X.ravel(), alpha, beta)[0]
    np.save(M / f"X_seed{seed:02d}.npy", interpolate(X))
    rows.append(dict(seed=seed, alpha=alpha, beta=beta, loglik=ll))
    print(f"seed {seed}: alpha {alpha:.3f} beta {beta:.4g} loglik {ll:.2f}", flush=True)
R = pd.DataFrame(rows).sort_values("loglik", ascending=False)
R.to_csv(P / "runs.tsv", sep="\t", index=False)

def norm_step(X):
    X = X - X.mean(0)
    return X / np.linalg.norm(np.diff(X[idx], axis=0), axis=1).mean()
def procrustes_rmsd(A, B):
    """RMSD after optimal rotation (+ reflection: Hi-C cannot tell mirror images apart) and translation."""
    A = A - A.mean(0); B = B - B.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    return np.sqrt(((B @ U @ Vt - A) ** 2).sum(1).mean())
top = [norm_step(np.load(M / f"X_seed{s:02d}.npy")) for s in R.seed[:min(10, len(R))]]
np.save(P / "top10_models.npy", np.array(top))
rm = [procrustes_rmsd(top[i][idx], top[j][idx]) for i in range(len(top)) for j in range(i + 1, len(top))]
best = top[0]
pd.DataFrame(best, columns=["x", "y", "z"]).assign(bin=bins.bin, start=bins.start, interpolated=~keep).to_csv(P / "best_model.tsv", sep="\t", index=False)

Dm = squareform(pdist(best[idx])); Cn = C * w[:, None] * w[None, :]
iu = np.triu_indices(len(idx), 1)
rho_all = stats.spearmanr(Dm[iu], Cn[iu]).correlation
off = np.abs(idx[iu[0]] - idx[iu[1]]) > 1
rho_nonadj = stats.spearmanr(Dm[iu][off], Cn[iu][off]).correlation
# within-diagonal (distance-corrected) agreement: Spearman of model distance vs contact after ranking within each diagonal
dg = idx[iu[1]] - idx[iu[0]]
def within(v):
    return pd.Series(v).groupby(dg).rank(pct=True).values
rho_diag = stats.spearmanr(within(Dm[iu])[off], within(Cn[iu])[off]).correlation
qc = dict(zeros=a.zeros, n_bins=len(bins), n_fit=len(idx), n_interpolated=int((~keep).sum()), best_seed=int(R.seed.iloc[0]),
          alpha=R.alpha.iloc[0], beta=R.beta.iloc[0], loglik=R.loglik.iloc[0],
          rmsd_top10_median=np.median(rm), rmsd_top10_iqr=f"{np.percentile(rm,25):.3f}-{np.percentile(rm,75):.3f}",
          rmsd_units="mean consecutive-bead distance",
          rg_best=np.sqrt(((best[idx] - best[idx].mean(0)) ** 2).sum(1).mean()),
          spearman_dist_vs_ICE_all_pairs=rho_all, spearman_dist_vs_ICE_nonadjacent=rho_nonadj,
          spearman_within_diagonal=rho_diag)
pd.Series(qc).to_csv(P / "model_qc.tsv", sep="\t", header=False)
print(pd.Series(qc).to_string())
