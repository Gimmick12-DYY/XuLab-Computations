"""Ensemble analysis for the exact-SE model: fit (Spearman of O/E vs Hi-C on held-out, unrestrained and restrained pixels, as se_hub/04_analyze.py), closure (< 2 sigma) per structure and per edge for loop-edges (restrained) and Cicero-only edges (not restrained; held-out test), each against 200 random bead pairs at the same separation in the window (same ensemble; pairs within 5 beads of a loop-edge excluded)."""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); a = ap.parse_args()
Bd = Path(a.base); I = Bd / "inputs"; A = Bd / "analysis"; A.mkdir(exist_ok=True); N = 910; rng = np.random.default_rng(31)
H = np.load(I / "hic_10kb.npz"); ice, keep = H["ice"], H["keep"]; n = ice.shape[0]; RS = pd.read_csv(I / "restraints.tsv", sep="\t"); HP = pd.read_csv(I / "heldout_pixels.tsv", sep="\t"); E = pd.read_csv(I / "hub_edges.tsv", sep="\t"); L0 = 430000
mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000; E["b1"] = [min(tb(mid(x)), tb(mid(y))) for x, y in zip(E.region1, E.region2)]; E["b2"] = [max(tb(mid(x)), tb(mid(y))) for x, y in zip(E.region1, E.region2)]; lo = E[E["on_loop_0.7"] == 1]
nu = []
for e in E.itertuples():
    d = e.b2 - e.b1; c = []
    while len(c) < 200:
        i = int(rng.integers(0, N - d)); j = i + d
        if ((np.abs(lo.b1 - i) <= 5) & (np.abs(lo.b2 - j) <= 5)).any(): continue
        c.append((i, j))
    nu.append(c)
nu = np.array(nu); C = np.zeros((N, N), np.int64); nconf = 0; cl = []; cn = np.zeros(nu.shape[:2]); conf = []
for t, f in enumerate(sorted(glob.glob(str(Bd / "sim/eps0/traj*.npz")))):
    X = np.load(f)["X"][:, :N]
    for k, P in enumerate(X):
        pr = cKDTree(P).query_pairs(2.0, output_type="ndarray"); np.add.at(C, (pr[:, 0], pr[:, 1]), 1); nconf += 1; cl.append(np.linalg.norm(P[E.b1] - P[E.b2], axis=1) < 2.0); cn += np.linalg.norm(P[nu[..., 0]] - P[nu[..., 1]], axis=2) < 2.0; conf.append(t * 50 + k + 1)
C = C + C.T; S = C.reshape(n, 5, n, 5).sum((1, 3)) / nconf; np.save(A / "sim_10kb.npy", S); cl = np.array(cl); isl = (E["on_loop_0.7"] == 1).values; nf = cn / nconf
def oe(M):
    R = np.full_like(M, np.nan, dtype=float)
    for d in range(n):
        i = np.arange(n - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(M[i, j])
        if ok.any():
            mu = M[i[ok], j[ok]].mean()
            if mu > 0: R[i, j] = R[j, i] = M[i, j] / mu
    return R
Ho, So = oe(ice), oe(S); Rm = np.zeros((n, n), bool); Rm[RS.bin1, RS.bin2] = True; Rm |= Rm.T; HO = np.zeros((n, n), bool); HO[HP.bin1, HP.bin2] = True; HO |= HO.T; iu = np.triu_indices(n, 2); ok = keep[iu[0]] & keep[iu[1]] & np.isfinite(Ho[iu]) & np.isfinite(So[iu]); kl = np.where(HO[iu], "held-out", np.where(Rm[iu], "restrained", "unrestrained")); res = dict(n_structures=int(nconf), n_loop=int(isl.sum()), n_cicero_only=int((~isl).sum()))
for c in ("held-out", "unrestrained", "restrained"):
    m = ok & (kl == c); res[f"spearman_{c}"] = float(stats.spearmanr(So[iu][m], Ho[iu][m])[0]); res[f"n_{c}"] = int(m.sum())
nl = cl[:, isl].sum(1); nc = cl[:, ~isl].sum(1); res.update(loop_closed_mean=float(nl.mean()), loop_closed_median=float(np.median(nl)), loop_closed_max=int(nl.max()), frac_all_loop_closed=float((nl == isl.sum()).mean()), cicero_closed_mean=float(nc.mean()), cicero_closed_median=float(np.median(nc)))
T = E.drop(columns=["tfs"]).copy(); T["closure"] = cl.mean(0); T["null_mean"] = nf.mean(1); T["ratio"] = T.closure / T.null_mean; T["null_pct"] = [(nf[i] < T.closure.iloc[i]).mean() for i in range(len(T))]; T.to_csv(A / "edge_closure_all.tsv", sep="\t", index=False)
for nm, m in (("loop", isl), ("cicero_only", ~isl)):
    res[f"{nm}_closure_mean"] = float(T.closure[m].mean()); res[f"{nm}_null_mean"] = float(T.null_mean[m].mean()); res[f"{nm}_ratio_mean"] = float(T.closure[m].mean() / T.null_mean[m].mean())
pd.DataFrame(dict(conf=conf, n_loop_closed=nl, n_cicero_closed=nc)).to_csv(A / "per_structure.tsv", sep="\t", index=False); np.save(A / "closed_matrix.npy", cl.astype(np.bool_)); json.dump(res, open(A / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
