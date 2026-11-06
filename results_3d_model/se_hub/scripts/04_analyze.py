"""SE-hub step 2 analysis: ensemble contact map (bead pairs < 2 sigma -> 10 kb), Spearman of O/E vs Hi-C on held-out / unrestrained /
restrained pixels (>= 20 kb, kept bins; O/E = per-distance mean over kept bins of the window), hub loop-edge closure (< 2 sigma) per
structure and per edge, representative = most hub edges closed, then most compact hub elements (Rg); its percentile is reported."""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); a = ap.parse_args()
Bd = Path(a.base); I = Bd / "inputs"; A = Bd / "analysis"; A.mkdir(exist_ok=True); N = 910
H = np.load(I / "hic_10kb.npz"); ice, keep = H["ice"], H["keep"]; n = ice.shape[0]
RS = pd.read_csv(I / "restraints.tsv", sep="\t"); HP = pd.read_csv(I / "heldout_pixels.tsv", sep="\t"); EL = pd.read_csv(I / "hub_elements.tsv", sep="\t")
RE = RS[RS.kind == "red_edge"].reset_index(drop=True); ends = EL.bead.values
C = np.zeros((N, N), np.int64); nconf = 0; closed, rg, conf = [], [], []
for t, f in enumerate(sorted(glob.glob(str(Bd / "sim/eps0/traj*.npz")))):
    X = np.load(f)["X"][:, :N]
    for k, P in enumerate(X):
        pr = cKDTree(P).query_pairs(2.0, output_type="ndarray"); np.add.at(C, (pr[:, 0], pr[:, 1]), 1); nconf += 1
        closed.append(np.linalg.norm(P[RE.b1] - P[RE.b2], axis=1) < 2.0); Q = P[ends]; rg.append(np.sqrt(((Q - Q.mean(0)) ** 2).sum(1).mean())); conf.append(t * 50 + k + 1)
C = C + C.T; S = C.reshape(n, 5, n, 5).sum((1, 3)) / nconf; np.save(A / "sim_10kb.npy", S)
closed = np.array(closed); rg = np.array(rg); conf = np.array(conf); nc = closed.sum(1)
def oe(M):
    E = np.full_like(M, np.nan, dtype=float)
    for d in range(n):
        i = np.arange(n - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(M[i, j])
        if ok.any():
            mu = M[i[ok], j[ok]].mean()
            if mu > 0: E[i, j] = E[j, i] = M[i, j] / mu
    return E
Ho, So = oe(ice), oe(S)
RES_ = np.zeros((n, n), bool); RES_[RS.bin1, RS.bin2] = True; RES_ |= RES_.T; HO = np.zeros((n, n), bool); HO[HP.bin1, HP.bin2] = True; HO |= HO.T
iu = np.triu_indices(n, 2); ok = keep[iu[0]] & keep[iu[1]] & np.isfinite(Ho[iu]) & np.isfinite(So[iu])
cl = np.where(HO[iu], "held-out", np.where(RES_[iu], "restrained", "unrestrained")); res = dict(n_structures=int(nconf))
for c in ("held-out", "unrestrained", "restrained"):
    m = ok & (cl == c); res[f"spearman_{c}"] = float(stats.spearmanr(So[iu][m], Ho[iu][m])[0]); res[f"n_{c}"] = int(m.sum())
order = np.lexsort((rg, -nc)); k = int(order[0])
res.update(mean_closed=float(nc.mean()), n_edges=int(len(RE)), frac_all_closed=float((nc == len(RE)).mean()), rep_conf=int(conf[k]), rep_closed=int(nc[k]),
           rep_frac_at_or_above=float((nc >= nc[k]).mean()), rep_rg=float(rg[k]), rg_median=float(np.median(rg)), per_edge_min=float(closed.mean(0).min()), per_edge_max=float(closed.mean(0).max()))
pd.DataFrame(dict(b1=RE.b1, b2=RE.b2, closure_freq=closed.mean(0), closed_in_rep=closed[k])).to_csv(A / "edge_closure.tsv", sep="\t", index=False)
pd.DataFrame(dict(conf=conf, n_closed=nc, rg=rg)).to_csv(A / "per_structure.tsv", sep="\t", index=False)
json.dump(res, open(A / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
