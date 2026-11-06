"""Representative structure of the exact-SE ensemble: median number of closed loop-edges, within the lowest quartile of self-overlap, Rg and crumpling closest to the ensemble median. Rg over the whole chain; crumpling = non-local 10 kb segment contacts < 1.5 sigma (|i-j| >= 3); self-overlap = same at < 0.5 sigma."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
R0 = Path(__file__).resolve().parents[1]; A = R0 / "model/analysis"; N = 910; P0 = pd.read_csv(A / "per_structure.tsv", sep="\t"); rows = []; iu = None
for f in sorted(glob.glob(sys.argv[1] + "/traj*.npz")):
    for X in np.load(f)["X"][:, :N].astype(float):
        S = X[: N // 5 * 5].reshape(-1, 5, 3).mean(1)
        if iu is None: iu = np.triu_indices(S.shape[0], 3)
        d = np.linalg.norm(S[iu[0]] - S[iu[1]], axis=1); rows.append((np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()), (d < 1.5).sum(), (d < 0.5).sum()))
M = pd.DataFrame(rows, columns=["rg", "crumple", "overlap"]); D = pd.concat([P0, M], axis=1)
for k in ("n_loop_closed", "rg", "crumple", "overlap"): D[k + "_pct"] = D[k].rank(pct=True) * 100
D["traj"] = (D.conf - 1) // 50; D.to_csv(R0 / "figure/conformation_metrics.tsv", sep="\t", index=False); med = int(D.n_loop_closed.median()); C = D[(D.n_loop_closed == med) & (D.overlap <= D.overlap.quantile(0.25))].copy(); C["score"] = (C.rg_pct - 50).abs() + (C.crumple_pct - 50).abs(); C = C.sort_values("score"); print("median", med, D.n_loop_closed.value_counts().sort_index().to_dict()); print(C.head(5).round(2).to_string(index=False))
