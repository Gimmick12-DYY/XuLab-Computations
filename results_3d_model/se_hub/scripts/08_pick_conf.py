"""Pick a representative conformation of the SE-hub Hi-C-first ensemble (not an extreme one). For each of the 10,000 structures, on the
chain smoothed to 10 kb: n_closed (SE-partner spokes < 2 sigma), Rg, crumpling (non-local 10 kb segment contacts < 1.5 sigma, |i-j| >= 3),
self-overlap (non-local pairs < 0.5 sigma). Representative = median n_closed, then smallest summed |percentile - 50| over Rg and crumpling,
among the structures with the lowest quartile of self-overlap."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_3d_v2"; T = sys.argv[1]; N = 910
EL = pd.read_csv(Bd / "model/inputs/hub_elements.tsv", sep="\t"); core = int(EL.bead[EL.is_core].iloc[0]); part = EL.bead[~EL.is_core].values
rows = []; iu = None; c = 0
for f in sorted(glob.glob(T + "/traj*.npz")):
    XX = np.load(f)["X"][:, :N].astype(float)
    for X in XX:
        c += 1; S = X[: N // 5 * 5].reshape(-1, 5, 3).mean(1); m = S.shape[0]
        if iu is None: iu = np.triu_indices(m, 3)
        d = np.linalg.norm(S[iu[0]] - S[iu[1]], axis=1); ds = np.linalg.norm(X[part] - X[core], axis=1)
        pp = np.linalg.norm(X[part][:, None] - X[part][None], axis=2)[np.triu_indices(len(part), 1)]
        rows.append((c, (ds < 2).sum(), np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()), (d < 1.5).sum(), (d < 0.5).sum(), np.median(ds), np.median(pp)))
D = pd.DataFrame(rows, columns="conf n_closed rg crumple overlap med_se_partner med_partner_partner".split())
for k in ("n_closed", "rg", "crumple", "overlap"): D[k + "_pct"] = D[k].rank(pct=True) * 100
D.to_csv(O / "conformation_metrics.tsv", sep="\t", index=False)
med = int(D.n_closed.median()); C = D[(D.n_closed == med) & (D.overlap <= D.overlap.quantile(0.25))].copy()
C["score"] = (C.rg_pct - 50).abs() + (C.crumple_pct - 50).abs(); C = C.sort_values("score")
print(D.describe().loc[["min", "25%", "50%", "75%", "max"]].round(2).T.to_string()); print("median n_closed", med, "candidates", len(C)); print(C.head(10).round(2).to_string(index=False))
