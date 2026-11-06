"""Pick 3 typical structures closing the median 4-5 SE-candidate loops, with partner subsets covering as many of the 12 partners as possible. Eligible: 4 or 5 closed (< 2 sigma), Rg and crumpling within the 25th-75th percentile, self-overlap in the lowest quartile, each from a different trajectory.
Among triples with the largest union and the least overlap between subsets, the one closest to median Rg and crumpling."""
import glob, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_v4_snapshots"; N = 910
EL = pd.read_csv(Bd / "model/inputs/hub_elements.tsv", sep="\t"); core = int(EL.bead[EL.is_core].iloc[0]); part = EL[~EL.is_core].sort_values("bead"); bd = part.bead.values; D = pd.read_csv(Bd / "figure_3d_v2/conformation_metrics.tsv", sep="\t")
mask = []
for f in sorted(glob.glob(sys.argv[1] + "/traj*.npz")):
    XX = np.load(f)["X"][:, :N].astype(float); c = np.linalg.norm(XX[:, bd] - XX[:, [core]], axis=2) < 2
    mask.extend((c * (1 << np.arange(12))).sum(1).tolist())
D["mask"] = mask; D["n"] = [bin(m).count("1") for m in D["mask"]]; assert (D.n == D.n_closed).all(); D["traj"] = (D.conf - 1) // 50
E = D[D.n.isin([4, 5]) & D.rg_pct.between(25, 75) & D.crumple_pct.between(25, 75) & (D.overlap <= D.overlap.quantile(0.25))].copy(); E["dev"] = (E.rg_pct - 50).abs() + (E.crumple_pct - 50).abs(); print("eligible", len(E), E["mask"].nunique())
B = E.sort_values("dev").drop_duplicates("mask").reset_index(drop=True); ms = B["mask"].values; dev = B.dev.values; tr = B.traj.values; best = None; m = len(B); print("distinct", m)
for i in range(min(m, 400)):
    for j in range(i + 1, min(m, 400)):
        if tr[i] == tr[j]: continue
        u = ms[i] | ms[j]; ov = bin(ms[i] & ms[j]).count("1")
        for k in range(j + 1, min(m, 400)):
            if tr[k] in (tr[i], tr[j]): continue
            U = bin(u | ms[k]).count("1"); key = (-U, ov + bin((u) & ms[k]).count("1"), dev[i] + dev[j] + dev[k])
            if best is None or key < best[0]: best = (key, i, j, k)
print("best", best[0]); labs = part.label.values; rows = []
for q in best[1:]:
    r = B.iloc[q]; cl = [labs[z] for z in range(12) if int(r["mask"]) >> z & 1]; print(int(r.conf), int(r.n), round(r.rg_pct, 1), round(r.crumple_pct, 1), int(r.overlap), cl); rows.append(dict(conf=int(r.conf), n_closed=int(r.n), rg_pct=r.rg_pct, crumple_pct=r.crumple_pct, overlap=int(r.overlap), n_closed_pct=r.n_closed_pct, closed=";".join(cl)))
pd.DataFrame(rows).to_csv(O / "snapshots.tsv", sep="\t", index=False)
c_ = np.zeros(12, int); ml = E["mask"].values
for z in range(12): c_[z] = ((ml >> z) & 1).sum()
print(dict(zip(labs, c_)))
