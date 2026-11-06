"""v2 step 6 (Mac): find hub flowers in every conformation of the chosen strength and pick a typical clean one.

Per conformation: DBSCAN (1.6 sigma, min 4) on ON TFs. For each cluster: anchors = beads within --anchor-r sigma of any
member TF; hub cores among anchors; petals = gaps between consecutive anchors of --pmin..--pmax beads (a loop that
leaves and re-enters the cluster; shorter gaps are part of the same contact stretch, longer ones are not counted).
Hub cluster = cluster with the most hub-core anchors.
Candidates: >=3 hub cores and >=4 petals. Typical = among candidates with the median hub-core count, the one with the
median petal count, then the TF count closest to the median. Writes flower_table.tsv (all conformations) and flower_pick.npz.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--sim", required=True, help="folder with traj*.npz of the chosen strength")
ap.add_argument("--anchor-r", type=float, default=2.0)
ap.add_argument("--pmin", type=int, default=10)
ap.add_argument("--pmax", type=int, default=100)
a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_flower"; O.mkdir(exist_ok=True)
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B)
hub = B.index[B.hub_label != ""].values
rows, store = [], {}
for t, f in enumerate(sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))):
    z = np.load(f); X, ON = z["X"], z["ON"]
    for k in range(len(X)):
        P = X[k, :N].astype(float); T = X[k, N:].astype(float); on = np.where(ON[k])[0]
        lab = np.full(len(T), -1)
        if len(on) >= 4: lab[on] = DBSCAN(eps=1.6, min_samples=4).fit_predict(T[on])
        best = None
        tree = cKDTree(P)
        for c in range(lab.max() + 1):
            anc = np.unique(np.concatenate(tree.query_ball_point(T[lab == c], a.anchor_r)).astype(int))
            nh = np.isin(hub, anc).sum()
            if best is None or nh > best[0]: best = (nh, c, anc)
        if best is None: rows.append((t, k, -1, 0, 0, 0, 0)); continue
        nh, c, anc = best; g = np.diff(anc); pet = int(((g >= a.pmin) & (g <= a.pmax)).sum())
        rows.append((t, k, c, nh, pet, int((lab == c).sum()), len(anc)))
D = pd.DataFrame(rows, columns=["traj", "frame", "cluster", "n_hub", "n_petals", "n_tf", "n_anchor"])
D.to_csv(O / "flower_table.tsv", sep="\t", index=False)
cand = D[(D.n_hub >= 3) & (D.n_petals >= 4)]
print(f"{len(D)} conformations; hub cluster with >=3 hub cores: {(D.n_hub >= 3).mean()*100:.1f}%; "
      f"and >=4 petals: {len(cand)/len(D)*100:.1f}%; n_hub dist {D.n_hub.value_counts().sort_index().to_dict()}")
mh = int(cand.n_hub.median()); c2 = cand[cand.n_hub == mh]; mp = int(c2.n_petals.median()); c3 = c2[c2.n_petals == mp]
r = c3.iloc[np.argmin(np.abs(c3.n_tf - c3.n_tf.median()))]
print(f"pick: traj {r.traj} frame {r.frame} (conformation {r.traj*50 + r.frame + 1}): hub cores {r.n_hub}, petals {r.n_petals}, TFs {r.n_tf}, anchors {r.n_anchor}; "
      f"medians among candidates: hub {mh}, petals {mp}")
f = sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))[int(r.traj)]
z = np.load(f); k = int(r.frame)
T = z["X"][k, N:]; on = np.where(z["ON"][k])[0]; lab = np.full(len(T), -1); lab[on] = DBSCAN(eps=1.6, min_samples=4).fit_predict(T[on])
np.savez(O / "flower_pick.npz", X=z["X"][k], LT=lab, hc=int(r.cluster), traj=int(r.traj), frame=k, conf=int(r.traj * 50 + k + 1),
         anchor_r=a.anchor_r, pmin=a.pmin, pmax=a.pmax)
