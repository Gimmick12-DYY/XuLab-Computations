"""Flower view search: for each structure closing the maximum SE loop-edges (14), sample 4,000 random viewing directions and score how evenly the petals (chain midpoints between consecutive closed anchors, weighted by reach) spread around the SE in projection: low circular resultant, many occupied 30-degree sectors. The structure's coordinates are not changed; only the camera. Writes flower_view.json."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter1d
R0 = Path(__file__).resolve().parents[1]; O = R0 / "figure_extreme"; N = 910; L0 = 430000; files = sorted(glob.glob(sys.argv[1] + "/traj*.npz")); H = pd.read_csv(R0 / "hub_edges_all.tsv", sep="\t"); mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000; H["b1"] = [tb(mid(x)) for x in H.reg_a]; H["b2"] = [tb(mid(x)) for x in H.reg_b]; se_b = tb((1248039 + 1254347) // 2); M = pd.read_csv(R0 / "figure/conformation_metrics.tsv", sep="\t"); rows = []; best = None; L = H[(H.from_se == 1) & (H.on_loop == 1)]
for c in M[M.n_loop_closed == 14].conf:
    t, k = divmod(int(c) - 1, 50); X = np.load(files[t])["X"][k, :N].astype(float); Y = (X - X[se_b]) * 10; Z = gaussian_filter1d(Y, 4.0, axis=0, mode="nearest"); ends = [int(a1 if str(a) != "SE" else a2) for a, a1, a2 in zip(L.a, L.b1, L.b2) if np.linalg.norm(X[a1] - X[a2]) < 2]; ch = sorted(set(ends + [se_b])); mids = np.array([Z[(p + q) // 2] for p, q in zip(ch[:-1], ch[1:]) if q - p > 8]); rs = Rot.random(4000, random_state=int(c)); bl = None
    for i in range(len(rs)):
        p = rs[i].apply(mids)[:, :2]; a = np.arctan2(p[:, 1], p[:, 0]); r = np.hypot(p[:, 0], p[:, 1]); res = np.abs(np.exp(1j * a).mean()); sec = len(set(((a + np.pi) // (np.pi / 6)).astype(int))); sc = sec / 12 - res + 0.3 * min(1, np.median(r) / (np.linalg.norm(mids, axis=1).max() + 1e-6))
        if bl is None or sc > bl[0]: bl = (sc, i, sec, res)
    m = M[M.conf == c].iloc[0]; rows.append(dict(conf=int(c), score=bl[0], sectors=bl[2], resultant=bl[3], n_petals=len(mids), max_reach=float(np.linalg.norm(mids, axis=1).max()), overlap=int(m.overlap), rg_pct=m.rg_pct, crumple_pct=m.crumple_pct))
    if best is None or bl[0] > best[0]: best = (bl[0], int(c), rs[bl[1]].as_matrix().tolist())
T = pd.DataFrame(rows).sort_values("score", ascending=False); T.to_csv(O / "flower_view_scores.tsv", sep="\t", index=False); print(T.round(2).to_string(index=False)); json.dump(dict(conf=best[1], R=best[2]), open(O / "flower_view.json", "w"))
