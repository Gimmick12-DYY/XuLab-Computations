"""v2 step 9 (Mac): pick the conformation whose largest flower best matches the browser figure's chr16p11.2 hub component.

Hub component = the 23 elements / 24 shared clique edges of hub_chr16_edges.tsv (the browser figure). Element -> bead of
its midpoint; label = gene with a TSS within 2 kb (browser gene track), else position in Mb.
Per conformation: DBSCAN (1.6 sigma, min 4) on ON TFs; anchors of a cluster = beads within 2 sigma of its TFs;
petals = gaps of 10-100 beads (20-200 kb) between consecutive anchors. For each conformation keep the cluster with the
most hub-component elements among its anchors; record elements, edges with both ends among the anchors, petals.
Pick: among conformations with 4-6 petals, the most elements, then the most edges, then fewest stray anchors. This is a
SELECTED example; the fraction of conformations reaching the same level is reported.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--sim", required=True)
a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_hub_flow"; I = O / "inputs"
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B); L0 = int(B.start[0])
E = pd.read_csv(I / "hub_chr16_edges.tsv", sep="\t")
G = pd.read_csv(I / "genes_chr16_29.7-31.6.tsv", sep="\t"); G["tss"] = np.where(G.strand == "+", G.txStart, G.txEnd)
def parse(r): c, s = r.split(":"); s, e = map(int, s.split("-")); return s, e
els = sorted({r for c in ("region1", "region2") for r in E[c]}, key=lambda r: parse(r)[0])
rows = []
for r in els:
    s, e = parse(r); d = np.minimum(np.abs(G.tss - s), np.abs(G.tss - e)); d[(G.tss >= s) & (G.tss <= e)] = 0
    g = sorted(set(G.loc[d <= 2000, "name"]))
    rows.append(dict(element=r, bead=int(((s + e) // 2 - L0) // 2000), label="/".join(g) if g else f"{(s + e) / 2e6:.3f} Mb"))
EL = pd.DataFrame(rows); EL.to_csv(O / "hub_elements.tsv", sep="\t", index=False)
bead_of = dict(zip(EL.element, EL.bead))
E["b1"] = E.region1.map(bead_of); E["b2"] = E.region2.map(bead_of)
E.to_csv(O / "hub_edges_beads.tsv", sep="\t", index=False)
eb = set(EL.bead)
print(f"{len(EL)} elements on {len(eb)} beads; {len(E)} edges ({int(E['on_loop_0.7'].sum())} on a 0.7 loop)")

res, files = [], sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))
for t, f in enumerate(files):
    z = np.load(f); X, ON = z["X"], z["ON"]
    for k in range(len(X)):
        P = X[k, :N].astype(float); T = X[k, N:].astype(float); on = np.where(ON[k])[0]
        lab = np.full(len(T), -1)
        if len(on) >= 4: lab[on] = DBSCAN(eps=1.6, min_samples=4).fit_predict(T[on])
        tree = cKDTree(P); best = (0, 0, -1, 0, 0)
        for c in range(lab.max() + 1):
            anc = np.unique(np.concatenate(tree.query_ball_point(T[lab == c], 2.0)).astype(int)); sa = set(anc)
            ne = len(eb & sa); nedge = int(sum((b1 in sa) and (b2 in sa) for b1, b2 in zip(E.b1, E.b2)))
            g = np.diff(anc); pet = int(((g >= 10) & (g <= 100)).sum())
            if (ne, nedge) > best[:2]: best = (ne, nedge, c, pet, len(anc))
        res.append((t, k, *best))
D = pd.DataFrame(res, columns=["traj", "frame", "n_elements", "n_edges", "cluster", "n_petals", "n_anchor"])
D["conf"] = D.traj * 50 + D.frame + 1
D.to_csv(O / "hub_flow_table.tsv", sep="\t", index=False)
cand = D[(D.n_petals >= 4) & (D.n_petals <= 6)].sort_values(["n_elements", "n_edges", "n_anchor"], ascending=[False, False, True])
r = cand.iloc[0]
lvl = ((D.n_elements >= r.n_elements) & (D.n_edges >= r.n_edges)).mean()
print(f"pick conf {r.conf} (traj {r.traj}, frame {r.frame + 1}): {r.n_elements}/{len(eb)} element beads, {r.n_edges}/{len(E)} edges, "
      f"{r.n_petals} petals, {r.n_anchor} anchors; conformations reaching this level: {lvl*100:.2f}%")
print("distribution of hub-component elements in the best flower (median, 90th pct):", D.n_elements.median(), D.n_elements.quantile(0.9),
      "; edges:", D.n_edges.median(), D.n_edges.quantile(0.9))
print("fraction with >=5 elements:", (D.n_elements >= 5).mean(), ">=3 edges:", (D.n_edges >= 3).mean())
z = np.load(files[int(r.traj)]); k = int(r.frame)
T = z["X"][k, N:]; on = np.where(z["ON"][k])[0]; lab = np.full(len(T), -1); lab[on] = DBSCAN(eps=1.6, min_samples=4).fit_predict(T[on])
np.savez(O / "hub_flow_pick.npz", X=z["X"][k], LT=lab, hc=int(r.cluster), conf=int(r.conf), traj=int(r.traj), frame=k,
         level_frac=lvl, n_elements=int(r.n_elements), n_edges=int(r.n_edges))
