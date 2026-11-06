"""Hybrid step 9 (Mac): as v2 09_pick_hub_flow.py, on the hybrid (TF bridging + Hi-C loop restraints) trajectories.
Also: a petal is loop-anchored if a restrained loop that is ON in that trajectory has its anchors within +-3 beads of the
petal's two ends. Writes hub_flow_pick_selected.npz (most elements/edges, 4-6 petals) and hub_flow_pick_typical.npz
(median elements and edges among conformations with 4-6 petals, then most loop-anchored petals is NOT used: first match).

v2 notes: pick the conformation whose largest flower best matches the browser figure's chr16p11.2 hub component.

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

LR = pd.read_csv(Bd / "inputs/loop_restraints.tsv", sep="\t")
def loop_anch(petals, lon):
    lb1, lb2 = LR.b1.values[lon], LR.b2.values[lon]
    return [bool(np.any((np.abs(lb1 - p) <= 3) & (np.abs(lb2 - q) <= 3))) for p, q in petals]
res, files = [], sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))
for t, f in enumerate(files):
    z = np.load(f); X, ON, LON = z["X"], z["ON"], z["loop_on"]
    for k in range(len(X)):
        P = X[k, :N].astype(float); T = X[k, N:].astype(float); on = np.where(ON[k])[0]
        lab = np.full(len(T), -1)
        if len(on) >= 4: lab[on] = DBSCAN(eps=1.6, min_samples=4).fit_predict(T[on])
        tree = cKDTree(P); best = (0, 0, -1, 0, 0, 0)
        for c in range(lab.max() + 1):
            anc = np.unique(np.concatenate(tree.query_ball_point(T[lab == c], 2.0)).astype(int)); sa = set(anc)
            ne = len(eb & sa); nedge = int(sum((b1 in sa) and (b2 in sa) for b1, b2 in zip(E.b1, E.b2)))
            g = np.diff(anc); pm = (g >= 10) & (g <= 100); pet = int(pm.sum())
            if (ne, nedge) > best[:2]:
                la = int(sum(loop_anch(list(zip(anc[:-1][pm], anc[1:][pm])), LON)))
                best = (ne, nedge, c, pet, len(anc), la)
        res.append((t, k, *best))
D = pd.DataFrame(res, columns=["traj", "frame", "n_elements", "n_edges", "cluster", "n_petals", "n_anchor", "n_loop_petals"])
D["conf"] = D.traj * 50 + D.frame + 1
D.to_csv(O / "hub_flow_table.tsv", sep="\t", index=False)
cand = D[(D.n_petals >= 4) & (D.n_petals <= 6)].sort_values(["n_elements", "n_edges", "n_anchor"], ascending=[False, False, True])
print(f"petals loop-anchored (hub flower, all conformations): {D.n_loop_petals.sum() / max(1, D.n_petals.sum()) * 100:.1f}%")
print("distribution of hub-component elements in the best flower (median, 90th pct):", D.n_elements.median(), D.n_elements.quantile(0.9),
      "; edges:", D.n_edges.median(), D.n_edges.quantile(0.9), "; fraction >=5 elements", (D.n_elements >= 5).mean(), ">=3 edges", (D.n_edges >= 3).mean())
me, md = int(D.n_elements.median()), int(D.n_edges.median())
typ = cand[(cand.n_elements == me) & (cand.n_edges == md)]
typ = typ.iloc[[np.argmin(np.abs(typ.n_anchor - typ.n_anchor.median()))]] if len(typ) else cand.iloc[[len(cand) // 2]]
for tag, r in (("selected", cand.iloc[0]), ("typical", typ.iloc[0])):
    lvl = ((D.n_elements >= r.n_elements) & (D.n_edges >= r.n_edges)).mean()
    print(f"{tag}: conf {r.conf} (traj {r.traj}, frame {r.frame + 1}): {r.n_elements}/{len(eb)} elements, {r.n_edges}/{len(E)} edges, "
          f"{r.n_petals} petals ({r.n_loop_petals} loop-anchored), {r.n_anchor} anchors; conformations at or above this level: {lvl*100:.2f}%")
    z = np.load(files[int(r.traj)]); k = int(r.frame)
    T = z["X"][k, N:]; on = np.where(z["ON"][k])[0]; lab = np.full(len(T), -1); lab[on] = DBSCAN(eps=1.6, min_samples=4).fit_predict(T[on])
    np.savez(O / f"hub_flow_pick_{tag}.npz", X=z["X"][k], LT=lab, hc=int(r.cluster), conf=int(r.conf), traj=int(r.traj), frame=k,
             level_frac=lvl, n_elements=int(r.n_elements), n_edges=int(r.n_edges), loop_on=z["loop_on"])
