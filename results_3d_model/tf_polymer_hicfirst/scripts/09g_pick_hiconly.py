"""Hi-C-only arm (no TF attraction): there are no TF clusters, so the 'hub' is defined spatially. Per conformation, the
browser hub elements are linked when < 2 sigma apart (single linkage); the largest linked group is the Hi-C hub. Anchors
= beads within 2 sigma of its elements; petals = 10-100 bead gaps between consecutive anchors (as for flowers).
Selected = most elements, then most shared edges inside, with 4-6 petals; typical = median elements and edges.
The pick files store the hub-element positions in place of TF positions (LT = 0) and nocore = 1, so 10f/11f draw no TF core.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--sim", required=True); a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_hub_flow"; I = O / "inputs"
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t"); N = len(B); L0 = int(B.start[0])
E = pd.read_csv(I / "hub_chr16_edges.tsv", sep="\t")
G = pd.read_csv(I / "genes_chr16_29.7-31.6.tsv", sep="\t"); G["tss"] = np.where(G.strand == "+", G.txStart, G.txEnd)
def parse(r): c, s = r.split(":"); s, e = map(int, s.split("-")); return s, e
els = sorted({r for c in ("region1", "region2") for r in E[c]}, key=lambda r: parse(r)[0]); rows = []
for r in els:
    s, e = parse(r); d = np.minimum(np.abs(G.tss - s), np.abs(G.tss - e)); d[(G.tss >= s) & (G.tss <= e)] = 0
    g = sorted(set(G.loc[d <= 2000, "name"])); rows.append(dict(element=r, bead=int(((s + e) // 2 - L0) // 2000), label="/".join(g) if g else f"{(s + e) / 2e6:.3f} Mb"))
EL = pd.DataFrame(rows); EL.to_csv(O / "hub_elements.tsv", sep="\t", index=False)
bo = dict(zip(EL.element, EL.bead)); E["b1"] = E.region1.map(bo); E["b2"] = E.region2.map(bo); E.to_csv(O / "hub_edges_beads.tsv", sep="\t", index=False)
KM = pd.read_csv(Bd / "inputs/restraints.tsv", sep="\t").kind.isin(["loop", "red_edge"]).values
hb = EL.bead.values; files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz"))); res = []
for t, f in enumerate(files):
    z = np.load(f); X = z["X"]
    for k in range(len(X)):
        P = X[k, :N].astype(float); Q = P[hb]
        pr = cKDTree(Q).query_pairs(2.0, output_type="ndarray")
        _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(hb), len(hb))), directed=False) if len(pr) else (0, np.arange(len(hb)))
        big = np.bincount(lab).argmax(); mem = hb[lab == big]
        anc = np.unique(np.concatenate(cKDTree(P).query_ball_point(P[mem], 2.0)).astype(int)); sa = set(anc)
        ne = len(set(hb) & sa); nedge = int(sum((x in sa) and (y in sa) for x, y in zip(E.b1, E.b2)))
        g = np.diff(anc); res.append((t, k, ne, nedge, int(((g >= 10) & (g <= 100)).sum()), len(anc), len(mem)))
D = pd.DataFrame(res, columns=["traj", "frame", "n_elements", "n_edges", "n_petals", "n_anchor", "n_linked"]); D["conf"] = D.traj * 50 + D.frame + 1
D.to_csv(O / "hub_flow_table.tsv", sep="\t", index=False)
cand = D[(D.n_petals >= 4) & (D.n_petals <= 6)].sort_values(["n_elements", "n_edges", "n_anchor"], ascending=[False, False, True])
print("Hi-C hub elements (median, 90th):", D.n_elements.median(), D.n_elements.quantile(0.9), "; edges:", D.n_edges.median(), D.n_edges.quantile(0.9))
me, md = int(D.n_elements.median()), int(D.n_edges.median()); typ = cand[(cand.n_elements == me) & (cand.n_edges == md)]
typ = typ.iloc[[np.argmin(np.abs(typ.n_anchor - typ.n_anchor.median()))]] if len(typ) else cand.iloc[[len(cand) // 2]]
for tag, r in (("selected", cand.iloc[0]), ("typical", typ.iloc[0])):
    lvl = ((D.n_elements >= r.n_elements) & (D.n_edges >= r.n_edges)).mean()
    z = np.load(files[int(r.traj)]); k = int(r.frame); P = z["X"][k, :N]; Q = P[hb]
    pr = cKDTree(Q).query_pairs(2.0, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(hb), len(hb))), directed=False)
    mem = hb[lab == np.bincount(lab).argmax()]
    np.savez(O / f"hub_flow_pick_{tag}.npz", X=np.vstack([P, P[mem]]), LT=np.zeros(len(mem), int), hc=0, conf=int(r.conf), traj=int(r.traj),
             frame=k, level_frac=lvl, n_elements=int(r.n_elements), n_edges=int(r.n_edges), loop_on=z["loop_on"][KM], nocore=1, nchain=N)
    print(f"{tag}: conf {r.conf}: {r.n_elements}/23 elements, {r.n_edges}/24 edges, {r.n_petals} petals, {r.n_anchor} anchors; at or above: {lvl*100:.2f}%")
