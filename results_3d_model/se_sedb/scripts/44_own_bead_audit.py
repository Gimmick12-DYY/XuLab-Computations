"""Audit of the bead mapping behind the SE hub numbers. hub_edges_all.tsv keeps, per edge, the first Cicero region seen in each 10 kb node, and the rosette kept the first of those per node; this recomputes every structure-level number with each partner's own 2 kb bead (midpoint of its region in edges_exact_se.tsv). SE end = the edge's own anchor region for SE edges (as 04_analyze_se.py). No new simulation. Usage: 44_own_bead_audit.py <trajectory dir>"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
R0 = Path(__file__).resolve().parents[1]; O = R0 / "own_bead_audit"; N = 910; L0 = 430000; mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000; se_b = tb((1248039 + 1254347) // 2)
X = np.concatenate([np.load(f)["X"][:, :N].astype(np.float32) for f in sorted(glob.glob(sys.argv[1] + "/traj*.npz"))]); print(X.shape)
T = pd.read_csv(R0 / "edges_exact_se.tsv", sep="\t"); T["bin"] = [mid(x) // 10000 for x in T.partner]; OWN = {str(b): tb(mid(x)) for b, x in zip(T.bin, T.partner)}; ANC = {str(b): tb(mid(x)) for b, x in zip(T.bin, T.anchor)}; CL = {str(b): c for b, c in zip(T.bin, T.cls)}; LAB = {str(b): l for b, l in zip(T.bin, T.label)}
H = pd.read_csv(R0 / "hub_edges_all.tsv", sep="\t"); H["h1"] = [tb(mid(x)) for x in H.reg_a]; H["h2"] = [tb(mid(x)) for x in H.reg_b]
H["o1"] = [ANC[str(b)] if a == "SE" else OWN[str(a)] for a, b in zip(H.a, H.b)]; H["o2"] = [ANC[str(a)] if b == "SE" else OWN[str(b)] for a, b in zip(H.a, H.b)]
print("beads differ, edges:", int(((H.h1 != H.o1) | (H.h2 != H.o2)).sum()), "of", len(H), "; partner-partner:", int((((H.h1 != H.o1) | (H.h2 != H.o2)) & (H.from_se == 0)).sum()), "; SE edges:", int((((H.h1 != H.o1) | (H.h2 != H.o2)) & (H.from_se == 1)).sum()))
print(H[(H.from_se == 1) & ((H.h1 != H.o1) | (H.h2 != H.o2))][["a", "b", "on_loop", "h1", "o1", "h2", "o2"]].to_string(index=False))
dist = lambda a, b: np.linalg.norm(X[:, a] - X[:, b], axis=2) < 2.0
Co, Ch = dist(H.o1.values, H.o2.values), dist(H.h1.values, H.h2.values); L = ((H.from_se == 1) & (H.on_loop == 1)).values
PS = pd.read_csv(R0 / "model/analysis/per_structure.tsv", sep="\t"); no, nh = Co[:, L].sum(1), Ch[:, L].sum(1); print("own = per_structure (title source) in", int((no == PS.n_loop_closed.values).sum()), "of", len(PS))
res = {}
for n_, v in (("per_structure", PS.n_loop_closed.values), ("own", no), ("hub_edges_all", nh)): res[n_] = dict(mean=round(float(v.mean()), 3), median=float(np.median(v)), max=int(v.max()), n_at_max=int((v == v.max()).sum()), top_pct=round(float((v == v.max()).mean() * 100), 3), n_all18=int((v == 18).sum())); print(n_, res[n_])
c = 9102; i = c - 1; print("#9102 closed loop-edges own", int(no[i]), "hub_edges_all", int(nh[i]), "all 135 own", int(Co[i].sum()), "hub_edges_all", int(Ch[i].sum()))
def far(r): return r.cls_b if r.a == "SE" else r.cls_a
H["far"] = [far(r) for r in H.itertuples()]; print("#9102 by far element, own", H[L & Co[i]].far.value_counts().to_dict(), "; hub_edges_all", H[L & Ch[i]].far.value_counts().to_dict())
# rosette node beads (first region per node, as in 21_se_rosette.py) vs own
Kb = {"SE": se_b}
for r in H.itertuples():
    for k_, b_ in ((str(r.a), r.h1), (str(r.b), r.h2)):
        if k_ not in Kb: Kb[k_] = int(b_)
OWNK = dict(OWN, SE=se_b)
def comp(P, beads):
    hb = np.array(sorted(set(beads))); pr = cKDTree(P[hb]).query_pairs(2.0, output_type="ndarray"); _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(hb), len(hb))), directed=False); return np.sort(hb[lab == lab[np.where(hb == se_b)[0][0]]])
M = pd.read_csv(R0 / "figure/conformation_metrics.tsv", sep="\t"); rows = []
for cf in M[M.n_loop_closed == M.n_loop_closed.max()].conf:
    P = X[cf - 1].astype(float); e = M[M.conf == cf].iloc[0]; row = dict(conf=int(cf), rg_pct=round(e.rg_pct, 1), crumple_pct=round(e.crumple_pct, 1), overlap=int(e.overlap))
    for n_, K in (("old", Kb), ("own", OWNK)):
        m_ = comp(P, list(K.values())); row["linked_" + n_] = len(m_) - 1; row["petals_" + n_] = sum((q - p) * 2 <= 200 for p, q in zip(m_[:-1], m_[1:]))
    rows.append(row)
S = pd.DataFrame(rows); S.to_csv(O / "max_structures.tsv", sep="\t", index=False); print(S.sort_values(["linked_own", "petals_own", "overlap"], ascending=[False, False, True]).to_string(index=False))
P = X[i].astype(float); memo, memn = comp(P, list(Kb.values())), comp(P, list(OWNK.values())); rev = {v: k for k, v in Kb.items()}; revn = {v: k for k, v in OWNK.items()}; print("in core, old map only:", sorted(LAB.get(rev[b]) for b in memo if rev[b] != "SE" and OWNK[rev[b]] not in memn), "; own map only:", sorted(LAB.get(revn[b]) for b in memn if revn[b] != "SE" and Kb[revn[b]] not in memo))
# ensemble closure against random pairs at the same separation, as 18_final_flower.py
rng = np.random.default_rng(37); good = np.arange(N); 
def null(a, b):
    s = abs(a - b); st = rng.integers(0, N - s, 200); return np.linalg.norm(X[::5][:, st] - X[::5][:, st + s], axis=2).__lt__(2.0).mean()
H["cl_own"], H["cl_old"] = Co.mean(0), Ch.mean(0); H["null_own"] = [null(a, b) for a, b in zip(H.o1, H.o2)]; H["null_old"] = [null(a, b) for a, b in zip(H.h1, H.h2)]; H.to_csv(O / "edges_own_vs_old.tsv", sep="\t", index=False)
G = H.groupby(["from_se", "on_loop"]).agg(n=("a", "size"), cl_old=("cl_old", "mean"), null_old=("null_old", "mean"), cl_own=("cl_own", "mean"), null_own=("null_own", "mean")); print((G * [1, 100, 100, 100, 100]).round(1).to_string())
# screen shift of the balls if moved to their own beads
Y0 = pd.read_csv(R0 / "figure_rosette_tf/rosette_nodes.tsv", sep="\t"); cen = P[Y0.bead].mean(0); Sc = 10.0
A = np.c_[(P[Y0.bead] - cen) * Sc, np.ones(len(Y0))]; cx, *_ = np.linalg.lstsq(A, Y0.px, rcond=None); cy, *_ = np.linalg.lstsq(A, Y0.py, rcond=None); print("projection fit residual px", round(float(np.abs(A @ cx - Y0.px).max()), 2), round(float(np.abs(A @ cy - Y0.py).max()), 2))
ppu = np.sqrt((cx[:3] ** 2).sum()); rows = []
for k_, ob in OWN.items():
    nb = Kb[k_]
    if nb == ob or nb not in set(Y0.bead): continue
    d3 = np.linalg.norm(P[nb] - P[ob]) * Sc; B_ = np.r_[(P[ob] - cen) * Sc, 1]; A_ = np.r_[(P[nb] - cen) * Sc, 1]; d2 = np.hypot(B_ @ cx - A_ @ cx, B_ @ cy - A_ @ cy) / ppu; rows.append(dict(element=LAB[k_], node_bead=nb, own_bead=ob, kb=abs(nb - ob) * 2, dist_3d_sigma=round(d3 / Sc, 2), shift_3d_ball_radii=round(d3 / 4.6, 2), shift_screen_ball_radii=round(d2 / 4.6, 2)))
B5 = pd.DataFrame(rows); B5.to_csv(O / "ball_shift.tsv", sep="\t", index=False); print(B5.to_string(index=False))
json.dump(res, open(O / "summary.json", "w"), indent=1)
