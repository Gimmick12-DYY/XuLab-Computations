"""Step 3: validate the TF-binding polymer against held-out data and pick the attraction strength.

Per strength (sim/eps*/traj*.npz):
  contacts      bead pairs closer than --rc (2 sigma) in each conformation; 10 kb map = sum of the 5x5 bead blocks / n_conf
  vs Hi-C       10 kb ICE map, kept bins, pairs >= 2 bins apart: Spearman on raw and on distance-normalised O/E
                (each diagonal divided by its mean over kept pairs), P(s)
  TF clusters   DBSCAN on TF particles (eps --db-eps, min_samples --db-min); a bead belongs to the cluster of the nearest
                clustered TF within the attraction cut-off (1.8 sigma)
The strength with the best O/E Spearman is the one reported; all three are summarised.
Validation sets (chosen strength; also tabulated for the others):
  loops         Peakachu 0.7 (10 kb bin pairs) vs random bin pairs at the same distance
  Cicero        hub-component edges, shared clique edges, step-1 pairs (bead pairs) vs
                (a) random bead pairs at the same separation and (b) bead pairs matched for separation (+-2 beads, widened
                to +-5/+-10 if empty) and for the strength class of each end
                statistic = median over pairs of observed / mean matched frequency; one-sided p from 1000 replicates
                in which every pair is replaced by one random matched pair
  rosette       hub-core pairs co-clustered vs class- and separation-matched (+-10 beads) bound bead pairs; per conformation,
                max number of the 9 hub cores in one TF cluster vs 1000 random class-matched sets of bound beads drawn
                within +-50 beads of each core
"""
import argparse, glob, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--rc", type=float, default=2.0)
ap.add_argument("--db-eps", type=float, default=1.6)
ap.add_argument("--db-min", type=int, default=3)
ap.add_argument("--db-eps2", type=float, default=1.25, help="stricter DBSCAN for a sensitivity check of the rosette metric")
ap.add_argument("--db-min2", type=int, default=4)
ap.add_argument("--n-rep", type=int, default=1000)
a = ap.parse_args()
Bd = Path(a.base); I = Bd / "inputs"; A = Bd / "analysis"; A.mkdir(exist_ok=True)
rng = np.random.default_rng(1)
B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B); cls = B.cls.values
H = np.load(I / "hic_10kb.npz"); ice = H["ice"]; keep = H["keep"]; nbin = ice.shape[0]
CP = pd.read_csv(I / "pairs_cicero.tsv", sep="\t"); LP = pd.read_csv(I / "pairs_loops.tsv", sep="\t")
hub = B[B.hub_label != ""]; hub_b = hub.bead.values; hub_lab = hub.hub_label.values
bound = np.where(cls > 0)[0]

def load(d):
    X = []
    for f in sorted(glob.glob(str(d / "traj*.npz"))):
        z = np.load(f); X.append(z["X"])
    return np.concatenate(X), len(X)

def per_conf(X):
    """bead contact counts, bead cluster labels (n_conf x N, -1 = none), TF cluster labels"""
    nc = len(X); C = np.zeros((N, N), np.int32); NT = X.shape[1] - N
    L = np.full((nc, N), -1, np.int16); LT = np.full((nc, NT), -1, np.int16); L2 = L.copy(); LT2 = LT.copy()
    for k in range(nc):
        P = X[k, :N].astype(float); T = X[k, N:].astype(float)
        pr = cKDTree(P).query_pairs(a.rc, output_type="ndarray")
        np.add.at(C, (pr[:, 0], pr[:, 1]), 1)
        for Lb, Lt, e_, m_ in ((L, LT, a.db_eps, a.db_min), (L2, LT2, a.db_eps2, a.db_min2)):
            lab = DBSCAN(eps=e_, min_samples=m_).fit_predict(T); Lt[k] = lab
            if (lab >= 0).any():
                ok = np.where(lab >= 0)[0]
                dd, ii = cKDTree(T[ok]).query(P, distance_upper_bound=1.8)
                m = np.isfinite(dd); Lb[k, m] = lab[ok[ii[m]]]
    C = C + C.T
    return C, L, LT, L2, LT2

def to10(Cb, nc):
    M = Cb.reshape(nbin, 5, nbin, 5).sum((1, 3)).astype(float) / nc
    return M

def oe(M, mask):
    E = np.full_like(M, np.nan)
    for d in range(nbin):
        i = np.arange(nbin - d); j = i + d
        ok = mask[i] & mask[j] & np.isfinite(M[i, j])
        if ok.sum() == 0: continue
        mu = M[i[ok], j[ok]].mean()
        if mu > 0: E[i, j] = M[i, j] / mu; E[j, i] = E[i, j]
    return E

iu = np.triu_indices(nbin, 2)
pm = keep[iu[0]] & keep[iu[1]] & np.isfinite(ice[iu])
def compare(S):
    r_raw = stats.spearmanr(S[iu][pm], ice[iu][pm])[0]
    So, Ho = oe(S, keep), oe(ice, keep)
    ok = pm & np.isfinite(So[iu]) & np.isfinite(Ho[iu])
    r_oe = stats.spearmanr(So[iu][ok], Ho[iu][ok])[0]
    # O/E restricted to <= 500 kb (where most Hi-C signal sits)
    ok5 = ok & ((iu[1] - iu[0]) <= 50)
    r_oe5 = stats.spearmanr(So[iu][ok5], Ho[iu][ok5])[0]
    return r_raw, r_oe, r_oe5

def ps(M, mask):
    v = []
    for d in range(1, nbin):
        i = np.arange(nbin - d); j = i + d; ok = mask[i] & mask[j] & np.isfinite(M[i, j])
        v.append(M[i[ok], j[ok]].mean() if ok.any() else np.nan)
    return np.array(v)

def matched_test(freq, pairs, cand_fn, n_rep):
    """pairs: list of (i, j); cand_fn(i, j) -> candidate pairs array (k x 2). Returns per-pair table and summary."""
    obs, exp, cands = [], [], []
    for i, j in pairs:
        c = cand_fn(i, j); cands.append(c)
        obs.append(freq[i, j]); exp.append(freq[c[:, 0], c[:, 1]].mean() if len(c) else np.nan)
    obs, exp = np.array(obs), np.array(exp)
    ok = np.isfinite(exp) & (exp > 0)
    ratio = np.where(ok, obs / np.where(ok, exp, 1), np.nan)
    mean_obs = obs[ok].mean(); rep = np.empty(n_rep)
    for r in range(n_rep):
        rep[r] = np.mean([freq[c[q, 0], c[q, 1]] for c, q in ((cands[m], rng.integers(len(cands[m]))) for m in np.where(ok)[0])])
    p = (1 + (rep >= mean_obs).sum()) / (n_rep + 1)
    # agg_ratio = mean observed / mean matched expectation (robust when most single pairs have zero contacts)
    return obs, exp, ratio, dict(n=int(ok.sum()), mean_obs=mean_obs, mean_exp=float(exp[ok].mean()),
                                 agg_ratio=float(mean_obs / exp[ok].mean()), median_ratio=float(np.nanmedian(ratio)),
                                 frac_obs_gt_exp=float(np.mean(obs[ok] > exp[ok])), p=p)

def cand_sep(i, j, tol=0, excl=set()):
    s = j - i; out = []
    for ds in range(-tol, tol + 1):
        ss = s + ds
        if ss <= 0: continue
        ii = np.arange(N - ss); out.append(np.c_[ii, ii + ss])
    c = np.concatenate(out)
    if excl: c = c[[tuple(x) not in excl for x in c]]
    return c

sep_cls_cache = {}
def cand_cls(i, j, excl=set()):
    ci, cj = cls[i], cls[j]; s = j - i
    for tol in (2, 5, 10):
        key = (min(ci, cj), max(ci, cj), s, tol)
        if key not in sep_cls_cache:
            out = []
            for ss in range(max(1, s - tol), s + tol + 1):
                ii = np.arange(N - ss); jj = ii + ss
                m = ((cls[ii] == ci) & (cls[jj] == cj)) | ((cls[ii] == cj) & (cls[jj] == ci))
                out.append(np.c_[ii[m], jj[m]])
            sep_cls_cache[key] = np.concatenate(out)
        c = sep_cls_cache[key]
        c = c[[tuple(x) not in excl for x in c]] if excl else c
        if len(c) >= 5: return c
    return c

summ, ps_rows, val_rows, pair_rows = [], [], [], []
store = {}
eps_dirs = sorted(glob.glob(str(Bd / "sim" / "eps*")))
t0 = time.time()
for d in eps_dirs:
    d = Path(d); X, ntraj = load(d); nc = len(X)
    eps = float(d.name[3:])
    Cb, L, LT, L2, LT2 = per_conf(X)
    S = to10(Cb, nc)
    r_raw, r_oe, r_oe5 = compare(S)
    ntf_clu = [(LT[k] >= 0).sum() for k in range(nc)]
    ncl = [len(set(LT[k][LT[k] >= 0])) for k in range(nc)]
    bf = (L[:, bound] >= 0).mean()
    ncl2 = [len(set(LT2[k][LT2[k] >= 0])) for k in range(nc)]
    big = [np.bincount(LT[k][LT[k] >= 0]).max() / LT.shape[1] if (LT[k] >= 0).any() else 0 for k in range(nc)]
    summ.append(dict(eps_top=eps, n_traj=ntraj, n_conf=nc, spearman_raw=r_raw, spearman_oe=r_oe, spearman_oe_le500kb=r_oe5,
                     mean_tf_clusters=np.mean(ncl), frac_tf_in_clusters=np.mean(ntf_clu) / LT.shape[1],
                     frac_bound_beads_in_cluster=bf, frac_tf_in_largest_cluster=np.mean(big), mean_tf_clusters_strict=np.mean(ncl2)))
    print(summ[-1], f"{time.time()-t0:.0f}s", flush=True)
    store[eps] = (X, Cb, L, LT, L2, LT2, S, nc)
    np.save(A / f"sim_10kb_eps{eps:g}.npy", S)
    for k, (v, h) in enumerate(zip(ps(S, keep), ps(ice, keep))):
        ps_rows.append(dict(eps_top=eps, s_kb=(k + 1) * 10, p_sim=v, p_hic=h))
SM = pd.DataFrame(summ); best = float(SM.loc[SM.spearman_oe.idxmax(), "eps_top"]); SM["chosen"] = SM.eps_top == best
SM.to_csv(A / "sweep_summary.tsv", sep="\t", index=False); pd.DataFrame(ps_rows).to_csv(A / "ps.tsv", sep="\t", index=False)
print("chosen eps_top", best, flush=True)

for eps, (X, Cb, L, LT, L2, LT2, S, nc) in store.items():
    F = Cb / nc  # bead contact probability
    # loops at 10 kb
    lp = [(int(r.bin1), int(r.bin2)) for r in LP.itertuples() if r.bin2 - r.bin1 >= 2]
    ex = set(lp)
    def cand_bin(i, j):
        s = j - i; ii = np.arange(nbin - s); c = np.c_[ii, ii + s]
        return c[[tuple(x) not in ex for x in c]]
    o, e, rt, sm = matched_test(S, lp, cand_bin, a.n_rep)
    val_rows.append(dict(eps_top=eps, set="loops_peakachu07", null="random_same_distance", **sm))
    for (i, j), oo, ee, rr in zip(lp, o, e, rt):
        pair_rows.append(dict(eps_top=eps, set="loops_peakachu07", null="random_same_distance", i=i, j=j, obs=oo, exp=ee, ratio=rr))
    for st in ["hub_edges", "shared_edges", "cicero_step1"]:
        sub = CP[CP.set == st]; pr = [(int(r.b1), int(r.b2)) for r in sub.itertuples() if r.b2 - r.b1 >= 2]
        ex = set(pr)
        for nl, fn in [("random_same_distance", lambda i, j: cand_sep(i, j, 0, ex)), ("bound_matched_class_distance", lambda i, j: cand_cls(i, j, ex))]:
            o, e, rt, sm = matched_test(F, pr, fn, a.n_rep)
            val_rows.append(dict(eps_top=eps, set=st, null=nl, **sm)); print(eps, st, nl, sm, f"{time.time()-t0:.0f}s", flush=True)
            for (i, j), oo, ee, rr in zip(pr, o, e, rt):
                pair_rows.append(dict(eps_top=eps, set=st, null=nl, i=i, j=j, obs=oo, exp=ee, ratio=rr))
    # rosette: co-clustering of hub-core pairs (main DBSCAN; strict DBSCAN as sensitivity)
    hp = [(min(x, y), max(x, y)) for k, x in enumerate(hub_b) for y in hub_b[k + 1:]]
    exh = set(hp)
    def cand_hub(i, j):
        c = cand_cls(i, j, exh)
        c = c[(cls[c[:, 0]] > 0) & (cls[c[:, 1]] > 0)]
        return c[np.abs((c[:, 1] - c[:, 0]) - (j - i)) <= 10]
    def rosette(Lx, tag):
        class _Q:
            def __getitem__(self, ij):
                i, j = ij
                f = lambda x, y: float(np.mean((Lx[:, x] == Lx[:, y]) & (Lx[:, x] >= 0)))
                if np.ndim(i): return np.array([f(x, y) for x, y in zip(i, j)])
                return f(i, j)
        o, e, rt, sm = matched_test(_Q(), hp, cand_hub, 200)
        val_rows.append(dict(eps_top=eps, set=f"hub_core_pairs_cocluster{tag}", null="bound_matched_class_distance", **sm))
        for (i, j), oo, ee, rr in zip(hp, o, e, rt):
            pair_rows.append(dict(eps_top=eps, set=f"hub_core_pairs_cocluster{tag}", null="bound_matched_class_distance", i=i, j=j, obs=oo, exp=ee, ratio=rr))
        def maxin(idx):
            Lh = Lx[:, idx]
            eq = (Lh[:, :, None] == Lh[:, None, :]) & (Lh[:, :, None] >= 0)
            return eq.sum(2).max(1)
        mh = maxin(hub_b)
        nullm, null3 = [], []
        for r in range(a.n_rep if eps == best else 200):
            idx = []
            for b in hub_b:
                c = bound[(cls[bound] == cls[b]) & (np.abs(bound - b) <= 50) & ~np.isin(bound, hub_b)]
                idx.append(rng.choice(c))
            m = maxin(np.array(idx)); nullm.append(m.mean()); null3.append((m >= 3).mean())
        nullm, null3 = np.array(nullm), np.array(null3)
        val_rows.append(dict(eps_top=eps, set=f"hub_cores_max_in_one_cluster{tag}", null="random_class_matched_sets_pm50beads", n=nc,
                             mean_obs=float(mh.mean()), mean_exp=float(nullm.mean()), agg_ratio=float(mh.mean() / nullm.mean()),
                             frac_obs_gt_exp=float((mh >= 3).mean()), p=float((1 + (nullm >= mh.mean()).sum()) / (len(nullm) + 1)),
                             frac_conf_ge3=float((mh >= 3).mean()), frac_conf_ge5=float((mh >= 5).mean()), null_frac_conf_ge3=float(null3.mean())))
        print(eps, tag, val_rows[-2], val_rows[-1], flush=True)
        np.savez(A / f"rosette{tag}_eps{eps:g}.npz", mh=mh, null_mean=nullm, null_frac3=null3, hub_b=hub_b, hub_lab=hub_lab)
        return mh
    mh = rosette(L, "")
    mh2 = rosette(L2, "_strict")
    if eps == best:
        # representative: most hub cores in one strict TF cluster (the main DBSCAN merges nearly all TFs into one or two
        # condensates, so it does not separate conformations); ties -> most hub cores in one main cluster, then most hub edges in contact
        he = CP[CP.set == "hub_edges"]
        d_he = np.array([np.linalg.norm(X[:, he.b1.values] - X[:, he.b2.values], axis=2)]).squeeze(0)  # nc x n_edges
        score = mh2 * 10000 + mh * 100 + (d_he < a.rc).sum(1)
        k = int(np.argmax(score))
        rep = dict(conf=k, traj=k // 50, frame=k % 50, max_hub_in_cluster=int(mh[k]), max_hub_in_strict_cluster=int(mh2[k]), hub_edges_in_contact=int((d_he[k] < a.rc).sum()))
        pool = np.setdiff1d(np.arange(0, nc, 50) + rng.integers(50, size=max(1, nc // 50)), [k]); pool = pool[pool < nc]
        gal = rng.choice(pool, min(6, len(pool)), replace=False)
        # L/LT = strict cluster labels (used for the render); L_main/LT_main = main DBSCAN
        np.savez(A / "representative.npz", X=X[k], L=L2[k], LT=LT2[k], L_main=L[k], LT_main=LT[k], conf=k, gallery=gal, XG=X[gal],
                 LG=L2[gal], LTG=LT2[gal], mh_g=mh2[gal], mh_main_g=mh[gal])
        json.dump(rep, open(A / "representative.json", "w"))
        print("representative", rep, "gallery", gal.tolist(), "max-in-strict-cluster", mh2[gal].tolist(), flush=True)
        np.save(A / "contact_bead_best.npy", F.astype(np.float32))

V = pd.DataFrame(val_rows); V.to_csv(A / "validation.tsv", sep="\t", index=False)
pd.DataFrame(pair_rows).to_csv(A / "validation_pairs.tsv", sep="\t", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(SM); print(V)
