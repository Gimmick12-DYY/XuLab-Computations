# M6 HELD-OUT scoring, single run (SPEC.md Addenda 1-2, sha 596a05b2...; candidate chosen from grid6/spec_summary.tsv sha aa0b86ef...).
# Test lists = the 24 held-out TFs (split_v1); own/off-target KDs = held-out KDs only (dev KDs not used). 1,000 random sets, seed 20261109.
# Rows: win100k raw @1000 (named candidate), ABC-linked raw sum @1000, controls at +-100 kb (element count, pooled other-TF CPM, ATAC CPM),
# descriptive borrowing rows win100k and ABC-sum with reliability alpha_t over the kNN5 graph (net1b definitions). Built from net1b.py.
# M6 net1b = net1 with the three declared fixes (NET1B_DECLARE.md): r_SB floor 0.05 and alpha_t cap 0.7; dense-W non-partners = W < 0.5,
# with their share of smoothing mass recorded; 5 kNN5 double-edge-swap draws. net1 (job 8700032) was cancelled before any output was read.
# M6 net1 (SPEC_NETWORK.md, development TFs only): borrowing strength across TFs over the Regulome network. Gene score S(t) = win100k raw sum
# (and ABC-linked raw sum), each TF's vector scaled to mean 1 over genes; S_sm(t) = (1-alpha_t) S(t) + alpha_t * sum_u W(t,u) S(u) / sum_u W(t,u).
# W (binding data only): depth-matched library-stratified Pearson percentile (r1 v2 pearson_domain_pct) x (1 + support), support = 1 if Boltz trimmed
# or AF3 strong, CAP-SELEX called or STRING physical present, 0.5 if Boltz/AF3 weak, else 0. Graphs: dense W over all other panel TFs, and kNN5
# (each TF's 5 highest-W neighbours). Comparators at the same alpha: uniform pooling (W = 1 over all others), dense weight permutation (2 draws),
# degree-preserving double-edge swap of the kNN5 graph (3 draws), alpha = 0. alpha: fixed 0.15/0.3/0.5/0.7 and reliability alpha_t =
# s2/(s2 + tau2), s2 = 1/r_SB - 1 from a split-half (cells) correlation of the TF's own win100k score, tau2 = median s2 over the panel.
# Scoring as grid6 (1,000 expression-matched random sets) on dev KDs; per list: own z, mean z over kNN5 partners with a dev KD (real graph),
# mean z over non-partners, specificity = own - mean of all other dev KDs, own rank. Writes net1/per_tf.tsv, net1/graph.tsv, net1/alpha_t.tsv.
# M6 grid 6 (development TFs only, SPEC.md rules, last round before the stopping rule): 1,000 random sets per list (seed fixed); raw sums at
# +-100/150/200 kb, exp decays L = 100/150/200 kb (elements within 600 kb), win100k residual vs pool, ABC-linked raw sum reference; element-count,
# pooled other-TF and ATAC controls at every matching window / decay; list sizes 500 and 1,000.
# M6 grid 5 (development TFs only, SPEC.md rules): element-level normalisation of binding against the pooled panel signal (ratio, log-ratio, residual),
# summed over ABC links / 50 kb / 100 kb windows; all three controls (element count, pooled other-TF binding, ATAC) per linking; own-KD top-5% logOR added.
# M6 grid 4 (development TFs only): tune TSS window / distance decay and list size against ceiling-weighted specificity (grid3 metric).
# Controls kept for every window: element count (no binding) and pooled other-TF binding. Reference: ABC-linked sum. Grid 3 header follows.
# M6 grid 3 (development TFs only): TF-specificity metric. Each dev TF list is scored under its own KD and under every other dev KD;
# specificity = own z minus mean off-KD z (same-panel KDs excluded). Variants = grid 2 + window element-count and pooled-other-TF controls.
# M6 grid 1: which TF->gene scoring rule tracks the HEK293T knockdown response best? Development benchmark (Perturb-seq used for tuning).
# Variants differ only in how each TF's raw binding (Gate 1 cellmats, CPM per active element) is turned into a gene score over ABC-linked elements
# (m2_abc EnhancerPredictionsFull_thr0.02). Each TF's top-N genes are scored with the B6 effect (b3b4 score(): mean |R| vs 1,000 expression-matched
# random sets, / SD) and the top-5% responder Fisher OR, on both arms of kd_response/out_v2. Writes grid1/per_tf.tsv, grid1/pooled.tsv.
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; G1 = f"{X}/regulatome/gate1/cellmats"; O = f"{X}/regulatome/m6_improve/heldout_v1"; os.makedirs(O, exist_ok=True)
SEED = 20261109; NR = 1000; NB = 10000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = E.cls.isin(["promoter", "distal_active", "bivalent"]).values; Ea = E[act].reset_index(drop=True)
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "distance", "isSelfPromoter"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int)
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float); atac_cpm = atac / atac.sum() * 1e6
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
A["sq"] = np.sqrt(A["ABC.Score"]); A["dk"] = np.exp(-A.distance.abs() / 5e4); prom = A.isSelfPromoter.astype(str).str.lower().eq("true")

Sp = pd.read_csv(f"{X}/regulatome/m6_improve/split_v1.tsv", sep="\t"); DEV = set(Sp.tf[Sp.set == "heldout"])  # NOTE: DEV holds the HELD-OUT TFs in this script
tfs_dev = [t for t in tfs if panel(t) in DEV]; print("dev TFs", len(tfs_dev), flush=True)
def cpm_of(t, kind):
    y = np.asarray(sp.load_npz(f"{G1}/{t}.{kind}.npz").sum(0)).ravel(); return y / max(y.sum(), 1e-12) * 1e6
RAW = {t: cpm_of(t, "raw") for t in tfs}; TOT = np.sum([RAW[t] for t in tfs], axis=0)
# distance windows (no ABC): elements with midpoint within W of the gene TSS (genes and TSS from the ABC table)
Gt = A.drop_duplicates("TargetGene")[["chr", "TargetGene"]].copy(); tss = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["TargetGene", "TargetGeneTSS"]).drop_duplicates("TargetGene").set_index("TargetGene").TargetGeneTSS
Gt["tss"] = Gt.TargetGene.map(tss).values; mid = ((Ea.start + Ea.end) // 2).values
WIN = {}
for W in (100000,):
    gi_, ei_ = [], []
    for ch, G in Gt.groupby("chr"):
        ix = np.flatnonzero(Ea.chrom.values == ch); o = np.argsort(mid[ix]); ix = ix[o]; m = mid[ix]
        lo = np.searchsorted(m, G.tss.values - W); hi = np.searchsorted(m, G.tss.values + W, side="right")
        for g, a_, b_ in zip(G.TargetGene.values, lo, hi): gi_ += [g] * (b_ - a_); ei_ += list(ix[a_:b_])
    WIN[W] = (np.array(gi_), np.array(ei_, int))
nlink = A.groupby("TargetGene").size().astype(float)

NT = len(tfs); PT = [panel(t) for t in tfs]
R1 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t", keep_default_na=False, na_values=[""])
R3 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2_final_v3/r1_pairs_frozen_v3.tsv", sep="\t", keep_default_na=False, na_values=[""])
key = lambda a, b: tuple(sorted((panel(a), panel(b))))
pct = {key(a, b): v for a, b, v in zip(R1.a, R1.b, R1.pearson_domain_pct)}
def sup(r):
    st = [str(r.boltz_trimmed_tier), str(r.af3_tier)]
    if "strong" in st or str(r.capselex).startswith("called") or pd.notna(r.string_physical): return 1.0
    return 0.5 if "weak" in st else 0.0
spt = {key(r.a, r.b): sup(r) for r in R3.itertuples()}
P_ = sorted(set(PT)); ix_ = {p: i for i, p in enumerate(P_)}; n = len(P_)
W = np.zeros((n, n))
for a in P_:
    for b in P_:
        if a != b: W[ix_[a], ix_[b]] = np.nan_to_num(pct.get(key(a, b), 0.0)) * (1 + spt.get(key(a, b), 0.0))
K = 5; knn = np.zeros_like(W)
for i in range(n): j = np.argsort(-W[i])[:K]; knn[i, j] = W[i, j]
rows_g = [dict(tf=P_[i], partners=",".join(P_[j] for j in np.flatnonzero(knn[i])), weights=",".join(f"{W[i, j]:.3f}" for j in np.flatnonzero(knn[i]))) for i in range(n)]
os.makedirs(O, exist_ok=True); pd.DataFrame(rows_g).to_csv(f"{O}/graph.tsv", sep="\t", index=False)
PARTNER = {P_[i]: set(P_[j] for j in np.flatnonzero(knn[i])) for i in range(n)}
DENSE_NONP = {P_[i]: set(P_[j] for j in range(n) if j != i and W[i, j] < 0.5) for i in range(n)}
DENSE_NONP_MASS = {P_[i]: float(W[i][[j for j in range(n) if j != i and W[i, j] < 0.5]].sum() / max(W[i].sum(), 1e-12)) for i in range(n)}
pd.DataFrame(dict(tf=P_, dense_nonpartner_n=[len(DENSE_NONP[p]) for p in P_], dense_nonpartner_mass=[DENSE_NONP_MASS[p] for p in P_])).to_csv(f"{O}/dense_nonpartner_mass.tsv", sep="\t", index=False, float_format="%.4f")
rng_g = np.random.default_rng(20261109)
def perm_dense(Wm):
    iu = np.triu_indices(n, 1); v = np.maximum(Wm[iu], Wm.T[iu]); v = rng_g.permutation(v); M = np.zeros_like(Wm); M[iu] = v; return M + M.T
def swap_knn(Am):
    E = {tuple(sorted((i, j))) for i, j in zip(*np.nonzero(Am))}; E = list(E); w = rng_g.permutation([max(W[i, j], W[j, i]) for i, j in E]); E = [list(e) for e in E]; S_ = {tuple(e) for e in E}
    for _ in range(20 * len(E)):
        x, y = rng_g.integers(0, len(E), 2)
        if x == y: continue
        a, b = E[x]; c, d = E[y]
        if rng_g.random() < .5: c, d = d, c
        if len({a, b, c, d}) < 4 or tuple(sorted((a, d))) in S_ or tuple(sorted((c, b))) in S_: continue
        S_ -= {tuple(sorted((a, b))), tuple(sorted((c, d)))}; E[x] = [a, d]; E[y] = [c, b]; S_ |= {tuple(sorted((a, d))), tuple(sorted((c, b)))}
    M = np.zeros_like(Am)
    for (i, j), ww in zip(E, w): M[i, j] = M[j, i] = ww
    return M
GRAPHS = {"graph_dense": W, "graph_knn5": knn, "uniform": 1.0 - np.eye(n)}
for k in range(2): GRAPHS[f"shufdense{k}"] = perm_dense(W)
for k in range(5): GRAPHS[f"shufknn{k}"] = swap_knn(knn)
# gene scores (all panel TFs; binding only), scaled to mean 1; panel-level average for HP1 paralogs
gg, ee = WIN[100000]
def base_scores(t):
    c = RAW[t]; return {"win100k": pd.Series(c[ee]).groupby(gg).sum(), "abc": pd.Series(c[A.e.values]).groupby(A.TargetGene.values).sum()}
B0 = {t: base_scores(t) for t in tfs}
GEN = sorted(set().union(*[set(B0[t]["win100k"].index) | set(B0[t]["abc"].index) for t in tfs]))
SM = {}
for b in ("win100k", "abc"):
    M = np.zeros((n, len(GEN)))
    for p in P_:
        v = np.mean([B0[t][b].reindex(GEN).fillna(0).values for t in tfs if panel(t) == p], axis=0); M[ix_[p]] = v / max(v.mean(), 1e-12)
    SM[b] = M
# reliability alpha_t from split-half binding (cells), win100k score
rngs = np.random.default_rng(20261109); s2 = {}
for t in tfs:
    X_ = sp.load_npz(f"{G1}/{t}.raw.npz").tocsr(); idx = rngs.permutation(X_.shape[0]); h = [np.asarray(X_[idx[k::2]].sum(0)).ravel() for k in (0, 1)]
    v = [np.log1p(pd.Series(x[ee]).groupby(gg).sum().reindex(GEN).fillna(0).values) for x in h]; r = np.corrcoef(*v)[0, 1]; rsb = max(2 * r / (1 + r), 0.05); s2.setdefault(panel(t), []).append(1 / rsb - 1)
s2 = {p: float(np.mean(v)) for p, v in s2.items()}; tau2 = float(np.median(list(s2.values()))); ALPHA_T = {p: min(s2[p] / (s2[p] + tau2), 0.7) for p in P_}; print("alpha_t at cap 0.7:", sum(v >= 0.7 for v in ALPHA_T.values()), "of", len(ALPHA_T), "quantiles", np.round(np.quantile(list(ALPHA_T.values()), [0, .25, .5, .75, 1]), 3), flush=True)
pd.DataFrame(dict(tf=P_, s2=[s2[p] for p in P_], alpha_t=[ALPHA_T[p] for p in P_])).assign(tau2=tau2).to_csv(f"{O}/alpha_t.tsv", sep="\t", index=False, float_format="%.4f")
print("tau2", round(tau2, 4), "alpha_t median", round(np.median(list(ALPHA_T.values())), 3), flush=True)
gg1, ee1 = WIN[100000]
def ctrl_lists(p):
    ts = [t for t in tfs if panel(t) == p]; out = {}
    for nm, vec in (("n_elem", lambda t: np.ones_like(RAW[t])), ("pool", lambda t: (TOT - RAW[t]) / (NT - 1)), ("atac", lambda t: atac_cpm)):
        v = np.mean([pd.Series(vec(t)[ee1]).groupby(gg1).sum().reindex(GEN).fillna(0).values for t in ts], axis=0); out[f"win100k_CTRL_{nm}@1000"] = v
    return out
SC = {}
for p in sorted(set(P_) & DEV):
    i = ix_[p]; L = {}
    for b, M in SM.items():
        s0 = M[i]; w = GRAPHS["graph_knn5"][i]; nb = (w @ M) / max(w.sum(), 1e-12)
        L[f"{'win100k' if b == 'win100k' else 'abc'}_raw@1000"] = s0; L[f"{'win100k' if b == 'win100k' else 'abc'}_borrow_knn5_arel@1000"] = (1 - ALPHA_T[p]) * s0 + ALPHA_T[p] * nb
    L.update(ctrl_lists(p))
    SC[p] = {k: [GEN[j] for j in np.argsort(-v, kind="stable")[:1000] if v[j] > 0] for k, v in L.items()}
print("lists per TF", len(next(iter(SC.values()))), flush=True)
def load(a):  # b3b4.py load() (no replicate part)
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_]); P_ = np.nan_to_num(P_); Rr = P_ - P_.mean(0); Rr = Rr - Rr.mean(1, keepdims=True)
    return dict(g=g, gi={x: k for k, x in enumerate(g)}, R=Rr, tfs=tf_, mean=d["ntc_mean"][uni].astype(float))
def bstat(e, rng):
    e = np.asarray(e, float); n = len(e); bs = e[rng.integers(0, n, (NB, n))].mean(1); pos = int((e > 0).sum()); nz = int((e != 0).sum())
    return dict(n=n, mean=round(e.mean(), 3), ci_lo=round(np.percentile(bs, 2.5), 3), ci_hi=round(np.percentile(bs, 97.5), 3), n_pos=pos, sign_p=(float(f"{stats.binomtest(pos, nz, .5).pvalue:.3g}") if nz else 1.0))
rows = []
for a in ARMS:
    D = load(a); rng = np.random.default_rng(SEED + (a == "arm2")); g, gi, Rr, tf_ = D["g"], D["gi"], D["R"], D["tfs"]
    pts = sorted({panel(t) for t in tf_} & DEV); Rp = np.vstack([Rr[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); Ap = np.abs(Rp).astype(np.float32); pi = {p: k for k, p in enumerate(pts)}
    bins = pd.qcut(D["mean"], 20, labels=False, duplicates="drop"); BG = [np.flatnonzero(bins == b) for b in range(bins.max() + 1)]
    for p in pts:
        if p not in SC: continue
        for v, genes in SC[p].items():
            ix = np.array(sorted({gi[x] for x in genes if x in gi}), int)
            if len(ix) < 50: continue
            cnt = np.bincount(bins[ix], minlength=len(BG)); rs = np.hstack([BG[b][rng.integers(0, len(BG[b]), (NR, c))] for b, c in enumerate(cnt) if c])
            obs = Ap[:, ix].mean(1); rm = np.stack([Ap[k][rs].mean(1) for k in range(len(pts))]); z = (obs - rm.mean(1)) / rm.std(1)
            own = z[pi[p]]; off = np.delete(z, pi[p]); part = [pi[q] for q in PARTNER.get(p, ()) if q in pi]; nonp = [k for k in range(len(pts)) if k != pi[p] and k not in part]; dnonp = [pi[q] for q in DENSE_NONP.get(p, ()) if q in pi and q != p]; ab = Ap[pi[p]]; ng = len(ab); resp = np.zeros(ng, bool); resp[np.argsort(-ab, kind="stable")[:int(round(ng * .05))]] = True; inl = np.zeros(ng, bool); inl[ix] = True
            t2 = [[int((inl & resp).sum()), int((~inl & resp).sum())], [int((inl & ~resp).sum()), int((~inl & ~resp).sum())]]; lor = np.log((t2[0][0] + .5) * (t2[1][1] + .5) / ((t2[0][1] + .5) * (t2[1][0] + .5)))
            rows.append(dict(arm=a, tf=p, variant=v, n=len(ix), z_own=own, z_off_mean=off.mean(), specificity=own - off.mean(), own_rank=int((z > own).sum() + 1), n_kd=len(pts), logor5=lor, z_partner=float(z[part].mean()) if part else np.nan, n_partner=len(part), z_nonpartner=float(z[nonp].mean()), z_dense_nonpartner=float(z[dnonp].mean()) if dnonp else np.nan))
    print(a, "done", flush=True)
P = pd.DataFrame(rows); P.to_csv(f"{O}/per_tf.tsv", sep="\t", index=False, float_format="%.4g")
S = []; rng = np.random.default_rng(SEED + 7)
for a, G in P.groupby("arm"):
    for col in ("z_own", "z_off_mean", "specificity"):
        W = G.pivot(index="tf", columns="variant", values=col).dropna()
        for v in W.columns:
            b = bstat(W[v], rng); d = bstat(W[v] - W["abc_raw@1000"], rng)
            S.append(dict(arm=a, metric=col, variant=v, n_tfs=b["n"], mean=b["mean"], ci=f"[{b['ci_lo']}, {b['ci_hi']}]", vs_abc_linked=d["mean"], vs_abc_linked_ci=f"[{d['ci_lo']}, {d['ci_hi']}]", npos_vs_abc=d["n_pos"]))
    R1 = G.pivot(index="tf", columns="variant", values="own_rank"); S += [dict(arm=a, metric="median_own_rank", variant=v, n_tfs=int(R1[v].notna().sum()), mean=float(R1[v].median()), ci=f"top10%: {round((R1[v] <= max(1, int(.1 * G.n_kd.iloc[0]))).mean(), 3)}") for v in R1.columns]
S = pd.DataFrame(S); S.to_csv(f"{O}/pooled.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
for m in ("specificity", "z_own", "z_off_mean", "median_own_rank"): print(S[S.metric == m].sort_values(["arm", "mean"], ascending=[True, m == "median_own_rank"]).to_string(index=False))
