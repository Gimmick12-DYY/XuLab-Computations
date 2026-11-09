# M6 grid 1: which TF->gene scoring rule tracks the HEK293T knockdown response best? Development benchmark (Perturb-seq used for tuning).
# Variants differ only in how each TF's raw binding (Gate 1 cellmats, CPM per active element) is turned into a gene score over ABC-linked elements
# (m2_abc EnhancerPredictionsFull_thr0.02). Each TF's top-N genes are scored with the B6 effect (b3b4 score(): mean |R| vs 1,000 expression-matched
# random sets, / SD) and the top-5% responder Fisher OR, on both arms of kd_response/out_v2. Writes grid1/per_tf.tsv, grid1/pooled.tsv.
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; G1 = f"{X}/regulatome/gate1/cellmats"; O = f"{X}/regulatome/m6_improve/grid1"; os.makedirs(O, exist_ok=True)
SEED = 20261109; NR = 1000; NB = 10000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = E.cls.isin(["promoter", "distal_active", "bivalent"]).values; Ea = E[act].reset_index(drop=True)
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "distance", "isSelfPromoter"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int)
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float); atac_cpm = atac / atac.sum() * 1e6
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
A["sq"] = np.sqrt(A["ABC.Score"]); A["dk"] = np.exp(-A.distance.abs() / 5e4); prom = A.isSelfPromoter.astype(str).str.lower().eq("true")
def variants(t):
    y = np.asarray(sp.load_npz(f"{G1}/{t}.raw.npz").sum(0)).ravel(); cpm = y / np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float).sum() * 1e6
    c = cpm[A.e.values]; enr = np.maximum(np.log2((c + .5) / (atac_cpm[A.e.values] + .5)), 0)
    V = {"sum_cpm": c, "cpm_x_abc": c * A["ABC.Score"].values, "cpm_x_sqrtabc": c * A.sq.values, "cpm_x_dist50k": c * A.dk.values, "enrich_log2_vs_atac": enr, "enrich_x_abc": enr * A["ABC.Score"].values}
    out = {k: pd.Series(v).groupby(A.TargetGene.values).sum() for k, v in V.items()}
    out["max_cpm"] = pd.Series(c).groupby(A.TargetGene.values).max(); out["promoter_cpm"] = pd.Series(np.where(prom, c, 0)).groupby(A.TargetGene.values).sum()
    out["atac_x_abc"] = pd.Series(atac_cpm[A.e.values] * A["ABC.Score"].values).groupby(A.TargetGene.values).sum()
    return out
TOPN = {"sum_cpm": [500, 1000, 2000], "cpm_x_abc": [500, 1000, 2000]}
SC = {}
for t in tfs:
    V = variants(t); L = {}
    for k, s in V.items():
        s = s[s > 0].sort_values(ascending=False, kind="stable")
        for n in TOPN.get(k, [1000]): L[f"{k}@{n}"] = list(s.index[:n])
    SC[panel(t)] = L
print("variants", list(next(iter(SC.values())).keys()), flush=True)
def load(a):  # b3b4.py load() (no replicate part)
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_]); P_ = np.nan_to_num(P_); Rr = P_ - P_.mean(0); Rr = Rr - Rr.mean(1, keepdims=True)
    return dict(g=g, gi={x: k for k, x in enumerate(g)}, R=Rr, tfs=tf_, mean=d["ntc_mean"][uni].astype(float))
def bstat(e, rng):
    e = np.asarray(e, float); n = len(e); bs = e[rng.integers(0, n, (NB, n))].mean(1); pos = int((e > 0).sum()); nz = int((e != 0).sum())
    return dict(n=n, mean=round(e.mean(), 3), ci_lo=round(np.percentile(bs, 2.5), 3), ci_hi=round(np.percentile(bs, 97.5), 3), n_pos=pos, sign_p=float(f"{stats.binomtest(pos, nz, .5).pvalue:.3g}"))
rows = []
for a in ARMS:
    D = load(a); rng = np.random.default_rng(SEED + (a == "arm2")); g, gi, Rr, tf_ = D["g"], D["gi"], D["R"], D["tfs"]; pts = sorted({panel(t) for t in tf_}); Rp = np.vstack([Rr[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); Ap = np.abs(Rp)
    bins = pd.qcut(D["mean"], 20, labels=False, duplicates="drop"); BG = [np.flatnonzero(bins == b) for b in range(bins.max() + 1)]
    def score(ab, ix):
        cnt = np.bincount(bins[ix], minlength=len(BG)); rs = np.hstack([BG[b][rng.integers(0, len(BG[b]), (NR, c))] for b, c in enumerate(cnt) if c]); m = ab[rs].mean(1); return (ab[ix].mean() - m.mean()) / m.std()
    for i, p in enumerate(pts):
        if p not in SC: continue
        ab = Ap[i]; ng = len(ab); resp = np.zeros(ng, bool); resp[np.argsort(-ab, kind="stable")[:int(round(ng * .05))]] = True
        for v, genes in SC[p].items():
            ix = np.array(sorted({gi[x] for x in genes if x in gi}), int)
            if len(ix) < 50: continue
            inl = np.zeros(ng, bool); inl[ix] = True; t2 = [[int((inl & resp).sum()), int((~inl & resp).sum())], [int((inl & ~resp).sum()), int((~inl & ~resp).sum())]]
            rows.append(dict(arm=a, tf=p, variant=v, n=len(ix), effect=score(ab, ix), logor5=np.log((t2[0][0] + .5) * (t2[1][1] + .5) / ((t2[0][1] + .5) * (t2[1][0] + .5)))))
    print(a, "done", flush=True)
P = pd.DataFrame(rows); P.to_csv(f"{O}/per_tf.tsv", sep="\t", index=False, float_format="%.4g")
S = []; rng = np.random.default_rng(SEED + 7)
for a, G in P.groupby("arm"):
    W = G.pivot(index="tf", columns="variant", values="effect"); Wo = G.pivot(index="tf", columns="variant", values="logor5"); W = W.dropna(); Wo = Wo.loc[W.index]
    for v in W.columns:
        b = bstat(W[v], rng); bo = bstat(Wo[v], rng); d = bstat(W[v] - W["sum_cpm@1000"], rng); dn = bstat(W[v] - W["atac_x_abc@1000"], rng)
        S.append(dict(arm=a, variant=v, n_tfs=b["n"], effect=b["mean"], effect_ci=f"[{b['ci_lo']}, {b['ci_hi']}]", logOR5=bo["mean"], logOR5_ci=f"[{bo['ci_lo']}, {bo['ci_hi']}]",
                      vs_sum_cpm=d["mean"], vs_sum_cpm_ci=f"[{d['ci_lo']}, {d['ci_hi']}]", vs_sum_cpm_npos=d["n_pos"], vs_atac=dn["mean"], vs_atac_ci=f"[{dn['ci_lo']}, {dn['ci_hi']}]"))
S = pd.DataFrame(S).sort_values(["arm", "effect"], ascending=[True, False]); S.to_csv(f"{O}/pooled.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); print(S.to_string(index=False))
