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
X = "/work/users/d/y/dyy12/XuLab"; M4 = f"{X}/regulatome/m4_validation"; G1 = f"{X}/regulatome/gate1/cellmats"; O = f"{X}/regulatome/m6_improve/grid6"; os.makedirs(O, exist_ok=True)
SEED = 20261109; NR = 1000; NB = 10000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = E.cls.isin(["promoter", "distal_active", "bivalent"]).values; Ea = E[act].reset_index(drop=True)
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "distance", "isSelfPromoter"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int)
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float); atac_cpm = atac / atac.sum() * 1e6
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
A["sq"] = np.sqrt(A["ABC.Score"]); A["dk"] = np.exp(-A.distance.abs() / 5e4); prom = A.isSelfPromoter.astype(str).str.lower().eq("true")

Sp = pd.read_csv(f"{X}/regulatome/m6_improve/split_v1.tsv", sep="\t"); DEV = set(Sp.tf[Sp.set == "development"])
tfs_dev = [t for t in tfs if panel(t) in DEV]; print("dev TFs", len(tfs_dev), flush=True)
def cpm_of(t, kind):
    y = np.asarray(sp.load_npz(f"{G1}/{t}.{kind}.npz").sum(0)).ravel(); return y / max(y.sum(), 1e-12) * 1e6
RAW = {t: cpm_of(t, "raw") for t in tfs}; TOT = np.sum([RAW[t] for t in tfs], axis=0)
# distance windows (no ABC): elements with midpoint within W of the gene TSS (genes and TSS from the ABC table)
Gt = A.drop_duplicates("TargetGene")[["chr", "TargetGene"]].copy(); tss = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["TargetGene", "TargetGeneTSS"]).drop_duplicates("TargetGene").set_index("TargetGene").TargetGeneTSS
Gt["tss"] = Gt.TargetGene.map(tss).values; mid = ((Ea.start + Ea.end) // 2).values
WIN = {}
for W in (100000, 150000, 200000, 600000):
    gi_, ei_ = [], []
    for ch, G in Gt.groupby("chr"):
        ix = np.flatnonzero(Ea.chrom.values == ch); o = np.argsort(mid[ix]); ix = ix[o]; m = mid[ix]
        lo = np.searchsorted(m, G.tss.values - W); hi = np.searchsorted(m, G.tss.values + W, side="right")
        for g, a_, b_ in zip(G.TargetGene.values, lo, hi): gi_ += [g] * (b_ - a_); ei_ += list(ix[a_:b_])
    WIN[W] = (np.array(gi_), np.array(ei_, int))
nlink = A.groupby("TargetGene").size().astype(float)
NT = len(tfs)
DIST600 = np.abs(mid[WIN[600000][1]] - Gt.set_index("TargetGene").tss.reindex(WIN[600000][0]).values)
def variants(t):
    c = RAW[t]; pm = (TOT - c) / (NT - 1); b = np.polyfit(pm, c, 1)
    E_ = {"raw": c, "CTRL_n_elem": np.ones_like(c), "CTRL_pool": pm, "CTRL_atac": atac_cpm}
    out = {"abc_raw": pd.Series(c[A.e.values]).groupby(A.TargetGene.values).sum()}
    for W in (100000, 150000, 200000):
        gg, ee = WIN[W]
        for k, v in E_.items(): out[f"win{W // 1000}k_{k}"] = pd.Series(v[ee]).groupby(gg).sum()
    gg, ee = WIN[100000]; out["win100k_resid"] = pd.Series(np.maximum(c - (b[0] * pm + b[1]), 0)[ee]).groupby(gg).sum()
    gg, ee = WIN[600000]
    for L_ in (100000, 150000, 200000):
        wd = np.exp(-DIST600 / L_)
        for k, v in E_.items(): out[f"decay{L_ // 1000}k_{k}"] = pd.Series(v[ee] * wd).groupby(gg).sum()
    return out
TOPN = {k: [500, 1000] for k in ["abc_raw", "win100k_raw", "win150k_raw", "win200k_raw", "win100k_resid", "decay100k_raw", "decay150k_raw", "decay200k_raw"]}
SC = {}
for t in tfs_dev:
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
            own = z[pi[p]]; off = np.delete(z, pi[p]); ab = Ap[pi[p]]; ng = len(ab); resp = np.zeros(ng, bool); resp[np.argsort(-ab, kind="stable")[:int(round(ng * .05))]] = True; inl = np.zeros(ng, bool); inl[ix] = True
            t2 = [[int((inl & resp).sum()), int((~inl & resp).sum())], [int((inl & ~resp).sum()), int((~inl & ~resp).sum())]]; lor = np.log((t2[0][0] + .5) * (t2[1][1] + .5) / ((t2[0][1] + .5) * (t2[1][0] + .5)))
            rows.append(dict(arm=a, tf=p, variant=v, n=len(ix), z_own=own, z_off_mean=off.mean(), specificity=own - off.mean(), own_rank=int((z > own).sum() + 1), n_kd=len(pts), logor5=lor))
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
