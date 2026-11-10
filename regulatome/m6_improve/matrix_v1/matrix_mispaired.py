# DESCRIPTIVE ONLY. matrix_v1 Correction 1 (README.md, declared before output read): mis-paired S as the primary control for the DC and partial
# variants. For each arm and score, 5 no-fixed-point permutations of TF labels on S (seed 20261109 + k); TF t's z is correlated with TF perm(t)'s S
# (DC S row, or the same rank-residualised S for partial). Per TF: rho_S, min/median/max rho_mispaired, delta = rho_S - median(rho_mispaired) with a
# 1,000-gene bootstrap CI (same multinomial weights for rho_S and all draws). Same inputs, loader, universe and functions as matrix_compare.py.
# Writes matrix_v1/mispaired_per_tf_DESCRIPTIVE.tsv and mispaired_summary_DESCRIPTIVE.tsv.
# DESCRIPTIVE ONLY (held-out spent). Full-matrix comparison of our TF x gene scores with the X-Atlas/Orion knockdown z matrix (SPEC_MATRIX.md).
# z: kd_response/out_v2, loaded as b3b4/b6/m6 load(): >= 10% NTC universe, units averaged per (panel) TF, both arms; raw z and double-centred z.
# Scores over all ABC-linked genes: S_v8 = raw Gate 1 CPM summed over ABC-linked elements (B6 list e / manifest v8 rule, no ABC weight);
# S_abc = r5_core manifest v7 S (raw binding x ABC, the v7 core rule); E = number of ABC-linked elements per gene (no TF index).
# Universe per arm = z universe & linked genes; TFs = all overlapping, labelled dev/heldout (split_v1), none excluded.
# Per TF x arm x score: Spearman rho of S vs |z| and vs signed z, raw and double-centred (DC: gene means then TF means removed over the same
# gene x TF block, for S and for z/|z|); partial Spearman (rank residuals on ranks of E and of the gene's mean |z| across TFs) vs |z| and signed z;
# control rho(E, .) for the same four variants (E has no TF index, so it is used uncentred against DC z); delta = rho_S - rho_E.
# CIs: 1,000 gene bootstrap resamples (multinomial weights, seed 20261109), rank-Pearson on resampled genes. Ceiling: b3b4 replicate ceiling.
# Writes matrix_v1/per_tf_DESCRIPTIVE.tsv, summary_DESCRIPTIVE.tsv, sign_calls_DESCRIPTIVE.tsv.
import os, subprocess
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; M4 = f"{R}/m4_validation"; O = f"{R}/m6_improve/matrix_v1"; G1 = f"{R}/gate1/cellmats"
assert not os.path.exists(f"{O}/mispaired_per_tf_DESCRIPTIVE.tsv"), "outputs exist"
v = subprocess.run("sha256sum -c --quiet regulatome/r5_core/manifest_v7.sha256 && (cd regulatome/r6_v8 && sha256sum -c --quiet manifest_v8.sha256)", shell=True, cwd=X, capture_output=True, text=True); assert v.returncode == 0, v.stdout + v.stderr
SEED = 20261109; NB = 1000; ARMS = {"arm1": "arm1_high_knockdown", "arm2": "arm2_all_cell"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); Ea = E_[E_.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
A = pd.read_csv(f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int)
Egene = A.groupby("TargetGene").size().astype(float); LINKED = set(Egene.index)
tfs = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]
SV8, SABC = {}, {}
for t in tfs:
    y = np.asarray(sp.load_npz(f"{G1}/{t}.raw.npz").sum(0)).ravel(); cpm = y / np.load(f"{G1}/{t}.cells.npz")["reads"].astype(float).sum() * 1e6
    SV8[panel(t)] = pd.Series(cpm[A.e.values]).groupby(A.TargetGene.values).sum()
    SABC[panel(t)] = pd.read_csv(f"{R}/r5_core/lists/{t}.tsv.gz", sep="\t", usecols=["gene", "S"]).set_index("gene").S
Sp = pd.read_csv(f"{R}/m6_improve/split_v1.tsv", sep="\t").set_index("tf"); C = pd.read_csv(f"{M4}/b3b4/reliability_ceilings.tsv", sep="\t")
def load(a):  # b3b4.py load() (no replicate part)
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}; uni = d["ntc_detect"] >= .1; g = d["genes"][uni]; Z = d["z"][:, uni].astype(float); kp = q[q.kept == 1]; prim = kp[kp.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_]); P_ = np.nan_to_num(P_)
    pts = sorted({panel(t) for t in tf_}); Zp = np.vstack([P_[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts]); return pd.DataFrame(Zp, index=pts, columns=g)
def dc(M):
    M = M - M.mean(0, keepdims=True); return M - M.mean(1, keepdims=True)
rk = lambda x: stats.rankdata(x, axis=-1)
def resid(y, Xc):
    Xc = np.column_stack([np.ones(len(y)), Xc]); b, *_ = np.linalg.lstsq(Xc, y, rcond=None); return y - Xc @ b
def bootr(x, y, W):  # rank-Pearson on bootstrap-weighted genes; x, y already ranked
    n = W.sum(1, keepdims=False); sx, sy = W @ x, W @ y; sxx, syy, sxy = W @ (x * x), W @ (y * y), W @ (x * y)
    cov = sxy / n - sx * sy / n ** 2; vx = sxx / n - (sx / n) ** 2; vy = syy / n - (sy / n) ** 2; return cov / np.sqrt(vx * vy)

def derange(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p
rows = []
for a in ARMS:
    Z = load(a); U = sorted(set(Z.columns) & LINKED); T_ = sorted(set(Z.index) & set(SV8)); Zr = Z.loc[T_, U].values; ng = len(U); nt = len(T_)
    S = {"S_v8": np.vstack([SV8[t].reindex(U).fillna(0).values for t in T_]), "S_abc": np.vstack([SABC[t].reindex(U).fillna(0).values for t in T_])}
    E = Egene.reindex(U).values; mabs = np.abs(Zr).mean(0); rC = np.column_stack([rk(E), rk(mabs)])
    Y = {"absz_DC": dc(np.abs(Zr)), "signed_DC": dc(Zr)}; YP = {"absz_partial": np.abs(Zr), "signed_partial": Zr}
    rng = np.random.default_rng(SEED + (a == "arm2")); W = rng.multinomial(ng, np.full(ng, 1 / ng), size=NB).astype(np.float32)
    for sname, Sm in S.items():
        Sdc = dc(Sm); XDC = np.vstack([rk(Sdc[i]) for i in range(nt)]); XP = np.vstack([rk(resid(rk(Sm[i]), rC)) for i in range(nt)])
        PERM = [derange(nt, np.random.default_rng(20261109 + k)) for k in range(5)]
        for i, t in enumerate(T_):
            r = dict(arm=a, tf=t, split=Sp.set.get(t, "na"), score=sname, n_genes=ng, n_tfs=nt)
            for vname, Ym in list(Y.items()) + list(YP.items()):
                part = vname.endswith("partial"); X_ = XP if part else XDC
                y = rk(resid(rk(Ym[i]), rC)) if part else rk(Ym[i])
                rho = np.corrcoef(X_[i], y)[0, 1]; mis = [np.corrcoef(X_[p[i]], y)[0, 1] for p in PERM]
                bs = bootr(X_[i], y, W); bm = np.median(np.vstack([bootr(X_[p[i]], y, W) for p in PERM]), axis=0); dl = bs - bm
                r.update({f"rho_{vname}": rho, f"mis_min_{vname}": min(mis), f"mis_med_{vname}": float(np.median(mis)), f"mis_max_{vname}": max(mis),
                          f"delta_mis_{vname}": rho - float(np.median(mis)), f"delta_mis_{vname}_lo": np.nanpercentile(dl, 2.5), f"delta_mis_{vname}_hi": np.nanpercentile(dl, 97.5)})
            rows.append(r)
    print(a, "TFs", nt, "genes", ng, flush=True)
P = pd.DataFrame(rows); os.makedirs(O, exist_ok=True); P.insert(0, "DESCRIPTIVE", "DESCRIPTIVE ONLY: held-out spent"); P.to_csv(f"{O}/mispaired_per_tf_DESCRIPTIVE.tsv", sep="\t", index=False, float_format="%.4g")
V = ["absz_DC", "signed_DC", "absz_partial", "signed_partial"]; S_ = []; rng = np.random.default_rng(SEED)
for excl in (False, True):
    for (a, s), G in P[~(excl & (P.tf == "NRF1"))].groupby(["arm", "score"]):
        for v in V: S_.append(dict(DESCRIPTIVE="DESCRIPTIVE", nrf1="excluded" if excl else "included", arm=a, score=s, variant=v, n_tfs=len(G), median_rho=round(G[f"rho_{v}"].median(), 4), median_mis=round(G[f"mis_med_{v}"].median(), 4),
                                   median_delta_mis=round(G[f"delta_mis_{v}"].median(), 4), n_delta_ci_gt0=int((G[f"delta_mis_{v}_lo"] > 0).sum()), n_delta_ci_lt0=int((G[f"delta_mis_{v}_hi"] < 0).sum())))
    for a, G in P[~(excl & (P.tf == "NRF1"))].groupby("arm"):
        for v in V:
            w = G.pivot(index="tf", columns="score", values=f"delta_mis_{v}").dropna(); d = (w.S_v8 - w.S_abc).values; bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            S_.append(dict(DESCRIPTIVE="DESCRIPTIVE", nrf1="excluded" if excl else "included", arm=a, score="S_v8_minus_S_abc(delta_mis)", variant=v, n_tfs=len(d), mean_paired_diff=round(d.mean(), 4), paired_ci=f"[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]", n_v8_better=int((d > 0).sum())))
pd.DataFrame(S_).to_csv(f"{O}/mispaired_summary_DESCRIPTIVE.tsv", sep="\t", index=False)
