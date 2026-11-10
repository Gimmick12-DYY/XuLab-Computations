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
assert not os.path.exists(f"{O}/per_tf_DESCRIPTIVE.tsv"), "outputs exist"
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
rows = []
for a in ARMS:
    Z = load(a); U = sorted(set(Z.columns) & LINKED); T_ = sorted(set(Z.index) & set(SV8)); Zr = Z.loc[T_, U].values; ng = len(U)
    S = {"S_v8": np.vstack([SV8[t].reindex(U).fillna(0).values for t in T_]), "S_abc": np.vstack([SABC[t].reindex(U).fillna(0).values for t in T_])}
    E = Egene.reindex(U).values; mabs = np.abs(Zr).mean(0)
    Y = {"absz_raw": np.abs(Zr), "signed_raw": Zr, "absz_DC": dc(np.abs(Zr)), "signed_DC": dc(Zr)}
    rng = np.random.default_rng(SEED + (a == "arm2")); W = rng.multinomial(ng, np.full(ng, 1 / ng), size=NB).astype(np.float32)
    rE = rk(E); rC = np.column_stack([rk(E), rk(mabs)]); ceil = C[C.arm == a].groupby("panel_tf").ceiling.mean()
    for sname, Sm in S.items():
        Sdc = dc(Sm)
        for i, t in enumerate(T_):
            r = dict(arm=a, tf=t, split=Sp.set.get(t, "na"), score=sname, n_genes=ng, ceiling=ceil.get(t, np.nan))
            for vname, Ym in Y.items():
                x = rk(Sdc[i] if vname.endswith("DC") else Sm[i]); y = rk(Ym[i]); rho = np.corrcoef(x, y)[0, 1]; rhoE = np.corrcoef(rE, y)[0, 1]
                bs, be = bootr(x, y, W), bootr(rE, y, W); dl = bs - be
                r.update({f"rho_{vname}": rho, f"rho_{vname}_ci": f"[{np.nanpercentile(bs, 2.5):.3f}, {np.nanpercentile(bs, 97.5):.3f}]", f"rhoE_{vname}": rhoE,
                          f"delta_{vname}": rho - rhoE, f"delta_{vname}_lo": np.nanpercentile(dl, 2.5), f"delta_{vname}_hi": np.nanpercentile(dl, 97.5), f"rho_{vname}_lo": np.nanpercentile(bs, 2.5), f"rho_{vname}_hi": np.nanpercentile(bs, 97.5)})
            for vname, yv in (("absz_partial", np.abs(Zr[i])), ("signed_partial", Zr[i])):
                x = rk(resid(rk(Sm[i]), rC)); y = rk(resid(rk(yv), rC)); rho = np.corrcoef(x, y)[0, 1]; bs = bootr(x, y, W)
                r.update({f"rho_{vname}": rho, f"rho_{vname}_lo": np.nanpercentile(bs, 2.5), f"rho_{vname}_hi": np.nanpercentile(bs, 97.5)})
            r["rho_absz_DC_over_ceiling"] = r["rho_absz_DC"] / r["ceiling"] if r["ceiling"] and r["ceiling"] > 0 else np.nan
            rows.append(r)
    print(a, "TFs", len(T_), "genes", ng, flush=True)
P = pd.DataFrame(rows); os.makedirs(O, exist_ok=True)
P.insert(0, "DESCRIPTIVE", "DESCRIPTIVE ONLY: held-out spent"); P.to_csv(f"{O}/per_tf_DESCRIPTIVE.tsv", sep="\t", index=False, float_format="%.4g")
VARS = ["absz_raw", "signed_raw", "absz_DC", "signed_DC"]; S_ = []; rng = np.random.default_rng(SEED)
q = P.copy(); q["tert"] = pd.qcut(q.ceiling, 3, labels=["low", "mid", "high"]).astype(str)
for excl in (False, True):
    for (a, s), G in q[~(excl & (q.tf == "NRF1"))].groupby(["arm", "score"]):
        for vname in VARS + ["absz_partial", "signed_partial"]:
            row = dict(DESCRIPTIVE="DESCRIPTIVE", nrf1="excluded" if excl else "included", arm=a, score=s, variant=vname, n_tfs=len(G), median_rho=round(G[f"rho_{vname}"].median(), 4))
            if vname in VARS: row.update(median_delta_vs_E=round(G[f"delta_{vname}"].median(), 4), n_delta_ci_gt0=int((G[f"delta_{vname}_lo"] > 0).sum()), n_rho_ci_gt0=int((G[f"rho_{vname}_lo"] > 0).sum()), n_rho_ci_lt0=int((G[f"rho_{vname}_hi"] < 0).sum()))
            else: row.update(n_rho_ci_gt0=int((G[f"rho_{vname}_lo"] > 0).sum()), n_rho_ci_lt0=int((G[f"rho_{vname}_hi"] < 0).sum()))
            for tt, H in G.groupby("tert"): row[f"median_rho_{tt}_ceiling"] = round(H[f"rho_{vname}"].median(), 4)
            S_.append(row)
    for a, G in q[~(excl & (q.tf == "NRF1"))].groupby("arm"):
        for vname in ("absz_DC", "signed_DC"):
            w = G.pivot(index="tf", columns="score", values=f"rho_{vname}").dropna(); d = (w.S_v8 - w.S_abc).values; bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            S_.append(dict(DESCRIPTIVE="DESCRIPTIVE", nrf1="excluded" if excl else "included", arm=a, score="S_v8_minus_S_abc", variant=vname, n_tfs=len(d), median_rho=round(np.median(d), 4), mean_paired_diff=round(d.mean(), 4), paired_ci=f"[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]", n_v8_better=int((d > 0).sum())))
pd.DataFrame(S_).to_csv(f"{O}/summary_DESCRIPTIVE.tsv", sep="\t", index=False)
V = P[P.score == "S_v8"].pivot(index="tf", columns="arm", values=["rho_signed_DC", "rho_signed_DC_lo", "rho_signed_DC_hi"]); out = []
for t in V.index:
    g = lambda c, a: V.loc[t, (c, a)] if (c, a) in V.columns else np.nan
    r2, r1 = g("rho_signed_DC", "arm2"), g("rho_signed_DC", "arm1"); ex = [(g("rho_signed_DC_lo", a) > 0) or (g("rho_signed_DC_hi", a) < 0) for a in ("arm1", "arm2") if not np.isnan(g("rho_signed_DC", a))]
    if np.isnan(r1): call = "single-arm"
    elif np.sign(r1) == np.sign(r2) and any(ex): call = "activator-like" if r2 < 0 else "repressor-like"
    else: call = "uncalled"
    out.append(dict(DESCRIPTIVE="DESCRIPTIVE", tf=t, split=Sp.set.get(t, "na"), rho_signed_DC_arm2=r2, ci_arm2=f"[{g('rho_signed_DC_lo', 'arm2'):.3f}, {g('rho_signed_DC_hi', 'arm2'):.3f}]", rho_signed_DC_arm1=r1,
                    ci_arm1=f"[{g('rho_signed_DC_lo', 'arm1'):.3f}, {g('rho_signed_DC_hi', 'arm1'):.3f}]" if not np.isnan(r1) else ".", call=call))
pd.DataFrame(out).to_csv(f"{O}/sign_calls_DESCRIPTIVE.tsv", sep="\t", index=False, float_format="%.4f")
print(pd.DataFrame(out).call.value_counts().to_dict())
