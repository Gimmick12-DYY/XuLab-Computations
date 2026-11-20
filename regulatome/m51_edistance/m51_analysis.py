# m51 (SPEC.md): Part A E-distance per TF, Part B Stage 1 by E-distance group, Part C paper-style per-gene d responders.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.utils.extmath import randomized_svd
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read()
exec(SRC50[:SRC50.index("# ---- responder counts")])  # m48 prefix (UG, TFS, Z, OK, AZ, LISTS, STRAT, BROAD, dec, ex) + L1, NC, TER, DEX, ALIAS, F (m50 unit CP10k means)
exec(SRC50[SRC50.index("# ---- Stage 1"):SRC50.index("ST, LOO = stage1(")])  # SETS, NP, pval, stage1()
O = "/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out"; M23 = "/work/users/d/y/dyy12/XuLab/regulatome/m23_sgrna_top25/out/matrices"
OWNG = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
GS = np.load(f"{O}/ntc_gem_stats.npz", allow_pickle=True); sym = GS["genes"].astype(str); gpos = pd.Series(np.arange(len(sym)), index=sym); gpos = gpos[~gpos.index.duplicated(keep="last")]; col = gpos.reindex(UG).values; assert not np.isnan(col).any(); col = col.astype(int)
KD = sp.load_npz(f"{O}/kd_cells.npz")[:, col].toarray().astype(np.float32); KO = pd.read_csv(f"{O}/kd_cells.tsv.gz", sep="\t")
NT = sp.load_npz(f"{O}/ntc_sample.npz")[:, col].toarray().astype(np.float32); NO = pd.read_csv(f"{O}/ntc_sample_cells.tsv.gz", sep="\t")
Q = pd.read_csv(f"{M23}/top25_own/guide_qc.tsv", sep="\t"); KEPT = set(Q[(Q.kept == 1) & Q.kind.isin(["all", "construct"])].unit); dual = set(Q[Q.kind == "construct"].tf)
cn = {}
for g in dual:
    for c, gt in enumerate(sorted(KO[KO.gene_target == g].guide_target.unique())): cn[(g, gt)] = f"{g}|construct{c + 1}"
KO["unit"] = [cn.get((g, gt), f"{g}|all") if g in dual else f"{g}|all" for g, gt in zip(KO.gene_target, KO.guide_target)]; KO["TF"] = [ALIAS.get(g, g) for g in KO.gene_target]
keep = KO.unit.isin(KEPT).values & KO.TF.isin(TFS).values; KD, KO = KD[keep], KO[keep].reset_index(drop=True); print("KD cells", len(KO), "TFs", KO.TF.nunique(), "NTC sample", len(NO), flush=True); assert KO.TF.nunique() == len(TFS)
# HVG from NTC sample (normalised dispersion within 20 mean bins)
cp = np.expm1(NT); mu = cp.mean(0); va = cp.var(0, ddof=1); disp = np.log(np.where(mu > 0, va / np.maximum(mu, 1e-12), np.nan)); b = pd.qcut(pd.Series(np.log1p(mu)).rank(method="first"), 20, labels=False).values
dz = pd.Series(disp).groupby(b).transform(lambda v: (v - v.mean()) / v.std()).values; HVG = np.zeros(len(UG), bool); HVG[np.argsort(-np.nan_to_num(dz, nan=-1e9))[:2000]] = True
rng = np.random.default_rng(20261120); REFN = rng.choice(len(NO), 5000, replace=False); REF = np.vstack([NT[REFN], KD])
def pcs(gmask):
    X = REF[:, gmask]; m = X.mean(0); s = X.std(0); ok = s > 0; X = (X[:, ok] - m[ok]) / s[ok]; _, _, Vt = randomized_svd(X, 20, random_state=20261120)
    return lambda Y: ((Y[:, gmask][:, ok] - m[ok]) / s[ok]) @ Vt.T
def mo(D): n = D.shape[0]; return D.sum() / (n * (n - 1))
from scipy.spatial.distance import cdist
def edist(kix, proj, rs):
    K = proj(KD[kix]); gems = KO.gem.values[kix]; n = len(kix); R, grp = [], []
    for g, kg in pd.Series(gems).value_counts().items():
        P = np.flatnonzero(NO.gem.values == g); rg = int(min(round(1000 * kg / n), len(P) - kg)); assert rg >= kg + 2, (g, kg, rg); R.append(rs.choice(P, rg, replace=False)); grp.append((kg, rg))
    Rc = np.concatenate(R); XR = proj(NT[Rc]); DRR = cdist(XR, XR); E = 2 * cdist(K, XR).mean() - mo(cdist(K, K)) - mo(DRR)
    off = np.cumsum([0] + [rg for _, rg in grp]); tot = DRR.sum(); r = len(Rc); null = np.empty(1000)
    for d in range(1000):
        S_ = np.concatenate([off[i] + rs.choice(rg, kg, replace=False) for i, (kg, rg) in enumerate(grp)]); ns = len(S_); m_ = r - ns
        sSR = DRR[S_].sum(); sSS = DRR[np.ix_(S_, S_)].sum(); null[d] = 2 * (sSR - sSS) / (ns * m_) - sSS / (ns * (ns - 1)) - (tot - 2 * sSR + sSS) / (m_ * (m_ - 1))
    return E, null, n, r
# on-target knockdown % from m50 unit CP10k means
fu = F["units"].astype(str); fg = pd.Series(np.arange(len(F["genes"])), index=F["genes"].astype(str)); fg = fg[~fg.index.duplicated(keep="last")]
def kdpct(units):
    ii = [k for k, u in enumerate(fu) if u in units]; w = F["cells"][ii].astype(float); own = OWNG.get(ALIAS.get(units[0].split("|")[0], units[0].split("|")[0]), [units[0].split("|")[0]]); v = []
    for o in own:
        if o in fg.index: j = fg[o]; v.append(100 * (1 - (w * F["kd_cp10k"][ii, j]).sum() / max((w * F["ntc_cp10k"][ii, j]).sum(), 1e-12)))
    return float(np.mean(v)) if v else np.nan
ROWS = []
for t in TFS:
    own = OWNG.get(t, [t]); gm_ = np.ones(len(UG), bool)
    for o in own:
        if o in set(UG): gm_[np.flatnonzero(np.array(UG) == o)] = False
    projs = {"11,228 genes": pcs(gm_), "2,000 HVG": pcs(gm_ & HVG)}
    tix = np.flatnonzero(KO.TF.values == t); units = sorted(KO.unit[tix].unique()); levels = [("TF", tix, units)] + ([(u, np.flatnonzero(KO.unit.values == u), [u]) for u in units] if len(units) > 1 else [])
    for lev, ix, us in levels:
        r = {"TF": t, "unit": "TF (pooled)" if lev == "TF" else lev, "cells": len(ix), "on-target knockdown (%)": kdpct(us)}
        for gn, pr in projs.items():
            E, nl, n, rr = edist(ix, pr, np.random.default_rng(20261120)); r.update({f"E ({gn})": E, f"null mean ({gn})": nl.mean(), f"null 95th pct ({gn})": np.percentile(nl, 95), f"E / null mean ({gn})": E / nl.mean(), f"p ({gn})": (1 + (nl >= E).sum()) / 1001, f"NTC reference cells ({gn})": rr})
        ROWS.append(r)
    print(t, round(ROWS[-len(levels)]["E / null mean (11,228 genes)"], 3), ROWS[-len(levels)]["p (11,228 genes)"], flush=True)
ED = pd.DataFrame(ROWS); TL = ED[ED.unit == "TF (pooled)"].set_index("TF").reindex(TFS)
def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
for gn in ("11,228 genes", "2,000 HVG"): TL[f"q BH ({gn})"] = bh(TL[f"p ({gn})"].values)
ED = ED.merge(TL[["q BH (11,228 genes)", "q BH (2,000 HVG)"]].reset_index(), on="TF", how="left"); ED.loc[ED.unit != "TF (pooled)", ["q BH (11,228 genes)", "q BH (2,000 HVG)"]] = np.nan
R3 = (OK & (AZ >= 3)).sum(1); EN = TL["E / null mean (11,228 genes)"].values; QV = TL["q BH (11,228 genes)"].values
COR = []
for gn in ("11,228 genes", "2,000 HVG"):
    e = TL[f"E / null mean ({gn})"].values
    for nm, v in (("cells", TL.cells.values), ("on-target knockdown (%)", TL["on-target knockdown (%)"].values), ("|z| >= 3 responders", R3)):
        k = np.isfinite(e) & np.isfinite(v.astype(float)); rr = spearmanr(e[k], v[k]); COR.append({"genes": gn, "E / null mean vs": nm, "TFs": int(k.sum()), "Spearman": rr.statistic, "p": rr.pvalue})
COR = pd.DataFrame(COR); pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(TL[["cells", "on-target knockdown (%)", "E (11,228 genes)", "null mean (11,228 genes)", "null 95th pct (11,228 genes)", "E / null mean (11,228 genes)", "p (11,228 genes)", "q BH (11,228 genes)", "E / null mean (2,000 HVG)", "q BH (2,000 HVG)"]].round(3).sort_values("E / null mean (11,228 genes)", ascending=False).to_string()); print(COR.round(3).to_string(index=False), flush=True)
print("q < 0.05 TFs (11,228 genes):", int((QV < .05).sum()), "| (2,000 HVG):", int((TL["q BH (2,000 HVG)"].values < .05).sum()), flush=True)
# ---- Part B
three = np.array([t in ("NRF1", "E4F1", "DNMT1") for t in TFS]); rk = pd.Series(EN).rank(method="first").values; thirds = np.where(rk > 42, "top third", np.where(rk > 21, "middle third", "bottom third"))
G0 = {"all 63 TFs": np.ones(len(TFS), bool), "E-distance q < 0.05": QV < .05, "E-distance q >= 0.05": QV >= .05, "top third by E / null": thirds == "top third", "middle third by E / null": thirds == "middle third", "bottom third by E / null": thirds == "bottom third"}
SETS = {}
for k, m in G0.items(): SETS[k] = m; SETS[k + ", without NRF1/E4F1/DNMT1"] = m & ~three
REFZ = [("reference |z|", f">= {c}", OK & (AZ >= c), Z) for c in (2, 3, 4, 5)]
STB, _ = stage1(REFZ, 20261118, False); STB = STB[STB.direction == "all"]
PT = []
B10 = np.vstack([LISTS["10 kb"][t] for t in TFS])
for c in (2, 3):
    M = OK & (AZ >= c); ob = (B10 & M).sum(1); ex_ = np.zeros(len(TFS))
    for i in range(len(TFS)):
        s_ = STRAT[OK[i]]; b_ = B10[i][OK[i]]; p_ = M[i][OK[i]]
        for st in np.unique(s_[p_]): m = s_ == st; ex_[i] += b_[m].sum() * p_[m].sum() / m.sum()
    k = ex_ > 0; rr = spearmanr(EN[k], ob[k] / ex_[k]); PT.append({"cutoff": f"|z| >= {c}", "TFs": int(k.sum()), "Spearman E / null vs observed / expected in both (10 kb)": rr.statistic, "p": rr.pvalue})
    TL[f"observed / expected in both, |z| >= {c}, 10 kb"] = np.where(k, ob / np.where(k, ex_, 1), np.nan)
PT = pd.DataFrame(PT); print(PT.round(3).to_string(index=False), flush=True)
print(STB[STB.padding == "10 kb"][["version", "cutoff", "TF set", "TFs contributing", "Perturb-seq genes", "recall (%)", "shuffle mean (%)", "gap (points)", "observed / shuffle", "p (one-sided)", "precision (%)", "precision shuffle (%)"]].round(3).to_string(index=False), flush=True)
# ---- Part C
GN = GS["n"].astype(float); NM = GS["mean"][:, col]; NV = GS["var"][:, col]; DU = {}
for u in sorted(KO.unit.unique()):
    ix = np.flatnonzero(KO.unit.values == u); ng = np.bincount(KO.gem.values[ix], minlength=len(GN)).astype(float); w = ng / ng.sum()
    diff = KD[ix].mean(0) - (w[:, None] * NM).sum(0); sd = np.sqrt((w[:, None] * NV).sum(0)); DU[u] = (diff, np.where(sd > 0, diff / np.where(sd > 0, sd, 1), np.nan))
D = {}; DIF = {}
for u, (df_, d_) in DU.items(): t = ALIAS.get(u.split("|")[0], u.split("|")[0]); D.setdefault(t, []).append(d_); DIF.setdefault(t, []).append(df_)
DM = np.vstack([np.mean(D[t], 0) for t in TFS]); DM[~OK] = np.nan
F49 = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m49_z_fold_change/out/unit_fold_change.npz", allow_pickle=True); g49 = pd.Series(np.arange(len(F49["genes"])), index=F49["genes"].astype(str)); g49 = g49[~g49.index.duplicated(keep="last")]
mx = 0.
for k, u in enumerate(F49["units"].astype(str)):
    if u in DU: c49 = g49.reindex(UG).values; m_ = ~np.isnan(c49); mx = max(mx, float(np.nanmax(np.abs(DU[u][0][m_] - F49["logdiff"][k][c49[m_].astype(int)]))))
print("check: KD - NTC log1p difference vs m49, max abs diff", mx, "| sign agreement d vs z", float((np.sign(DM[OK]) == np.sign(Z[OK])).mean()), flush=True)
AD = np.abs(np.nan_to_num(DM)); VSC = [("paper-style d", f">= {c}", OK & (AD >= c), DM) for c in (0.2, 0.3, 0.4, 0.5)]
RC = []
for v, c, M, S_ in VSC:
    n = M.sum(1); dn = (M & (np.nan_to_num(S_) < 0)).sum(); rho = spearmanr(n, NC); tot = max(M.sum(), 1)
    d = {"version": v, "cutoff": c, "pairs": int(M.sum()), "down (%)": 100 * dn / tot, "TFs >= 1 responder": int((n >= 1).sum()), "TFs >= 10 responders": int((n >= 10).sum()), "median responders per TF": float(np.median(n)), "median |z| of pairs": float(np.median(AZ[M])) if M.any() else np.nan,
         "Spearman responders vs cells per TF": rho.statistic, "Spearman p": rho.pvalue, "median responders, TFs with 25-42 cells": float(np.median(n[TER == "25-42"])), "43-73 cells": float(np.median(n[TER == "43-73"])), "74-216 cells": float(np.median(n[TER == "74-216"]))}
    for k in range(10): d[f"decile {k + 1} (%)"] = 100 * M[:, DEX == k].sum() / tot
    RC.append(d)
RC = pd.DataFrame(RC); print(RC.round(3).to_string(index=False), flush=True)
SETS = {"all 63 TFs": np.ones(len(TFS), bool), "55 non-broad TFs": np.array([t not in BROAD for t in TFS]), "without NRF1, E4F1, DNMT1": ~three, "E-distance q < 0.05": QV < .05, "E-distance q < 0.05, without NRF1/E4F1/DNMT1": (QV < .05) & ~three}
STC, _ = stage1(VSC, 20261118, False)
print(STC[(STC.padding == "10 kb")][["cutoff", "direction", "TF set", "TFs contributing", "Perturb-seq genes", "recall (%)", "shuffle mean (%)", "gap (points)", "observed / shuffle", "p (one-sided)", "precision (%)", "precision shuffle (%)"]].round(3).to_string(index=False), flush=True)
note = pd.DataFrame({"note": [
 "E-distance (Part A): cells = m23 top-25% sgRNA cells of the kept units; log1p(CP10k); genes = 11,228-gene Stage 1 universe or the top 2,000 HVG (normalised dispersion of NTC CP10k in 20 mean bins), own target gene(s) removed; per gene z-scaling and 20-component PCA (randomized SVD, seed 20261120) fitted per TF on 5,000 random NTC cells (from a sample of up to 100 NTC per GEM) plus all top-25% knockdown cells. E = 2 mean d(KD, NTC) - mean d(KD, KD) - mean d(NTC, NTC), Euclidean, within-group means without self pairs. NTC reference: per GEM r_g = min(round(1000 k_g / n), available - k_g) cells of the knockdown cells' GEMs. Null: 1,000 draws of k_g NTC per GEM from the reference scored against the rest; p = (1 + #null >= E)/1001; BH over 63 TFs.",
 "On-target knockdown % = 100 (1 - mean CP10k of the own gene in the top-25% cells / GEM-weighted NTC mean), constructs cell-weighted, HP1 = mean over CBX1/3/5.",
 "Part B: Stage 1 with the existing |z| responders (m23 z), top 10,000 lab peaks, link = promoter or canonical Hi-C ABC element + padding; matched gene shuffle 1,000, seed 20261118, one run per table, TF groups share per-TF draws. Per-TF observed / expected in both: expected = exact hypergeometric mean over NTC expression x promoter ATAC strata.",
 "Part C: d = (mean log1p(CP10k) in the top-25% cells - GEM-weighted NTC mean) / sqrt(sum_g (n_g/n) V_g), V_g = within-GEM NTC variance of log1p(CP10k) (all NTC cells); per TF mean over constructs / CBX units; own gene removed."]})
with pd.ExcelWriter(f"{O}/stage1_edistance.xlsx") as w:
    TL.reset_index().to_excel(w, sheet_name="E-distance per TF", index=False); ED.to_excel(w, sheet_name="E-distance units", index=False); COR.to_excel(w, sheet_name="E-distance correlations", index=False)
    STB.to_excel(w, sheet_name="Stage 1 by E-distance group", index=False); PT.to_excel(w, sheet_name="per-TF E vs Stage 1", index=False)
    RC.to_excel(w, sheet_name="paper-style d responders", index=False); STC.to_excel(w, sheet_name="Stage 1 d", index=False); note.to_excel(w, sheet_name="notes", index=False)
print("done")
