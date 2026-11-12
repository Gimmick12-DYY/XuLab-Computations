# m22 Amendment 2: high-confidence Perturb-seq pairs vs the Regulome edge table. Tables only. Diagnostic.
import sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; O_ = f"{RG_}/m22_edge_table"; OUT_ = f"{O_}/out"; rg = np.random.default_rng(20261112); LAB_ = "diagnostic"
sys.argv = ["m15.py", "dc"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA, d, keep, gix
E = pd.read_csv(f"{OUT_}/regulome_edges.tsv.gz", sep="\t", low_memory=False, usecols=["tf", "gene", "promoter_peak", "score_signal_x_abc", "shared_by", "n_distal_elements_with_peak"])
gi_ = {g: k for k, g in enumerate(UG)}; ti_ = {t: k for k, t in enumerate(TF)}; nT_, nG_ = len(TF), len(UG)
BND = np.zeros((nT_, nG_), bool); PRM = np.zeros((nT_, nG_), bool); SCO = np.zeros((nT_, nG_))
for r in E.itertuples():
    if r.tf in ti_ and r.gene in gi_: BND[ti_[r.tf], gi_[r.gene]] = True; PRM[ti_[r.tf], gi_[r.gene]] = bool(r.promoter_peak); SCO[ti_[r.tf], gi_[r.gene]] = r.score_signal_x_abc
SB = E.groupby("gene").shared_by.first()
FLt = pd.read_csv(f"{RG_}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf").tertile; tert = np.array([str(FLt.get(t, "na")) for t in TF])
GL_ = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); ac = [c for c in GL_.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac_ = GL_[ac].reindex(UG).fillna(0).values; expr_ = d["ntc_mean"][keep][gix]; GW_ = pd.read_csv(f"{RG_}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW_ = np.load(f"{RG_}/m7_step2/features/raw_hicfixed_wide/base.npz"); cgi_ = pd.Series(BW_["base"][:, 4], index=GW_.gene).reindex(UG).fillna(0).values > 0
strat_ = (pd.qcut(pd.Series(expr_).rank(method="first"), 10, labels=False).values * 10 + pd.qcut(pd.Series(atac_).rank(method="first"), 10, labels=False).values) * 2 + cgi_
def topmask(A, frac):  # as m15: per-TF top fraction by |z|
    m = np.zeros_like(A, bool)
    for i in range(A.shape[0]):
        a = np.abs(A[i]); ok = np.isfinite(a); k = int(round(ok.sum() * frac)); o = np.argsort(-np.where(ok, a, -1))[:k]; m[i, o] = True
    return m
def topk_set(A, K):
    a = np.abs(A); flat = np.where(np.isfinite(a), a, -1).ravel(); o = np.argsort(-flat)[:K]; return set(zip(*np.unravel_index(o, A.shape)))
def sets(K):
    s1, s2 = topk_set(Z1, K), topk_set(Z2, K); both = {p for p in s1 & s2 if np.sign(Z1[p]) == np.sign(Z2[p])}
    T2_, T1_ = topmask(Z2, .05), topmask(Z1, .05); a1 = [p for p in s1 if T2_[p]]; a2 = [p for p in s2 if T1_[p]]; un = set(a1) | set(a2)
    return {"both_halves_topK_same_sign": both, "m15_union": un}
anyresp = (np.abs(np.nan_to_num(Z1)) >= 2) | (np.abs(np.nan_to_num(Z2)) >= 2)
ROWS = []; PAIRS = []; REV = []
def analyse(name, K, P_, sub_label, sel):
    P_ = [p for p in P_ if sel(p)]
    if not P_: return
    ti, gj = np.array([p[0] for p in P_]), np.array([p[1] for p in P_]); b = BND[ti, gj]; pr = PRM[ti, gj]
    sw = np.array([BND[np.setdiff1d(np.flatnonzero(tert == tert[i]), [i]), j].mean() for i, j in zip(ti, gj)])
    mb = []; mrows = []
    for i, j in zip(ti, gj):
        pool = np.flatnonzero((strat_ == strat_[j]) & ~anyresp[i] & np.isfinite(ZA[i])); pick = rg.choice(pool, min(20, len(pool)), replace=False) if len(pool) else np.array([], int); mb.append(BND[i, pick].mean() if len(pick) else np.nan); mrows.append((i, pick))
    mb = np.array(mb)
    def orr(bh, bm): a, c = bh.sum(), (~bh).sum(); return ((a + .5) / (c + .5)) / ((bm + .5 / 20).mean() / (1 - bm + .5 / 20).mean()) if len(bh) else np.nan
    def or2(idx):
        bh = b[idx]; m_ = np.concatenate([BND[mrows[k][0], mrows[k][1]] for k in idx]) if len(idx) else np.array([])
        n11, n10, n01, n00 = bh.sum(), (~bh).sum(), m_.sum(), (~m_).sum(); return (n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5))
    obs = or2(np.arange(len(P_))); bp = [or2(rg.integers(0, len(P_), len(P_))) for _ in range(2000)]
    utf = np.unique(ti); bc = []
    for _ in range(2000):
        s = rg.choice(utf, len(utf)); idx = np.concatenate([np.flatnonzero(ti == x) for x in s]); bc.append(or2(idx))
    ROWS.append(dict(label=LAB_, set=name, K=K, subset=sub_label, n_pairs=len(P_), n_tfs=len(utf), frac_bound=float(b.mean()), frac_promoter=float(pr.mean()), swapped_tf_frac_bound=float(np.nanmean(sw)), matched_gene_frac_bound=float(np.nanmean(mb)),
                     OR_vs_matched=obs, OR_lo_pair_boot=float(np.percentile(bp, 2.5)), OR_hi_pair_boot=float(np.percentile(bp, 97.5)), OR_lo_tf_boot=float(np.percentile(bc, 2.5)), OR_hi_tf_boot=float(np.percentile(bc, 97.5)),
                     tfs=";".join(f"{TF[k]}:{v}" for k, v in pd.Series(ti).value_counts().items())))
for K in (100, 500, 2000):
    for name, P_ in sets(K).items():
        for sub, sel in (("all", lambda p: True), ("without E4F1, NRF1, DNMT1", lambda p: TF[p[0]] not in ("E4F1", "NRF1", "DNMT1")), ("gene down on knockdown", lambda p: ZA[p] < 0), ("gene up on knockdown", lambda p: ZA[p] > 0)):
            analyse(name, K, P_, sub, sel)
        if name == "both_halves_topK_same_sign":
            for t in sorted({TF[p[0]] for p in P_}):
                analyse(name, K, P_, f"TF {t}", lambda p, t=t: TF[p[0]] == t)
            for (i, j) in P_:
                if BND[i, j]: PAIRS.append(dict(K=K, tf=TF[i], gene=UG[j], z_all=ZA[i, j], z_half1=Z1[i, j], z_half2=Z2[i, j], promoter_peak=bool(PRM[i, j]), score_signal_x_abc=SCO[i, j], shared_by=int(SB.get(UG[j], 0))))
            hb = [(i, j) for (i, j) in P_ if BND[i, j]]
            for t in sorted({TF[i] for i, _ in hb}):
                i = ti_[t]; bg = np.flatnonzero(BND[i]); pct_ = pd.Series(SCO[i, bg]).rank(pct=True).values; pm = dict(zip(bg, pct_)); hs = [pm[j] for (ii, j) in hb if ii == i]; nr = [pm[j] for j in bg if not anyresp[i, j]]
                REV.append(dict(label=LAB_, K=K, tf=t, n_bound_highconf=len(hs), median_pct_highconf=float(np.median(hs)), median_pct_bound_nonresponders=float(np.median(nr)) if nr else np.nan, mannwhitney_p=float(mannwhitneyu(hs, nr).pvalue) if len(hs) >= 3 and len(nr) >= 3 else np.nan))
R = pd.DataFrame(ROWS); R.to_csv(f"{OUT_}/highconf_summary.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(PAIRS).drop_duplicates(["tf", "gene"]).to_csv(f"{OUT_}/highconf_pairs.tsv", sep="\t", index=False, float_format="%.4g"); pd.DataFrame(REV).to_csv(f"{OUT_}/highconf_reverse.tsv", sep="\t", index=False, float_format="%.4g")
print(R[~R.subset.str.startswith("TF ")].drop(columns=["label", "tfs"]).round(3).to_string(index=False)); print(R[R.subset.str.startswith("TF ") & (R.K == 2000)].drop(columns=["label", "tfs", "set", "K"]).round(3).to_string(index=False))
print(pd.DataFrame(REV).round(3).to_string(index=False))
