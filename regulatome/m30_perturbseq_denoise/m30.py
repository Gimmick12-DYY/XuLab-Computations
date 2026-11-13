# m30 (SPEC.md): denoise Perturb-seq, then redo the simple per-TF overlap. Tables only. Diagnostic.
import sys, re
import numpy as np, pandas as pd
from scipy.stats import hypergeom, spearmanr
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; OUT_ = f"{RG_}/m30_perturbseq_denoise/out"; rg = np.random.default_rng(20261113); LAB_ = "diagnostic"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA, d, keep, gix
TFL, N0 = list(TF), len(UG); gi_ = {g: k for k, g in enumerate(UG)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
def own_rm(M, names):
    M = M.copy()
    for i, t in enumerate(names):
        for o in OWN.get(t, [t]):
            if o in gi_: M[i, gi_[o]] = np.nan
    return M
ZA, Z1, Z2 = own_rm(ZA, TFL), own_rm(Z1, TFL), own_rm(Z2, TFL)
# cohesin perturbations from m26
COH = ["RAD21", "SMC3", "STAG1", "NIPBL", "WAPL", "PDS5A", "PDS5B", "MAU2", "CTCFL"]
m = np.load(f"{RG_}/m26_perturbseq_known_pairs/out/matrices/arm2_all_cell/kd_response.npz", allow_pickle=True); mu = list(m["units"].astype(str)); mg = pd.Series(np.arange(len(m["genes"])), index=m["genes"].astype(str)); mg = mg[~mg.index.duplicated(keep="last")]; mix = mg.reindex(UG).fillna(-1).astype(int).values; mz = m["z"].astype(float)
def mrow(names):
    r = [mu.index(x) for x in names if x in mu]
    if not r: return np.full(N0, np.nan)
    v = np.nanmean(mz[r], 0); return np.where(mix >= 0, v[np.maximum(mix, 0)], np.nan)
CA, C1_, C2_ = [], [], []
for g in COH:
    al = [x for x in mu if x.startswith(g + "|") and (x.endswith("|all") or "construct" in x)]; CA.append(mrow([x for x in al if x.endswith("|all")] or al))
    C1_.append(mrow([f"{g}|half1"] if f"{g}|half1" in mu else [f"{g}|construct1"])); C2_.append(mrow([f"{g}|half2"] if f"{g}|half2" in mu else [f"{g}|construct2"]))
CA, C1_, C2_ = own_rm(np.vstack(CA), COH), own_rm(np.vstack(C1_), COH), own_rm(np.vstack(C2_), COH)
ALLZ, ALL1, ALL2, NAMES = np.vstack([ZA, CA]), np.vstack([Z1, C1_]), np.vstack([Z2, C2_]), TFL + COH; NP = len(TFL); print("perturbations", len(NAMES), "genes", N0, flush=True)
# 1. response frequency
def dc(M): M = M - np.nanmean(M, 0, keepdims=True); return M - np.nanmean(M, 1, keepdims=True)
def freq(M): ok = np.isfinite(M); return (np.where(ok, np.abs(M), 0) >= 2).sum(0) / np.maximum(ok.sum(0), 1)
FR, FD = freq(ALLZ), freq(dc(ALLZ)); MA = np.nanmean(np.abs(ALLZ), 0)
STRESS = re.compile(r"^(HSP|DNAJ)|^(ATF3|ATF4|DDIT3|DDIT4|JUN|JUNB|FOS|FOSB|EGR1|IER2|IER3|XBP1|HERPUD1|TRIB3|SESN2|CHAC1|GADD45A|GADD45B|PPP1R15A)$")
cls = lambda g: "ribosomal" if re.match(r"^RP[LS]\d", g) else ("mitochondrial" if re.match(r"^(MT-|MRPL|MRPS)", g) else ("stress" if STRESS.search(g) else ""))
GF = pd.DataFrame(dict(label=LAB_, gene=UG, freq_raw=FR, freq_doublecentred=FD, mean_abs_z=MA, gene_class=[cls(g) for g in UG])).sort_values(["freq_raw", "mean_abs_z"], ascending=False)
GF["rank"] = np.arange(1, N0 + 1); GF.to_csv(f"{OUT_}/gene_response_frequency.tsv", sep="\t", index=False, float_format="%.4g")
DROP = {"D1 freq>=20%": set(UG[FR >= .2]), "D2 freq>=10%": set(UG[FR >= .1]), "D3 top 10% by freq": set(GF.gene[: int(round(N0 * .1))])}
base_share = GF.gene_class.value_counts() / N0; RM = []
for nm, s in DROP.items():
    sub = GF[GF.gene.isin(s)]; sh = sub.gene_class.value_counts() / max(len(sub), 1)
    RM.append(dict(label=LAB_, denoiser=nm, genes_removed=len(s), n_over_cutoff_on_doublecentred=int((FD >= (.2 if "20" in nm else .1)).sum()) if "D3" not in nm else np.nan,
                   pct_ribosomal=100 * sh.get("ribosomal", 0), pct_mitochondrial=100 * sh.get("mitochondrial", 0), pct_stress=100 * sh.get("stress", 0),
                   universe_pct_ribosomal=100 * base_share.get("ribosomal", 0), universe_pct_mitochondrial=100 * base_share.get("mitochondrial", 0), universe_pct_stress=100 * base_share.get("stress", 0), top30=",".join(sub.gene[:30])))
RM = pd.DataFrame(RM); RM.to_csv(f"{OUT_}/genes_removed.tsv", sep="\t", index=False, float_format="%.3g")
print(RM.drop(columns="label").to_string(), flush=True)
print("frequency raw vs double-centred: Spearman", round(spearmanr(FR, FD)[0], 3), "| median freq raw", round(np.median(FR), 4), "dc", round(np.median(FD), 4), "| genes >=20%: raw", int((FR >= .2).sum()), "dc", int((FD >= .2).sum()), "| >=10%: raw", int((FR >= .1).sum()), "dc", int((FD >= .1).sum()), flush=True)
# 2. PCs
F0 = np.nan_to_num(ALLZ); U, S, Vt = np.linalg.svd(F0, full_matrices=False); ve = S ** 2 / (S ** 2).sum(); print("variance explained PC1-5:", np.round(ve[:5], 4), flush=True)
def pcrm(M, k): V = Vt[:k].T; R = np.nan_to_num(M) - np.nan_to_num(M) @ V @ V.T; return np.where(np.isfinite(M), R, np.nan)
VARIANTS = {"baseline": (ALLZ, ALL1, ALL2, set())}
for nm, s in DROP.items(): VARIANTS[nm] = (ALLZ, ALL1, ALL2, s)
for k in (1, 3, 5): VARIANTS[f"D4 remove PC1-{k}"] = (pcrm(ALLZ, k), pcrm(ALL1, k), pcrm(ALL2, k), set())
# Regulome
E = pd.read_csv(f"{RG_}/m22_edge_table/out/regulome_edges.tsv.gz", sep="\t", usecols=["tf", "gene", "shared_by"]); RT = sorted(set(E.tf)); REG = {t: np.zeros(N0, bool) for t in RT}
for t, g in zip(E.tf, E.gene):
    if g in gi_: REG[t][gi_[g]] = True
SB = E.groupby("gene").shared_by.max().reindex(UG).fillna(0).values; HOT = set(UG[SB >= 40]); print("genes bound by >= 40 of 79 TFs:", len(HOT), flush=True)
GL_ = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); ac = [c for c in GL_.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac_ = GL_[ac].reindex(UG).fillna(0).values; expr_ = d["ntc_mean"][keep][gix]; GW_ = pd.read_csv(f"{RG_}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW_ = np.load(f"{RG_}/m7_step2/features/raw_hicfixed_wide/base.npz"); cgi_ = pd.Series(BW_["base"][:, 4], index=GW_.gene).reindex(UG).fillna(0).values > 0
strat = (pd.qcut(pd.Series(expr_).rank(method="first"), 10, labels=False).values * 10 + pd.qcut(pd.Series(atac_).rank(method="first"), 10, labels=False).values) * 2 + cgi_
FLt = pd.read_csv(f"{RG_}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf").tertile
F3 = set(pd.read_csv(f"{RG_}/m27_kd_quality_subset/out/kd_quality_per_tf.tsv", sep="\t").query("F3 == True").tf)
ROWS = []
for uf, extra in (("all genes", set()), ("drop genes bound by >= 50% of the 79 TFs", HOT)):
    for vn, (A, H1, H2, drop) in VARIANTS.items():
        U_ = np.array([g not in drop and g not in extra for g in UG]); N = int(U_.sum()); SI = {s: np.flatnonzero(U_ & (strat == s)) for s in np.unique(strat)}
        for lt in ("|z|>=2", "reproducible halves"):
            for i, t in enumerate(TFL):
                if t not in REG: continue
                P = (np.abs(np.nan_to_num(A[i])) >= 2) & np.isfinite(A[i]) if lt == "|z|>=2" else (np.isfinite(H1[i]) & np.isfinite(H2[i]) & (np.abs(np.nan_to_num(H1[i])) >= 2) & (np.abs(np.nan_to_num(H2[i])) >= 2) & (np.sign(H1[i]) == np.sign(H2[i])))
                P = P & U_; R = REG[t] & U_; a, b, k = int(P.sum()), int(R.sum()), int((R & P).sum()); exp = a * b / N if N else np.nan
                r = dict(label=LAB_, universe_filter=uf, denoiser=vn, list_type=lt, universe_N=N, TF=t, reproducible_kd_F3=t in F3, n_regulome_targets=b, n_kd_responding_genes=a, n_shared=k,
                         pct_of_responding_genes_that_are_bound=100 * k / a if a else np.nan, pct_of_bound_genes_that_respond=100 * k / b if b else np.nan, pct_of_all_genes_bound=100 * b / N, expected_shared=exp, fold=k / exp if exp else np.nan, testable=a >= 5 and b >= 5)
                if r["testable"]:
                    same = [u for u in TFL if u != t and u in REG and str(FLt.get(u, "na")) == str(FLt.get(t, "na"))]
                    sw = np.array([((REG[u] & U_) & P).sum() / (a * (REG[u] & U_).sum() / N) if (REG[u] & U_).sum() else np.nan for u in same])
                    cnt = pd.Series(strat[np.flatnonzero(P)]).value_counts(); dr = np.zeros(1000)
                    for s, c in cnt.items():
                        pool = SI[s]; c = min(c, len(pool)); dr += R[pool[np.argsort(rg.random((1000, len(pool))), 1)[:, :c]]].sum(1)
                    r.update(hyper_p=float(hypergeom.sf(k - 1, N, b, a)), swapped_median_fold=float(np.nanmedian(sw)), own_rank_same_tertile=int(1 + (sw > r["fold"]).sum()), n_swapped=len(sw),
                             swapped_p=float((1 + (sw >= r["fold"]).sum()) / (1 + len(sw))), matched_mean_shared=float(dr.mean()), matched_p=float((1 + (dr >= k).sum()) / 1001))
                ROWS.append(r)
            print(uf, vn, lt, flush=True)
T = pd.DataFrame(ROWS); T["bh_q"] = np.nan
for key, g in T[T.testable].groupby(["universe_filter", "denoiser", "list_type"]):
    p = g.hyper_p.values; o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1); T.loc[g.index, "bh_q"] = qq
T["pass_all_three"] = T.testable & (T.bh_q < .05) & (T.swapped_p <= .05) & (T.matched_p <= .05)
T.sort_values(["universe_filter", "denoiser", "list_type", "n_shared"], ascending=[True, True, True, False]).to_csv(f"{OUT_}/per_tf_overlap_denoised.tsv", sep="\t", index=False, float_format="%.4g")
FOC = ["NRF1", "E4F1", "MAZ", "CTCF", "DNMT1", "NME2"]; SM = []
for (uf, vn, lt), g in T.groupby(["universe_filter", "denoiser", "list_type"], sort=False):
    h = g[g.testable]; f = g.set_index("TF").fold
    SM.append(dict(label=LAB_, universe_filter=uf, denoiser=vn, list_type=lt, genes_removed=N0 - int(g.universe_N.iloc[0]), universe_N=int(g.universe_N.iloc[0]), tfs_testable=len(h), tfs_q_lt_0_05=int((h.bh_q < .05).sum()), tfs_pass_all_three=int(h.pass_all_three.sum()),
                   median_fold_own=float(h.fold.median()), median_fold_swapped=float(h.swapped_median_fold.median()), **{f"fold_{x}": float(f.get(x, np.nan)) for x in FOC}, q_lt_0_05_tfs=",".join(h.TF[h.bh_q < .05]), pass_tfs=",".join(h.TF[h.pass_all_three])))
SM = pd.DataFrame(SM); SM.to_csv(f"{OUT_}/summary.tsv", sep="\t", index=False, float_format="%.4g"); pd.set_option("display.width", 250); print(SM.drop(columns="label").round(3).to_string(index=False)); print("done")
