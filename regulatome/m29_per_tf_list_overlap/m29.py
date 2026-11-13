# m29 (SPEC.md): per-TF list overlap, Regulome gene list vs Perturb-seq responder list. Tables only. Diagnostic.
import sys
import numpy as np, pandas as pd
from scipy.stats import hypergeom
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; OUT_ = f"{RG_}/m29_per_tf_list_overlap/out"; rg = np.random.default_rng(20261113); LAB_ = "diagnostic"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA, d, keep, gix
TFL, N = list(TF), len(UG); print("universe genes (m9 universe, NTC detection >= 10%):", N, "TFs with knockdown:", len(TFL), flush=True)
gi_ = {g: k for k, g in enumerate(UG)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
E = pd.read_csv(f"{RG_}/m22_edge_table/out/regulome_edges.tsv.gz", sep="\t", usecols=["tf", "gene"]); RT = sorted(set(E.tf))
REG = {t: np.zeros(N, bool) for t in RT}
for t, g in zip(E.tf, E.gene):
    if g in gi_: REG[t][gi_[g]] = True
GL_ = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); ac = [c for c in GL_.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac_ = GL_[ac].reindex(UG).fillna(0).values; expr_ = d["ntc_mean"][keep][gix]; GW_ = pd.read_csv(f"{RG_}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW_ = np.load(f"{RG_}/m7_step2/features/raw_hicfixed_wide/base.npz"); cgi_ = pd.Series(BW_["base"][:, 4], index=GW_.gene).reindex(UG).fillna(0).values > 0
strat = (pd.qcut(pd.Series(expr_).rank(method="first"), 10, labels=False).values * 10 + pd.qcut(pd.Series(atac_).rank(method="first"), 10, labels=False).values) * 2 + cgi_
SI = {s: np.flatnonzero(strat == s) for s in np.unique(strat)}
FLt = pd.read_csv(f"{RG_}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf").tertile
F3 = set(pd.read_csv(f"{RG_}/m27_kd_quality_subset/out/kd_quality_per_tf.tsv", sep="\t").query("F3 == True").tf)
def lists(i, t):
    z, z1, z2 = ZA[i].copy(), Z1[i].copy(), Z2[i].copy()
    for o in OWN.get(t, [t]):
        if o in gi_: z[gi_[o]] = z1[gi_[o]] = z2[gi_[o]] = np.nan
    a = np.abs(z); ok = np.isfinite(z); top = np.zeros(N, bool); top[np.argsort(-np.where(ok, a, -1))[:200]] = True; top &= ok
    c4 = np.isfinite(z1) & np.isfinite(z2) & (np.abs(z1) >= 2) & (np.abs(z2) >= 2) & (np.sign(z1) == np.sign(z2))
    return {"C1 |z|>=2": ok & (a >= 2), "C1_down z<=-2": ok & (z <= -2), "C1_up z>=2": ok & (z >= 2), "C2 |z|>=3": ok & (a >= 3), "C3 top200 |z|": top, "C4 reproducible halves": c4}
def fold(R, P): b, a = R.sum(), P.sum(); return (R & P).sum() / (a * b / N) if a and b else np.nan
ROWS = []
for i, t in enumerate(TFL):
    if t not in REG: continue
    R = REG[t]; same = [u for u in TFL if u != t and u in REG and str(FLt.get(u, "na")) == str(FLt.get(t, "na"))]
    for cut, P in lists(i, t).items():
        a, b, k = int(P.sum()), int(R.sum()), int((R & P).sum()); r = dict(label=LAB_, tf=t, reproducible_kd_F3=t in F3, depth_tertile=FLt.get(t, np.nan), cutoff=cut, n_perturb=a, n_regulome=b, overlap=k)
        if a < 5 or b < 5: ROWS.append(dict(r, testable=False)); continue
        exp = a * b / N; sw = np.array([fold(REG[u], P) for u in same]); f = k / exp
        idx = np.flatnonzero(P); cnt = pd.Series(strat[idx]).value_counts(); dr = np.zeros(1000)
        for s, c in cnt.items(): dr += np.array([R[rg.choice(SI[s], c, replace=False)].sum() for _ in range(1000)]) if len(SI[s]) >= c else np.nan
        ROWS.append(dict(r, testable=True, expected=exp, fold=f, jaccard=k / (a + b - k), hyper_p=float(hypergeom.sf(k - 1, N, b, a)), n_swapped=len(sw), swapped_median_fold=float(np.nanmedian(sw)), own_rank_among_same_tertile=int(1 + (sw > f).sum()),
                         swapped_p=float((1 + (sw >= f).sum()) / (1 + len(sw))), matched_mean_overlap=float(np.nanmean(dr)), matched_p=float((1 + (dr >= k).sum()) / 1001)))
    print(t, flush=True)
T = pd.DataFrame(ROWS); T["bh_q"] = np.nan
for cut, g in T[T.testable].groupby("cutoff"):
    p = g.hyper_p.values; o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1); T.loc[g.index, "bh_q"] = qq
T["pass"] = T.testable & (T.bh_q < .05) & (T.swapped_p <= .05) & (T.matched_p <= .05)
T.to_csv(f"{OUT_}/per_tf_overlap.tsv", sep="\t", index=False, float_format="%.4g")
S = []
for cut, g in T.groupby("cutoff", sort=False):
    h = g[g.testable]; S.append(dict(label=LAB_, cutoff=cut, universe_N=N, tfs_testable=len(h), median_perturb_list=float(h.n_perturb.median()), median_regulome_list=float(h.n_regulome.median()), tfs_q_lt_0_05=int((h.bh_q < .05).sum()),
                                      tfs_q_and_beat_swapped=int(((h.bh_q < .05) & (h.swapped_p <= .05)).sum()), tfs_pass_all=int(h["pass"].sum()), median_fold_own=float(h.fold.median()), median_fold_swapped=float(h.swapped_median_fold.median()),
                                      tfs_own_fold_above_swapped_median=int((h.fold > h.swapped_median_fold).sum()), F3_testable=int(h.reproducible_kd_F3.sum()), F3_pass=int(h[h.reproducible_kd_F3]["pass"].sum()),
                                      tfs_q_lt_0_05_list=",".join(h.tf[h.bh_q < .05]), pass_list=",".join(h.tf[h["pass"]])))
S = pd.DataFrame(S); S.to_csv(f"{OUT_}/summary_by_cutoff.tsv", sep="\t", index=False, float_format="%.4g"); print(S.drop(columns="label").T.to_string())
print(T[T.reproducible_kd_F3 & T.testable & (T.cutoff == "C1 |z|>=2")][["tf", "n_perturb", "n_regulome", "overlap", "fold", "bh_q", "swapped_median_fold", "own_rank_among_same_tertile", "n_swapped", "swapped_p", "matched_p"]].round(4).to_string(index=False)); print("done")
