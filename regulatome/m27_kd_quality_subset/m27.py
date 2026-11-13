# m27_kd_quality_subset (SPEC.md): knockdown-quality table, filters F1-F3 (knockdown quality only), and the m22 agreement on each subset (top-5,000 edge table). Tables only. Diagnostic.
import json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; O_ = f"{RG_}/m27_kd_quality_subset"; OUT_ = f"{O_}/out"; rg = np.random.default_rng(20261113); LAB_ = "diagnostic"
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}
# 1. quality table
q = pd.read_csv(f"{RG_}/m4_validation/kd_response/out_v2/arm2_all_cell/guide_qc.tsv", sep="\t"); q = q[q.kind.isin(["all", "construct"])]; q["p"] = q.tf.map(lambda t: ALIAS.get(t, t))
cp = q.groupby("p").agg(cells_total=("n_cells_all", "sum"), min_cells_per_guide_pair=("n_cells_all", "min"), n_guide_pairs=("unit", "size"))
st = pd.read_csv(f"{RG_}/phase1/panel_tf_xatlas_status.tsv", sep="\t"); st["p"] = st.panel_tf; st = st.groupby("p").agg(knockdown_pct=("knockdown_pct", "min"), ntc_detect=("ntc_frac_cells_detected", "min"))
FT = pd.read_csv(f"{RG_}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF").ceiling; RC = pd.read_csv(f"{RG_}/m19_binary_matrices/out/responder_counts.tsv", sep="\t", index_col=0)["P4_reproducible_halves|all"]
SP = pd.read_csv(f"{RG_}/m16_strong_phenotype/out/strong_phenotype_per_tf.tsv", sep="\t").set_index("panel_tf").status
tfs79 = [l.strip() for l in open(f"{RG_}/m1_binding/v2/tfs.txt") if l.strip()]
Q = pd.DataFrame(index=pd.Index(tfs79, name="tf")).join(cp).join(st); Q["reliability"] = FT.reindex(Q.index); Q["reproducible_responders"] = RC.reindex(Q.index); Q["m16_rebuilt_strong_phenotype"] = SP.reindex(Q.index); Q["authors_label"] = "not released"
def first_fail(r, lvl):
    checks = [("no knockdown data", pd.notna(r.cells_total)), ("< 25 cells", (r.cells_total or 0) >= 25), ("< 30% knockdown", (r.knockdown_pct or 0) >= 30), ("target not detected in NTC", (r.ntc_detect or 0) > 0)]
    if lvl >= 2: checks.append(("< 50 cells in a guide pair", (r.min_cells_per_guide_pair or 0) >= 50))
    if lvl >= 3: checks.append(("reliability < 0.2 or undefined", pd.notna(r.reliability) and r.reliability >= .2))
    for nm, ok in checks:
        if not ok: return nm
    return ""
for k in (1, 2, 3): Q[f"F{k}_fail_reason"] = [first_fail(r, k) for r in Q.itertuples()]; Q[f"F{k}"] = Q[f"F{k}_fail_reason"] == ""
Q.insert(0, "label", LAB_); Q.reset_index().to_csv(f"{OUT_}/kd_quality_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
for k in (1, 2, 3): print(f"F{k}", int(Q[f"F{k}"].sum()), ",".join(Q.index[Q[f"F{k}"]]), "| fail reasons:", Q[f"F{k}_fail_reason"][~Q[f"F{k}"]].value_counts().to_dict(), flush=True)
# 2. agreement inputs (top-5,000 edge table)
E = pd.read_csv(f"{RG_}/m22_edge_table/out/regulome_edges.tsv.gz", sep="\t", low_memory=False, usecols=["tf", "gene", "score_signal_x_abc"])
sys.argv = ["m15.py", "dc"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); _o = (O_, OUT_); exec(src[:src.index("# ---- binding percentiles")]); O_, OUT_ = _o
gi_ = {g: k for k, g in enumerate(UG)}; ti_ = {t: k for k, t in enumerate(TF)}; nT_, nG_ = len(TF), len(UG)
BND = np.zeros((nT_, nG_), bool); SC = np.zeros((nT_, nG_))
for r in E.itertuples():
    if r.tf in ti_ and r.gene in gi_: BND[ti_[r.tf], gi_[r.gene]] = True; SC[ti_[r.tf], gi_[r.gene]] = r.score_signal_x_abc
FLt = pd.read_csv(f"{RG_}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf").tertile; tert = np.array([str(FLt.get(t, "na")) for t in TF])
GL_ = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); ac = [c for c in GL_.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac_ = GL_[ac].reindex(UG).fillna(0).values; expr_ = d["ntc_mean"][keep][gix]; GW_ = pd.read_csv(f"{RG_}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW_ = np.load(f"{RG_}/m7_step2/features/raw_hicfixed_wide/base.npz"); cgi_ = pd.Series(BW_["base"][:, 4], index=GW_.gene).reindex(UG).fillna(0).values > 0
strat_ = (pd.qcut(pd.Series(expr_).rank(method="first"), 10, labels=False).values * 10 + pd.qcut(pd.Series(atac_).rank(method="first"), 10, labels=False).values) * 2 + cgi_
anyresp = (np.abs(np.nan_to_num(Z1)) >= 2) | (np.abs(np.nan_to_num(Z2)) >= 2)
def topk(A, K): a = np.abs(A); flat = np.where(np.isfinite(a), a, -1).ravel(); o = np.argsort(-flat)[:K]; return set(zip(*np.unravel_index(o, A.shape)))
HC = [p for p in topk(Z1, 2000) & topk(Z2, 2000) if np.sign(Z1[p]) == np.sign(Z2[p])]
MATCH = {}
for (i, j) in HC:
    pool = np.flatnonzero((strat_ == strat_[j]) & ~anyresp[i] & np.isfinite(ZA[i])); MATCH[(i, j)] = rg.choice(pool, min(20, len(pool)), replace=False) if len(pool) else np.array([], int)
SimT = pd.read_csv(f"{RG_}/m22_edge_table/out/simple_correlation.tsv", sep="\t"); SimT = SimT[SimT.version == "all_cells"].set_index("tf")
Rz = np.load(f"{RG_}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True); KT = list(Rz["tfs"].astype(str)); Zr = Rz["r1"].astype(float)
W = np.load(f"{RG_}/m22_edge_table/work_matrices.npz", allow_pickle=True); w_t = list(W["tfs"]); Sw = W["score"]; Ug = W["genes"].astype(str)
MIS = dict(zip(tfs79, json.load(open(f"{RG_}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}; ug2 = {g: k for k, g in enumerate(Ug)}
def mh(tabs):
    num = den = P_ = Q_ = S_ = 0.0
    for a_, b_, c_, d_ in tabs:
        N = a_ + b_ + c_ + d_
        if N == 0: continue
        r_, s_ = a_ * d_ / N, b_ * c_ / N; num += r_; den += s_; p_, q_ = (a_ + d_) / N, (b_ + c_) / N; P_ += p_ * r_; Q_ += p_ * s_ + q_ * r_; S_ += q_ * s_
    if num == 0 or den == 0: return np.nan, np.nan, np.nan
    o = num / den; se = np.sqrt(P_ / (2 * num ** 2) + Q_ / (2 * num * den) + S_ / (2 * den ** 2)); return o, o * np.exp(-1.96 * se), o * np.exp(1.96 * se)
def agreement(tset, name):
    out = dict(label=LAB_, subset=name, n_tfs=len(tset))
    P_ = [(i, j) for (i, j) in HC if TF[i] in tset]; out["hc_pairs"] = len(P_); out["hc_tfs"] = len({TF[i] for i, _ in P_})
    if P_:
        ti = np.array([p[0] for p in P_]); gj = np.array([p[1] for p in P_]); b = BND[ti, gj]
        sw = [BND[np.setdiff1d(np.flatnonzero(tert == tert[i]), [i]), j].mean() for i, j in P_]; mb = [BND[i, MATCH[(i, j)]].mean() if len(MATCH[(i, j)]) else np.nan for i, j in P_]
        out.update(frac_bound_own=float(b.mean()), frac_bound_swapped_tertile=float(np.mean(sw)), frac_bound_matched_genes=float(np.nanmean(mb)))
        def tabs_for(idx):
            T = {}
            for k in idx:
                i, j = P_[k]; m = BND[i, MATCH[(i, j)]]; t = T.setdefault(i, [0, 0, 0, 0]); t[0] += BND[i, j]; t[1] += 1 - BND[i, j]; t[2] += m.sum(); t[3] += (~m).sum()
            return [tuple(v) for v in T.values()]
        o, lo, hi = mh(tabs_for(range(len(P_)))); bp = [mh(tabs_for(rg.integers(0, len(P_), len(P_))))[0] for _ in range(1000)]
        ut = np.unique(ti); bc = []
        for _ in range(1000):
            s = rg.choice(ut, len(ut)); idx = np.concatenate([np.flatnonzero(ti == x) for x in s]); bc.append(mh(tabs_for(idx))[0])
        out.update(MH_OR=o, MH_lo_RBG=lo, MH_hi_RBG=hi, pair_boot_lo=float(np.nanpercentile(bp, 2.5)), pair_boot_hi=float(np.nanpercentile(bp, 97.5)), tf_boot_lo=float(np.nanpercentile(bc, 2.5)), tf_boot_hi=float(np.nanpercentile(bc, 97.5)))
    S = SimT.reindex([t for t in tset if t in SimT.index]); beat = (S.sp_lo > 0) & (S.spearman_abs_z > S.atac_only_spearman_abs_z) & (S.spearman_abs_z > S.swapped_spearman_abs_z.fillna(-9))
    out.update(per_tf_n=len(S), per_tf_median_spearman=float(S.spearman_abs_z.median()) if len(S) else np.nan, n_tfs_interval_above_0=int((S.sp_lo > 0).sum()), n_tfs_beat_both_controls=int(beat.sum()), tfs_beat_both=",".join(S.index[beat]))
    tl = [t for t in tset if t in KT and t in w_t]
    if tl:
        Z = np.vstack([Zr[KT.index(t)] for t in tl]).copy()
        for k, t in enumerate(tl):
            for o_ in OWN.get(t, [t]):
                if o_ in ug2: Z[k, ug2[o_]] = np.nan
        Sm = np.vstack([Sw[w_t.index(t)] for t in tl]); Ss = np.vstack([Sw[w_t.index(MIS[t])] if MIS.get(t) in w_t else np.zeros(len(Ug)) for t in tl]); A = np.abs(Z); m = np.isfinite(A)
        def sp(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
        def pct(M): return np.vstack([rankdata(np.where(np.isfinite(r), r, 0)) / len(r) for r in M])
        out.update(whole_raw=sp(Sm[m], A[m]), whole_raw_swapped=sp(Ss[m], A[m]), whole_within_tf_pct=sp(pct(Sm)[m], pct(np.where(m, A, np.nan))[m]), whole_within_tf_pct_swapped=sp(pct(Ss)[m], pct(np.where(m, A, np.nan))[m]))
    return out
SUBS = {"all TFs with a knockdown": [t for t in TF], "F1": list(Q.index[Q.F1]), "F2": list(Q.index[Q.F2]), "F3": list(Q.index[Q.F3])}
R = []
for nm, tl in SUBS.items():
    tl = [t for t in tl if t in ti_ or t in KT]
    R.append(agreement(set(tl), nm)); R.append(agreement(set(tl) - {"NRF1", "E4F1"}, nm + " without NRF1, E4F1")); print(nm, "done", flush=True)
R = pd.DataFrame(R); R.to_csv(f"{OUT_}/agreement_by_subset.tsv", sep="\t", index=False, float_format="%.4g"); pd.set_option("display.width", 250); print(R.drop(columns=["label", "tfs_beat_both"]).round(3).T.to_string()); print(R[["subset", "tfs_beat_both"]].to_string(index=False))
