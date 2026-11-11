# m9_abc_diagnosis sections A and D (SPEC.md). Diagnostic, benchmark already read: no claim of validation comes from these tables.
# A: canonical ABC (HEK293T_hicfixed) target sets per TF, as m2c tf_targets_overlap.py (lab MACS2 peaks q <= 0.05, summit in candidate, ABC >= 0.02 link).
# D: SVD of the testable-TF x gene raw z matrix (genes standardised across TFs), loadings vs promoter features, r3 = z minus top-k components,
#    and the step-1 Fisher ORs (ABC targets vs top-5% |response|) recomputed for r1, r2, r3(k = 1, 2, 3, 5), beside the ATAC-only control.
# usage: m9_ad.py -> A_targets_per_tf.tsv, A_summary.tsv, D_variance.tsv, D_loadings.tsv, D_tf_loading.tsv, D_step1_or_per_tf.tsv, D_step1_or_pooled.tsv, work/responses_<arm>.npz
import os
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m9_abc_diagnosis"; os.makedirs(f"{O}/work", exist_ok=True)
LAB = "diagnostic, benchmark already read"; R = f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed"; PK = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"; M4 = f"{RG}/m4_validation"
ARMS = {"arm2": "arm2_all_cell", "arm1": "arm1_high_knockdown"}; ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; panel = lambda t: ALIAS.get(t, t)
QMIN = -np.log10(0.05)
LEAD = ["NRF1", "NFYA", "CTCF", "E4F1", "ZNF143", "NME2", "XPA", "MAZ", "E2F3", "MLX", "SOX4", "DDIT3"]
# ---- canonical candidates, links (self-promoter vs distal)
E = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t"); ac = [c for c in E.columns if c.startswith("ATAC.") and c.endswith(".readCount")][0]
E["key"] = E.chr + ":" + E.start.astype(str) + "-" + E.end.astype(str); NE = len(E); ekey = dict(zip(E.key, range(NE)))
L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "isSelfPromoter"])
L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ekey).astype(int); L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true")
GL = pd.read_csv(f"{R}/Neighborhoods/GeneList.txt", sep="\t", usecols=["name"]).name.astype(str).unique(); genes = np.array(sorted(set(GL))); gix = {g: i for i, g in enumerate(genes)}
mk = lambda d: sp.csr_matrix((np.ones(len(d)), (d.e.values, d.TargetGene.map(gix).values)), shape=(NE, len(genes)))
Mp = mk(L[L.self]); Md = mk(L[~L.self]); Mp.data[:] = 1; Md.data[:] = 1
def reach(el):
    if not len(el): z = np.zeros(len(genes), bool); return z, z, z
    p = np.asarray(Mp[el].sum(0)).ravel() > 0; d = np.asarray(Md[el].sum(0)).ravel() > 0; return p | d, p & ~d, d & ~p
CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in E.sort_values(["chr", "start"]).groupby("chr")}
pk_tfs = sorted(d[3:] for d in os.listdir(PK) if d.startswith("TF.")); B = {}; npk = {}
for t in pk_tfs:
    P = pd.read_csv(f"{PK}/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 8, 9]); P.columns = ["chr", "start", "q", "off"]; P = P[P.q >= QMIN]; npk[t] = len(P); el = []
    for c, s in zip(P.chr.values, (P.start + P.off).values):
        if c not in CH: continue
        st, en, ix = CH[c]; k = np.searchsorted(st, s, side="right") - 1
        if k >= 0 and s < en[k]: el.append(ix[k])
    B[t] = np.unique(np.array(el, int))
atac_order = np.argsort(-E[ac].values, kind="stable")
# ---- knockdown responses (r1 raw per-TF z, r2 double-centred as B6), panel-collapsed, genes >= 10% NTC detection
def load(a):
    d = np.load(f"{M4}/kd_response/out_v2/{ARMS[a]}/kd_response.npz", allow_pickle=True); q = pd.read_csv(f"{M4}/kd_response/out_v2/{ARMS[a]}/guide_qc.tsv", sep="\t"); u = d["units"].tolist(); ui = {x: k for k, x in enumerate(u)}
    uni = d["ntc_detect"] >= .1; g = d["genes"][uni].astype(str); Z = d["z"][:, uni].astype(float); prim = q[(q.kept == 1) & q.kind.isin(["all", "construct"])]; tf_ = sorted(prim.tf.unique())
    P_ = np.nan_to_num(np.vstack([np.nanmean(Z[[ui[x] for x in prim.unit[prim.tf == t]]], axis=0) for t in tf_])); R2 = P_ - P_.mean(0); R2 = R2 - R2.mean(1, keepdims=True)
    pts = sorted({panel(t) for t in tf_}); col = lambda M: np.vstack([M[[i for i, t in enumerate(tf_) if panel(t) == p]].mean(0) for p in pts])
    cells = {p: int(prim[prim.tf.map(panel) == p].n_used.sum()) for p in pts}
    s = pd.Series(np.arange(len(g)), index=g); s = s[~s.index.duplicated(keep="last")]
    return dict(g=s, r1=col(P_), r2=col(R2), pts=pts, mean=d["ntc_mean"][uni].astype(float), cells=cells, n_units=len(u), n_tf_matrix=len(tf_))
FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF"); FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf")
# promoter features for loadings: m7_step2 wide raw_hicfixed base (promoter ATAC, H3K27ac, CGI), genes of gene_split_wide
GW = pd.read_csv(f"{RG}/m7_step2/splits/gene_split_wide.tsv", sep="\t", usecols=["gene"]).gene.values; BW = np.load(f"{RG}/m7_step2/features/raw_hicfixed_wide/base.npz"); BWdf = pd.DataFrame(BW["base"], index=GW, columns=[str(x) for x in BW["names"]])
def fisher(inl, resp):
    a = int((inl & resp).sum()); b = int((inl & ~resp).sum()); c = int((~inl & resp).sum()); d = int((~inl & ~resp).sum()); return stats.fisher_exact([[a, b], [c, d]])[0]
Arows = []; Drows_v = []; Drows_l = []; Drows_t = []; ORr = []
for a in ARMS:
    D = load(a); uni = np.array([g in D["g"].index for g in genes]); ug = genes[uni]; kix = D["g"][ug].values; nu = len(ug)
    if a == "arm2":
        # ---- A (described on the Arm 2 universe)
        T = {}; Ta = {}
        for t in pk_tfs:
            al, po, do = reach(B[t]); al, po, do = al[uni], po[uni], do[uni]; T[t] = al
            aal, apo, ado = reach(atac_order[:len(B[t])]); aal, apo, ado = aal[uni], apo[uni], ado[uni]; Ta[t] = aal
            pt = panel(t); Arows.append(dict(label=LAB, tf=t, cells=FL.cells.get(pt, np.nan), tertile=FL.tertile.get(pt, ""), d=FL.d.get(pt, np.nan), n_peaks_q05=npk[t], n_bound_elements=len(B[t]), n_targets=int(al.sum()), frac_universe=al.mean(),
                frac_targets_promoter_only=po.sum() / max(al.sum(), 1), frac_targets_distal_only=do.sum() / max(al.sum(), 1), atac_ctrl_n_targets=int(aal.sum()), atac_ctrl_frac_universe=aal.mean(), atac_ctrl_frac_promoter_only=apo.sum() / max(aal.sum(), 1), atac_ctrl_frac_distal_only=ado.sum() / max(aal.sum(), 1),
                jaccard_with_atac_ctrl=(al & aal).sum() / max((al | aal).sum(), 1)))
        At = pd.DataFrame(Arows).sort_values("cells", ascending=False); At.to_csv(f"{O}/A_targets_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
        ok = [t for t in pk_tfs if T[t].sum() >= 20]; TM = np.vstack([T[t] for t in ok]).astype(float); TA = np.vstack([Ta[t] for t in ok]).astype(float)
        def jac(M):
            inter = M @ M.T; s = M.sum(1); un = s[:, None] + s[None, :] - inter; J = inter / np.maximum(un, 1); iu = np.triu_indices(len(M), 1); return J[iu]
        jt, ja = jac(TM), jac(TA); cnt = TM.sum(0)
        AS = [("universe_genes_arm2", nu), ("tfs_with_peaks", len(pk_tfs)), ("tfs_with_ge20_targets", len(ok)), ("median_n_targets", float(At.n_targets.median())), ("median_frac_universe", float(At.frac_universe.median())),
              ("frac_universe_target_of_gt_half_tfs", float((cnt > len(ok) / 2).mean())), ("frac_universe_target_of_ge1_tf", float((cnt >= 1).mean())), ("frac_universe_target_of_0_tfs", float((cnt == 0).mean())),
              ("median_jaccard_tf_pairs", float(np.median(jt))), ("iqr_jaccard_tf_pairs", f"{np.percentile(jt, 25):.3f}-{np.percentile(jt, 75):.3f}"), ("median_jaccard_atac_ctrl_pairs", float(np.median(ja))),
              ("median_frac_targets_promoter_only", float(At.frac_targets_promoter_only.median())), ("median_frac_targets_distal_only", float(At.frac_targets_distal_only.median())), ("median_jaccard_tf_vs_own_atac_ctrl", float(At.jaccard_with_atac_ctrl.median())),
              ("kd_matrix_units", D["n_units"]), ("kd_matrix_tfs", D["n_tf_matrix"])]
        pd.DataFrame([dict(label=LAB, key=k, value=v) for k, v in AS]).to_csv(f"{O}/A_summary.tsv", sep="\t", index=False)
    # ---- D: SVD on testable TFs with knockdown in this arm
    tt = [t for t in FT.index if t in D["pts"] and np.abs(D["r1"][D["pts"].index(t)]).sum() > 0]; Z = np.vstack([D["r1"][D["pts"].index(t)][kix] for t in tt])
    mu, sd = Z.mean(0), Z.std(0); sd[sd == 0] = 1; Zs = (Z - mu) / sd; U, S, Vt = np.linalg.svd(Zs, full_matrices=False); ve = S ** 2 / (S ** 2).sum()
    for k in range(10): Drows_v.append(dict(label=LAB, arm=a, n_tfs=len(tt), n_genes=nu, component=k + 1, var_explained=ve[k], cumulative=ve[:k + 1].sum()))
    feat = {"promoter_atac": BWdf.promoter_atac.reindex(ug).values, "promoter_h3k27ac": BWdf.promoter_h3k27ac.reindex(ug).values, "ntc_expression": D["mean"][kix], "cgi": BWdf.cgi_flag.reindex(ug).values}
    for k in range(5):
        r = dict(label=LAB, arm=a, component=k + 1)
        for f, v in feat.items(): m = np.isfinite(v); r[f"spearman_{f}"] = stats.spearmanr(Vt[k][m], v[m])[0]
        Drows_l.append(r)
    for i, t in enumerate(tt): Drows_t.append(dict(label=LAB, arm=a, tf=t, loading_c1=U[i, 0] * S[0], loading_c2=U[i, 1] * S[1], ceiling=FT.ceiling.get(t), kd_cells=D["cells"].get(t, np.nan)))
    resp = {"r1": Z, "r2": np.vstack([D["r2"][D["pts"].index(t)][kix] for t in tt])}
    for k in (1, 2, 3, 5): resp[f"r3_k{k}"] = Zs - (U[:, :k] * S[:k]) @ Vt[:k]
    np.savez(f"{O}/work/responses_{a}.npz", tfs=np.array(tt), genes=ug, **resp)
    # step-1 Fisher OR: canonical ABC targets (and ATAC-only control) vs top-5% |response|, for each reading
    for rn, Rm in resp.items():
        for i, t in enumerate(tt):
            pkn = [x for x in pk_tfs if panel(x) == t]
            if not pkn: continue
            al = reach(B[pkn[0]])[0][uni]
            if al.sum() < 20: continue
            rs = np.zeros(nu, bool); rs[np.argsort(-np.abs(Rm[i]), kind="stable")[:int(round(nu * .05))]] = True; aal = reach(atac_order[:len(B[pkn[0]])])[0][uni]
            ORr.append(dict(label=LAB, arm=a, reading=rn, tf=t, ceiling=FT.ceiling.get(t), n_targets=int(al.sum()), OR=fisher(al, rs), OR_atac_ctrl=fisher(aal, rs)))
pd.DataFrame(Drows_v).to_csv(f"{O}/D_variance.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(Drows_l).to_csv(f"{O}/D_loadings.tsv", sep="\t", index=False, float_format="%.4f")
DT = pd.DataFrame(Drows_t); DT.to_csv(f"{O}/D_tf_loading.tsv", sep="\t", index=False, float_format="%.4f")
TL = []
for a, G in DT.groupby("arm"):
    m = G.ceiling.notna(); TL.append(dict(label=LAB, arm=a, spearman_c1_vs_ceiling=stats.spearmanr(G.loading_c1[m], G.ceiling[m])[0], spearman_absc1_vs_ceiling=stats.spearmanr(G.loading_c1[m].abs(), G.ceiling[m])[0], spearman_c1_vs_kd_cells=stats.spearmanr(G.loading_c1, G.kd_cells)[0], spearman_absc1_vs_kd_cells=stats.spearmanr(G.loading_c1.abs(), G.kd_cells)[0], n_tfs=len(G)))
pd.DataFrame(TL).to_csv(f"{O}/D_tf_loading_summary.tsv", sep="\t", index=False, float_format="%.4f")
OR = pd.DataFrame(ORr); OR["OR_minus_atac"] = OR.OR - OR.OR_atac_ctrl; OR.to_csv(f"{O}/D_step1_or_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
rng = np.random.default_rng(20261111); PO = []
for (a, rn), G in OR.groupby(["arm", "reading"], sort=False):
    for sn, H in (("all", G), ("ceiling_ge_0.2", G[G.ceiling >= .2])):
        if not len(H): continue
        lo_ = np.log(H.OR.clip(lower=1e-3)).values; la = np.log(H.OR_atac_ctrl.clip(lower=1e-3)).values; dd = lo_ - la; b = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
        PO.append(dict(label=LAB, arm=a, reading=rn, tf_set=sn, n_tfs=len(H), median_OR=H.OR.median(), frac_OR_gt1=(H.OR > 1).mean(), median_OR_atac_ctrl=H.OR_atac_ctrl.median(), frac_OR_gt_atac_ctrl=(H.OR > H.OR_atac_ctrl).mean(), mean_logOR_minus_atac=dd.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
pd.DataFrame(PO).to_csv(f"{O}/D_step1_or_pooled.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(PO).to_string(index=False))
