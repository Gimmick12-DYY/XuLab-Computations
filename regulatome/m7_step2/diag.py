# m7_step2 Amendment 6: post-hoc EXPLORATORY diagnostics D1-D3, declared after the test chromosomes were read. Cannot change any verdict of section 7.
# No model of the frozen set is refitted or changed. D1 uses train and validation chromosomes only. D2 reads per_tf_test.tsv only. D3 scores unfitted references.
# usage: diag.py <d1|d2|d3>  -> diagnostics_exploratory/*.tsv, *.json
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome"; O = f"{R}/m7_step2"; D = f"{O}/diagnostics_exploratory"; os.makedirs(D, exist_ok=True)
LAB = "EXPLORATORY (Amendment 6, post-hoc)"; NB = 2000; SEED = 20261111
TESTC, VALC = {"chr3", "chr11", "chr17", "chr20"}, {"chr5", "chr12"}
PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
ABC = {"powerlaw": f"{R}/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", "hicfixed": f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv"}
def chrset(c): return "test" if c in TESTC else ("val" if c in VALC else "train")
def boot_cols(M, rng):  # M: TFs x stats; paired bootstrap over TFs (same draws for every column)
    n = len(M); idx = rng.integers(0, n, (NB, n)); out = []
    for j in range(M.shape[1]):
        v = M[:, j]; b = v[idx].mean(1); out.append((float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))))
    return out
def fmt(t): return f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def elements(link):
    """active elements with ATAC CPM and H3K27ac for one link set, as build_features.py"""
    if link == "hicfixed":
        CL = pd.read_csv(f"{R}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end", "ATAC.RPM", "H3K27ac.RPM"]).rename(columns={"chr": "chrom"})
        return CL[["chrom", "start", "end"]], CL["ATAC.RPM"].values.astype(float), CL["H3K27ac.RPM"].values.astype(float)
    E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); act = E_.cls.isin(["promoter", "distal_active", "bivalent"]).values; Ea = E_[act].reset_index(drop=True)
    atac = np.load(f"{R}/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float); atac = atac / atac.sum() * 1e6
    coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
    K = pd.read_csv(f"{R}/m2b_abc_k27ac/results/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end", "H3K27ac.RPM"]); K["e"] = (K.chr + ":" + K.start.astype(str) + "-" + K.end.astype(str)).map(coord); K = K.dropna(subset=["e"]); k27 = np.zeros(len(Ea)); k27[K.e.astype(int).values] = K["H3K27ac.RPM"].values
    return Ea[["chrom", "start", "end"]], atac, k27

def cgi_overlap(Ea):
    cg = pd.read_csv(f"{O}/annotation/cpgIslandExt.txt.gz", sep="\t", header=None, usecols=[1, 2, 3], names=["c", "s", "e"], compression="gzip"); ov = np.zeros(len(Ea), bool)
    for c, H in cg.groupby("c"):
        ix = np.flatnonzero(Ea.chrom.values == c)
        if not len(ix): continue
        Hs = H.sort_values("s"); hi = np.searchsorted(Hs.s.values, Ea.end.values[ix], side="left"); cm = np.maximum.accumulate(Hs.e.values); ok = hi > 0; ov[ix[ok]] = cm[hi[ok] - 1] > Ea.start.values[ix[ok]]
    return ov

def rung1_features(link, genes_keep):
    """rung-1 columns (promoter ATAC, promoter H3K27ac, distal ABC sum, n links, CGI flag) for an arbitrary gene list, as build_features.py, unstandardised"""
    Ea, atac, k27 = elements(link); coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
    gidx = pd.Series(np.arange(len(genes_keep)), index=genes_keep)
    A = pd.read_csv(ABC[link], sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"]); A = A[A.TargetGene.isin(gidx.index)].copy()
    A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord); n_unmapped = int(A.e.isna().sum()); A = A.dropna(subset=["e"]); A["e"] = A.e.astype(int); A["g"] = A.TargetGene.map(gidx).values; A["self"] = A.isSelfPromoter.astype(str).str.lower().eq("true").values
    NG, NE = len(genes_keep), len(Ea)
    Lp = sp.csr_matrix((np.ones(int(A.self.sum())), (A.g[A.self], A.e[A.self])), shape=(NG, NE))
    ABCd = A[~A.self].groupby("g")["ABC.Score"].sum().reindex(np.arange(NG)).fillna(0).values; nl = A.groupby("g").size().reindex(np.arange(NG)).fillna(0).values
    cgi = np.asarray(Lp @ cgi_overlap(Ea).astype(float)).ravel() > 0
    M = np.column_stack([np.log1p(np.asarray(Lp @ atac).ravel()), np.log1p(np.asarray(Lp @ k27).ravel()), np.log1p(ABCd), np.log1p(nl), cgi.astype(float)])
    return M, n_unmapped

def d1():
    import m7
    from sklearn.linear_model import ElasticNet
    d = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True)
    allmean = pd.Series(d["ntc_mean"].astype(float), index=d["genes"].astype(str)); allmean = allmean[~allmean.index.duplicated(keep="last")]
    y = m7.expr_target(); res = {"label": LAB}
    fin = np.isfinite(y); q = np.percentile(y[fin], [0, 25, 50, 75, 100])
    res["i_target_distribution_universe"] = dict(n_universe=int(m7.NG), n_with_target=int(fin.sum()), n_without_target_arm1_only=int((~fin).sum()), min=q[0], q25=q[1], median=q[2], q75=q[3], max=q[4], fraction_zero=float((y[fin] == 0).mean()), target="NTC mean log1p(CP10k), Arm 2, genes >= 10% NTC detection")
    q = np.percentile(allmean.values, [0, 25, 50, 75, 100]); res["i_target_distribution_all_measured_genes"] = dict(n=int(len(allmean)), min=q[0], q25=q[1], median=q[2], q75=q[3], max=q[4], fraction_zero=float((allmean.values == 0).mean()))
    tr = m7.set_mask("train", "arm2") & fin; va = m7.set_mask("val", "arm2") & fin
    for link in ("hicfixed", "powerlaw"):
        r = {}; Bz = np.load(f"{O}/features/raw_{link}/base.npz"); base = Bz["base"]; names = [str(x) for x in Bz["names"]]
        r["ii_spearman_with_target_train_genes"] = {n: float(spearmanr(base[tr, j], y[tr])[0]) for j, n in enumerate(names)}; r["ii_n_train_genes"] = int(tr.sum())
        r["iii_val_r2_as_fitted"] = {f"{m}_rung{k}": json.load(open(f"{O}/fits/raw_{link}/final/expr/{m}_rung{k}_real/val.json"))["val_r2"] for m in ("enet", "lgb", "mlp") for k in (1, 2)}
        p = json.load(open(f"{O}/frozen/params_{link}.json"))["expr|enet|rung1"]
        # (iv) same rung-1 elastic net, frozen parameters, refitted on train and scored on validation: (a) the gene universe recomputed here, (b) all ABC genes on train/val chromosomes, undetected genes at their NTC mean, 0 if absent
        A = pd.read_csv(ABC[link], sep="\t", usecols=["chr", "TargetGene"]).drop_duplicates("TargetGene"); A = A[A.chr.isin(PRIMARY)]; A["set"] = A.chr.map(chrset); A = A[A.set != "test"]
        wide = A.TargetGene.values; Mw, nun = rung1_features(link, wide); yw = allmean.reindex(wide).fillna(0.0).values; sw = A.set.values
        det = pd.Series(d["ntc_detect"].astype(float), index=d["genes"].astype(str)); det = det[~det.index.duplicated(keep="last")].reindex(wide).fillna(0).values >= .1
        ins = pd.Series(wide).isin(m7.GENES[tr | va]).values
        out = {}
        for nm, keep in (("a_universe_genes_recomputed", ins), ("b_all_abc_genes_widened", np.ones(len(wide), bool)), ("c_abc_genes_detected_ge10pct", det)):
            t_ = keep & (sw == "train"); v_ = keep & (sw == "val"); mu, sd = Mw[t_].mean(0), Mw[t_].std(0); sd[sd == 0] = 1; Z = (Mw - mu) / sd; Z[:, 4] = Mw[:, 4]
            m = ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=5000, tol=1e-4).fit(Z[t_], yw[t_])
            out[nm] = dict(n_train=int(t_.sum()), n_val=int(v_.sum()), val_r2=float(r2(yw[v_], m.predict(Z[v_]))), frac_target_zero_val=float((yw[v_] == 0).mean()), frac_absent_from_matrix_val=float((~pd.Series(wide[v_]).isin(allmean.index)).mean()), spearman_promoter_atac_train=float(spearmanr(Mw[t_, 0], yw[t_])[0]))
        out["n_links_with_unmapped_element"] = nun; r["iv_widened_universe_rung1_enet"] = out
        # (v) promoter feature taken at the gene's own TSS element
        Ea, atac, k27 = elements(link); coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}
        S = pd.read_csv(ABC[link], sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "isSelfPromoter"]); S = S[S.TargetGene.isin(m7.GENES)]; S = S[S.isSelfPromoter.astype(str).str.lower().eq("true")].copy()
        S["dist"] = np.where((S.TargetGeneTSS >= S.start) & (S.TargetGeneTSS < S.end), 0, np.minimum((S.TargetGeneTSS - S.start).abs(), (S.TargetGeneTSS - S.end).abs()))
        gd = S.groupby("TargetGene").dist.min()
        # independent join: the element whose interval contains the gene's TSS, found by coordinates, versus the promoter_atac feature
        T = pd.read_csv(ABC[link], sep="\t", usecols=["chr", "TargetGene", "TargetGeneTSS"]).drop_duplicates("TargetGene").set_index("TargetGene").reindex(m7.GENES)
        cont = np.full(m7.NG, -1)
        for c, H in Ea.reset_index().groupby("chrom"):
            o = np.argsort(H.start.values); st, en, ii = H.start.values[o], H.end.values[o], H["index"].values[o]
            gi = np.flatnonzero(T.chr.values == c); ts = T.TargetGeneTSS.values[gi]; k = np.searchsorted(st, ts, side="right") - 1; ok = (k >= 0) & (ts < en[np.maximum(k, 0)])
            cont[gi[ok]] = ii[k[ok]]
        has = cont >= 0; tss_atac = np.where(has, np.log1p(atac[np.maximum(cont, 0)]), np.nan)
        rho = float(spearmanr(tss_atac[has], base[has, 0])[0])
        S["e"] = (S.chr + ":" + S.start.astype(str) + "-" + S.end.astype(str)).map(coord); pairs = set(zip(S.TargetGene, S.e.astype("Int64")))
        same = int(sum((gname, int(e)) in pairs for gname, e in zip(m7.GENES[has], cont[has])))
        r["v_promoter_at_own_tss"] = dict(genes_with_self_promoter_link=int(len(gd)), genes_without=int(m7.NG - len(gd)), self_element_contains_tss=int((gd == 0).sum()), within_1kb=int((gd <= 1000).sum()), max_distance_bp=int(gd.max()), genes_with_an_element_containing_tss=int(has.sum()), tss_element_is_one_of_the_self_promoter_elements=same, spearman_tss_element_atac_vs_promoter_atac_feature=rho)
        res[link] = r
    json.dump(res, open(f"{D}/d1_expression_audit.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))

def d2():
    M = pd.read_csv(f"{O}/per_tf_test.tsv", sep="\t"); M = M[(M.split == "gene") & (M.arm == "arm2") & (M.centring == "train") & (M.model == "enet") & (M["mode"] == "kd_only")]
    rng = np.random.default_rng(SEED); rows = []
    for kind, G in M.groupby("kind"):
        for sn, sel in (("claim19", G.claim_tier.astype(bool)), ("all76", np.ones(len(G), bool)), ("ceiling_ge_0.2", G.ceiling >= 0.2)):
            H = G[np.asarray(sel)]; V = np.column_stack([H.real - H.base, H.pseudo - H.base, H.mis - H.base, H.real - H.mis]); b = boot_cols(V, rng)
            rows.append(dict(label=LAB, kind=kind, tf_set=sn, n_tfs=len(H), real_minus_base=fmt(b[0]), pseudo_minus_base=fmt(b[1]), mis_minus_base=fmt(b[2]), real_minus_mis=fmt(b[3])))
    T = pd.DataFrame(rows); T.to_csv(f"{D}/d2_claim_tier_controls.tsv", sep="\t", index=False); print(T.to_string(index=False))

def d3():
    os.environ["M7_LINK"] = "powerlaw"; import m7
    tfs = m7.tfs; NT = len(tfs); NG = m7.NG; G = m7.GS
    # panel build (power-law candidate elements), as build_features.py references
    E_ = pd.read_csv(f"{R}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"]); act = E_.cls.isin(["promoter", "distal_active", "bivalent"]).values; Ea = E_[act].reset_index(drop=True); NE = len(Ea)
    B = np.column_stack([(lambda y: y / max(y.sum(), 1e-12) * 1e6)(np.asarray(sp.load_npz(f"{R}/gate1/cellmats/{t}.raw.npz").sum(0)).ravel().astype(float)) for t in tfs])
    atac_raw = np.load(f"{R}/m1_binding/v2/counts/atac_counts.npz")["atac"][np.flatnonzero(act)].astype(float); atac = atac_raw / atac_raw.sum() * 1e6
    A = pd.read_csv(ABC["powerlaw"], sep="\t", usecols=["TargetGene", "TargetGeneTSS"]); A = A[A.TargetGene.isin(m7.GENES)].drop_duplicates("TargetGene").set_index("TargetGene").TargetGeneTSS.reindex(m7.GENES).values
    mid = ((Ea.start + Ea.end) // 2).values; ro, co = [], []
    for ch, Gs in G.groupby("chromosome"):
        ix = np.flatnonzero(Ea.chrom.values == ch); o = np.argsort(mid[ix]); ix = ix[o]; m = mid[ix]; gi = m7.GIDX[Gs.gene.values].values; t_ = A[gi]
        lo = np.searchsorted(m, t_ - 100000); hi = np.searchsorted(m, t_ + 100000, side="right")
        for g_, a_, b_ in zip(gi, lo, hi): ro += [g_] * int(b_ - a_); co += list(ix[int(a_):int(b_)])
    W = sp.csr_matrix((np.ones(len(ro)), (ro, co)), shape=(NG, NE))
    win = np.asarray(W @ B).T; refs = np.load(f"{O}/features/raw_powerlaw/refs.npz"); assert np.allclose(win, refs["win100k"], rtol=1e-3, atol=1e-2), "window sum does not reproduce refs.npz"
    v8 = refs["v8"].astype(float); watac = np.asarray(W @ atac).ravel(); pool = B.sum(1); wpool = np.asarray(W @ pool).ravel()
    qc = json.load(open(f"{O}/features/raw_powerlaw/feature_qc.json")); MIS = {d: [tfs.index(x) for x in qc["mis_paired_map"][f"mis{d}"]] for d in range(5)}
    dec = pd.qcut(pd.Series(atac_raw).rank(method="first"), 10, labels=False).values; DEC = [np.flatnonzero(dec == k) for k in range(10)]
    def pseudo_col(i, d):
        rng = np.random.default_rng(SEED + 1000 * (d + 1) + i); b = B[:, i].copy()
        for ix in DEC: b[ix] = B[ix[rng.permutation(len(ix))], i]
        return b
    Zp = m7.load_kd("arm2")[0]; arm_tfs = [t for t in m7.TESTABLE if np.isfinite(Zp[m7.PTF[t]]).any()]
    trm = m7.set_mask("train", "arm2"); tem = m7.set_mask("test", "arm2")
    Y = {"train": m7.targets(Zp, arm_tfs, "train", trm), "raw": m7.targets(Zp, arm_tfs, "raw", trm)}
    tt_tr, gg_tr = m7.kd_rows(arm_tfs, trm, Zp); tt_te, gg_te = m7.kd_rows(arm_tfs, tem, Zp)
    PT = pd.read_csv(f"{O}/per_tf_test.tsv", sep="\t"); PT = PT[(PT.split == "gene") & (PT.arm == "arm2") & (PT.model == "enet") & (PT["mode"] == "kd_only")]
    rows = []
    for t in arm_tfs:
        ti = m7.PTF[t]; i = m7.TF_IDX[t]; a_tr = gg_tr[tt_tr == ti]; a_te = gg_te[tt_te == ti]; zte = Zp[ti, a_te]
        preds = {"a_own_win100k": [win[i]], "b_v8": [v8[i]], "c_atac_win100k": [watac], "d_pool_other78_win100k": [wpool - win[i]], "e_mispaired_win100k": [win[MIS[d][i]] for d in range(5)], "f_pseudo_win100k": [np.asarray(W @ pseudo_col(i, d)).ravel() for d in range(5)]}
        r = dict(tf=t)
        for cen in ("train", "raw"):
            ztr = Y[cen][ti, a_tr]
            for k, L in preds.items():
                vals = []
                for p in L:
                    s = np.sign(m7.rank_sp(p[a_tr], ztr)); vals.append(s * m7.rank_sp(p[a_te], zte))
                r[f"{k}__{cen}"] = float(np.mean(vals))
            for kind in ("raw_hicfixed", "raw_powerlaw"):
                q = PT[(PT.kind == kind) & (PT.centring == cen) & (PT.tf == t)]; r[f"g_enet_base_{kind}__{cen}"] = float(q.base.iloc[0]) if len(q) else np.nan
        rows.append(r)
    T = pd.DataFrame(rows).set_index("tf"); F = pd.read_csv(f"{O}/splits/tf_folds.tsv", sep="\t").set_index("TF"); T["claim_tier"] = F.claim_tier.reindex(T.index).astype(bool); T["ceiling"] = F.ceiling.reindex(T.index)
    T.insert(0, "label", LAB); T.to_csv(f"{D}/d3_signed_refs_per_tf.tsv", sep="\t", float_format="%.5f")
    rng = np.random.default_rng(SEED + 3); rows = []
    for cen in ("train", "raw"):
        for sn, sel in (("all76", np.ones(len(T), bool)), ("claim19", T.claim_tier.values), ("ceiling_ge_0.2", (T.ceiling >= 0.2).values)):
            H = T[sel]; a = H[f"a_own_win100k__{cen}"].values
            cols = {"a_own": a, "b_v8": H[f"b_v8__{cen}"].values, "c_atac": H[f"c_atac_win100k__{cen}"].values, "d_pool_other78": H[f"d_pool_other78_win100k__{cen}"].values, "e_mispaired": H[f"e_mispaired_win100k__{cen}"].values, "f_pseudo": H[f"f_pseudo_win100k__{cen}"].values,
                    "enet_base_hicfixed": H[f"g_enet_base_raw_hicfixed__{cen}"].values, "enet_base_powerlaw": H[f"g_enet_base_raw_powerlaw__{cen}"].values,
                    "a_minus_c": a - H[f"c_atac_win100k__{cen}"].values, "a_minus_d": a - H[f"d_pool_other78_win100k__{cen}"].values, "a_minus_e": a - H[f"e_mispaired_win100k__{cen}"].values, "a_minus_f": a - H[f"f_pseudo_win100k__{cen}"].values,
                    "a_minus_enet_base_hicfixed": a - H[f"g_enet_base_raw_hicfixed__{cen}"].values, "a_minus_enet_base_powerlaw": a - H[f"g_enet_base_raw_powerlaw__{cen}"].values}
            b = boot_cols(np.column_stack(list(cols.values())), rng); rows.append(dict(label=LAB, centring=cen, tf_set=sn, n_tfs=len(H), **{k: fmt(x) for k, x in zip(cols, b)}))
    S = pd.DataFrame(rows); S.to_csv(f"{D}/d3_signed_refs_pooled.tsv", sep="\t", index=False); print(S.to_string(index=False))

{"d1": d1, "d2": d2, "d3": d3}[sys.argv[1]]()
