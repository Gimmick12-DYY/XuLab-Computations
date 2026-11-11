# m10_promoter_regulome Amendment 3: H1-H3 (diagnostic, benchmark already read; no model fitting).
# usage: m10_h.py count <i>  -> work/h/hcount_<TF>.npz: TF i's site-proximal fragments per gene at every motif TF's promoter sites
#        m10_h.py eval       -> H1_within_motif.tsv, H1_pooled.tsv, H2_motif_within_top1000.tsv, H3_distal.tsv
import json, os, sys
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m10_promoter_regulome"; M9 = f"{RG}/m9_abc_diagnosis"; GW_ = f"{O}/work/g"; W = f"{O}/work/h"; os.makedirs(W, exist_ok=True)
LAB = "diagnostic, benchmark already read"; SEED = 20261111
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; MOT = sorted(f[:-5] for f in os.listdir(f"{M9}/work/motifs") if f.endswith(".meme"))
Pm = pd.read_csv(f"{M9}/work/promoters.bed", sep="\t", header=None, names=["c", "s", "e", "gene", "x", "strand"]); gi = pd.Series(np.arange(len(Pm)), index=Pm.gene)
def psites(t):
    f = f"{GW_}/psites_{t}.tsv"
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None
    S = pd.read_csv(f, sep="\t", header=None, names=["gene", "s", "t"]); S["g"] = S.gene.map(gi); S["ctr"] = Pm.s.values[S.g] + (S.s + S.t) // 2 - 1; S["c"] = Pm.c.values[S.g]; return S
if sys.argv[1] == "count":
    o = tfs79[int(sys.argv[2])]
    F = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{o}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"], dtype={"c": str, "s": np.int64, "e": np.int64, "st": str})
    pos = np.where(F.st.values == "+", F.s.values, F.e.values - 1); POS = {c: np.sort(pos[ix]) for c, ix in F.groupby("c", sort=False).indices.items()}; out = {}
    for t in MOT:
        S = psites(t); v = np.zeros(len(Pm))
        if S is not None:
            for g, H in S.groupby("g"):
                c = H.c.iloc[0]
                if c not in POS: continue
                s = np.sort(H.ctr.values); ms = []
                for x in s:
                    if ms and x - 100 <= ms[-1][1]: ms[-1][1] = x + 100
                    else: ms.append([x - 100, x + 100])
                ms = np.array(ms); q = POS[c]; v[g] = int((np.searchsorted(q, ms[:, 1], side="right") - np.searchsorted(q, ms[:, 0], side="left")).sum())
        out[t] = v
    np.savez(f"{W}/hcount_{o}.npz", **out); print(o, flush=True)
else:
    rng = np.random.default_rng(SEED); FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF")
    MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
    HC = {o: np.load(f"{W}/hcount_{o}.npz") for o in tfs79}
    Rz = np.load(f"{M9}/work/responses_arm2.npz", allow_pickle=True); tt = list(Rz["tfs"]); ug = pd.Index(Rz["genes"].astype(str)); n = len(ug); pidx = pd.Index(Pm.gene).get_indexer(ug); assert (pidx >= 0).all()
    GLf = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); acol = [c for c in GLf.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]; atac = GLf[acol].reindex(ug).fillna(0).values
    TESTABLE = sorted(set(pd.read_csv(f"{O}/F1_testable.tsv", sep="\t").query("arm == 'arm2'").tf))
    CT = {t: np.load(f"{O}/work/counts/{t}.npz", allow_pickle=True) for t in tfs79}; pg = pd.Index(CT[tfs79[0]]["genes"].astype(str)); cidx = pg.get_indexer(ug)
    def resp_of(t): y = Rz["r1"][tt.index(t)]; m = np.zeros(n, bool); m[np.argsort(-np.abs(y), kind="stable")[:int(round(n * .05))]] = True; return m
    def or2(top, resp): a = (top & resp).sum(-1); b = (top & ~resp).sum(-1); c = (~top & resp).sum(-1); d = (~top & ~resp).sum(-1); return (a + .5) * (d + .5) / ((b + .5) * (c + .5))
    def topfrac(v, f): k = int(round(len(v) * f)); o = np.lexsort((rng.random(len(v)), -v)); m = np.zeros(len(v), bool); m[o[:k]] = True; return m
    def ci(top, resp): bi = rng.integers(0, len(top), (1000, len(top))); b = or2(top[bi], resp[bi]); return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    H1 = []
    for t in MOT:
        if t not in tt: continue
        own = HC[t][t][pidx]; S = psites(t)
        if S is None: continue
        M = np.zeros(n, bool); M[pd.Index(ug).get_indexer(Pm.gene.values[S.g.unique()])[pd.Index(ug).get_indexer(Pm.gene.values[S.g.unique()]) >= 0]] = True
        if M.sum() < 40: continue
        resp = resp_of(t)[M]; a_ = own[M]; pool = sum(HC[o][t][pidx] for o in tfs79 if o != t)[M].astype(np.int64); K = int(a_.sum())
        sw = MIS.get(t); c_ = HC[sw][t][pidx][M] if sw in HC else None
        fm = pd.read_csv(f"{M9}/work/fimo/{t}.tsv", sep="\t", header=None, names=["motif", "gene", "score", "p"]); fm = fm[~fm.motif.str.contains("_shuf")].groupby("gene").score.max().reindex(ug).values[M]
        for frac, lab in ((.5, "top_half"), (.25, "top_quartile")):
            r = dict(label=LAB, tf=t, testable=t in TESTABLE, n_motif_genes=int(M.sum()), n_responders_in_M=int(resp.sum()), own_fragments_total=K, split=lab)
            ta = topfrac(a_.astype(float), frac); r["OR_a_own"] = float(or2(ta, resp)); r["a_lo"], r["a_hi"] = ci(ta, resp)
            if K > 0 and pool.sum() >= K:
                ob = [or2(topfrac(rng.multivariate_hypergeometric(pool, K).astype(float), frac), resp) for _ in range(200)]; r["OR_b_pooled_thinned"] = float(np.median(ob))
            if c_ is not None: r["swap_tf"] = sw; r["OR_c_swapped"] = float(or2(topfrac(c_.astype(float), frac), resp))
            r["OR_d_atac"] = float(or2(topfrac(atac[M], frac), resp)); r["OR_e_motif_score"] = float(or2(topfrac(np.nan_to_num(fm, nan=-1e3), frac), resp))
            r["reads_add_to_motif"] = bool(r["a_lo"] > max(r.get("OR_b_pooled_thinned", np.inf), r.get("OR_c_swapped", np.inf), r["OR_d_atac"]))
            H1.append(r)
    H1 = pd.DataFrame(H1); H1.to_csv(f"{O}/H1_within_motif.tsv", sep="\t", index=False, float_format="%.4f"); P = []
    for lab, Q in H1.groupby("split"):
        for nm, col in (("a_minus_b", "OR_b_pooled_thinned"), ("a_minus_c", "OR_c_swapped"), ("a_minus_d", "OR_d_atac"), ("a_minus_e", "OR_e_motif_score")):
            d = (Q.OR_a_own - Q[col]).dropna().values; b = np.median(d[rng.integers(0, len(d), (2000, len(d)))], axis=1)
            P.append(dict(label=LAB, split=lab, comparison=nm, n_tfs=len(d), median=float(np.median(d)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n_reads_add=int(Q.reads_add_to_motif.sum())))
    pd.DataFrame(P).to_csv(f"{O}/H1_pooled.tsv", sep="\t", index=False, float_format="%.4f")
    # H2
    H2 = []
    for t in MOT:
        if t not in tt: continue
        S = psites(t)
        if S is None: continue
        mot = np.zeros(n, bool); ix = pd.Index(ug).get_indexer(Pm.gene.values[S.g.unique()]); mot[ix[ix >= 0]] = True
        w = CT[t]["w500"][cidx].astype(float); top = np.zeros(n, bool); top[np.lexsort((rng.random(n), -w))[:1000]] = True; resp = resp_of(t)[top]; m = mot[top]
        bi = rng.integers(0, top.sum(), (1000, top.sum())); b = or2(m[bi], resp[bi])
        H2.append(dict(label=LAB, tf=t, testable=t in TESTABLE, n_top1000_with_motif=int(m.sum()), n_without=int((~m).sum()), resp_rate_with=float(resp[m].mean()) if m.any() else np.nan, resp_rate_without=float(resp[~m].mean()) if (~m).any() else np.nan,
                       OR_with_vs_without=float(or2(m, resp)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
    H2 = pd.DataFrame(H2); d = H2.OR_with_vs_without.values; b = np.median(d[rng.integers(0, len(d), (2000, len(d)))], axis=1)
    H2 = pd.concat([H2, pd.DataFrame([dict(label=LAB, tf="POOLED_median", OR_with_vs_without=float(np.median(d)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))])]); H2.to_csv(f"{O}/H2_motif_within_top1000.tsv", sep="\t", index=False, float_format="%.4f")
    # H3: distal own-motif sites (Amendment 2 site lists, open chromatin, not within 1 kb of a TSS) linked by canonical ABC distal links
    Z = np.load(f"{GW_}/lists.npz", allow_pickle=True); R = f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed"; EL = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"])
    LK = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "isSelfPromoter"])
    LK = LK[~LK.isSelfPromoter.astype(str).str.lower().eq("true") & LK.TargetGene.isin(ug)]; ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}; LK["e"] = (LK.chr + ":" + LK.start.astype(str) + "-" + LK.end.astype(str)).map(ek)
    CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in EL.sort_values(["chr", "start"]).groupby("chr")}; GC = {o: np.load(f"{GW_}/count_{o}.npz") for o in tfs79}; H3 = []
    for t in ("CTCF", "NRF1", "ZNF143"):
        nm = f"motif_{t}"; c, p, pr = Z[f"{nm}|chr"], Z[f"{nm}|pos"], Z[f"{nm}|prom"]; el = np.full(len(p), -1)
        for ch in np.unique(c):
            if ch not in CH: continue
            st, en, ix = CH[ch]; m = c == ch; k = np.searchsorted(st, p[m], side="right") - 1; ok = (k >= 0) & (p[m] < en[np.clip(k, 0, None)]); e_ = np.full(m.sum(), -1); e_[ok] = ix[k[ok]]; el[m] = e_
        keep = (~pr) & (el >= 0); own_site = GC[t][f"{nm}|site"]; pool_site = sum(GC[o][f"{nm}|site"].astype(np.int64) for o in tfs79 if o != t)
        df = pd.DataFrame(dict(e=el[keep], own=own_site[keep], pool=pool_site[keep])).groupby("e").sum(); L2 = LK[LK.e.isin(df.index)]; L2 = L2.join(df, on="e"); gs = L2.groupby("TargetGene")[["own", "pool"]].sum().reindex(ug)
        M = gs.own.notna().values; resp = resp_of(t)[M]; a_ = gs.own.values[M]; pool = gs.pool.values[M].astype(np.int64); K = int(a_.sum())
        ta = topfrac(a_.astype(float), .5); r = dict(label=LAB, tf=t, n_genes_linked_to_distal_motif_site=int(M.sum()), n_responders=int(resp.sum()), OR_own=float(or2(ta, resp)))
        r["lo"], r["hi"] = ci(ta, resp); r["OR_pooled_thinned"] = float(np.median([or2(topfrac(rng.multivariate_hypergeometric(pool, K).astype(float), .5), resp) for _ in range(200)])) if K > 0 and pool.sum() >= K else np.nan
        r["OR_atac"] = float(or2(topfrac(atac[M], .5), resp)); H3.append(r)
    pd.DataFrame(H3).to_csv(f"{O}/H3_distal.tsv", sep="\t", index=False, float_format="%.4f"); print("done")
