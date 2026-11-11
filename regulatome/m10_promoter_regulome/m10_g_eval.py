# m10_promoter_regulome Amendment 2: G1-G3 tables from work/g/count_<TF>.npz (diagnostic, benchmark already read; no model fitting).
# usage: m10_g_eval.py -> G1_signal_fraction.tsv, G2_lists.tsv, G2_ctcf_recovery.tsv, G3_ctcf_groups.tsv
import json, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m10_promoter_regulome"; M9 = f"{RG}/m9_abc_diagnosis"; W = f"{O}/work/g"; LAB = "diagnostic, benchmark already read"; SEED = 20261111
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; FL = pd.read_csv(f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv", sep="\t").set_index("tf")
MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
C = {t: np.load(f"{W}/count_{t}.npz", allow_pickle=True) for t in tfs79}; names = sorted({k.split("|")[0] for k in C[tfs79[0]].files if "|" in k and not k.split("|")[0].endswith("__bg")})
rng = np.random.default_rng(SEED)
# ---- G1
G1 = []; Z = np.load(f"{W}/lists.npz", allow_pickle=True)
for t in tfs79:
    own = [nm for nm in names if nm == f"motif_{t}" or nm.startswith(f"chip_{t}_")]
    for nm in own:
        z = C[t]; tot = float(z["total"]); otot = sum(float(C[o]["total"]) for o in tfs79 if o != t)
        pr = Z[f"{nm}|prom"]; bpr = Z[f"{nm}__bg|prom"]
        for sub, m, bm in (("all", np.ones(len(pr), bool), np.ones(len(bpr), bool)), ("promoter", pr, bpr), ("distal", ~pr, ~bpr)):
            if m.sum() < 20 or bm.sum() < 20: continue
            s_own = z[f"{nm}|site"][m].astype(float); s_bg = z[f"{nm}__bg|site"][bm].astype(float); s_pool = sum(C[o][f"{nm}|site"][m].astype(float) for o in tfs79 if o != t)
            f_own = z[f"{nm}|union_{ {'all': 'all', 'promoter': 'prom', 'distal': 'dist'}[sub] }"] / tot
            f_pool = sum(C[o][f"{nm}|union_{ {'all': 'all', 'promoter': 'prom', 'distal': 'dist'}[sub] }"] for o in tfs79 if o != t) / otot
            f_bg = z[f"{nm}__bg|union_{ {'all': 'all', 'promoter': 'prom', 'distal': 'dist'}[sub] }"] / tot
            e_bg = (s_own.mean() / max(s_bg.mean(), 1e-12)); e_pool = (s_own.sum() / tot) / max(s_pool.sum() / otot, 1e-12)
            bi = rng.integers(0, len(s_own), (1000, len(s_own))); bj = rng.integers(0, len(s_bg), (1000, len(s_bg)))
            eb = s_own[bi].mean(1) / np.maximum(s_bg[bj].mean(1), 1e-12); ep = (s_own[bi].sum(1) / tot) / np.maximum(s_pool[bi].sum(1) / otot, 1e-12)
            G1.append(dict(label=LAB, tf=t, site_list=nm, sites=sub, n_sites=int(m.sum()), cells=FL.cells.get(t, np.nan), total_fragments=tot, frac_fragments_on_sites=f_own, frac_pooled_other78_on_sites=f_pool, frac_on_background_positions=f_bg,
                           enrichment_vs_background=e_bg, eb_lo=float(np.percentile(eb, 2.5)), eb_hi=float(np.percentile(eb, 97.5)), enrichment_vs_pooled_other78=e_pool, ep_lo=float(np.percentile(ep, 2.5)), ep_hi=float(np.percentile(ep, 97.5))))
pd.DataFrame(G1).to_csv(f"{O}/G1_signal_fraction.tsv", sep="\t", index=False, float_format="%.5g"); print("G1", len(G1), flush=True)
# ---- G2
Rz = np.load(f"{M9}/work/responses_arm2.npz", allow_pickle=True); tt = list(Rz["tfs"]); ug = pd.Index(Rz["genes"].astype(str)); n = len(ug)
GLf = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t").drop_duplicates("name", keep="last").set_index("name"); acol = [c for c in GLf.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]
atac = GLf[acol].reindex(ug).fillna(0).values; GW = pd.read_csv(f"{RG}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); BW = np.load(f"{RG}/m7_step2/features/raw_hicfixed_wide/base.npz"); cgi = pd.Series(BW["base"][:, 4], index=GW.gene).reindex(ug).fillna(0).values
dec = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values; strat = dec * 2 + (cgi > 0); SI = [np.flatnonzero(strat == s) for s in range(20)]; BIDX = rng.integers(0, n, (1000, n))
CT = {t: np.load(f"{O}/work/counts/{t}.npz", allow_pickle=True) for t in tfs79}; pg = pd.Index(CT[tfs79[0]]["genes"].astype(str)); pidx = pg.get_indexer(ug); assert (pidx >= 0).all()
pgi = pd.Index(C[tfs79[0]]["prom_genes"].astype(str)).get_indexer(ug); assert (pgi >= 0).all()
tot = {t: float(CT[t]["total"]) for t in tfs79}; w500 = {t: CT[t]["w500"][pidx].astype(float) for t in tfs79}; allw = sum(w500.values()); allt = sum(tot.values())
def or_(tg, resp): a = (tg & resp).sum(-1); b = (tg & ~resp).sum(-1); c = resp.sum(-1) - a; d = tg.shape[-1] - a - b - c; return (a + .5) * (d + .5) / ((b + .5) * (c + .5))
def topk(v, N=1000):
    m = np.zeros(len(v), bool); o = np.lexsort((rng.random(len(v)), -v)); k = min(N, int((v > 0).sum())); m[o[:k]] = True; return m
def lists_for(t):
    L = {}
    if t not in tfs79: return L
    L["i_peak_within_1kb"] = C[t]["peak_within_1kb"][pgi].astype(bool)
    pm = C[t]["prom_motif_counts"][pgi].astype(float)
    if pm.sum() > 0: L["ii_motif_proximal_fragments"] = topk(pm)
    cp = w500[t] / tot[t] * 1e6; po = (allw - w500[t]) / (allt - tot[t]) * 1e6; L["iii_ratio_to_pooled_other78"] = topk((cp + 1) / (po + 1))
    return L
def resp_of(t): y = Rz["r1"][tt.index(t)]; m = np.zeros(n, bool); m[np.argsort(-np.abs(y), kind="stable")[:int(round(n * .05))]] = True; return m
TESTABLE = sorted(set(pd.read_csv(f"{O}/F1_testable.tsv", sep="\t").query("arm == 'arm2'").tf)); G2 = []; REC = []
p_atac = (atac + 1e-3) / (atac + 1e-3).sum()
BKs = dict(np.load(f"{M9}/work/scores_bulk.npz", allow_pickle=True)); bg = pd.Index(BKs.pop("genes").astype(str))
for t in TESTABLE:
    resp = resp_of(t); L = lists_for(t); Ls = lists_for(MIS.get(t, ""))
    for nm, tg in L.items():
        k = int(tg.sum()); cnt = np.bincount(strat[tg], minlength=20); nul = np.zeros((200, n), bool)
        for s_, c_ in enumerate(cnt):
            if c_: np.put_along_axis(nul, np.array([rng.choice(SI[s_], c_, replace=len(SI[s_]) < c_) for _ in range(200)]), True, axis=1)
        ps = np.array([topk(rng.multinomial(max(k * 5, 1), p_atac).astype(float), k) for _ in range(20)])
        bo = or_(tg[BIDX], resp[BIDX])
        G2.append(dict(label=LAB, tf=t, list=nm, n_genes=k, OR=float(or_(tg, resp)), OR_lo=float(np.percentile(bo, 2.5)), OR_hi=float(np.percentile(bo, 97.5)), OR_matched=float(np.median(or_(nul, resp))), OR_pseudoTF=float(np.median(or_(ps, resp))),
                       swap_tf=MIS.get(t), OR_swapped=float(or_(Ls[nm], resp)) if nm in Ls else np.nan, n_swapped=int(Ls[nm].sum()) if nm in Ls else np.nan))
    if t == "CTCF":
        chip = pd.Series(BKs["CTCF"], index=bg).reindex(ug).values; cpos = topk(np.nan_to_num(chip, nan=-1)) & resp
        for nm, tg in L.items(): REC.append(dict(label=LAB, tf=t, list=nm, n_list=int(tg.sum()), chip_HEK293_positive_responders=int(cpos.sum()), recovered=int((cpos & tg).sum())))
        REC.append(dict(label=LAB, tf=t, list="raw_tss500_top1000 (m10)", n_list=1000, chip_HEK293_positive_responders=int(cpos.sum()), recovered=int((cpos & topk(w500[t])).sum())))
G2 = pd.DataFrame(G2); G2.loc[G2.n_genes == 0, ["OR", "OR_lo", "OR_hi", "OR_matched", "OR_pseudoTF", "OR_swapped"]] = np.nan; G2.loc[G2.n_swapped == 0, "OR_swapped"] = np.nan  # Amendment 3: empty lists are NA
G2.to_csv(f"{O}/G2_lists.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(REC).to_csv(f"{O}/G2_ctcf_recovery.tsv", sep="\t", index=False)
# ---- G3
chip_t = pd.Series(BKs["CTCF_293T"], index=bg).reindex(ug).values; ours = w500["CTCF"]; ct = topk(np.nan_to_num(chip_t, nan=-1)); ot = topk(ours); rk = pd.Series(ours).rank(pct=True).values
A = ct & (rk < .5); B = ot & ~ct; pool = allw / allt * 1e6; G3 = []
for col, v in (("promoter_atac_RPM", atac), ("cpg_island", (cgi > 0).astype(float)), ("pooled_panel_promoter_CPM", pool), ("our_ctcf_promoter_fragments", ours), ("chip_HEK293T_promoter_signal", chip_t)):
    a, b = v[A], v[B]; G3.append(dict(label=LAB, feature=col, n_A_chip_top_we_rank_low=int(A.sum()), A_mean=float(np.nanmean(a)), A_median=float(np.nanmedian(a)), A_q25=float(np.nanpercentile(a, 25)), A_q75=float(np.nanpercentile(a, 75)),
                                     n_B_our_top_not_chip=int(B.sum()), B_mean=float(np.nanmean(b)), B_median=float(np.nanmedian(b)), B_q25=float(np.nanpercentile(b, 25)), B_q75=float(np.nanpercentile(b, 75)), mannwhitney_p=float(mannwhitneyu(a[np.isfinite(a)], b[np.isfinite(b)]).pvalue)))
pd.DataFrame(G3).to_csv(f"{O}/G3_ctcf_groups.tsv", sep="\t", index=False, float_format="%.4g"); print(pd.DataFrame(G3).to_string(index=False))
