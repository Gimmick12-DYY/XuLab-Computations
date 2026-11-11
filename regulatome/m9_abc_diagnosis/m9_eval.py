# m9_abc_diagnosis sections B, C, E (SPEC.md, coordinator notes 2026-11-11). Diagnostic, benchmark already read: no claim of validation comes from it.
# Every target definition is scored per TF against that TF's knockdown response, read as r1 (raw per-TF z), r2 (double-centred z as B6) and
# r3 (z with the top k = 1, 2, 3, 5 components removed; section D), Arm 2 and Arm 1, gene universe = ABC genes in the matrix.
# Statistics: OR = (a+.5)(d+.5)/((b+.5)(c+.5)) of the target set vs top-5% |response| genes; Spearman of the continuous score with |z| and with z.
# Controls beside every statistic: (m) accessibility-matched: target sets redrawn with the same promoter-ATAC-decile x CpG-island composition and
# scores permuted within those strata, 200 draws (mean of the null); (x) mis-paired TF: the definition of the TF paired in m7_step2 section 6 (mis0,
# same depth tertile) scored against this TF's response; for motifs also (s) the 20 column-permuted motifs (mean). "atac_alone" = promoter ATAC as the score.
# usage: m9_eval.py -> BCE_long.tsv, E_summary.tsv, E_per_tf.tsv, B_coverage.tsv
import json, os
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m9_abc_diagnosis"; W = f"{O}/work"; LAB = "diagnostic, benchmark already read"
ND = 200; SEED = 20261111; TOPK = 1000
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF")
GW = pd.read_csv(f"{RG}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); gw = pd.Series(np.arange(len(GW)), index=GW.gene)
def wide(kind, name): return np.load(f"{RG}/m7_step2/features/{kind}_hicfixed_wide/{name}.npz")
BASE = np.load(f"{RG}/m7_step2/features/raw_hicfixed_wide/base.npz"); BN = [str(x) for x in BASE["names"]]; BASE = BASE["base"]
# ---- ABC target sets and continuous ABC score (sum of ABC.Score over links from bound elements), both link sets, as m9_ad.py / m2c
PK = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"; QMIN = -np.log10(0.05); ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}
def abc_scores(bs):
    R = f"{RG}/m2c_abc_canonical/results/{bs}"; E = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); E["key"] = E.chr + ":" + E.start.astype(str) + "-" + E.end.astype(str); NE = len(E); ek = dict(zip(E.key, range(NE)))
    L = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score"])
    L["e"] = (L.chr + ":" + L.start.astype(str) + "-" + L.end.astype(str)).map(ek).astype(int); gl = np.array(sorted(L.TargetGene.unique())); gi = {g: i for i, g in enumerate(gl)}
    M = sp.csr_matrix((L["ABC.Score"].values, (L.e.values, L.TargetGene.map(gi).values)), shape=(NE, len(gl)))
    CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in E.sort_values(["chr", "start"]).groupby("chr")}; out = {}
    for t in sorted(d[3:] for d in os.listdir(PK) if d.startswith("TF.")):
        P = pd.read_csv(f"{PK}/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 8, 9]); P.columns = ["chr", "start", "q", "off"]; P = P[P.q >= QMIN]; el = []
        for c, s in zip(P.chr.values, (P.start + P.off).values):
            if c not in CH: continue
            st, en, ix = CH[c]; k = np.searchsorted(st, s, side="right") - 1
            if k >= 0 and s < en[k]: el.append(ix[k])
        el = np.unique(np.array(el, int)); v = np.asarray(M[el].sum(0)).ravel() if len(el) else np.zeros(len(gl)); out[ALIAS.get(t, t)] = pd.Series(v, index=gl)
    return out
ABC = {"abc_hicfixed": abc_scores("HEK293T_hicfixed"), "abc_powerlaw": abc_scores("HEK293T_powerlaw")}
# ---- motif FIMO tables
MW = pd.read_csv(f"{O}/B_motif_workbook.tsv", sep="\t").set_index("tf"); FIMO = {}
for t in FT.index:
    f = f"{W}/fimo/{t}.tsv"
    if os.path.exists(f) and os.path.getsize(f) > 0: FIMO[t] = pd.read_csv(f, sep="\t", header=None, names=["motif", "gene", "score", "p"])
BULK = dict(np.load(f"{W}/scores_bulk.npz", allow_pickle=True)); CHR = dict(np.load(f"{W}/scores_chromnitron.npz", allow_pickle=True))
bulk_genes = pd.Index(BULK.pop("genes").astype(str)); chr_genes = pd.Index(CHR.pop("genes").astype(str))
rows = []; cover = []
for arm in ("arm2", "arm1"):
    Rz = np.load(f"{W}/responses_{arm}.npz", allow_pickle=True); tt = list(Rz["tfs"]); ug = pd.Index(Rz["genes"].astype(str)); n = len(ug); READ = {k: Rz[k] for k in ("r1", "r2", "r3_k1", "r3_k2", "r3_k3", "r3_k5")}
    wi = gw.reindex(ug).values; okw = ~np.isnan(wi); wi = np.where(okw, wi, 0).astype(int)
    atac = np.where(okw, BASE[wi, BN.index("promoter_atac")], np.nan); k27 = np.where(okw, BASE[wi, BN.index("promoter_h3k27ac")], np.nan); cgi = np.where(okw, BASE[wi, BN.index("cgi_flag")], 0)
    dec = pd.qcut(pd.Series(np.nan_to_num(atac, nan=np.nanmin(atac))).rank(method="first"), 10, labels=False).values; strat = dec * 2 + (cgi > 0); SI = [np.flatnonzero(strat == s) for s in range(20)]
    rng = np.random.default_rng(SEED + (arm == "arm2")); PERM = np.tile(np.arange(n), (ND, 1))
    for ix in SI:
        for d in range(ND): PERM[d, ix] = ix[rng.permutation(len(ix))]
    train = np.where(okw, (GW.set.values[wi] == "train"), False)
    # ---- definitions: name -> {tf: score vector over ug (nan = not scorable), optional explicit set}
    DEF = {}
    def put(name, t, score, st=None): DEF.setdefault(name, {})[t] = (np.asarray(score, float), st)
    for t, H in FIMO.items():
        real = H[~H.motif.str.contains("_shuf")]; sc = real.groupby("gene").score.max().reindex(ug).values; pmin = real.groupby("gene").p.min().reindex(ug).values
        sc = np.where(np.isnan(sc), -1e3, sc); put("B1_motif_top1000", t, sc); put("B1_motif_p1e-4", t, sc, np.nan_to_num(pmin, nan=1) < 1e-4)
        for k in range(20):
            Sh = H[H.motif.str.endswith(f"_shuf{k:02d}")]; s2 = Sh.groupby("gene").score.max().reindex(ug).values; s2 = np.where(np.isnan(s2), -1e3, s2); p2 = Sh.groupby("gene").p.min().reindex(ug).values
            put(f"S_motif_top1000_shuf{k:02d}", t, s2); put(f"S_motif_p1e-4_shuf{k:02d}", t, s2, np.nan_to_num(p2, nan=1) < 1e-4)
    for key, v in BULK.items(): put("B2_bulk_chip" + ("_HEK293T" if key == "CTCF_293T" else ""), key.replace("_293T", ""), pd.Series(v, index=bulk_genes).reindex(ug).values)
    for key, v in CHR.items(): put("B3_chromnitron", key, pd.Series(v, index=chr_genes).reindex(ug).values)
    for kind in ("raw", "imp"):
        D = wide(kind, "real")["prom"]  # TF x wide gene, standardised log1p CPM (rank-equivalent to log1p CPM)
        Pm = np.where(okw[None, :], D[:, wi], np.nan); rk = np.vstack([rankdata(r) for r in np.nan_to_num(Pm, nan=-9)]) / n; pool = np.where(okw, BASE[wi, BN.index("pool_promoter")], np.nan)
        Xr = np.column_stack([np.ones(n), np.nan_to_num(atac), np.nan_to_num(k27)])
        for i, t in enumerate(tfs79):
            put(f"C_{kind}_i_log1pCPM", t, Pm[i]); put(f"C_{kind}_ii_rank", t, rk[i]); put(f"C_{kind}_iii_rank_minus_others", t, rk[i] - (rk.sum(0) - rk[i]) / (len(tfs79) - 1))
            y = np.nan_to_num(Pm[i]); beta = np.linalg.lstsq(Xr[train], y[train], rcond=None)[0]; put(f"C_{kind}_iv_resid_atac_k27_trainfit", t, y - Xr @ beta)
            put(f"C_{kind}_v_minus_pooled_panel", t, Pm[i] - pool)
    for nm, dct in ABC.items():
        for t, s in dct.items(): sc = s.reindex(ug).fillna(0).values; put(f"C_{nm}", t, sc, sc > 0)
    for t in FT.index: put("atac_alone", t, atac)
    for nm, dct in DEF.items(): cover.append(dict(arm=arm, definition=nm, n_tfs=len(dct), tfs=",".join(sorted(dct)) if not nm.startswith("S_") else ""))
    # ---- evaluate
    def set_of(score, st):
        if st is not None: return st & np.isfinite(score)
        s = np.where(np.isfinite(score), score, -np.inf); m = np.zeros(n, bool); m[np.argsort(-s, kind="stable")[:TOPK]] = True; return m
    def or_(a_in, resp):
        a = (a_in & resp).sum(-1); b = (a_in & ~resp).sum(-1); c = resp.sum() - a; d = n - a - b - c; return (a + .5) * (d + .5) / ((b + .5) * (c + .5))
    def sp_(sc_rank, y):  # Spearman via ranks; sc_rank may be (ND, n)
        ry = rankdata(y); ry = (ry - ry.mean()) / ry.std(); x = sc_rank - sc_rank.mean(-1, keepdims=True); x = x / x.std(-1, keepdims=True); return (x * ry).mean(-1)
    cache = {}
    for nm, dct in DEF.items():
        for t, (score, st) in dct.items():
            if t not in tt: continue
            ok = np.isfinite(score)
            if ok.sum() < 100: continue
            sv = np.where(ok, score, np.nanmin(score[ok]) - 1); srank = rankdata(sv); tg = set_of(score, st)
            # matched set null: same strata composition
            cnt = np.bincount(strat[tg], minlength=20); nul = np.zeros((ND, n), bool)
            for s_, c_ in enumerate(cnt):
                if c_: idx = np.array([rng.choice(SI[s_], c_, replace=len(SI[s_]) < c_) for _ in range(ND)]); np.put_along_axis(nul, idx, True, axis=1)
            cache[(nm, t)] = (srank, tg)
            for rn, Rm in READ.items():
                y = Rm[tt.index(t)]; resp = np.zeros(n, bool); resp[np.argsort(-np.abs(y), kind="stable")[:int(round(n * .05))]] = True
                r = dict(arm=arm, definition=nm, tf=t, reading=rn, n_set=int(tg.sum()), OR=float(or_(tg, resp)), OR_matched=float(np.median(or_(nul, resp))),
                         sp_abs=float(sp_(srank, np.abs(y))), sp_signed=float(sp_(srank, y)), sp_abs_matched=float(sp_(srank[PERM], np.abs(y)).mean()), sp_signed_matched=float(sp_(srank[PERM], y).mean()))
                rows.append(r)
    # mis-paired: partner's definition scored against this TF's response (partner must be a TF of the same definition)
    L = pd.DataFrame([r for r in rows if r["arm"] == arm]); mis = []
    for (nm, t), (srank, tg) in cache.items():
        pt = MIS.get(t)
        if pt is None or (nm, pt) not in cache: continue
        ps, ptg = cache[(nm, pt)]
        for rn, Rm in READ.items():
            y = Rm[tt.index(t)]; resp = np.zeros(n, bool); resp[np.argsort(-np.abs(y), kind="stable")[:int(round(n * .05))]] = True
            mis.append(dict(arm=arm, definition=nm, tf=t, reading=rn, mis_partner=pt, OR_mis=float(or_(ptg, resp)), sp_abs_mis=float(sp_(ps, np.abs(y))), sp_signed_mis=float(sp_(ps, y))))
    rows = [r for r in rows if r["arm"] != arm] + L.merge(pd.DataFrame(mis), on=["arm", "definition", "tf", "reading"], how="left").to_dict("records")
    print(arm, "done", flush=True)
B = pd.DataFrame(rows); B.insert(0, "label", LAB)
# motif shuffle control: mean over 20 permuted motifs, joined beside the real motif rows
Sh = B[B.definition.str.startswith("S_")].copy(); Sh["base"] = Sh.definition.str.replace(r"^S_(motif_(top1000|p1e-4))_shuf\d+$", r"B1_\1", regex=True)
Sm = Sh.groupby(["arm", "base", "tf", "reading"])[["OR", "sp_abs", "sp_signed"]].mean().add_suffix("_motifshuffle").reset_index().rename(columns={"base": "definition"})
B = B[~B.definition.str.startswith("S_")].merge(Sm, on=["arm", "definition", "tf", "reading"], how="left"); B["ceiling"] = B.tf.map(FT.ceiling)
B.to_csv(f"{O}/BCE_long.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(cover).assign(label=LAB).to_csv(f"{O}/B_coverage.tsv", sep="\t", index=False)
# ---- E summary
E = []
for (arm, nm, rn), G in B.groupby(["arm", "definition", "reading"]):
    for sn, H in (("all", G), ("ceiling_ge_0.2", G[G.ceiling >= .2])):
        if not len(H): continue
        r = dict(label=LAB, arm=arm, definition=nm, reading=rn, tf_set=sn, n_tfs=len(H), median_OR=H.OR.median(), median_OR_matched=H.OR_matched.median(), frac_OR_gt_matched=(H.OR > H.OR_matched).mean(),
                 median_OR_mis=H.OR_mis.median(), frac_OR_gt_mis=(H.OR > H.OR_mis).mean() if H.OR_mis.notna().any() else np.nan, mean_sp_abs=H.sp_abs.mean(), mean_sp_abs_matched=H.sp_abs_matched.mean(), mean_sp_abs_mis=H.sp_abs_mis.mean(),
                 mean_sp_signed=H.sp_signed.mean(), mean_abs_sp_signed=H.sp_signed.abs().mean(), mean_sp_signed_matched=H.sp_signed_matched.mean(), mean_sp_signed_mis=H.sp_signed_mis.mean())
        if "OR_motifshuffle" in H and H.OR_motifshuffle.notna().any(): r.update(median_OR_motifshuffle=H.OR_motifshuffle.median(), frac_OR_gt_motifshuffle=(H.OR > H.OR_motifshuffle).mean(), mean_sp_abs_motifshuffle=H.sp_abs_motifshuffle.mean())
        E.append(r)
pd.DataFrame(E).to_csv(f"{O}/E_summary.tsv", sep="\t", index=False, float_format="%.4f")
LEAD = ["NRF1", "NFYA", "CTCF", "E4F1", "ZNF143", "NME2", "XPA", "MAZ", "E2F3", "MLX", "SOX4", "DDIT3"]; keep = set(LEAD) | set(FT.index[FT.ceiling >= .2])
B[B.tf.isin(keep)].to_csv(f"{O}/E_per_tf.tsv", sep="\t", index=False, float_format="%.4f"); print("done")
