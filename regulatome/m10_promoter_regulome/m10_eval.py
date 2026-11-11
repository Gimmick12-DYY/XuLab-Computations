# m10_promoter_regulome (SPEC.md sections 0-4). Label: diagnostic / development, benchmark already read.
# usage: m10_eval.py -> gap_per_tf.tsv, gap_pooled.tsv, distal_tuning.tsv, distal_per_split.tsv, ctcf_znf143.tsv, ctcf_znf143_responders.tsv
import json, os
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m10_promoter_regulome"; M9 = f"{RG}/m9_abc_diagnosis"; LAB = "diagnostic / development, benchmark already read"
SEED = 20261111; NM = 200; NBG = 1000; NBT = 2000; NS = (500, 1000)
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; MIS = dict(zip(tfs79, json.load(open(f"{RG}/m7_step2/features/raw_hicfixed/feature_qc.json"))["mis_paired_map"]["mis0"]))
FT = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t").set_index("TF"); LEAD = ["NRF1", "E4F1", "DDIT3", "MAZ", "CTCF", "ZNF143", "NFYA", "NME2", "XPA"]
TEST, VAL = {"chr3", "chr11", "chr17", "chr20"}, {"chr5", "chr12"}
# ---- promoter-window counts (step 1)
CT = {t: np.load(f"{O}/work/counts/{t}.npz", allow_pickle=True) for t in tfs79}; pg = pd.Index(CT[tfs79[0]]["genes"].astype(str))
def cpm(t, k): z = CT[t]; return z[k] / max(float(z["total"]), 1) * 1e6 if k.startswith("w") else z[k]
GLf = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t"); GLf = GLf.drop_duplicates("name", keep="last").set_index("name")
acol = [c for c in GLf.columns if c.startswith("ATAC.") and c.endswith(".RPM")][0]; chrom_of = GLf.chr
# ---- m9 sources: motif (FIMO), bulk ChIP, Chromnitron, canonical ABC summed score; distal ABC x element CPM for step 3
FIMO = {f[:-4]: pd.read_csv(f"{M9}/work/fimo/{f}", sep="\t", header=None, names=["motif", "gene", "score", "p"]) for f in os.listdir(f"{M9}/work/fimo") if f.endswith(".tsv")}
BK = dict(np.load(f"{M9}/work/scores_bulk.npz", allow_pickle=True)); bg = pd.Index(BK.pop("genes").astype(str)); CN = dict(np.load(f"{M9}/work/scores_chromnitron.npz", allow_pickle=True)); cg = pd.Index(CN.pop("genes").astype(str))
R = f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed"; EL = pd.read_csv(f"{R}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); EL["key"] = EL.chr + ":" + EL.start.astype(str) + "-" + EL.end.astype(str); NE = len(EL); ek = dict(zip(EL.key, range(NE)))
LK = pd.read_csv(sorted(f"{R}/Predictions/{f}" for f in os.listdir(f"{R}/Predictions") if f.startswith("EnhancerPredictionsFull_threshold0.02") and f.endswith(".tsv"))[0], sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score", "isSelfPromoter"])
LK["e"] = (LK.chr + ":" + LK.start.astype(str) + "-" + LK.end.astype(str)).map(ek).astype(int); LK["self"] = LK.isSelfPromoter.astype(str).str.lower().eq("true")
CC = np.load(f"{RG}/m7_step2/features/canon_counts/raw_counts.npy").astype(float); tot_reads = pd.read_csv(f"{RG}/m7_step2/features/canon_counts/read_totals.tsv", sep="\t").set_index("tf").reads
PK = "/vast/som/xujie_lab/TAGATG-293T/bigwig/macs2"; QMIN = -np.log10(0.05)
CH = {c: (g.start.values, g.end.values, g.index.values) for c, g in EL.sort_values(["chr", "start"]).groupby("chr")}
def bound_elements(t):
    P = pd.read_csv(f"{PK}/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 2, 8, 9]); P.columns = ["chr", "start", "end", "q", "off"]; P = P[P.q >= QMIN]; el = []
    for c, s in zip(P.chr.values, (P.start + P.off).values):
        if c not in CH: continue
        st, en, ix = CH[c]; k = np.searchsorted(st, s, side="right") - 1
        if k >= 0 and s < en[k]: el.append(ix[k])
    return np.unique(np.array(el, int)), P
# ---- per arm
def or_(tg, resp, n):
    a = (tg & resp).sum(-1); b = (tg & ~resp).sum(-1); c = resp.sum(-1) - a; d = n - a - b - c; return (a + .5) * (d + .5) / ((b + .5) * (c + .5))
def topn(score, N):
    s = np.where(np.isfinite(score), score, -np.inf); m = np.zeros(len(s), bool); o = np.argsort(-s, kind="stable"); m[o[:N]] = True; m &= np.isfinite(score) & (s > -np.inf); return m
GAP = []; POOL = []; DT = []; DS = []; CZ = []; CR = []
for arm in ("arm2", "arm1"):
    Rz = np.load(f"{M9}/work/responses_{arm}.npz", allow_pickle=True); tt = list(Rz["tfs"]); ug = pd.Index(Rz["genes"].astype(str)); n = len(ug)
    atac = GLf[acol].reindex(ug).fillna(0).values; BW = np.load(f"{RG}/m7_step2/features/raw_hicfixed_wide/base.npz"); GW = pd.read_csv(f"{RG}/m7_step2/splits/gene_split_wide.tsv", sep="\t")
    cgi = pd.Series(BW["base"][:, 4], index=GW.gene).reindex(ug).fillna(0).values
    dec = pd.qcut(pd.Series(atac).rank(method="first"), 10, labels=False).values; strat = dec * 2 + (cgi > 0); SI = [np.flatnonzero(strat == s) for s in range(20)]
    rng = np.random.default_rng(SEED + (arm == "arm2")); BIDX = rng.integers(0, n, (NBG, n))
    pidx = pg.get_indexer(ug); okp = pidx >= 0
    def prom(t, k): v = cpm(t, k); out = np.full(n, np.nan); out[okp] = v[pidx[okp]]; return out
    # scores
    S = {}
    for t in tfs79:
        for k, nm in (("w500", "raw_tss500"), ("w1k", "raw_tss1k"), ("imp500", "imp_tss500"), ("imp1k", "imp_tss1k")): S.setdefault(nm, {})[t] = prom(t, k)
    for t, H in FIMO.items():
        r = H[~H.motif.str.contains("_shuf")]; S.setdefault("motif", {})[t] = r.groupby("gene").score.max().reindex(ug).fillna(-1e3).values
    for k, v in BK.items(): S.setdefault("bulk_chip_HEK293T" if k == "CTCF_293T" else "bulk_chip_HEK293orK562", {})[k.replace("_293T", "")] = pd.Series(v, index=bg).reindex(ug).values
    for k, v in CN.items(): S.setdefault("chromnitron", {})[k] = pd.Series(v, index=cg).reindex(ug).values
    gi = pd.Series(np.arange(n), index=ug); LKu = LK[LK.TargetGene.isin(ug)]; BE = {}
    for t in [d[3:] for d in os.listdir(PK) if d.startswith("TF.")]:
        el, _ = bound_elements(t); BE[t] = el; L2 = LKu[LKu.e.isin(set(el))]; S.setdefault("abc_hicfixed_sum", {})[t] = L2.groupby("TargetGene")["ABC.Score"].sum().reindex(ug).fillna(0).values
    for t in FT.index: S.setdefault("atac_alone", {})[t] = atac
    READ = {"r1": Rz["r1"], "r2": Rz["r2"]}
    def resp_of(rn, t): y = READ[rn][tt.index(t)]; m = np.zeros(n, bool); m[np.argsort(-np.abs(y), kind="stable")[:int(round(n * .05))]] = True; return m
    # ---- section 0/1/2: fixed top N, gene-bootstrap interval, matched, pseudo-TF (raw/imputed windows), swapped TF
    for nm, D in S.items():
        for t, sc in D.items():
            if t not in tt or np.isfinite(sc).sum() < 100: continue
            for N in NS:
                tg = topn(sc, N); cnt = np.bincount(strat[tg], minlength=20); nul = np.zeros((NM, n), bool)
                for s_, c_ in enumerate(cnt):
                    if c_: np.put_along_axis(nul, np.array([rng.choice(SI[s_], c_, replace=len(SI[s_]) < c_) for _ in range(NM)]), True, axis=1)
                ps = None
                if nm.startswith(("raw_", "imp_")):
                    tot = int(round(np.nansum(sc) if nm.startswith("imp_") else np.nansum(sc) * float(CT[t]["total"]) / 1e6)); p = (atac + 1e-3) / (atac + 1e-3).sum()
                    ps = np.array([topn(rng.multinomial(max(tot, 1), p).astype(float) + rng.random(n) * 1e-6, N) for _ in range(20)])
                sw = MIS.get(t); swsc = D.get(sw) if sw is not None else None
                for rn in READ:
                    resp = resp_of(rn, t); o = float(or_(tg, resp, n)); bo = or_(tg[BIDX], resp[BIDX], n)
                    r = dict(label=LAB, arm=arm, reading=rn, definition=nm, N=N, tf=t, ceiling=FT.ceiling.get(t), n_set=int(tg.sum()), OR=o, OR_lo=float(np.percentile(bo, 2.5)), OR_hi=float(np.percentile(bo, 97.5)), OR_matched=float(np.median(or_(nul, resp, n))))
                    if ps is not None: r["OR_pseudoTF"] = float(np.median(or_(ps, resp, n)))
                    if swsc is not None: r["swap_tf"] = sw; r["OR_swapped"] = float(or_(topn(swsc, N), resp, n))
                    GAP.append(r)
    print(arm, "gap done", flush=True)
    # ---- section 3: promoter + w x distal (raw), w chosen on train chromosomes
    chrom = chrom_of.reindex(ug).values; split = np.where(pd.Series(chrom).isin(TEST), "test", np.where(pd.Series(chrom).isin(VAL), "val", "train"))
    Ld = LKu[~LKu.self]; Md = sp.csr_matrix((Ld["ABC.Score"].values, (Ld.TargetGene.map(gi).values, Ld.e.values)), shape=(n, NE)); W = (0, .1, .3, 1, 3)
    if arm == "arm2":
        dist = {t: np.asarray(Md @ (CC[:, j] / max(float(tot_reads.get(t, CC[:, j].sum())), 1) * 1e6)).ravel() for j, t in enumerate(tfs79)}
        def dist_or(w, msk, N):
            out = {}
            for t in tt:
                p = S["raw_tss500"][t]
                if not np.isfinite(p).any(): continue
                sc = np.where(msk, np.nan_to_num(p) + w * dist[t], np.nan); out[t] = float(or_(topn(sc, N)[msk], resp_of("r1", t)[msk], int(msk.sum())))
            return out
        trm = split == "train"; Ntr = int(round(1000 * trm.sum() / n)); best = None
        for w in W:
            o = dist_or(w, trm, Ntr); med = float(np.median(list(o.values()))); DT.append(dict(label=LAB, w=w, split="train", N=Ntr, median_OR=med, n_tfs=len(o)))
            if best is None or med > best[1]: best = (w, med)
        for sp_ in ("train", "val", "test"):
            msk = split == sp_; Nn = int(round(1000 * msk.sum() / n))
            for w in sorted({0, best[0]}):
                o = dist_or(w, msk, Nn)
                for t, v in o.items(): DS.append(dict(label=LAB, split=sp_, n_genes=int(msk.sum()), N=Nn, w=w, chosen=w == best[0], tf=t, ceiling=FT.ceiling.get(t), OR=v))
    # ---- section 4: CTCF and ZNF143
    if arm == "arm2":
        for t in ("CTCF", "ZNF143"):
            ours = {"raw_tss500": S["raw_tss500"][t], "imp_tss500": S["imp_tss500"][t]}
            refs = {}
            if t in S.get("bulk_chip_HEK293orK562", {}): refs["bulk_chip_HEK293"] = S["bulk_chip_HEK293orK562"][t]
            if t in S.get("bulk_chip_HEK293T", {}): refs["bulk_chip_HEK293T_pval"] = S["bulk_chip_HEK293T"][t]
            if t in FIMO: r_ = FIMO[t][~FIMO[t].motif.str.contains("_shuf")]; refs["motif_p1e-4"] = (r_.groupby("gene").p.min().reindex(ug).fillna(1).values < 1e-4).astype(float); refs["motif_score"] = S["motif"][t]
            if t in S.get("chromnitron", {}): refs["chromnitron"] = S["chromnitron"][t]
            z = CT[t]; CZ.append(dict(label=LAB, tf=t, item="reads_total", value=float(z["total"]))); CZ.append(dict(label=LAB, tf=t, item="frac_reads_within_tss1kb_any_gene", value=float(z["in_tss1k_any"]) / float(z["total"])))
            CZ.append(dict(label=LAB, tf=t, item="frac_reads_in_canonical_elements", value=float(z["in_canonical_elements"]) / float(z["total"])))
            el, Pk = bound_elements(t); tss = GLf.tss.dropna()
            def frac_near_tss(df):
                f = 0
                for c, G in df.groupby(df.columns[0]):
                    ts = np.sort(GLf.tss[GLf.chr == c].dropna().values.astype(int))
                    if not len(ts): continue
                    mid = ((G.iloc[:, 1] + G.iloc[:, 2]) // 2).values; k = np.searchsorted(ts, mid); d = np.minimum(np.abs(mid - ts[np.clip(k - 1, 0, len(ts) - 1)]), np.abs(mid - ts[np.clip(k, 0, len(ts) - 1)])); f += int((d <= 1000).sum())
                return f / max(len(df), 1)
            CZ.append(dict(label=LAB, tf=t, item="our_macs2_peaks_q05_n", value=len(Pk))); CZ.append(dict(label=LAB, tf=t, item="frac_our_peaks_within_1kb_of_tss", value=frac_near_tss(Pk[["chr", "start", "end"]])))
            bp = f"{X}/downstream/tracks/bulk_chipseq/{t}_peaks.bed"
            if os.path.exists(bp):
                Bk = pd.read_csv(bp, sep="\t", header=None, usecols=[0, 1, 2]); CZ.append(dict(label=LAB, tf=t, item="bulk_chip_HEK293_peaks_n", value=len(Bk))); CZ.append(dict(label=LAB, tf=t, item="frac_bulk_chip_peaks_within_1kb_of_tss", value=frac_near_tss(Bk)))
            for on, ov in ours.items():
                for rn_, rv in refs.items():
                    m = np.isfinite(ov) & np.isfinite(rv); CZ.append(dict(label=LAB, tf=t, item=f"spearman_{on}_vs_{rn_}", value=float(spearmanr(ov[m], rv[m])[0])))
                    for N in NS:
                        if rn_ == "motif_p1e-4": rs = rv > 0
                        else: rs = topn(rv, N)
                        a = topn(ov, N); CZ.append(dict(label=LAB, tf=t, item=f"overlap_top{N}_{on}_vs_{rn_}", value=int((a & rs).sum()), n_ref=int(rs.sum())))
            # ChIP- or motif-positive responders: missed (0 reads), bound but ranked low, in our top N
            resp = resp_of("r1", t); raw_reads = prom(t, "w500") * float(CT[t]["total"]) / 1e6
            for rn_, rv in refs.items():
                if rn_ == "motif_score": continue
                pos_ = (rv > 0) if rn_ == "motif_p1e-4" else topn(rv, 1000); g_ = pos_ & resp
                for on in ours:
                    a = topn(ours[on], 1000); val = ours[on]
                    CR.append(dict(label=LAB, tf=t, reference=rn_, ours=on, n_ref_positive_responders=int(g_.sum()), in_our_top1000=int((g_ & a).sum()), bound_ranked_low=int((g_ & ~a & (np.nan_to_num(val) > 0)).sum()), no_signal=int((g_ & (np.nan_to_num(val) <= 0)).sum()),
                                    median_our_rank_pct_of_those=float(np.median(rankdata(np.nan_to_num(val))[g_] / n * 100)) if g_.any() else np.nan))
            # distal-site OR test: genes linked by a distal ABC link to an element with (a) a bulk ChIP peak, (b) our peak
            Ldt = LKu[~LKu.self]
            sets = {"distal_elements_with_our_peak": set(el)}
            if os.path.exists(bp):
                Bk = pd.read_csv(bp, sep="\t", header=None, usecols=[0, 1, 2]); ce = []
                for c, s_, e_ in zip(Bk[0].values, Bk[1].values, Bk[2].values):
                    mid = (s_ + e_) // 2
                    if c not in CH: continue
                    st, en, ix = CH[c]; k = np.searchsorted(st, mid, side="right") - 1
                    if k >= 0 and mid < en[k]: ce.append(ix[k])
                sets["distal_elements_with_bulk_chip_peak"] = set(ce)
            for sn, es in sets.items():
                tg = np.zeros(n, bool); tg[gi[Ldt[Ldt.e.isin(es)].TargetGene.unique()].values] = True; cnt = np.bincount(strat[tg], minlength=20); nul = np.zeros((NM, n), bool)
                for s_, c_ in enumerate(cnt):
                    if c_: np.put_along_axis(nul, np.array([rng.choice(SI[s_], c_, replace=len(SI[s_]) < c_) for _ in range(NM)]), True, axis=1)
                bo = or_(tg[BIDX], resp[BIDX], n); CZ.append(dict(label=LAB, tf=t, item=f"OR_{sn}", value=float(or_(tg, resp, n)), lo=float(np.percentile(bo, 2.5)), hi=float(np.percentile(bo, 97.5)), OR_matched=float(np.median(or_(nul, resp, n))), n_set=int(tg.sum())))
G = pd.DataFrame(GAP); G.to_csv(f"{O}/gap_per_tf.tsv", sep="\t", index=False, float_format="%.4f")
rng = np.random.default_rng(SEED + 7)
for (arm, rn, nm, N), H in G.groupby(["arm", "reading", "definition", "N"]):
    for sn, Q in (("all", H), ("ceiling_ge_0.2", H[H.ceiling >= .2])):
        if not len(Q): continue
        d = (Q.OR - Q.OR_matched).values; b = np.median(d[rng.integers(0, len(d), (NBT, len(d)))], axis=1)
        r = dict(label=LAB, arm=arm, reading=rn, definition=nm, N=N, tf_set=sn, n_tfs=len(Q), median_OR=Q.OR.median(), median_OR_matched=Q.OR_matched.median(), median_OR_minus_matched=float(np.median(d)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), frac_OR_gt_matched=float((d > 0).mean()))
        if "OR_pseudoTF" in Q and Q.OR_pseudoTF.notna().any(): r["median_OR_pseudoTF"] = Q.OR_pseudoTF.median(); r["frac_OR_gt_pseudoTF"] = float((Q.OR > Q.OR_pseudoTF).mean())
        if "OR_swapped" in Q and Q.OR_swapped.notna().any(): r["median_OR_swapped"] = Q.OR_swapped.median(); r["frac_OR_gt_swapped"] = float((Q.OR > Q.OR_swapped).mean())
        POOL.append(r)
# paired comparisons: raw vs imputed promoter, raw promoter vs ABC, per N
for (arm, rn, N), H in G.groupby(["arm", "reading", "N"]):
    P_ = H.pivot_table(index="tf", columns="definition", values="OR")
    for a, b_ in (("raw_tss500", "imp_tss500"), ("raw_tss500", "abc_hicfixed_sum"), ("raw_tss1k", "imp_tss1k"), ("raw_tss500", "raw_tss1k")):
        if a in P_ and b_ in P_:
            Q = P_[[a, b_]].dropna(); d = (Q[a] - Q[b_]).values; bb = np.median(d[rng.integers(0, len(d), (NBT, len(d)))], axis=1)
            POOL.append(dict(label=LAB, arm=arm, reading=rn, definition=f"{a}_minus_{b_}", N=N, tf_set="paired", n_tfs=len(Q), median_OR_minus_matched=float(np.median(d)), lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)), frac_OR_gt_matched=float((d > 0).mean())))
pd.DataFrame(POOL).to_csv(f"{O}/gap_pooled.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(DT).to_csv(f"{O}/distal_tuning.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(DS).to_csv(f"{O}/distal_per_split.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(CZ).to_csv(f"{O}/ctcf_znf143.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(CR).to_csv(f"{O}/ctcf_znf143_responders.tsv", sep="\t", index=False, float_format="%.3f")
print("done")
