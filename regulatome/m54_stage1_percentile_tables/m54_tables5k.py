# m54: generated from m52_tables5k.py (m52) by gen54.py: responders = top X % of |score| within the TF (m53 full-precision score).
# m48 (SPEC.md): m45 Stage 1 tables with the top 10,000 lab peaks per TF (m45 code, only the peak number and output folder changed; broad grouping kept as m45 plus a top-10,000 regrouping).
# m45 (SPEC.md): Regulome Stage 1 tables. Padding 0 / 10 kb / 50 kb x |z| >= 2-5 x all / without broad / broad only.
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])  # UG, gi, NG, rule_abc, window, V, NB
O = f"{R}/m54_stage1_percentile_tables/out5k"; os.makedirs(O, exist_ok=True); sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
NPEAK = 5000
PADS = {"0": 0, "10 kb": 10000, "50 kb": 50000}
tfs79 = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; LISTS = {p: {} for p in PADS}
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= NPEAK]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values
    A, _ = rule_abc(c5, p5)
    for p, w in PADS.items(): LISTS[p][t] = A | window(c5, p5, w) if w else A
ALLT = sorted(LISTS["0"]); print("lists", len(ALLT), flush=True)
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf) & set(ALLT)); NT = len(TFS)
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
_F = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True); SC_ = pd.DataFrame(_F["score"], index=_F["tfs"].astype(str), columns=_F["genes"].astype(str)).reindex(index=TFS, columns=UG).values.astype(float)
assert (np.isfinite(SC_) == np.isfinite(Z)).all(), "score matrix does not cover exactly the z pairs"
from scipy.stats import rankdata
PA_ = np.full(SC_.shape, np.nan)
for i_ in range(len(TFS)):
    ok_ = np.isfinite(SC_[i_]); PA_[i_, ok_] = 100 * (rankdata(np.abs(SC_[i_, ok_])) - 0.5) / ok_.sum()
# m54: within-TF percentile of |score| mapped so that the m48 cutoffs read as top 10 / 5 / 2 / 1 % (|Z| >= 2 / 3 / 4 / 5) and "non-changing" (|Z| < 1) = bottom 50 %
Z = np.sign(SC_) * np.interp(np.nan_to_num(PA_) - 1e-9, [0, 50, 90, 95, 98, 99, 100], [0, 1, 2, 3, 4, 5, 6]); Z[~np.isfinite(SC_)] = np.nan
CUTL = {2: 10, 3: 5, 4: 2, 5: 1}; PE = {0: 0, 1: 50, 2: 90, 3: 95, 4: 98, 5: 99}
assert all(int((np.abs(Z[i_][np.isfinite(Z[i_])]) >= 2).sum()) in (1122, 1123) for i_ in range(len(TFS)))
OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z))
kd = np.load(f"{R}/m4_validation/kd_response/out_v2/arm2_all_cell/kd_response.npz", allow_pickle=True); keep = kd["ntc_detect"] >= .1
ex = pd.Series(kd["ntc_mean"][keep], index=kd["genes"][keep].astype(str)); ex = ex[~ex.index.duplicated(keep="last")].reindex(UG).fillna(0).values
TB_ = pd.read_csv(f"{NB}/Neighborhoods/GeneList.TSS1kb.bed", sep="\t", header=None, usecols=[3], names=["g"]); AT_ = pd.read_csv(f"{NB}/Neighborhoods/Genes.TSS1kb.ATAC.omniatac_merged.tn5.tagAlign.gz.CountReads.bedgraph", sep="\t", header=None, usecols=[3], names=["n"])
at = pd.Series(AT_.n.values, index=TB_.g.values); at = at[~at.index.duplicated(keep="last")].reindex(UG).fillna(0).values
dec = lambda v: pd.qcut(pd.Series(v).rank(method="first"), 10, labels=False).values; STRAT = dec(ex) * 10 + dec(at)
BROAD = {"RBBP4", "XPA", "RBBP7", "NFYA", "E2F3", "ZNF507", "DDIT3", "SOX4", "NME2", "ZNF703"}  # m45 grouping (top 5,000, padding 0, > 4,000 genes), kept fixed
BROAD10 = {t for t in ALLT if LISTS["0"][t].sum() > 4000}; print("m45 broad", sorted(BROAD), "| top-10k > 4,000 genes at padding 0:", sorted(BROAD10), "with knockdown", sorted(BROAD10 & set(TFS)), flush=True)
n10 = len(BROAD10 & set(TFS)); VERS = {"all 63 TFs": (None, set()), "without broad TFs (55)": ("drop", BROAD), "broad TFs only (8)": ("only", BROAD), f"without broad TFs, top-10k grouping ({63 - n10})": ("drop", BROAD10), f"broad TFs only, top-10k grouping ({n10})": ("only", BROAD10)}
rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1)
T1, T1T, T2, T3, T4, T5, INFO = [], [], [], [], [], [], []
for pad in PADS:
    B = np.vstack([LISTS[pad][t] for t in TFS]); ALLB = np.vstack([LISTS[pad][t] for t in ALLT]); sizes = ALLB.sum(1); ai = {t: k for k, t in enumerate(ALLT)}
    for t in ALLT: INFO.append({"padding": pad, "TF": t, "our genes": int(LISTS[pad][t].sum()), "broad TF (m45 grouping)": t in BROAD, "broad TF (top-10k grouping)": t in BROAD10, "usable knockdown": t in TFS})
    for vname, (mode, BS) in VERS.items():
        ev = np.array([(mode is None) or (mode == "drop" and t not in BS) or (mode == "only" and t in BS) for t in TFS]); pool = np.array([(mode != "drop") or (t not in BS) for t in ALLT])
        base = {"padding": pad, "TFs": vname}
        for c in (2, 3, 4, 5):
            ch = OK & (AZ >= c); nc = OK & (AZ < 1); den = ch.sum(1); sel = ev & (den > 0); cut = f"top {CUTL[c]}% of genes"; ours_i = (B & OK).sum(1); inb_i = (B & ch).sum(1)
            T1.append({**base, "cutoff": cut, "TFs left (>= 1 Perturb-seq gene)": int(sel.sum()), "median our genes per TF": float(np.median(ours_i[ev])), "our genes": int(ours_i[sel].sum()), "Perturb-seq genes": int(den[sel].sum()), "in both": int(inb_i[sel].sum()),
                       "recall: share of Perturb-seq genes found in our Regulome (%)": 100 * inb_i[sel].sum() / den[sel].sum(), "share of our genes found in Perturb-seq (%)": 100 * inb_i[sel].sum() / ours_i[sel].sum()})
            if mode is None:
                for i in np.flatnonzero(sel): T1T.append({"padding": pad, "cutoff": cut, "TF": TFS[i], "broad TF": TFS[i] in BROAD, "our genes": int(ours_i[i]), "Perturb-seq genes": int(den[i]), "in both": int(inb_i[i]), "recall (%)": 100 * inb_i[i] / den[i], "share of our genes found in Perturb-seq (%)": 100 * inb_i[i] / max(ours_i[i], 1)})
            nC = bC = mT = nU = bU = 0
            for i in np.flatnonzero(ev):
                s_ch = STRAT[ch[i]]; b_ch = B[i][ch[i]]; nb = pd.Series(B[i][nc[i]]).groupby(STRAT[nc[i]]).mean(); okm = pd.Series(s_ch).isin(nb.index).values
                nC += okm.sum(); bC += b_ch[okm].sum(); mT += nb.reindex(s_ch[okm]).sum(); nU += nc[i].sum(); bU += B[i][nc[i]].sum()
            T2.append({**base, "cutoff": cut, "TFs left": int(sel.sum()), "changed genes (matched)": int(nC), "recall, changed genes (%)": 100 * bC / max(nC, 1), "matched background (%)": 100 * mT / max(nC, 1), "all non-changing genes (%)": 100 * bU / max(nU, 1)})
            own_f, near_f, ranks = [], [], []
            for i in np.flatnonzero(sel):
                t = TFS[i]; f_all = ALLB[:, ch[i]].sum(1).astype(float); m = pool.copy(); m[ai[t]] = False; oth = f_all[m]; near = np.argsort(np.abs(sizes[m] - sizes[ai[t]]))[:5]
                own_f.append(f_all[ai[t]]); near_f.append(oth[near].mean()); ranks.append(1 + int((oth > f_all[ai[t]]).sum()))
            d_ = den[sel].sum(); ranks = np.array(ranks)
            T4.append({**base, "cutoff": cut, "TFs left": int(sel.sum()), "own list (%)": 100 * np.sum(own_f) / d_, "5 other TFs nearest in list size (%)": 100 * np.sum(near_f) / d_, "gap (points)": 100 * (np.sum(own_f) - np.sum(near_f)) / d_,
                       "lists ranked among": int(pool.sum()), "median rank of own list": float(np.median(ranks)), "TFs with own list in top 5": int((ranks <= 5).sum())})
            # gene shuffle within strata (and unstratified), pooled over evaluated TFs
            for strat in (True, False):
                obs = 0; dr = np.zeros(NP)
                for i in np.flatnonzero(sel):
                    s = STRAT[OK[i]] if strat else np.zeros(OK[i].sum(), int); b = B[i][OK[i]]; p_ = ch[i][OK[i]]; obs += int((b & p_).sum())
                    for st in np.unique(s):
                        m = s == st; n = int(m.sum()); nb_ = int(b[m].sum()); k = int(p_[m].sum())
                        if k and nb_: dr += rng.hypergeometric(nb_, n - nb_, k, size=NP)
                o = 100 * obs / d_; dd = 100 * dr / d_
                T5.append({**base, "cutoff": cut, "shuffle": "genes within expression x ATAC strata" if strat else "genes, unstratified (reference only)", "TFs left": int(sel.sum()), "recall observed (%)": o, "shuffle mean (%)": dd.mean(),
                           "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd), "shuffles": NP})
        for lo_, hi_ in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, np.inf)]:
            m = OK & (AZ >= lo_) & (AZ < hi_); nb_ = m.sum(1); bb_ = (m & B).sum(1); s2 = ev & (nb_ > 0)
            T3.append({**base, "percentile bin": f"{PE[lo_]}-{PE[hi_]}" if np.isfinite(hi_) else f"{PE[lo_]}-100", "TFs with genes in bin": int(s2.sum()), "genes": int(nb_[s2].sum()), "share bound (%)": 100 * bb_[s2].sum() / max(nb_[s2].sum(), 1)})
    print("padding", pad, flush=True)
TT = {"1_recall_pooled": pd.DataFrame(T1), "1_recall_per_tf": pd.DataFrame(T1T), "2_recall_vs_background": pd.DataFrame(T2), "3_dose_response": pd.DataFrame(T3), "4_own_list_vs_similar_size": pd.DataFrame(T4), "5_gene_shuffle": pd.DataFrame(T5), "list_sizes": pd.DataFrame(INFO)}
for n, d in TT.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_tables.xlsx") as w:
    for n, d in TT.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 40)
for n in ("1_recall_pooled", "2_recall_vs_background", "3_dose_response", "4_own_list_vs_similar_size", "5_gene_shuffle"): print("==", n); print(TT[n].round(2).to_string(index=False))
print("median genes per TF (63 / 79 lists):", {p: (float(np.median([LISTS[p][t].sum() for t in TFS])), float(np.median([LISTS[p][t].sum() for t in ALLT]))) for p in PADS})
print("done")
