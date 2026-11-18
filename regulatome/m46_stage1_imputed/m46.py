# m46 (SPEC.md): Stage 1 with imputed binding lists vs lab-peak lists.
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
R = "/work/users/d/y/dyy12/XuLab/regulatome"; X = "/work/users/d/y/dyy12/XuLab"; M45 = f"{R}/m45_stage1_summary"
s = open(f"{M45}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])  # LISTS (lab, paddings), PADS, TFS, OK, AZ, STRAT, BROAD, rule_abc, window, ALLT
O = f"{R}/m46_stage1_imputed/out"; os.makedirs(O, exist_ok=True)  # set after the exec, which defines its own O
U = f"{X}/unified/work"; reg = [l.strip() for l in open(f"{U}/nfya/impute/regions.tsv")]; chrom = np.array([r.split(":")[0] for r in reg]); start = np.array([int(r.split(":")[1].split("-")[0]) for r in reg])
mi = np.load(f"{X}/complex_screen/motif_shared_20261103/fg/signal_mask.npz", allow_pickle=True)["bins"]; assert len(mi) == 104738, len(mi)
bc, bp = chrom[mi], start[mi] + 500
LZ = np.load(f"{X}/complex_screen/pair_motif_composite/recall/imputed_scores.npz", allow_pickle=True); ltf = [str(x).upper() for x in LZ["tfs"]]
have = {t: os.path.exists(f"{U}/{t.lower()}/impute/matrix_csr.npz") for t in ALLT}; print("imputed matrices:", sum(have.values()), "of", len(ALLT), "lists;", sum(have[t] for t in TFS), "of 63 knockdown TFs", flush=True)
def link(sel, w): c, p = bc[sel], bp[sel]; A, _ = rule_abc(c, p); return A | window(c, p, w) if w else A
IMP = {v: {p: {} for p in PADS} for v in ("I1a", "I1b", "I2", "I2loo", "I3")}; NB = []
for t in ALLT:
    if not have[t]: continue
    m = sp.load_npz(f"{U}/{t.lower()}/impute/matrix_csr.npz"); nc = m.shape[1]; v = np.asarray(m.sum(1)).ravel()[mi].astype(float); del m
    S = v / max(v.sum(), 1) * 1e6; rk = np.argsort(-S, kind="stable"); lo = LZ["loo"][ltf.index(t)] if t in ltf else None; rkl = np.argsort(-lo, kind="stable") if lo is not None else None
    for p, w in PADS.items():
        IMP["I1a"][p][t] = link(v >= max(3, 0.01 * nc), w); IMP["I1b"][p][t] = link(v >= max(3, 0.05 * nc), w)
        top = np.zeros(len(S), bool); top[rk[:5000]] = True; IMP["I3"][p][t] = link(top, w)
        target = int(LISTS[p][t].sum())
        for key, order in (("I2", rk), ("I2loo", rkl)):
            if order is None: continue
            a, b = 1, len(order)
            while a < b:  # smallest N whose linked-gene count reaches the lab list size
                mid = (a + b) // 2; sel = np.zeros(len(S), bool); sel[order[:mid]] = True
                if link(sel, w).sum() >= target: b = mid
                else: a = mid + 1
            sel = np.zeros(len(S), bool); sel[order[:a]] = True; IMP[key][p][t] = link(sel, w); NB.append({"TF": t, "padding": p, "ranking": key, "bins used": a, "lab list genes": target, "imputed list genes": int(IMP[key][p][t].sum())})
    NB.append({"TF": t, "padding": "-", "ranking": "cells", "bins used": int((v >= max(3, 0.01 * nc)).sum()), "lab list genes": nc, "imputed list genes": int((v >= max(3, 0.05 * nc)).sum())})
    print(t, nc, flush=True)
pd.DataFrame(NB).to_csv(f"{O}/0_bins_and_sizes.tsv", sep="\t", index=False)
SRC = {"lab peaks": {p: LISTS[p] for p in PADS}, **IMP}
ti = {t: i for i, t in enumerate(TFS)}; TFI = [t for t in TFS if have[t]]; EX3 = {"NRF1", "E4F1", "DNMT1"}
SETS = {"all": set(), "without NRF1, E4F1, DNMT1": EX3, "without broad TFs": BROAD}
rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1)
P1, PT, JA, OWN = [], [], [], []
for src, LL in SRC.items():
    for p in PADS:
        L = LL[p]
        if not all(t in L for t in TFI): continue
        for c in (2, 3, 4, 5):
            ob, de, ou, dr = np.zeros(len(TFI)), np.zeros(len(TFI)), np.zeros(len(TFI)), np.zeros((len(TFI), NP))
            for k, t in enumerate(TFI):
                i = ti[t]; s_ = STRAT[OK[i]]; b = L[t][OK[i]]; p_ = (AZ[i] >= c)[OK[i]]; ob[k] = (b & p_).sum(); de[k] = p_.sum(); ou[k] = b.sum()
                for st in np.unique(s_):
                    m = s_ == st; n = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
                    if kk and nb: dr[k] += rng.hypergeometric(nb, n - nb, kk, size=NP)
                if de[k]: PT.append({"list": src, "padding": p, "cutoff": f"|z| >= {c}", "TF": t, "our genes": int(ou[k]), "Perturb-seq genes": int(de[k]), "in both": int(ob[k]), "recall (%)": 100 * ob[k] / de[k], "shuffle mean (%)": 100 * dr[k].mean() / de[k], "p (one-sided)": pval(ob[k], dr[k])})
            for sn, ex_ in SETS.items():
                sel = np.array([t not in ex_ for t in TFI]) & (de > 0); o = 100 * ob[sel].sum() / de[sel].sum(); dd = 100 * dr[sel].sum(0) / de[sel].sum()
                P1.append({"list": src, "padding": p, "cutoff": f"|z| >= {c}", "TFs": sn, "n TFs": int(sel.sum()), "our genes": int(ou[sel].sum()), "Perturb-seq genes": int(de[sel].sum()), "in both": int(ob[sel].sum()), "recall (%)": o, "share of our genes found in Perturb-seq (%)": 100 * ob[sel].sum() / max(ou[sel].sum(), 1),
                           "shuffle mean (%)": dd.mean(), "observed minus shuffle mean (points)": o - dd.mean(), "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd)})
            # own list vs 5 similar-size other TFs (pool = this source's lists for all TFs that have one)
            pool = [t for t in ALLT if t in L]; ALLB = np.vstack([L[t] for t in pool]); sizes = ALLB.sum(1); ai = {t: k for k, t in enumerate(pool)}; of, nf, dsum = 0, 0, 0
            for t in TFI:
                i = ti[t]; ch = OK[i] & (AZ[i] >= c)
                if not ch.any(): continue
                fa = ALLB[:, ch].sum(1).astype(float); mm = np.ones(len(pool), bool); mm[ai[t]] = False; near = np.argsort(np.abs(sizes[mm] - sizes[ai[t]]))[:5]; of += fa[ai[t]]; nf += fa[mm][near].mean(); dsum += ch.sum()
            OWN.append({"list": src, "padding": p, "cutoff": f"|z| >= {c}", "own list (%)": 100 * of / dsum, "5 other TFs nearest in list size (%)": 100 * nf / dsum, "gap (points)": 100 * (of - nf) / dsum})
        if src != "lab peaks":
            for t in TFI:
                a_, b_ = L[t], LISTS[p][t]; JA.append({"list": src, "padding": p, "TF": t, "imputed genes": int(a_.sum()), "lab genes": int(b_.sum()), "shared": int((a_ & b_).sum()), "Jaccard": (a_ & b_).sum() / max((a_ | b_).sum(), 1)})
    print("done", src, flush=True)
TT = {"1_pooled": pd.DataFrame(P1), "2_per_tf": pd.DataFrame(PT), "3_jaccard_imputed_vs_lab": pd.DataFrame(JA), "4_own_list_vs_similar_size": pd.DataFrame(OWN), "0_bins_and_sizes": pd.DataFrame(NB)}
for n, d in TT.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_imputed.xlsx") as w:
    for n, d in TT.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
print(TT["1_pooled"][TT["1_pooled"].TFs == "all"].drop(columns=["TFs"]).round(2).to_string(index=False)); print(TT["4_own_list_vs_similar_size"].round(2).to_string(index=False))
print(TT["3_jaccard_imputed_vs_lab"].groupby(["list", "padding"]).Jaccard.median().round(3).to_string()); print("done")
