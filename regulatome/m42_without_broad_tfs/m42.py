# m42_without_broad_tfs (SPEC.md): overlap, own-list, matched-gene and dose tables with and without broad TFs.
import sys, os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
s40 = open(f"{R}/m40_overlap_reference/m40.py").read(); exec(s40[:s40.index("R1, R1T, DOSE")])
O = f"{R}/m42_without_broad_tfs/out"
TSS = pd.read_csv(f"{NB}/Neighborhoods/GeneList.TSS1kb.bed", sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"]); MG = {}
for ch, g in TSS.sort_values(["c", "s"]).groupby("c"):
    s_, e_ = [], []
    for a, b in zip(g.s.values, g.e.values):
        if s_ and a <= e_[-1]: e_[-1] = max(e_[-1], b)
        else: s_.append(a); e_.append(b)
    MG[ch] = (np.array(s_), np.array(e_))
def prom_share(t):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]; n = 0
    for ch, g in P.groupby("c"):
        if ch not in MG: continue
        st, en = MG[ch]; sm = (g.s + g.off).values; k = np.searchsorted(st, sm, side="right") - 1; n += ((k >= 0) & (sm < en[np.maximum(k, 0)])).sum()
    return n / len(P)
INFO = pd.DataFrame({"TF": ALLT, "rule A genes": [int(LISTS["A"][t].sum()) for t in ALLT], "rule G genes": [int(LISTS["G"][t].sum()) for t in ALLT], "share of top-5,000 summits in promoters (%)": [100 * prom_share(t) for t in ALLT], "usable top-25% knockdown": [t in TFS for t in ALLT]})
INFO["broad: rule A list > 4,000"] = INFO["rule A genes"] > 4000; INFO["broad: > 40% of peaks in promoters"] = INFO["share of top-5,000 summits in promoters (%)"] > 40
INFO = INFO.sort_values("rule A genes", ascending=False); INFO.to_csv(f"{O}/broad_tfs.tsv", sep="\t", index=False, float_format="%.4g")
print(INFO[INFO["broad: rule A list > 4,000"] | INFO["broad: > 40% of peaks in promoters"]].round(1).to_string(index=False), flush=True)
DEF = {"list > 4,000 genes": set(INFO.TF[INFO["broad: rule A list > 4,000"]]), "> 40% peaks in promoters": set(INFO.TF[INFO["broad: > 40% of peaks in promoters"]])}
VERS = [("(i) all TFs", None, None)] + [(f"(ii) without broad TFs [{k}]", "drop", v) for k, v in DEF.items()] + [(f"(iii) only broad TFs [{k}]", "only", v) for k, v in DEF.items()]
T1, T2, T3, T4 = [], [], [], []
for rule in ("A", "G"):
    B = np.vstack([LISTS[rule][t] for t in TFS]); ALLB = np.vstack([LISTS[rule][t] for t in ALLT]); sizes = ALLB.sum(1); ai = {t: k for k, t in enumerate(ALLT)}
    for vname, mode, bset in VERS:
        ev = np.array([(mode is None) or (mode == "drop" and t not in bset) or (mode == "only" and t in bset) for t in TFS])
        pool = np.array([(mode != "drop") or (t not in bset) for t in ALLT])
        base = {"rule": rule, "version": vname}
        for c in (2, 3, 4, 5):
            ch = OK & (AZ >= c); nc = OK & (AZ < 1); den = ch.sum(1); sel = ev & (den > 0); cut = f"|z| >= {c}"
            ours = (B & OK)[sel].sum(); inb = (B & ch)[sel].sum()
            T1.append({**base, "cutoff": cut, "TFs evaluated": int(ev.sum()), "TFs with Perturb-seq genes left": int(sel.sum()), "our genes": int(ours), "Perturb-seq genes": int(den[sel].sum()), "in both": int(inb),
                       "share of Perturb-seq genes found in our Regulome (%)": 100 * inb / max(den[sel].sum(), 1), "share of our Regulome genes found in Perturb-seq (%)": 100 * inb / max(ours, 1)})
            own_f, near_f, ranks = [], [], []; npool = int(pool.sum())
            for i in np.flatnonzero(sel):
                t = TFS[i]; f_all = ALLB[:, ch[i]].sum(1).astype(float); own = f_all[ai[t]]; m = pool.copy(); m[ai[t]] = False
                oth = f_all[m]; osz = sizes[m]; near = np.argsort(np.abs(osz - sizes[ai[t]]))[:5]; own_f.append(own); near_f.append(oth[near].mean()); ranks.append(1 + int((oth > own).sum()))
            nl = npool if (mode != "drop") else npool  # own list is in the pool unless dropped; evaluated TFs in (ii) are not broad, so own list is in the pool
            ranks = np.array(ranks); d_ = den[sel].sum()
            T2.append({**base, "cutoff": cut, "TFs": int(sel.sum()), "share found, own list (%)": 100 * np.sum(own_f) / d_, "share found, 5 nearest-size other TFs (%)": 100 * np.sum(near_f) / d_,
                       "gap (points)": 100 * (np.sum(own_f) - np.sum(near_f)) / d_, "lists ranked among": nl, "median rank of own list": float(np.median(ranks)),
                       "TFs with own list in top 5": int((ranks <= 5).sum()), "TFs with own list in top 10%": int((ranks <= np.ceil(.1 * nl)).sum())})
            nC = bC = mT = nU = bU = 0
            for i in np.flatnonzero(ev):
                s_ch = STRAT[ch[i]]; b_ch = B[i][ch[i]]; nb = pd.Series(B[i][nc[i]]).groupby(STRAT[nc[i]]).mean(); okm = pd.Series(s_ch).isin(nb.index).values
                nC += okm.sum(); bC += b_ch[okm].sum(); mT += nb.reindex(s_ch[okm]).sum(); nU += nc[i].sum(); bU += B[i][nc[i]].sum()
            T3.append({**base, "cutoff": cut, "TFs": int(sel.sum()), "changed genes": int(nC), "share bound, changed (%)": 100 * bC / max(nC, 1), "share bound, matched not-changed (%)": 100 * mT / max(nC, 1), "share bound, all not-changed (%)": 100 * bU / max(nU, 1)})
        for lo_, hi_ in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, np.inf)]:
            m = OK & (AZ >= lo_) & (AZ < hi_); nb_ = m.sum(1); bb_ = (m & B).sum(1); s2 = ev & (nb_ > 0)
            T4.append({**base, "|z| bin": f"{lo_}-{hi_}" if np.isfinite(hi_) else f">= {lo_}", "TFs": int(s2.sum()), "genes": int(nb_[s2].sum()), "share bound (%)": 100 * bb_[s2].sum() / max(nb_[s2].sum(), 1)})
    print(rule, flush=True)
TT = {"broad_tfs": INFO, "1_simple_overlap": pd.DataFrame(T1), "2_own_list_vs_other_tfs": pd.DataFrame(T2), "3_changed_vs_matched": pd.DataFrame(T3), "4_dose_response": pd.DataFrame(T4)}
for n, d in TT.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/without_broad_tfs.xlsx") as w:
    for n, d in TT.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 45)
for n in list(TT)[1:]: print("==", n); print(TT[n].round(1).to_string(index=False))
print("done")
