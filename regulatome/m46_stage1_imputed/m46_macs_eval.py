# m46 Amendment 1: Stage 1 evaluation of MACS peaks called on imputed data (P1, P2, P3a, P3b), top 5,000 and equal list size, beside lab / I2 / I2loo.
import os, json
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M45 = f"{R}/m45_stage1_summary"; M = f"{R}/m46_stage1_imputed"
s = open(f"{M45}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])  # LISTS (lab), PADS, TFS, OK, AZ, STRAT, BROAD, rule_abc, window
O = f"{M}/out"; PK = f"{M}/peaks"
def read(t, k):
    f = f"{PK}/{t}/{k}.narrowPeak" if k in ("P1", "P2") else f"{PK}/{t}/{k}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 4, 6, 8, 9], names=["c", "s", "score", "sig", "q", "off"]); P["sm"] = P.s + P.off
    P = P.sort_values(["q", "sig"] if k.startswith("P3") else ["score", "sig"], ascending=False).reset_index(drop=True); return P
def link(P, n, w): x = P.iloc[:n]; c, p = x.c.values.astype(str), x.sm.values; A, _ = rule_abc(c, p); return A | window(c, p, w) if w else A
INFO = pd.DataFrame([json.load(open(f"{PK}/{t}/info.json")) for t in TFS if os.path.exists(f"{PK}/{t}/info.json")]); INFO.to_csv(f"{O}/6_macs_on_imputed_info.tsv", sep="\t", index=False, float_format="%.4g")
TFI = [t for t in TFS if os.path.exists(f"{PK}/{t}/info.json")]; print("TFs with peaks:", len(TFI), flush=True)
L = {}
for k in ("P1", "P2", "P3a", "P3b"):
    for p, w in PADS.items():
        L[(k, "top 5,000", p)] = {}; L[(k, "equal size", p)] = {}
    for t in TFI:
        P = read(t, k)
        for p, w in PADS.items():
            if P is None: L[(k, "top 5,000", p)][t] = np.zeros(len(UG), bool); L[(k, "equal size", p)][t] = np.zeros(len(UG), bool); continue
            L[(k, "top 5,000", p)][t] = link(P, 5000, w); target = int(LISTS[p][t].sum()); a, b = 1, len(P)
            if link(P, b, w).sum() < target: L[(k, "equal size", p)][t] = link(P, b, w); continue
            while a < b:
                mid = (a + b) // 2
                if link(P, mid, w).sum() >= target: b = mid
                else: a = mid + 1
            L[(k, "equal size", p)][t] = link(P, a, w)
    print("lists", k, flush=True)
for p in PADS: L[("lab peaks", "top 5,000", p)] = LISTS[p]
ti = {t: i for i, t in enumerate(TFS)}; EX3 = {"NRF1", "E4F1", "DNMT1"}; SETS = {"all": set(), "without NRF1, E4F1, DNMT1": EX3, "without broad TFs": BROAD}
rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1)
P1, PT, JA = [], [], []
for (k, ver, p), LL in L.items():
    for c in (2, 3, 4, 5):
        ob, de, ou, dr = np.zeros(len(TFI)), np.zeros(len(TFI)), np.zeros(len(TFI)), np.zeros((len(TFI), NP))
        for j, t in enumerate(TFI):
            i = ti[t]; s_ = STRAT[OK[i]]; b = LL[t][OK[i]]; p_ = (AZ[i] >= c)[OK[i]]; ob[j] = (b & p_).sum(); de[j] = p_.sum(); ou[j] = b.sum()
            for st in np.unique(s_):
                m = s_ == st; n = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
                if kk and nb: dr[j] += rng.hypergeometric(nb, n - nb, kk, size=NP)
            if de[j]: PT.append({"list": k, "version": ver, "padding": p, "cutoff": f"|z| >= {c}", "TF": t, "our genes": int(ou[j]), "Perturb-seq genes": int(de[j]), "in both": int(ob[j]), "recall (%)": 100 * ob[j] / de[j], "shuffle mean (%)": 100 * dr[j].mean() / de[j], "p (one-sided)": pval(ob[j], dr[j])})
        for sn, ex_ in SETS.items():
            sel = np.array([t not in ex_ for t in TFI]) & (de > 0); o = 100 * ob[sel].sum() / de[sel].sum(); dd = 100 * dr[sel].sum(0) / de[sel].sum()
            per = [r_ for r_ in PT[-int((de > 0).sum()):] if r_["TF"] not in ex_ and r_["Perturb-seq genes"] >= 10]
            P1.append({"list": k, "version": ver, "padding": p, "cutoff": f"|z| >= {c}", "TFs": sn, "n TFs": int(sel.sum()), "our genes": int(ou[sel].sum()), "median genes per TF": float(np.median(ou[sel])), "Perturb-seq genes": int(de[sel].sum()), "in both": int(ob[sel].sum()),
                       "recall (%)": o, "shuffle mean (%)": dd.mean(), "observed minus shuffle mean (points)": o - dd.mean(), "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd),
                       "TFs above shuffle (p < 0.05, >= 10 genes)": sum(r_["p (one-sided)"] < 0.05 for r_ in per), "TFs with >= 10 genes": len(per)})
    if k != "lab peaks":
        for t in TFI: a_, b_ = LL[t], LISTS[p][t]; JA.append({"list": k, "version": ver, "padding": p, "TF": t, "imputed-peak genes": int(a_.sum()), "lab genes": int(b_.sum()), "Jaccard": (a_ & b_).sum() / max((a_ | b_).sum(), 1)})
    print("eval", k, ver, p, flush=True)
prev = pd.read_csv(f"{O}/1_pooled.tsv", sep="\t", dtype={"padding": str}); prev = prev[prev.list.isin(["I2", "I2loo"])].assign(version="equal size (bins)")
T1 = pd.concat([pd.DataFrame(P1), prev], ignore_index=True)
TT = {"7_macs_pooled": T1, "8_macs_per_tf": pd.DataFrame(PT), "9_macs_jaccard_vs_lab": pd.DataFrame(JA), "6_macs_on_imputed_info": INFO}
for n, d in TT.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_imputed_macs.xlsx") as w:
    for n, d in TT.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 1000)
cols = ["list", "version", "padding", "cutoff", "median genes per TF", "recall (%)", "observed minus shuffle mean (points)", "p (one-sided)", "TFs above shuffle (p < 0.05, >= 10 genes)"]
print(T1[T1.TFs == "all"][cols].round(2).to_string(index=False))
print(INFO.describe().T[["50%", "min", "max"]].round(3).to_string()); print(INFO["matrix used"].value_counts().to_string())
print(pd.DataFrame(JA).groupby(["list", "version", "padding"]).Jaccard.median().round(3).to_string()); print("done")
