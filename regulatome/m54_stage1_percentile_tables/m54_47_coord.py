# m54: generated from m52_47_coord.py (m52) by gen54.py.
# m47 Amendment 4: base lists (top 5k / 10k / 20k, padding 0 / 10 / 50 kb) with and without the coordinated-binding links (m38 rule E).
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M45 = f"{R}/m45_stage1_summary"
s = open("/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/m54_tables5k.py").read(); exec(s[:s.index("VERS = {")])
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[src.index("def cicero(t):"):src.index("T25 = pd.read_csv")])  # cicero(t) exactly as m38
O = "/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out47"  # set after the execs
NS = [("top 5,000", 5000), ("top 10,000", 10000), ("top 20,000", 20000)]; L = {}; CI = {}
for t in TFS:
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P["rk"] = P.sig.rank(ascending=False, method="first")
    ci, has = cicero(t); CI[t] = (ci, has)
    for sn, n in NS:
        x = P[P.rk <= n]; c5, p5 = x.c.values.astype(str), (x.s + x.off).values; A, _ = rule_abc(c5, p5)
        for p, w in PADS.items():
            b = A | window(c5, p5, w) if w else A; L[(sn, p, "base")] = L.get((sn, p, "base"), {}); L[(sn, p, "base")][t] = b
            L[(sn, p, "base + coordinated binding")] = L.get((sn, p, "base + coordinated binding"), {}); L[(sn, p, "base + coordinated binding")][t] = b | ci
    print(t, flush=True)
print("TFs with coordinated-binding links:", sum(CI[t][1] for t in TFS), "of", len(TFS), flush=True)
rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1); NT = len(TFS)
GRP = {"all 63 TFs": np.ones(NT, bool), "55 non-broad TFs": np.array([t not in BROAD for t in TFS])}; ROWS, ADD = [], []
for (sn, p, ver), LL in L.items():
    B = np.vstack([LL[t] for t in TFS]); ours = (B & OK).sum(1)
    if ver != "base":
        B0 = np.vstack([L[(sn, p, "base")][t] for t in TFS]); Aadd = B & ~B0 & OK
    for c in (2, 3, 4, 5):
        ch = OK & (AZ >= c); de = ch.sum(1); ob = (B & ch).sum(1).astype(float); dr = np.zeros((NT, NP))
        for i in range(NT):
            s_ = STRAT[OK[i]]; b = B[i][OK[i]]; p_ = ch[i][OK[i]]
            for st in np.unique(s_):
                m = s_ == st; nn = int(m.sum()); nb = int(b[m].sum()); k = int(p_[m].sum())
                if k and nb: dr[i] += rng.hypergeometric(nb, nn - nb, k, size=NP)
        for gn, g in GRP.items():
            sel = g & (de > 0); o = ob[sel].sum(); dd = dr[sel].sum(0); nP = de[sel].sum(); nO = ours[sel].sum()
            ROWS.append({"peak set": sn, "padding": p, "links": ver, "cutoff": f"top {CUTL[c]}% of genes", "TFs": gn, "TFs pooled": int(sel.sum()), "total links": int(nO), "median genes per TF": float(np.median(ours[g])), "TFs with > 4,000 genes": int((ours[g] > 4000).sum()),
                         "Perturb-seq genes": int(nP), "in both": int(o), "recall (%)": 100 * o / nP, "shuffle mean recall (%)": 100 * dd.mean() / nP, "gap (points)": 100 * (o - dd.mean()) / nP, "observed / shuffle": o / dd.mean(),
                         "shuffle 2.5-97.5 (%)": f"{100*np.percentile(dd,2.5)/nP:.1f}-{100*np.percentile(dd,97.5)/nP:.1f}", "p (one-sided)": pval(o, dd), "precision (%)": 100 * o / nO})
            if ver != "base":
                B0s = (B0 & OK)[sel]; ad = Aadd[sel]
                ADD.append({"peak set": sn, "padding": p, "cutoff": f"top {CUTL[c]}% of genes", "TFs": gn, "added links": int(ad.sum()), "added links responding": int((ad & ch[sel]).sum()), "share of added links responding (%)": 100 * (ad & ch[sel]).sum() / max(ad.sum(), 1),
                            "base links": int(B0s.sum()), "share of base links responding (%)": 100 * (B0s & ch[sel]).sum() / B0s.sum()})
    print(sn, p, ver, flush=True)
T1, T2 = pd.DataFrame(ROWS), pd.DataFrame(ADD)
T1.to_csv(f"{O}/6_coordinated_binding.tsv", sep="\t", index=False, float_format="%.4g"); T2.to_csv(f"{O}/7_coordinated_binding_added_links.tsv", sep="\t", index=False, float_format="%.4g")
chk = T1[(T1["peak set"] == "top 5,000") & (T1.padding == "10 kb") & (T1.links != "base") & (T1.TFs == "all 63 TFs")]["recall (%)"].round(2).tolist(); print("top 5,000 + 10 kb + coordinated binding recall:", chk, "(|z| version m40 rule G was 24.74 / 28.26 / 32.95 / 37.46; not expected to match)")
with pd.ExcelWriter(f"{O}/stage1_peaks_padding_coordinated_binding.xlsx") as w:
    T1.to_excel(w, sheet_name="recall, shuffle, precision", index=False); T2.to_excel(w, sheet_name="added links alone", index=False)
    pd.DataFrame({"note": ["Coordinated-binding links = m38 rule E (frozen Cicero edges > 25 kb, edge end within +/- 1 kb of a TSS); top 5,000 + 10 kb + coordinated binding = m38/m40 rule G.", "Co-binding partner rule not included in this run; it can be added the same way.", f"Recall check vs rule G: {chk}"]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 25); pd.set_option("display.max_rows", 500)
print(T1[T1.TFs == "all 63 TFs"][["peak set", "padding", "links", "cutoff", "total links", "median genes per TF", "TFs with > 4,000 genes", "recall (%)", "gap (points)", "observed / shuffle", "p (one-sided)", "precision (%)"]].round(3).to_string(index=False))
print(T2[T2.TFs == "all 63 TFs"].round(2).to_string(index=False)); print("done")
