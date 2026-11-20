# m47 (SPEC.md): Stage 1 (m45 code) with different numbers of lab peaks per TF.
import os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M45 = f"{R}/m45_stage1_summary"
s = open(f"{M45}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])  # LISTS (top 5,000, PADS), TFS, OK, AZ, STRAT, BROAD, rule_abc, window, V, UG
O = f"{R}/m47_stage1_peak_set_size/out"; os.makedirs(O, exist_ok=True)
SETS = [("top 2,000", 2000), ("top 5,000", 5000), ("top 10,000", 10000), ("top 20,000", 20000), ("top 50,000", 50000), ("all q < 0.05", "q"), ("full peak set", None)]
L = {(sn, p): {} for sn, _ in SETS for p in PADS}; NPK = {sn: {} for sn, _ in SETS}
for t in TFS:
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"])
    P["rk"] = P.sig.rank(ascending=False, method="first")
    for sn, n in SETS:
        x = P if n is None else (P[P.q >= -np.log10(0.05)] if n == "q" else P[P.rk <= n]); NPK[sn][t] = len(x)
        c5, p5 = x.c.values.astype(str), (x.s + x.off).values; A, _ = rule_abc(c5, p5)
        for p, w in PADS.items(): L[(sn, p)][t] = A | window(c5, p5, w) if w else A
    print(t, len(P), flush=True)
for p in PADS: assert all((L[("top 5,000", p)][t] == LISTS[p][t]).all() for t in TFS), "top 5,000 does not reproduce m45"
print("top 5,000 lists identical to m45", flush=True)
rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1); NT = len(TFS)
GRP = {"all 63 TFs": np.ones(NT, bool), "55 non-broad TFs": np.array([t not in BROAD for t in TFS])}
ROWS, SIZE = [], []
for sn, n in SETS:
    for p in PADS:
        B = np.vstack([L[(sn, p)][t] for t in TFS]); ours = (B & OK).sum(1); npk = np.array([NPK[sn][t] for t in TFS])
        for gn, g in GRP.items():
            SIZE.append({"peak set": sn, "padding": p, "TFs": gn, "peaks per TF, median": float(np.median(npk[g])), "peaks per TF, min": int(npk[g].min()), "peaks per TF, max": int(npk[g].max()),
                         "TFs with fewer peaks than N": int((npk[g] < n).sum()) if isinstance(n, int) else None, "linked genes per TF, median": float(np.median(ours[g])), "total links": int(ours[g].sum()), "TFs with > 4,000 linked genes": int((ours[g] > 4000).sum())})
        for c in (2, 3, 4, 5):
            ch = OK & (AZ >= c); ob = np.zeros(NT); de = ch.sum(1).astype(float); dr = np.zeros((NT, NP)); pt = np.ones(NT)
            for i in range(NT):
                s_ = STRAT[OK[i]]; b = B[i][OK[i]]; p_ = ch[i][OK[i]]; ob[i] = (b & p_).sum()
                for st in np.unique(s_):
                    m = s_ == st; nn = int(m.sum()); nb = int(b[m].sum()); k = int(p_[m].sum())
                    if k and nb: dr[i] += rng.hypergeometric(nb, nn - nb, k, size=NP)
                if de[i]: pt[i] = pval(ob[i], dr[i])
            for gn, g in GRP.items():
                sel = g & (de > 0); o = 100 * ob[sel].sum() / de[sel].sum(); dd = 100 * dr[sel].sum(0) / de[sel].sum()
                ROWS.append({"peak set": sn, "padding": p, "cutoff": f"|z| >= {c}", "TFs": gn, "TFs left": int(sel.sum()), "Perturb-seq genes": int(de[sel].sum()), "in both": int(ob[sel].sum()),
                             "recall (%)": o, "shuffle mean (%)": dd.mean(), "gap (points)": o - dd.mean(), "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd),
                             "TFs above shuffle (p < 0.05, >= 10 genes)": int(((pt < 0.05) & g & (de >= 10)).sum()), "TFs with >= 10 genes": int((g & (de >= 10)).sum())})
        print(sn, p, flush=True)
T1, T2 = pd.DataFrame(ROWS), pd.DataFrame(SIZE)
T1.to_csv(f"{O}/1_recall_and_shuffle.tsv", sep="\t", index=False, float_format="%.4g"); T2.to_csv(f"{O}/2_list_sizes.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_peak_set_size.xlsx") as w:
    T1.to_excel(w, sheet_name="recall and shuffle", index=False); T2.to_excel(w, sheet_name="list sizes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 400)
print(T2.round(1).to_string(index=False)); print(T1[["peak set", "padding", "cutoff", "TFs", "recall (%)", "gap (points)", "p (one-sided)", "TFs above shuffle (p < 0.05, >= 10 genes)"]].round(3).to_string(index=False)); print("done")
