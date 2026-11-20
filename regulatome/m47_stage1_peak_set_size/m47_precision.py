# m47 Amendment 1: precision beside recall from table 1 (same shuffle), our links over TFs with >= 1 Perturb-seq gene.
import os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M45 = f"{R}/m45_stage1_summary"
s = open(f"{M45}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m47_stage1_peak_set_size/out"  # set after the exec, which defines its own O
SETS = [("top 5,000", 5000), ("top 10,000", 10000), ("top 20,000", 20000)]; L = {(sn, p): {} for sn, _ in SETS for p in PADS}
for t in TFS:
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P["rk"] = P.sig.rank(ascending=False, method="first")
    for sn, n in SETS:
        x = P[P.rk <= n]; c5, p5 = x.c.values.astype(str), (x.s + x.off).values; A, _ = rule_abc(c5, p5)
        for p, w in PADS.items(): L[(sn, p)][t] = A | window(c5, p5, w) if w else A
T1 = pd.read_csv(f"{O}/1_recall_and_shuffle.tsv", sep="\t", dtype={"padding": str}); NT = len(TFS)
GRP = {"all 63 TFs": np.ones(NT, bool), "55 non-broad TFs": np.array([t not in BROAD for t in TFS])}; rows = []
for (sn, p), LL in L.items():
    B = np.vstack([LL[t] for t in TFS]); ours = (B & OK).sum(1)
    for c in (2, 3, 4, 5):
        de = (OK & (AZ >= c)).sum(1)
        for gn, g in GRP.items():
            sel = g & (de > 0); r = T1[(T1["peak set"] == sn) & (T1.padding == p) & (T1.cutoff == f"|z| >= {c}") & (T1.TFs == gn)].iloc[0]
            assert int(r["TFs left"]) == int(sel.sum()) and int(r["Perturb-seq genes"]) == int(de[sel].sum())
            our = int(ours[sel].sum()); ib = int(r["in both"]); sib = r["shuffle mean (%)"] * r["Perturb-seq genes"] / 100
            rows.append({"peak set": sn, "padding": p, "cutoff": f"|z| >= {c}", "TFs": gn, "TFs pooled (>= 1 Perturb-seq gene)": int(sel.sum()), "our links": our, "Perturb-seq genes": int(r["Perturb-seq genes"]), "in both": ib,
                         "precision observed (%)": 100 * ib / our, "precision shuffle mean (%)": 100 * sib / our, "precision gap (points)": 100 * (ib - sib) / our, "observed / shuffle": ib / sib,
                         "recall (%)": r["recall (%)"], "recall gap (points)": r["gap (points)"], "p (one-sided, same for both)": r["p (one-sided)"]})
D = pd.DataFrame(rows); D.to_csv(f"{O}/3_precision.tsv", sep="\t", index=False, float_format="%.4g")
T2 = pd.read_csv(f"{O}/2_list_sizes.tsv", sep="\t")
with pd.ExcelWriter(f"{O}/stage1_peak_set_size.xlsx") as w:
    T1.to_excel(w, sheet_name="recall and shuffle", index=False); T2.to_excel(w, sheet_name="list sizes", index=False); D.to_excel(w, sheet_name="precision", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
print(D.drop(columns=["Perturb-seq genes"]).round(3).to_string(index=False)); print("done")
