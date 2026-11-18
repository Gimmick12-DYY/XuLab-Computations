import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
import sys; sys.argv = ["m15.py", "raw"]; s15 = open(f"{R}/m15_gene_to_tf/m15.py").read()
exec(s15[:s15.index("# ---- binding percentiles")]); TFA, ZALL, UGA = list(TF), ZA.copy(), list(UG)
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])
assert list(UG) == UGA
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); T63 = sorted(set(T25.tf))
Z25 = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=T63, columns=UG).values.astype(float)
OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(T63):
    for o in OWN.get(t, [t]):
        if o in gi: Z25[i, gi[o]] = np.nan
LA, LG = {}, {}
for t in sorted(set(TFA) | set(T63)):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]
    c5, p5 = P.c.values.astype(str), (P.s + P.off).values; A, _ = rule_abc(c5, p5); LA[t] = A; LG[t] = A | window(c5, p5, 10000) | cicero(t)[0]
rows = []
for cells, Z, tfl in (("all cells", ZALL, TFA), ("top-25% sgRNA cells", Z25, T63)):
    for tset in ("all TFs in this matrix", "same 63 TFs"):
        sel = [k for k, t in enumerate(tfl) if tset.startswith("all") or t in T63]
        for rule, L in (("A current", LA), ("G (A + 10 kb + coordinated binding)", LG)):
            for c in (2, 3, 4, 5):
                pg = ob = bt = 0; nt = 0
                for k in sel:
                    ok = np.isfinite(Z[k]); p = ok & (np.abs(np.nan_to_num(Z[k])) >= c); l = L[tfl[k]] & ok
                    pg += p.sum(); ob += (p & l).sum(); bt += l.sum(); nt += p.any()
                rows.append({"cells": cells, "TFs": tset, "n TFs": len(sel), "rule": rule, "cutoff": f"|z| >= {c}", "Perturb-seq genes left": int(pg), "TFs with any gene left": int(nt), "our genes": int(bt), "in both": int(ob),
                             "share of Perturb-seq genes found (%)": 100 * ob / pg if pg else np.nan, "share of our genes found (%)": 100 * ob / bt})
T = pd.DataFrame(rows); T.to_csv(f"{R}/m41_all_cells_cutoff/out/cutoff_all_vs_top25.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); print(T.round(1).to_string(index=False))
