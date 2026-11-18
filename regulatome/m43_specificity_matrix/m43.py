# m43 (SPEC.md): TF x TF specificity matrices (tables). Plots in m43_plot.py.
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
R = "/work/users/d/y/dyy12/XuLab/regulatome"
s40 = open(f"{R}/m40_overlap_reference/m40.py").read(); exec(s40[:s40.index("R1, R1T, DOSE")])
O = f"{R}/m43_specificity_matrix/out"; BROAD = sorted(t for t in ALLT if LISTS["A"][t].sum() > 4000); print("broad", BROAD, flush=True)
# order: Jaccard clustering of rule-A lists of the 63
B = np.vstack([LISTS["A"][t] for t in TFS]).astype(float); inter = B @ B.T; un = B.sum(1)[:, None] + B.sum(1)[None, :] - inter; J = 1 - inter / np.maximum(un, 1); np.fill_diagonal(J, 0)
ORDER = [TFS[k] for k in leaves_list(linkage(squareform(J, checks=False), "average"))]
pd.DataFrame({"position": range(1, len(ORDER) + 1), "TF": ORDER, "broad": [t in BROAD for t in ORDER]}).to_csv(f"{O}/tf_order.tsv", sep="\t", index=False)
oi = [TFS.index(t) for t in ORDER]
for rule in ("A", "G"):
    for c in (2, 3, 4, 5):
        rows = []
        for t in ORDER:
            i = TFS.index(t); ch = OK[i] & (AZ[i] >= c); nc = OK[i] & (AZ[i] < 1)
            if ch.sum() < 10: continue
            for u in ALLT:
                l = LISTS[rule][u]; raw = 100 * (l & ch).sum() / ch.sum(); bg = 100 * (l & nc).sum() / nc.sum()
                rows.append({"row TF (knocked down)": t, "column TF (Regulome list)": u, "Perturb-seq genes": int(ch.sum()), "raw share (%)": raw, "share of non-changing genes (%)": bg, "adjusted (points)": raw - bg})
        L_ = pd.DataFrame(rows); L_.to_csv(f"{O}/long_rule{rule}_z{c}.tsv.gz", sep="\t", index=False, float_format="%.4g")
        for col, nm in (("raw share (%)", "raw"), ("adjusted (points)", "adjusted")):
            M = L_.pivot(index="row TF (knocked down)", columns="column TF (Regulome list)", values=col); rr = [t for t in ORDER if t in M.index]
            M.reindex(index=rr, columns=ORDER + [u for u in ALLT if u not in ORDER]).to_csv(f"{O}/matrix_{nm}_rule{rule}_z{c}.tsv", sep="\t", float_format="%.4g")
        M = L_.pivot(index="row TF (knocked down)", columns="column TF (Regulome list)", values="raw share (%)"); Zc = (M - M.mean(0)) / M.std(0, ddof=0)
        Zc.reindex(index=[t for t in ORDER if t in Zc.index], columns=ORDER + [u for u in ALLT if u not in ORDER]).to_csv(f"{O}/matrix_columnz_rule{rule}_z{c}.tsv", sep="\t", float_format="%.4g")
        print(rule, c, L_["row TF (knocked down)"].nunique(), flush=True)
# Spearman: binding score = max over linked elements of (max top-5,000 summit signal in element) x ABC
P2 = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score"]); P2 = P2[P2.TargetGene.isin(gi)]
Pe = (P2.chr + ":" + P2.start.astype(str) + "-" + P2.end.astype(str)).map(ek).values.astype(int); Pg = P2.TargetGene.map(gi).values; Pa = P2["ABC.Score"].values
SC = {}
for t in ORDER:
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 5000]
    e = elem_of(P.c.values.astype(str), (P.s + P.off).values); ev = np.zeros(NE); np.maximum.at(ev, e[e >= 0], P.sig.values[e >= 0]); g = np.zeros(NG); np.maximum.at(g, Pg, ev[Pe] * Pa); SC[t] = g
SP = np.full((len(ORDER), len(ORDER)), np.nan)
for a, t in enumerate(ORDER):
    i = TFS.index(t); ok = OK[i]; y = rankdata(AZ[i][ok])
    for b, u in enumerate(ORDER): SP[a, b] = np.corrcoef(rankdata(SC[u][ok]), y)[0, 1]
pd.DataFrame(SP, index=ORDER, columns=ORDER).to_csv(f"{O}/matrix_spearman_ruleA.tsv", sep="\t", float_format="%.4g"); print("spearman done", flush=True)
