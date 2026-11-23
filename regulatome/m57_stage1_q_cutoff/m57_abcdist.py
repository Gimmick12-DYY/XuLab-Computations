# m57 Amendment 5: ABC map vs plain distance to the TSS, observed recall and precision (top 10,000 peaks).
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m57_stage1_q_cutoff/out"; NT = len(TFS); NGn = len(UG)
LX = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["TargetGene", "distance"]); LX = LX[LX.TargetGene.isin(gi)]; DIST = np.abs(LX["distance"].values.astype(float)); assert len(DIST) == len(Le)
by_el = pd.Series(np.arange(len(Le))).groupby(Le).apply(np.array).to_dict()
F = np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)
SC = pd.DataFrame(F["score"], index=F["tfs"].astype(str), columns=F["genes"].astype(str)).reindex(index=TFS, columns=UG).values; assert (np.isfinite(SC) == OK).all()
PA = np.full(SC.shape, np.nan); PS = np.full(SC.shape, np.nan)
for i in range(NT): ok = OK[i]; n = ok.sum(); PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / n; PS[i, ok] = 100 * (rankdata(SC[i, ok]) - 0.5) / n
DS = [1, 2, 5, 10, 20, 50, 100]; Z0 = lambda: np.zeros((NT, NGn))
PROM = np.zeros((NT, NGn), bool); DIS = np.zeros((NT, NGn), bool); DMIN = np.full((NT, NGn), np.inf); WIN = {d: np.zeros((NT, NGn), bool) for d in DS}
S_prom = Z0(); S_d50 = Z0(); S_dall = Z0(); S_dle200 = Z0(); S_win = {d: Z0() for d in (10, 50)}; NEAR = np.full((NT, NGn), np.inf)
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 10000]
    c5, p5, sg = P.c.values.astype(str), (P.s + P.off).values, P.sig.values; e = elem_of(c5, p5)
    for k in np.flatnonzero(e >= 0):
        rows = by_el.get(e[k])
        if rows is None: continue
        g = Lg[rows]; slf = Ls[rows]; ds = DIST[rows]
        PROM[i, g[slf]] = True; np.maximum.at(S_prom[i], g[slf], sg[k]); gd = g[~slf]; dd = ds[~slf]
        DIS[i, gd] = True; np.minimum.at(DMIN[i], gd, dd); np.maximum.at(S_dall[i], gd, sg[k]); np.maximum.at(S_d50[i], gd[dd <= 50e3], sg[k]); np.maximum.at(S_dle200[i], gd[dd <= 200e3], sg[k])
    for ch in np.unique(c5):
        if ch not in TB: continue
        tp, ix = TB[ch]; m = c5 == ch; x = np.sort(p5[m]); sx = sg[m][np.argsort(p5[m])]
        for d in DS:
            a = np.searchsorted(tp, x - d * 1000); b = np.searchsorted(tp, x + d * 1000, side="right")
            for aa, bb, sv in zip(a, b, sx):
                gg = ix[aa:bb]; WIN[d][i, gg] = True
                if d in (10, 50): S_win[d][i, gg] = np.maximum(S_win[d][i, gg], sv)
        k = np.clip(np.searchsorted(x, tp), 1, len(x) - 1) if len(x) > 1 else np.zeros(len(tp), int)
        dist = np.minimum(np.abs(x[k] - tp), np.abs(x[np.maximum(k - 1, 0)] - tp)) if len(x) > 1 else np.abs(x[0] - tp)
        NEAR[i, ix] = np.minimum(NEAR[i, ix], dist)
    print(t, flush=True)
M_ = lambda M: M & OK
ABC0 = PROM | DIS; CUR = PROM | DIS | WIN[10]; D50 = DIS & (DMIN <= 50e3); D200 = DIS & (DMIN <= 200e3)
assert int(M_(ABC0).sum()) == 169483 and int(M_(CUR).sum()) == 212087
TG = {}
for x in (5, 1): TG[(x, "either")] = OK & (PA > 100 - x); TG[(x, "up")] = OK & (PS > 100 - x); TG[(x, "down")] = OK & (PS < x)
assert int((M_(CUR) & TG[(5, "either")]).sum()) == 11536 and int((M_(CUR) & TG[(1, "either")]).sum()) == 2613; print("sanity passed", flush=True)
def row(fam, name, M):
    M = M_(M); r = {"family": fam, "rule": name, "links": int(M.sum()), "median genes per TF": float(np.median(M.sum(1)))}
    for (x, d), T in TG.items():
        ib = int((M & T).sum()); r[f"top {x}% {d}: in both"] = ib; r[f"top {x}% {d}: recall (%)"] = 100 * ib / T.sum(); r[f"top {x}% {d}: precision (%)"] = 100 * ib / max(M.sum(), 1); r[f"top {x}% {d}: precision / chance"] = (ib / max(M.sum(), 1)) / (x / 100)
    return r
rows = [row("A distance only", f"summit within +-{d} kb of TSS", WIN[d]) for d in DS]
rows += [row("B ABC only", "promoter element only", PROM), row("B ABC only", "promoter + distal ABC <= 50 kb", PROM | D50), row("B ABC only", "promoter + all distal ABC (0 kb rule)", ABC0)]
rows += [row("C current variants", "current (promoter + ABC + 10 kb)", CUR), row("C current variants", "current without distal ABC > 50 kb", PROM | D50 | WIN[10]), row("C current variants", "current without distal ABC > 200 kb", PROM | D200 | WIN[10])]
T = pd.DataFrame(rows)
S_cur = np.maximum.reduce([S_prom, S_dall, S_win[10]]); S_c50 = np.maximum.reduce([S_prom, S_d50, S_win[10]]); S_abc = np.maximum(S_prom, S_dall)
RANKS = [("(i) current rule, by peak signalValue", CUR, S_cur), ("(ii) distance-only +-10 kb, by peak signalValue", WIN[10], S_win[10]), ("(iii) distance-only +-50 kb, by peak signalValue", WIN[50], S_win[50]),
         ("(iv) ABC-only (0 kb rule), by peak signalValue", ABC0, S_abc), ("(v) all genes, by distance of nearest summit to TSS", OK.copy(), -NEAR), ("(vi) current without distal ABC > 50 kb, by peak signalValue", PROM | D50 | WIN[10], S_c50)]
DR = []
for nm, M, Sx in RANKS:
    M = M_(M)
    for Nn in (100, 250, 500, 1000, 2000):
        K = np.zeros_like(M)
        for i in range(NT): g = np.flatnonzero(M[i]); o = g[np.argsort(-Sx[i, g], kind="stable")]; K[i, o[:Nn]] = True
        r = row("D matched list size", nm, K); r["N genes per TF"] = Nn; DR.append(r)
D = pd.DataFrame(DR)
with pd.ExcelWriter(f"{O}/stage1_abc_vs_distance_raw.xlsx") as w:
    T.to_excel(w, sheet_name="rules A-C", index=False); D.to_excel(w, sheet_name="D matched list size", index=False)
    pd.DataFrame({"note": ["Observed only, no shuffle. 63 TFs, top 10,000 lab peaks by signalValue, summits. ABC = canonical m2c Hi-C build, ABC >= 0.02 (self-promoter = the gene's promoter element; distal = non-self linked element; distance = |ABC distance|). Distance-only: summit within +-D of the TSS (ABC GeneList TSS).",
      "Targets: top X % of |score| within the TF (paper per-gene score); up / down from the signed score. Recall = in both / targets; precision = in both / links; precision / chance = precision / (X/100).", "Ranking: per gene the strongest signalValue among the peaks that link it under that rule; (v) ranks every gene by the distance from its TSS to the nearest top-10,000 summit (closest first)."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 340); pd.set_option("display.max_columns", 30)
k = ["rule", "links", "median genes per TF", "top 5% either: in both", "top 5% either: recall (%)", "top 5% either: precision (%)", "top 5% either: precision / chance", "top 1% either: recall (%)", "top 1% either: precision / chance", "top 5% up: precision / chance", "top 5% down: precision / chance"]
print(T[k].round(3).to_string(index=False)); print(D[["N genes per TF"] + k].round(3).to_string(index=False)); print("done")
