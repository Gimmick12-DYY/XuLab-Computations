# m49: |z| cutoffs as fold change over the Stage 1 TF x gene pairs.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])  # UG, TFS, Z, OK, AZ, ex, STRAT, dec
O = f"{R}/m49_z_fold_change/out"
F = np.load(f"{O}/unit_fold_change.npz", allow_pickle=True); units = F["units"].astype(str); g_ = F["genes"].astype(str); gix = pd.Series(np.arange(len(g_)), index=g_); gix = gix[~gix.index.duplicated(keep="last")]; cols = gix.reindex(UG).values
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; LA, LB, CELLS = {}, {}, {}
for k, u in enumerate(units):
    t = ALIAS.get(u.split("|")[0], u.split("|")[0]); a = np.full(len(UG), np.nan); b = a.copy(); m = ~np.isnan(cols); a[m] = F["logdiff"][k][cols[m].astype(int)] / np.log(2); b[m] = F["log2fc_b"][k][cols[m].astype(int)]
    LA.setdefault(t, []).append(a); LB.setdefault(t, []).append(b); CELLS[t] = CELLS.get(t, 0) + int(F["cells"][k])
A_ = np.vstack([np.mean(LA[t], 0) for t in TFS]); B_ = np.vstack([np.mean(LB[t], 0) for t in TFS]); NC = np.array([CELLS[t] for t in TFS])
assert np.isfinite(A_[OK]).all() and np.isfinite(B_[OK]).all()
# check: z recomputed sign agrees with the log1p difference
print("sign(z) == sign(a) share:", float((np.sign(Z[OK]) == np.sign(A_[OK])).mean()), flush=True)
pct = lambda x: 100 * (2 ** x - 1)
def summ(m, lab, extra):
    za = AZ[m]; a = np.abs(A_[m]); b = np.abs(B_[m]); zs = Z[m]
    if m.sum() == 0: return dict(extra, **{"group": lab, "pairs": 0})
    q = lambda v: np.percentile(v, [25, 50, 75])
    qa, qb = q(a), q(b)
    return dict(extra, **{"group": lab, "pairs": int(m.sum()), "share down (%)": 100 * (zs < 0).mean(), "share up (%)": 100 * (zs > 0).mean(),
        "|log2FC| (a) median": qa[1], "(a) IQR": f"{qa[0]:.3f}-{qa[2]:.3f}", "(a) median as % change": pct(qa[1]), "(a) IQR as % change": f"{pct(qa[0]):.1f}-{pct(qa[2]):.1f}",
        "|log2FC| (b) median": qb[1], "(b) IQR": f"{qb[0]:.3f}-{qb[2]:.3f}", "(b) median as % change": pct(qb[1]), "(b) IQR as % change": f"{pct(qb[0]):.1f}-{pct(qb[2]):.1f}",
        "down: median log2FC (a)": float(np.median(A_[m][zs < 0])) if (zs < 0).any() else np.nan, "up: median log2FC (a)": float(np.median(A_[m][zs > 0])) if (zs > 0).any() else np.nan})
BANDS = [("|z| 1.9-2.1", 1.9, 2.1), ("|z| 2.9-3.1", 2.9, 3.1), ("|z| 3.8-4.2", 3.8, 4.2), ("|z| 4.5-5.5", 4.5, 5.5)]; CUTS = [("|z| >= 2", 2), ("|z| >= 3", 3), ("|z| >= 4", 4), ("|z| >= 5", 5)]
def sets(): return [(l, OK & (AZ >= lo) & (AZ <= hi)) for l, lo, hi in BANDS] + [(l, OK & (AZ >= c)) for l, c in CUTS]
MAIN = pd.DataFrame([summ(m, l, {}) for l, m in sets()])
dex = dec(ex); DEC = pd.DataFrame([summ(m & (dex[None, :] == k), l, {"NTC expression decile (1 = lowest)": k + 1}) for k in range(10) for l, m in sets()])
ter = pd.qcut(pd.Series(NC).rank(method="first"), 3, labels=["fewest cells", "middle", "most cells"]).values
TER = pd.DataFrame([summ(m & (ter == tt)[:, None], l, {"TF tertile by top-25% cells": tt, "cells per TF (range)": f"{NC[ter == tt].min()}-{NC[ter == tt].max()}"}) for tt in ["fewest cells", "middle", "most cells"] for l, m in sets()])
o = np.argsort(NC); pick = list(o[:3]) + list(o[-3:])
ONE = pd.DataFrame([summ(m & (np.arange(len(TFS)) == i)[:, None], l, {"TF": TFS[i], "top-25% cells": int(NC[i])}) for i in pick for l, m in sets() if l.startswith("|z| >=")])
CELLT = pd.DataFrame({"TF": TFS, "top-25% cells": NC}).sort_values("top-25% cells")
print("cells per TF: median", float(np.median(NC)), "range", NC.min(), NC.max(), flush=True)
note = pd.DataFrame({"note": [
 "z (m23 kd_response_m23.py resp(), read by Stage 1 via m23_eval.py load() -> observed_kd_top25.tsv.gz): per knockdown unit, top 25% of its own cells by GEM-percentile sgRNA UMI; d = mean log1p(CP10k) of those cells minus the GEM-weighted NTC mean (all NTC cells of the same GEMs, weighted by the unit's cells per GEM); z = d / sqrt(sum_g [n_g/n^2 + n_g^2/(n^2 N_g)] * V_g + 1e-6), V_g = NTC variance within GEM g. Raw z, not double-centred. Dual-construct TFs: mean of the construct z; HP1: mean of CBX1/3/5; own gene removed; 11,228-gene universe with NTC detection >= 10%.",
 "(a) log2FC = d / ln 2, i.e. the log1p(CP10k) difference the z is built on, in log2 units. (b) log2FC = log2((mean CP10k knockdown + 0.1) / (GEM-weighted mean CP10k NTC + 0.1)); same cells and reference. Per TF, aggregated like z (mean over constructs / CBX units).",
 "% change of a |log2FC| value x: 100 (2^x - 1). Bands are |z| windows; cutoffs are all pairs at or above. Down/up from the sign of z. Expression deciles = NTC mean expression deciles used for the matched gene shuffle. 63 Stage 1 TFs."]})
with pd.ExcelWriter("/work/users/d/y/dyy12/XuLab/regulatome/m49_z_fold_change/out/z_to_fold_change.xlsx") as w:
    MAIN.to_excel(w, sheet_name="main", index=False); DEC.to_excel(w, sheet_name="by expression decile", index=False); TER.to_excel(w, sheet_name="by cells per TF tertile", index=False); ONE.to_excel(w, sheet_name="TFs fewest-most cells", index=False); CELLT.to_excel(w, sheet_name="cells per TF", index=False); note.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(MAIN.round(3).to_string(index=False)); print(DEC[DEC.group.isin(["|z| 1.9-2.1", "|z| >= 2"])][["NTC expression decile (1 = lowest)", "group", "pairs", "|log2FC| (a) median", "|log2FC| (b) median", "(a) median as % change", "(b) median as % change"]].round(3).to_string(index=False))
print(TER[TER.group.isin(["|z| 1.9-2.1", "|z| >= 2"])].iloc[:, :8].round(3).to_string(index=False)); print(ONE[ONE.group == "|z| >= 2"].iloc[:, :9].round(3).to_string(index=False)); print("done")
