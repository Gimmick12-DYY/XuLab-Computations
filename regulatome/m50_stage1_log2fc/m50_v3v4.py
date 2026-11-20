# m50 Amendment 1: responder definitions V3 (|log2FC| and |z| >= 3) and V4 (log1p-scale fold change / ln 2, alone and with |z| >= 2), same Stage 1 evaluation as m50_stage1.py.
import numpy as np, pandas as pd
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read()
exec(SRC50[:SRC50.index("# ---- responder counts")])  # L1, NC, TER, DEX, Z, OK, AZ, TFS, UG, LISTS, STRAT, BROAD
st = SRC50[SRC50.index("# ---- Stage 1"):SRC50.index("ST, LOO = stage1(")]; exec(st)  # SETS, NP, pval, stage1()
from scipy.stats import spearmanr
O = "/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/out"
F4 = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m49_z_fold_change/out/unit_fold_change.npz", allow_pickle=True); u4 = F4["units"].astype(str); g4 = F4["genes"].astype(str); gx = pd.Series(np.arange(len(g4)), index=g4); gx = gx[~gx.index.duplicated(keep="last")]; c4 = gx.reindex(UG).values; m4 = ~np.isnan(c4); i4 = c4[m4].astype(int)
D = {}
for k, u in enumerate(u4):
    t = ALIAS.get(u.split("|")[0], u.split("|")[0]); a = np.full(len(UG), np.nan); a[m4] = F4["logdiff"][k][i4] / np.log(2); D.setdefault(t, []).append(a)
L4 = np.vstack([np.mean(D[t], 0) for t in TFS]); L4[~OK] = np.nan; assert np.isfinite(L4[OK]).all()
print("sign agreement V4 vs z:", float((np.sign(L4[OK]) == np.sign(Z[OK])).mean()), flush=True)
A1, A4 = np.abs(np.nan_to_num(L1)), np.abs(np.nan_to_num(L4))
VS = [("V3 |log2FC| and |z| >= 3", f">= {c}", OK & (A1 >= c) & (AZ >= 3), L1) for c in (0.25, 0.5, 0.75, 1.0)]
VS += [("V4a |log2 log1p-FC|", f">= {c}", OK & (A4 >= c), L4) for c in (0.1, 0.2, 0.3, 0.5)]
VS += [("V4b |log2 log1p-FC| and |z| >= 2", f">= {c}", OK & (A4 >= c) & (AZ >= 2), L4) for c in (0.1, 0.2, 0.3, 0.5)]
rem = OK & (AZ >= 3) & (A1 < 0.25); print("pairs with |z| >= 3 and |log2FC| < 0.25:", int(rem.sum()), "of", int((OK & (AZ >= 3)).sum()), "| down among them (%):", float(100 * (Z[rem] < 0).mean()), "| deciles 1/5/10 (%):", [round(100 * rem[:, DEX == k].sum() / rem.sum(), 1) for k in (0, 4, 9)], flush=True)
RC = []
for v, c, M, S_ in VS:
    n = M.sum(1); dn = (M & (np.nan_to_num(S_) < 0)).sum(); rho = spearmanr(n, NC); tot = max(M.sum(), 1)
    d = {"version": v, "cutoff": c, "pairs": int(M.sum()), "down (%)": 100 * dn / tot, "TFs >= 1 responder": int((n >= 1).sum()), "TFs >= 10 responders": int((n >= 10).sum()), "median responders per TF": float(np.median(n)),
         "median |z| of pairs": float(np.median(AZ[M])) if M.any() else np.nan, "Spearman responders vs cells per TF": rho.statistic, "Spearman p": rho.pvalue,
         "median responders, TFs with 25-42 cells": float(np.median(n[TER == "25-42"])), "43-73 cells": float(np.median(n[TER == "43-73"])), "74-216 cells": float(np.median(n[TER == "74-216"]))}
    for k in range(10): d[f"decile {k + 1} (%)"] = 100 * M[:, DEX == k].sum() / tot
    RC.append(d)
RC = pd.DataFrame(RC); RC = pd.concat([RC, pd.DataFrame([{"version": "|z| >= 3 and |log2FC| < 0.25 (removed by V3 at 0.25)", "cutoff": "", "pairs": int(rem.sum()), "down (%)": 100 * (Z[rem] < 0).mean(), **{f"decile {k + 1} (%)": 100 * rem[:, DEX == k].sum() / rem.sum() for k in range(10)}}])], ignore_index=True)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(RC[["version", "cutoff", "pairs", "down (%)", "TFs >= 1 responder", "TFs >= 10 responders", "median responders per TF", "median |z| of pairs", "Spearman responders vs cells per TF", "Spearman p", "median responders, TFs with 25-42 cells", "43-73 cells", "74-216 cells", "decile 1 (%)", "decile 5 (%)", "decile 10 (%)"]].round(3).to_string(index=False), flush=True)
ST, LOO = stage1(VS, 20261118, True)
note = pd.DataFrame({"note": ["V3: |log2FC| (log2((KD CP10k + 0.1)/(NTC CP10k + 0.1))) >= cutoff and |z| >= 3; direction = sign of log2FC.",
  "V4: (mean log1p(CP10k) in the top-25% cells - GEM-weighted NTC mean) / ln 2, i.e. the numerator of z in log2 units (m49 measure a); V4a alone, V4b with |z| >= 2; direction = sign of V4.",
  "Same cells, universe, lists and Stage 1 evaluation as stage1_log2fc.xlsx. Matched gene shuffle: same procedure and seed 20261118, one run for all rows of this file; draws are not identical to job 9984606 (different responder sets)."]})
with pd.ExcelWriter(f"{O}/stage1_log2fc_v3_v4.xlsx") as w:
    RC.to_excel(w, sheet_name="responder counts", index=False)
    for v, sh in (("V3", "Stage 1 V3"), ("V4a", "Stage 1 V4a"), ("V4b", "Stage 1 V4b")): ST[ST.version.str.startswith(v)].to_excel(w, sheet_name=sh, index=False)
    LOO.to_excel(w, sheet_name="leave one TF out 10 kb", index=False); note.to_excel(w, sheet_name="notes", index=False)
K = ST[(ST.padding == "10 kb") & (ST["TF set"] == "all 63 TFs")]
print(K.drop(columns=["padding", "TF set", "shuffle 2.5-97.5 (%)"]).round(3).to_string(index=False))
print(ST[(ST.padding == "10 kb") & (ST["TF set"] != "all 63 TFs") & (ST.direction == "all")][["version", "cutoff", "TF set", "TFs contributing", "recall (%)", "gap (points)", "p (one-sided)"]].round(3).to_string(index=False))
print(LOO[LOO.direction == "all"].round(2).to_string(index=False)); print("done")
