# m50 (SPEC.md): Stage 1 with log2 fold change responders (V1 fold change alone, V2 fold change and |z| >= 2) beside the |z| reference.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])  # UG, TFS, Z, OK, AZ, ex, STRAT, dec, LISTS (top 10,000; 0 / 10 / 50 kb), BROAD
O = f"{R}/m50_stage1_log2fc/out"
F = np.load(f"{O}/unit_mean_cp10k.npz", allow_pickle=True); units = F["units"].astype(str); g_ = F["genes"].astype(str); gix = pd.Series(np.arange(len(g_)), index=g_); gix = gix[~gix.index.duplicated(keep="last")]; cols = gix.reindex(UG).values; mc = ~np.isnan(cols); ci = cols[mc].astype(int)
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}
def lfc(pc):
    D, C = {}, {}
    for k, u in enumerate(units):
        t = ALIAS.get(u.split("|")[0], u.split("|")[0]); a = np.full(len(UG), np.nan); a[mc] = np.log2((F["kd_cp10k"][k][ci] + pc) / (F["ntc_cp10k"][k][ci] + pc)); D.setdefault(t, []).append(a); C[t] = C.get(t, 0) + int(F["cells"][k])
    L = np.vstack([np.mean(D[t], 0) for t in TFS]); L[~OK] = np.nan; return L, np.array([C[t] for t in TFS])
L1, NC = lfc(0.1); L5, _ = lfc(0.5); assert np.isfinite(L1[OK]).all()
print("sign agreement log2FC vs z:", float((np.sign(L1[OK]) == np.sign(Z[OK])).mean()), "| cells per TF median", float(np.median(NC)), flush=True)
TER = np.where(NC <= 42, "25-42", np.where(NC <= 73, "43-73", "74-216")); DEX = dec(ex)
def versions(L):
    V = []
    for c in (0.25, 0.5, 0.75, 1.0): V.append(("V1 |log2FC|", f">= {c}", OK & (np.abs(np.nan_to_num(L)) >= c), L))
    for c in (0.25, 0.5, 0.75, 1.0): V.append(("V2 |log2FC| and |z| >= 2", f">= {c}", OK & (np.abs(np.nan_to_num(L)) >= c) & (AZ >= 2), L))
    return V
REF = [("reference |z|", f">= {c}", OK & (AZ >= c), Z) for c in (2, 3, 4, 5)]
VALL = versions(L1) + REF
# ---- responder counts (before any Stage 1 number)
RC, DE = [], []
for v, c, M, S_ in VALL:
    n = M.sum(1); dn = (M & (np.nan_to_num(S_) < 0)).sum(); rho = spearmanr(n, NC)
    RC.append({"version": v, "cutoff": c, "pairs": int(M.sum()), "down": int(dn), "up": int(M.sum() - dn), "down (%)": 100 * dn / max(M.sum(), 1), "TFs >= 1 responder": int((n >= 1).sum()), "TFs >= 10 responders": int((n >= 10).sum()), "median responders per TF": float(np.median(n)),
               "median |z| of pairs": float(np.median(AZ[M])) if M.any() else np.nan, "median |log2FC| of pairs": float(np.median(np.abs(L1[M]))) if M.any() else np.nan, "Spearman responders vs cells per TF": rho.statistic, "Spearman p": rho.pvalue,
               "median responders, TFs with 25-42 cells": float(np.median(n[TER == "25-42"])), "43-73 cells": float(np.median(n[TER == "43-73"])), "74-216 cells": float(np.median(n[TER == "74-216"]))})
    d = {"version": v, "cutoff": c}; tot = M.sum()
    for k in range(10): d[f"decile {k + 1} (%)"] = 100 * M[:, DEX == k].sum() / max(tot, 1)
    DE.append(d)
U = {"version": "all tested pairs (universe)", "cutoff": ""}
for k in range(10): U[f"decile {k + 1} (%)"] = 100 * OK[:, DEX == k].sum() / OK.sum()
DE.append(U); RC, DE = pd.DataFrame(RC), pd.DataFrame(DE)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(RC.round(3).to_string(index=False)); print(DE.round(1).to_string(index=False), flush=True)
# ---- Stage 1
SETS = {"all 63 TFs": np.ones(len(TFS), bool), "55 non-broad TFs": np.array([t not in BROAD for t in TFS]), "without NRF1, E4F1, DNMT1": np.array([t not in ("NRF1", "E4F1", "DNMT1") for t in TFS])}
NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1)
def stage1(VS, seed, loo):
    rng = np.random.default_rng(seed); rows, LO = [], []
    for pad in ("10 kb", "0", "50 kb"):
        B = np.vstack([LISTS[pad][t] for t in TFS]); ours = (B & OK).sum(1)
        for v, c, M, S_ in VS:
            for dr_, Md in (("all", M), ("down", M & (np.nan_to_num(S_) < 0)), ("up", M & (np.nan_to_num(S_) > 0))):
                de = Md.sum(1); ob = (B & Md).sum(1); dr = np.zeros((len(TFS), NP))
                for i in range(len(TFS)):
                    if de[i] == 0: continue
                    s_ = STRAT[OK[i]]; b = B[i][OK[i]]; p_ = Md[i][OK[i]]
                    for st in np.unique(s_[p_]):
                        m = s_ == st; nn = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
                        if nb: dr[i] += rng.hypergeometric(nb, nn - nb, kk, size=NP)
                for sn, sm in SETS.items():
                    sel = (de > 0) & sm
                    if not sel.any(): rows.append({"padding": pad, "version": v, "cutoff": c, "direction": dr_, "TF set": sn, "TFs contributing": 0}); continue
                    o = 100 * ob[sel].sum() / de[sel].sum(); dd = 100 * dr[sel].sum(0) / de[sel].sum(); ol = ours[sel].sum()
                    rows.append({"padding": pad, "version": v, "cutoff": c, "direction": dr_, "TF set": sn, "TFs contributing": int(sel.sum()), "Perturb-seq genes": int(de[sel].sum()), "in both": int(ob[sel].sum()), "recall (%)": o, "shuffle mean (%)": dd.mean(), "gap (points)": o - dd.mean(), "observed / shuffle": o / dd.mean() if dd.mean() > 0 else np.nan,
                                 "shuffle 2.5-97.5 (%)": f"{np.percentile(dd, 2.5):.1f}-{np.percentile(dd, 97.5):.1f}", "p (one-sided)": pval(o, dd), "our links": int(ol), "precision (%)": 100 * ob[sel].sum() / ol, "precision shuffle (%)": 100 * dr[sel].sum(0).mean() / ol})
                if loo and pad == "10 kb":
                    sel0 = de > 0; rr = []
                    for i in np.flatnonzero(sel0):
                        s2 = sel0.copy(); s2[i] = False
                        if s2.any(): o = 100 * ob[s2].sum() / de[s2].sum(); rr.append((TFS[i], o, o - 100 * dr[s2].sum(0).mean() / de[s2].sum()))
                    if rr:
                        rr = pd.DataFrame(rr, columns=["TF", "r", "g"]); LO.append({"version": v, "cutoff": c, "direction": dr_, "TFs contributing": int(sel0.sum()), "Perturb-seq genes": int(de.sum()), "recall min (%)": rr.r.min(), "recall max (%)": rr.r.max(), "TF left out at min recall": rr.TF[rr.r.idxmin()], "gap min (points)": rr.g.min(), "gap max (points)": rr.g.max(), "TF left out at min gap": rr.TF[rr.g.idxmin()]})
        print("padding", pad, "done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(LO)
ST, LOO = stage1(VALL, 20261118, True); SEN, _ = stage1(versions(L5), 20261118, False)
ST.to_csv(f"{O}/stage1_log2fc.tsv", sep="\t", index=False, float_format="%.4g"); RC.to_csv(f"{O}/responder_counts.tsv", sep="\t", index=False, float_format="%.4g")
# sensitivity responder counts (pseudocount 0.5)
SRC = []
for v, c, M, S_ in versions(L5):
    n = M.sum(1); dn = (M & (np.nan_to_num(S_) < 0)).sum(); SRC.append({"version": v + " (pseudocount 0.5)", "cutoff": c, "pairs": int(M.sum()), "down (%)": 100 * dn / max(M.sum(), 1), "TFs >= 10 responders": int((n >= 10).sum()), "median responders per TF": float(np.median(n)), "Spearman responders vs cells per TF": spearmanr(n, NC).statistic})
note = pd.DataFrame({"note": [
 "log2FC = log2((mean CP10k in the TF's top-25% sgRNA cells + 0.1) / (GEM-weighted NTC mean CP10k + 0.1)); NTC = all NTC cells of the unit's GEMs, GEM means weighted by the unit's cells per GEM; two constructs averaged; HP1 = mean of CBX1/3/5; own gene removed; 11,228-gene universe, NTC detection >= 10%. Same cells as the m23 z (top 25% by GEM-percentile sgRNA UMI).",
 "V1: |log2FC| >= cutoff. V2: |log2FC| >= cutoff and |z| >= 2. Reference: |z| >= cutoff (m23 z = log1p CP10k difference / NTC-variance standard error). Direction: sign of log2FC (sign of z for the reference).",
 "Stage 1: top 10,000 lab peaks per TF; link = summit in a promoter or canonical Hi-C ABC-linked element, plus summit within the padding of the TSS. Recall pooled over TFs with >= 1 responder; precision = in both / our links of those TFs. Matched gene shuffle within NTC expression x promoter ATAC deciles, exact hypergeometric per TF and stratum, 1,000 shuffles, seed 20261118, one run for all rows of a sheet; one-sided p = (1 + #>= observed)/1001.",
 "Expression deciles = NTC mean expression deciles of the matched shuffle (decile 1 = lowest). Cell tertiles = top-25% cells per TF 25-42 / 43-73 / 74-216."]})
pick = lambda v: ST[ST.version.str.startswith(v)]
with pd.ExcelWriter(f"{O}/stage1_log2fc.xlsx") as w:
    RC.to_excel(w, sheet_name="responder counts", index=False); DE.to_excel(w, sheet_name="expression deciles", index=False)
    pick("V1").to_excel(w, sheet_name="Stage 1 V1", index=False); pick("V2").to_excel(w, sheet_name="Stage 1 V2", index=False); pick("reference").to_excel(w, sheet_name="reference |z|", index=False)
    LOO.to_excel(w, sheet_name="leave one TF out 10 kb", index=False); pd.concat([pd.DataFrame(SRC), SEN.assign(pseudocount=0.5)], ignore_index=True).to_excel(w, sheet_name="sensitivity pc 0.5", index=False); note.to_excel(w, sheet_name="notes", index=False)
K = ST[(ST.padding == "10 kb") & (ST["TF set"] == "all 63 TFs")]
print(K.drop(columns=["padding", "TF set", "shuffle 2.5-97.5 (%)"]).round(3).to_string(index=False))
print(ST[(ST.padding == "10 kb") & (ST["TF set"] != "all 63 TFs") & (ST.direction == "all")][["version", "cutoff", "TF set", "TFs contributing", "recall (%)", "gap (points)", "p (one-sided)"]].round(3).to_string(index=False))
print(LOO.round(2).to_string(index=False)); print(pd.DataFrame(SRC).round(3).to_string(index=False))
print(SEN[(SEN.padding == "10 kb") & (SEN["TF set"] == "all 63 TFs") & (SEN.direction == "all")][["version", "cutoff", "TFs contributing", "Perturb-seq genes", "recall (%)", "gap (points)", "observed / shuffle", "p (one-sided)"]].round(3).to_string(index=False)); print("done")
