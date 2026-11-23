# m57 Amendment 9: precision / recall report with top 10 % per TF as the headline target rule (observed only).
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M57 = f"{R}/m57_stage1_q_cutoff"
SA = open(f"{M57}/m57_abcdist.py").read(); exec(SA[:SA.index("M_ = lambda M: M & OK")])
A_PROM, A_DIS, A_DMIN, A_WIN, A_S = PROM.copy(), DIS.copy(), DMIN.copy(), {d: WIN[d].copy() for d in (10, 20, 50)}, np.maximum.reduce([S_prom, S_dall, S_win[10]])
SD = open(f"{M57}/m57_distal.py").read(); exec(SD[:SD.index("TG = {}")])  # PD, LOOP, LOOPONLY, NEAR, NEAR100, NONPW, BABC, BSIG, BCLS, CUR
O = f"{M57}/out"; NPAIR = int(OK.sum()); X = lambda f, sh: pd.read_excel(f"{O}/{f}", sheet_name=sh)
T10 = OK & (PA > 90); T10u = OK & (PS > 90); T10d = OK & (PS < 10); ntf = T10.sum(1); TMED, TTOT = float(np.median(ntf)), int(T10.sum())
CURm = (A_PROM | A_DIS | A_WIN[10]) & OK; assert int(CURm.sum()) == 212087
rows = []
def add(block, setting, peaks, gmed, links, tmed, ttot, both, rec, prec, zero):
    rows.append({"block": block, "setting": setting, "peaks per TF (median)": peaks, "our linked genes per TF (median)": gmed, "our links (total)": int(links), "Perturb-seq targets per TF (median)": tmed, "targets (total)": int(ttot), "targets as % of all genes": 100 * ttot / NPAIR,
                 "in both": int(both), "recall %": rec, "precision %": prec, "TFs with no links": zero})
def obs(block, setting, M, T=T10, gmed=None, zero=0):
    M = M & OK; ib = int((M & T).sum()); add(block, setting, 10000.0, float(np.median(M.sum(1))) if gmed is None else gmed, M.sum(), float(np.median(T.sum(1))), T.sum(), ib, 100 * ib / T.sum(), 100 * ib / max(M.sum(), 1), zero)
obs("headline", "HEADLINE: top 10,000 peaks, promoter + ABC + 10 kb, top 10 % of genes per TF, 63 TFs", CURm)
LO = X("stage1_looser_targets_recall_precision_raw.xlsx", "pooled"); LO["padding"] = LO.padding.astype(str); FC = X("stage1_fold_change_recall_precision_raw.xlsx", "pooled"); FC["padding"] = FC.padding.astype(str)
RAW = X("stage1_q_cutoff_recall_precision_raw.xlsx", "pooled"); RAW["padding"] = RAW.padding.astype(str); PK = X("stage1_q_cutoff_recall_precision_raw.xlsx", "peak sets").set_index("peak set")
for x in (50, 20, 5, 2, 1):
    M = OK & (PA > 100 - x); obs("Perturb-seq rule, same links", f"top {x} % of genes per TF", CURm, T=M)
obs("Perturb-seq rule, same links", "up only, top 10 %", CURm, T=T10u); obs("Perturb-seq rule, same links", "down only, bottom 10 %", CURm, T=T10d)
for fam, d, lab in (("A fold change alone", "either", "fold >= 1.1"), ("B fold change and |z| >= 2", "either", "fold >= 1.1 and |z| >= 2"), ("B fold change and |z| >= 2", "up", "fold >= 1.1 and |z| >= 2, up only")):
    r = FC[(FC["peak set"] == "top 10,000") & (FC.padding == "10 kb") & (FC["rule family"] == fam) & (FC.rule == "fold >= 1.1") & (FC.direction == d)].iloc[0]
    add("Perturb-seq rule, same links", lab, 10000, 2892.0, r["links"], r["targets per TF, median"], r["total targets"], r["in both"], r["recall (%)"], r["precision (%)"], 0)
for ps in ("top 5,000", "q <= 0.05", "q <= 0.01", "q <= 1e-3", "q <= 1e-5"):
    r = RAW[(RAW["peak set"] == ps) & (RAW.padding == "10 kb") & (RAW.rule == "top 10%") & (RAW["TF set"] == "all 63 TFs (main)")].iloc[0]
    add("Peak set, top 10 % targets", ps, float(PK.loc[ps, "peaks per TF, median"]), float(PK.loc[ps, "genes per TF, 10 kb: median"]), r["candidate links"], TMED, r["Perturb-seq targets"], r["links in both"], r["recall (%)"], r["precision (%)"], int(r["TFs with 0 links"]))
for pad, lab in (("0", "padding 0 kb"), ("50 kb", "padding 50 kb")):
    r = LO[(LO["peak set"] == "top 10,000") & (LO.padding == pad) & (LO.rule == "top 10%")].iloc[0]
    add("Padding, top 10 %", lab, 10000, float(PK.loc["top 10,000", f"genes per TF, {pad}: median"]), r["candidate links"], TMED, r["Perturb-seq targets"], r["links in both"], r["recall (%)"], r["precision (%)"], 0)
for lab, M in (("ABC only (promoter + all ABC elements, no window)", A_PROM | A_DIS), ("distance only, within 10 kb", A_WIN[10]), ("distance only, within 20 kb", A_WIN[20]), ("distance only, within 50 kb", A_WIN[50]), ("current rule without distal ABC > 50 kb", A_PROM | (A_DIS & (A_DMIN <= 50e3)) | A_WIN[10])): obs("Link rule, top 10 %", lab, M)
for n in (100, 250, 500, 1000, 2000):
    K = np.zeros_like(CURm)
    for i in range(NT): g = np.flatnonzero(CURm[i]); K[i, g[np.argsort(-A_S[i, g], kind="stable")][:n]] = True
    obs("Strongest linked genes per TF (current rule, by peak signalValue), top 10 %", f"top {n:,} genes per TF", K, gmed=float(n))
C2 = A_DIS & ~A_PROM
for lab, M in (("own promoter", A_PROM), ("distal ABC < 10 kb", C2 & (A_DMIN < 10e3)), ("distal ABC 10-50 kb", C2 & (A_DMIN >= 10e3) & (A_DMIN < 50e3)), ("distal ABC 50-200 kb", C2 & (A_DMIN >= 50e3) & (A_DMIN < 200e3)), ("distal ABC > 200 kb", C2 & (A_DMIN >= 200e3)), ("10 kb window only", A_WIN[10] & ~A_PROM & ~A_DIS)):
    obs("By link type (current rule, top 10 %)", lab, M, zero=np.nan)
assert sum(r["in both"] for r in rows if r["block"].startswith("By link type")) == int((CURm & T10).sum())
Q = np.zeros((NT, NGn), int)
for i in range(NT):
    g = np.flatnonzero(PD[i])
    if len(g): Q[i, g] = pd.qcut(pd.Series(BSIG[i, g]).rank(method="first"), 5, labels=False).values + 1
DL = [("pure distal links (ABC, no promoter summit, no summit within 10 kb)", PD)]
for lo, hi in ((0.02, 0.03), (0.03, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, np.inf)): DL.append((f"ABC score {lo}-{hi}" if np.isfinite(hi) else f"ABC score > {lo}", PD & (BABC >= lo) & (BABC < hi)))
for n in (100, 250, 500):
    M = np.zeros((NT, NGn), bool)
    for i in range(NT): g = np.flatnonzero(PD[i]); M[i, g[np.argsort(-BSIG[i, g], kind="stable")][:n]] = True
    DL.append((f"top {n} distal genes per TF by peak strength", M))
DL += [("loop-supported element-TSS pair", PD & LOOP), ("no loop support", PD & ~LOOP), ("loop-only rule (no ABC), all", LOOPONLY), ("loop-only rule, no promoter / 10 kb link", LOOPONLY & NONPW), ("ABC class genic", PD & (BCLS == "genic")), ("ABC class intergenic", PD & (BCLS == "intergenic")),
       ("nearest-gene rule, all", NEAR), ("nearest-gene rule, <= 100 kb", NEAR100), ("nearest-gene rule, no promoter / 10 kb link", NEAR & NONPW), ("nearest-gene rule, <= 100 kb, no promoter / 10 kb link", NEAR100 & NONPW),
       ("ABC > 0.05 and strongest peak quintile", PD & (BABC > 0.05) & (Q == 5)), ("loop-supported and strongest peak quintile", PD & LOOP & (Q == 5))]
for lab, M in DL: obs("distal links (top 10 %)", lab, M, zero=np.nan)
REP = pd.DataFrame(rows)
P = X("stage1_looser_targets_recall_precision_raw.xlsx", "per TF, top10k 10kb"); P = P[P.rule == "top 10%"]
PT = pd.DataFrame({"TF": P.TF, "peaks": 10000, "our linked genes": P["candidate target genes"], "Perturb-seq targets": P["Perturb-seq targets"], "in both": P["genes in both"], "recall %": P["recall (%)"], "precision %": P["precision (%)"]}).sort_values("precision %", ascending=False)
assert int(PT["in both"].sum()) == 22328 and REP.iloc[0]["in both"] == 22328
NOTES = pd.DataFrame({"block": ["definitions", "sources"], "source": ["Recall = in both / Perturb-seq targets; precision = in both / our links. 63 TFs, 707,239 TF-gene pairs, own gene removed. Targets: top 10 % of |score| within the TF (paper per-gene score, full precision; 1,123 genes per TF) unless stated. Links: top 10,000 lab peaks; summit in a promoter or canonical Hi-C ABC-linked element, or within 10 kb of the TSS, unless stated. Observed only (no shuffle).",
  "Headline, Perturb-seq top X %, link rule, strongest N, link type and distal rows: m57_report10.py (spec Amendment 9), code of m57_abcdist.py / m57_distal.py. Fold-change rows: job 10493662 (790c25d). Peak sets: job 10492917 (f02b3a7). Padding and per-TF: job 10493172 (83371a0). The top 5 % version is stage1_precision_recall_report.xlsx."]})
with pd.ExcelWriter(f"{O}/stage1_precision_recall_report_top10pct.xlsx") as w:
    REP.to_excel(w, sheet_name="report", index=False); PT.to_excel(w, sheet_name="per_TF", index=False); NOTES.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 62)
print(REP.drop(columns=["block", "peaks per TF (median)", "targets as % of all genes"]).round(2).to_string(index=False))
for c in ("recall %", "precision %"): v = PT[c]; print(c, "median", round(v.median(), 2), "IQR", round(v.quantile(.25), 2), "-", round(v.quantile(.75), 2), "min", round(v.min(), 2), PT.loc[v.idxmin(), "TF"], "max", round(v.max(), 2), PT.loc[v.idxmax(), "TF"])
print("done")
