# m57 Amendment 10: precision / recall report for four target rules (|z| >= 2, |z| >= 3, top 10 %, top 5 %), observed only.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M57 = f"{R}/m57_stage1_q_cutoff"
SA = open(f"{M57}/m57_abcdist.py").read(); exec(SA[:SA.index("M_ = lambda M: M & OK")])  # m48 prefix (Z, AZ, OK, LISTS, TFS, V, rule_abc, window), PA, PS, PROM, DIS, DMIN, WIN{}, S_*
O = f"{M57}/out"; NPAIR = int(OK.sum()); assert NPAIR == 707239
CUR = (PROM | DIS | WIN[10]) & OK; assert int(CUR.sum()) == 212087; S_cur = np.maximum.reduce([S_prom, S_dall, S_win[10]])
# fold change (m50 measure) and cells per TF
FU = np.load(f"{R}/m50_stage1_log2fc/out/unit_mean_cp10k.npz", allow_pickle=True); u_ = FU["units"].astype(str); g_ = pd.Series(np.arange(len(FU["genes"])), index=FU["genes"].astype(str)); g_ = g_[~g_.index.duplicated(keep="last")]; cc = g_.reindex(UG).values; mc = ~np.isnan(cc); ci = cc[mc].astype(int)
AL = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; D_, C_ = {}, {}
for k, u in enumerate(u_):
    t = AL.get(u.split("|")[0], u.split("|")[0]); a = np.full(len(UG), np.nan); a[mc] = np.log2((FU["kd_cp10k"][k][ci] + .1) / (FU["ntc_cp10k"][k][ci] + .1)); D_.setdefault(t, []).append(a); C_[t] = C_.get(t, 0) + int(FU["cells"][k])
LFC = np.vstack([np.mean(D_[t], 0) for t in TFS]); LFC[~OK] = np.nan; CELLS = np.array([C_[t] for t in TFS])
ED = pd.read_excel(f"{R}/m51_edistance/out/stage1_edistance.xlsx", sheet_name="E-distance per TF").set_index("TF"); KDP = ED["on-target knockdown (%)"].reindex(TFS).values
SC = pd.DataFrame(np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)["score"]).values
Zs = np.nan_to_num(Z); SCs = np.nan_to_num(np.where(OK, pd.DataFrame(np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)["score"], index=np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)["tfs"].astype(str), columns=np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)["genes"].astype(str)).reindex(index=TFS, columns=UG).values, 0))
RULES = {"z2": ("|z| >= 2", OK & (AZ >= 2), Zs), "z3": ("|z| >= 3", OK & (AZ >= 3), Zs), "top10": ("top 10 % of |score| per TF", OK & (PA > 90), SCs), "top5": ("top 5 % of |score| per TF", OK & (PA > 95), SCs)}
assert int(RULES["z2"][1].sum()) == 63954 and int(RULES["z3"][1].sum()) == 11237 and int((CUR & RULES["top10"][1]).sum()) == 22328 and int((CUR & RULES["top5"][1]).sum()) == 11536
# peak-set lists (10 kb)
QS = [("q <= 0.05", 0.05), ("q <= 0.01", 0.01), ("q <= 1e-3", 1e-3), ("q <= 1e-5", 1e-5)]; PSL = {}; NPK = {}
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]); P["rk"] = P.sig.rank(ascending=False, method="first")
    for nm, sub in [("top 5,000", P[P.rk <= 5000])] + [(n, P[P.q >= -np.log10(x)]) for n, x in QS]:
        NPK[(nm, t)] = len(sub)
        if len(sub): c5, p5 = sub.c.values.astype(str), (sub.s + sub.off).values; A, _ = rule_abc(c5, p5); PSL.setdefault(nm, np.zeros((NT, NGn), bool))[i] = A | window(c5, p5, 10000)
        else: PSL.setdefault(nm, np.zeros((NT, NGn), bool))
    print(t, flush=True)
def topn(n):
    K = np.zeros_like(CUR)
    for i in range(NT): g = np.flatnonzero(CUR[i]); K[i, g[np.argsort(-S_cur[i, g], kind="stable")][:n]] = True
    return K
C2 = DIS & ~PROM
LINKSETS = [("headline", "top 10,000 peaks, promoter + ABC + 10 kb", CUR, 10000.0)]
LINKSETS += [("peak set (10 kb)", nm, PSL[nm], float(np.median([NPK[(nm, t)] for t in TFS]))) for nm in ("top 5,000",) + tuple(n for n, _ in QS)]
LINKSETS += [("padding", "padding 0 kb", np.vstack([LISTS["0"][t] for t in TFS]), 10000.0), ("padding", "padding 50 kb", np.vstack([LISTS["50 kb"][t] for t in TFS]), 10000.0)]
LINKSETS += [("link rule", "ABC only (promoter + all ABC elements, no window)", PROM | DIS, 10000.0), ("link rule", "distance only, within 10 kb", WIN[10], 10000.0), ("link rule", "distance only, within 20 kb", WIN[20], 10000.0), ("link rule", "distance only, within 50 kb", WIN[50], 10000.0), ("link rule", "current rule without distal ABC > 50 kb", PROM | (DIS & (DMIN <= 50e3)) | WIN[10], 10000.0)]
LINKSETS += [("strongest linked genes per TF (by peak signalValue)", f"top {n:,} genes per TF", topn(n), 10000.0) for n in (100, 250, 500, 1000, 2000)]
LINKSETS += [("link type (exclusive classes of the headline links)", lab, M, 10000.0) for lab, M in (("own promoter", PROM), ("distal ABC < 10 kb", C2 & (DMIN < 10e3)), ("distal ABC 10-50 kb", C2 & (DMIN >= 10e3) & (DMIN < 50e3)), ("distal ABC 50-200 kb", C2 & (DMIN >= 50e3) & (DMIN < 200e3)), ("distal ABC > 200 kb", C2 & (DMIN >= 200e3)), ("10 kb window only", WIN[10] & ~PROM & ~DIS))]
def block_rows(T, sign):
    rows = []
    tn = T.sum(1)
    def one(block, setting, L, peaks, TT):
        L = L & OK; ib = int((L & TT).sum()); n_ = TT.sum(1)
        rows.append({"block": block, "setting": setting, "peaks per TF (median)": peaks, "our linked genes per TF (median)": float(np.median(L.sum(1))), "our links (total)": int(L.sum()), "targets per TF (median)": float(np.median(n_)), "targets per TF (min-max)": f"{int(n_.min())}-{int(n_.max())}",
                     "targets (total)": int(TT.sum()), "targets as % of all TF-gene pairs": 100 * TT.sum() / NPAIR, "in both": ib, "recall %": 100 * ib / TT.sum(), "precision %": 100 * ib / max(L.sum(), 1),
                     "TFs with no links": int((L.sum(1) == 0).sum()) if not block.startswith("link type") else np.nan, "TFs with < 10 targets": int((n_ < 10).sum())})
    for b, s, L, pk in LINKSETS[:1]: one(b, s, L, pk, T)
    one("direction", "up only", CUR, 10000.0, T & (sign > 0)); one("direction", "down only", CUR, 10000.0, T & (sign < 0))
    for b, s, L, pk in LINKSETS[1:]: one(b, s, L, pk, T)
    return pd.DataFrame(rows)
SHEETS = {f"report_{k}": block_rows(M, sg) for k, (lab, M, sg) in RULES.items()}
for k, d in SHEETS.items(): d.insert(0, "target rule", RULES[k.replace("report_", "")][0])
TRR = []
for lab, M, sg in [("|z| >= 2", OK & (AZ >= 2), Zs), ("|z| >= 3", OK & (AZ >= 3), Zs), ("|z| >= 4", OK & (AZ >= 4), Zs), ("|z| >= 5", OK & (AZ >= 5), Zs), ("top 10 % of |score| per TF", OK & (PA > 90), SCs), ("top 5 % of |score| per TF", OK & (PA > 95), SCs),
                   ("fold >= 1.1", OK & (np.abs(np.nan_to_num(LFC)) >= np.log2(1.1)), np.nan_to_num(LFC)), ("fold >= 1.1 and |z| >= 2", OK & (np.abs(np.nan_to_num(LFC)) >= np.log2(1.1)) & (AZ >= 2), np.nan_to_num(LFC))]:
    n_ = M.sum(1); ib = int((CUR & M).sum())
    TRR.append({"target rule": lab, "targets (total)": int(M.sum()), "targets as % of pairs": 100 * M.sum() / NPAIR, "per TF median": float(np.median(n_)), "per TF min": int(n_.min()), "per TF max": int(n_.max()), "TFs with >= 1 target": int((n_ >= 1).sum()), "TFs with >= 10 targets": int((n_ >= 10).sum()),
                "% up": 100 * float((sg[M] > 0).mean()), "headline links: in both": ib, "headline recall %": 100 * ib / M.sum(), "headline precision %": 100 * ib / CUR.sum()})
TRR = pd.DataFrame(TRR)
PT = pd.DataFrame({"TF": TFS, "top-25% cells": CELLS, "on-target knockdown % (top-25% cells)": KDP, "our linked genes": CUR.sum(1)})
for k in ("z2", "z3", "top10"):
    lab, M, _ = RULES[k]; n_ = M.sum(1); ib = (CUR & M).sum(1); PT[f"{lab}: targets"] = n_; PT[f"{lab}: in both"] = ib; PT[f"{lab}: recall %"] = np.where(n_ > 0, 100 * ib / np.maximum(n_, 1), np.nan); PT[f"{lab}: precision %"] = 100 * ib / CUR.sum(1)
PT = PT.sort_values("|z| >= 2: precision %", ascending=False)
NOTES = pd.DataFrame({"item": ["definitions", "z", "score / percentile", "fold change", "links", "sources", "percentile note", "headline history"], "text": [
  "Recall = in both / Perturb-seq targets (all 63 TFs); precision = in both / our links. 63 TFs, 11,228-gene universe (707,239 TF-gene pairs), own gene removed. Observed only, no shuffle.",
  "z (m23): (mean log1p CP10k of the gene in the TF's top-25% sgRNA cells - GEM-weighted NTC mean) / standard error of that difference from the within-GEM NTC variance; two constructs averaged, HP1 = CBX1/3/5.",
  "Paper per-gene score (m51/m53): per top-25% cell (log1p CP10k - own-GEM NTC mean) / own-GEM NTC SD, averaged over the TF's cells; top X % = within-TF percentile of |score|.",
  "Fold change (m50): log2((mean CP10k in top-25% cells + 0.1) / (GEM-weighted NTC mean CP10k + 0.1)).",
  "Top 10,000 lab peaks by signalValue unless the row says otherwise; gene linked if a summit lies in its promoter or a canonical Hi-C ABC-linked element (ABC >= 0.02), or within the padding (10 kb unless stated) of its TSS. q <= X = narrowPeak column 9 (-log10 q) >= -log10(X).",
  "All sheets: m57_report4.py (spec Amendment 10), one sbatch job; link sets from the m57_abcdist code (job 10494793, bba7b58); peak-set lists rebuilt in the job; fold change from m50 (unit_mean_cp10k.npz); knockdown % from m51.",
  "Percentile rules (top 10 % / 5 %) give every TF the same number of targets by construction (1,123 / 561), also when the knockdown changed little; |z| rules give each TF the number of genes that pass the cutoff (from 0 or a few up to thousands).",
  "2026-11-23: headline target rule top 5 % -> top 10 % (user, 19:01Z), then |z| >= 2 as the first sheet (user leaning back to |z|, 19:02Z); all four kept as sheets."]})
with pd.ExcelWriter(f"{O}/stage1_precision_recall_report.xlsx") as w:
    for k in ("report_z2", "report_z3", "report_top10", "report_top5"): SHEETS[k].to_excel(w, sheet_name=k, index=False)
    TRR.to_excel(w, sheet_name="target_rules", index=False); PT.to_excel(w, sheet_name="per_TF", index=False); NOTES.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 340); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 50)
print(TRR.round(2).to_string(index=False))
for k in ("report_z2", "report_z3", "report_top10"): print("==", k); print(SHEETS[k].drop(columns=["target rule", "peaks per TF (median)", "targets as % of all TF-gene pairs"]).round(2).to_string(index=False))
for k in ("z2", "z3", "top10"):
    lab = RULES[k][0]
    for c in ("recall %", "precision %"): v = PT[f"{lab}: {c}"].dropna(); print(lab, c, "median", round(v.median(), 2), "IQR", round(v.quantile(.25), 2), "-", round(v.quantile(.75), 2), "min", round(v.min(), 2), PT.loc[v.idxmin(), "TF"], "max", round(v.max(), 2), PT.loc[v.idxmax(), "TF"])
print("done")
