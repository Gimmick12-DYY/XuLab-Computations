# m57 Amendment 11: report sheets for the q <= 0.05 link setting (observed only).
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m57_stage1_q_cutoff/out"; NT = len(TFS); NGn = len(UG); NPAIR = int(OK.sum())
LX = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["TargetGene", "distance"]); LX = LX[LX.TargetGene.isin(gi)]; DIST = np.abs(LX["distance"].values.astype(float))
by_el = pd.Series(np.arange(len(Le))).groupby(Le).apply(np.array).to_dict()
F = np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)
SC = pd.DataFrame(F["score"], index=F["tfs"].astype(str), columns=F["genes"].astype(str)).reindex(index=TFS, columns=UG).values
PA = np.full(SC.shape, np.nan)
for i in range(NT): ok = OK[i]; PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / ok.sum()
Zs = np.nan_to_num(Z); SCs = np.nan_to_num(SC)
FU = np.load(f"{R}/m50_stage1_log2fc/out/unit_mean_cp10k.npz", allow_pickle=True); u_ = FU["units"].astype(str); g_ = pd.Series(np.arange(len(FU["genes"])), index=FU["genes"].astype(str)); g_ = g_[~g_.index.duplicated(keep="last")]; cc = g_.reindex(UG).values; mc = ~np.isnan(cc); ci = cc[mc].astype(int)
AL = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; D_ = {}
for k, u in enumerate(u_):
    t = AL.get(u.split("|")[0], u.split("|")[0]); a = np.full(NGn, np.nan); a[mc] = np.log2((FU["kd_cp10k"][k][ci] + .1) / (FU["ntc_cp10k"][k][ci] + .1)); D_.setdefault(t, []).append(a)
LFC = np.nan_to_num(np.vstack([np.mean(D_[t], 0) for t in TFS])); LFC[~OK] = 0
sh = lambda dt, v: np.full((NT, NGn), v, dtype=dt)
PROM = sh(bool, False); DIS = sh(bool, False); DMIN = sh(float, np.inf); WIN = {d: sh(bool, False) for d in (10, 20, 50)}; S_abc = sh(float, 0.); S_w10 = sh(float, 0.); NPK = np.zeros(NT, int)
for i, t in enumerate(TFS):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]); P = P[P.q >= -np.log10(0.05)]; NPK[i] = len(P)
    if not len(P): print(t, 0, flush=True); continue
    c5, p5, sg = P.c.values.astype(str), (P.s + P.off).values, P.sig.values; e = elem_of(c5, p5)
    for k in np.flatnonzero(e >= 0):
        rows = by_el.get(e[k])
        if rows is None: continue
        g = Lg[rows]; slf = Ls[rows]; PROM[i, g[slf]] = True; DIS[i, g[~slf]] = True; np.minimum.at(DMIN[i], g[~slf], DIST[rows][~slf]); np.maximum.at(S_abc[i], g, sg[k])
    for ch in np.unique(c5):
        if ch not in TB: continue
        tp, ix = TB[ch]; m = c5 == ch; x = p5[m]; sx = sg[m]
        for d in (10, 20, 50):
            a = np.searchsorted(tp, x - d * 1000); b = np.searchsorted(tp, x + d * 1000, side="right")
            for aa, bb, sv in zip(a, b, sx):
                gg = ix[aa:bb]; WIN[d][i, gg] = True
                if d == 10: S_w10[i, gg] = np.maximum(S_w10[i, gg], sv)
    print(t, len(P), flush=True)
CUR = (PROM | DIS | WIN[10]) & OK; assert int(CUR.sum()) == 183470, int(CUR.sum()); S_cur = np.maximum(S_abc, S_w10)
RULES = {"|z| >= 2": (OK & (AZ >= 2), Zs), "|z| >= 3": (OK & (AZ >= 3), Zs), "top 10 % of |score| per TF": (OK & (PA > 90), SCs), "top 5 % of |score| per TF": (OK & (PA > 95), SCs)}
assert int((CUR & RULES["|z| >= 2"][0]).sum()) == 17820
def topn(n):
    K = np.zeros_like(CUR)
    for i in range(NT): g = np.flatnonzero(CUR[i]); K[i, g[np.argsort(-S_cur[i, g], kind="stable")][:n]] = True
    return K
C2 = DIS & ~PROM
LS = [("padding", "padding 0 kb", PROM | DIS), ("padding", "padding 10 kb (setting)", CUR), ("padding", "padding 50 kb", PROM | DIS | WIN[50]),
      ("link rule", "ABC only (promoter + all ABC elements, no window)", PROM | DIS), ("link rule", "distance only, within 10 kb", WIN[10]), ("link rule", "distance only, within 20 kb", WIN[20]), ("link rule", "distance only, within 50 kb", WIN[50]), ("link rule", "current rule without distal ABC > 50 kb", PROM | (DIS & (DMIN <= 50e3)) | WIN[10])]
LS += [("strongest linked genes per TF (by peak signalValue)", f"top {n:,} genes per TF", topn(n)) for n in (100, 250, 500, 1000, 2000)]
LS += [("link type (exclusive classes)", lab, M) for lab, M in (("own promoter", PROM), ("distal ABC < 10 kb", C2 & (DMIN < 10e3)), ("distal ABC 10-50 kb", C2 & (DMIN >= 10e3) & (DMIN < 50e3)), ("distal ABC 50-200 kb", C2 & (DMIN >= 50e3) & (DMIN < 200e3)), ("distal ABC > 200 kb", C2 & (DMIN >= 200e3)), ("10 kb window only", WIN[10] & ~PROM & ~DIS))]
rows = []
def one(rule, block, setting, L, T):
    L = L & OK; ib = int((L & T).sum()); n_ = T.sum(1)
    rows.append({"target rule": rule, "block": block, "setting": setting, "peaks per TF (median)": float(np.median(NPK)), "our linked genes per TF (median)": float(np.median(L.sum(1))), "our links (total)": int(L.sum()), "targets per TF (median)": float(np.median(n_)), "targets per TF (min-max)": f"{int(n_.min())}-{int(n_.max())}",
                 "targets (total)": int(T.sum()), "in both": ib, "precision %": 100 * ib / max(L.sum(), 1), "recall %": 100 * ib / T.sum(), "TFs with no links": int((L.sum(1) == 0).sum()) if not block.startswith("link type") else np.nan})
FOLD = OK & (np.abs(LFC) >= np.log2(1.1))
for lab, T in (("|z| >= 2", RULES["|z| >= 2"][0]), ("top 10 % of |score| per TF", RULES["top 10 % of |score| per TF"][0]), ("fold >= 1.1", FOLD)): one(lab, "slide rows (q <= 0.05, promoter + ABC + 10 kb)", lab, CUR, T)
for rule, (T, sg) in RULES.items():
    one(rule, "direction", "up only", CUR, T & (sg > 0)); one(rule, "direction", "down only", CUR, T & (sg < 0))
    for b, s_, L in LS: one(rule, b, s_, L, T)
REP = pd.DataFrame(rows)
PT = pd.DataFrame({"TF": TFS, "peaks (q <= 0.05)": NPK, "our linked genes": CUR.sum(1), "no links": CUR.sum(1) == 0})
for lab in ("|z| >= 2", "top 10 % of |score| per TF"):
    T = RULES[lab][0]; n_ = T.sum(1); ib = (CUR & T).sum(1); PT[f"{lab}: targets"] = n_; PT[f"{lab}: in both"] = ib; PT[f"{lab}: precision %"] = np.where(CUR.sum(1) > 0, 100 * ib / np.maximum(CUR.sum(1), 1), np.nan); PT[f"{lab}: recall %"] = np.where(n_ > 0, 100 * ib / np.maximum(n_, 1), np.nan)
PT = PT.sort_values("|z| >= 2: precision %", ascending=False, na_position="last")
f = f"{O}/stage1_precision_recall_report.xlsx"; old = pd.read_excel(f, sheet_name=None)
NO = old.get("notes")
add = pd.DataFrame({"item": ["setting (q <= 0.05)", "report_q05 source"], "text": ["2026-11-23 (user): our link setting = MACS2 peaks with q <= 0.05 (narrowPeak column 9 >= -log10 0.05), promoter + ABC + 10 kb, 63 TFs (7 TFs have no linked genes). Sheets report_q05 / per_TF_q05.", "m57_report_q05.py (spec Amendment 11), one sbatch job; fold change from m50 unit means; observed only."]})
NO = pd.concat([add, NO], ignore_index=True) if NO is not None else add
with pd.ExcelWriter(f) as w:
    REP.to_excel(w, sheet_name="report_q05", index=False); PT.to_excel(w, sheet_name="per_TF_q05", index=False)
    for k, d in old.items():
        if k not in ("report_q05", "per_TF_q05", "notes"): d.to_excel(w, sheet_name=k, index=False)
    NO.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 50)
print(REP[REP["target rule"].isin(["|z| >= 2", "top 10 % of |score| per TF", "fold >= 1.1"])].drop(columns=["peaks per TF (median)"]).round(2).to_string(index=False))
print("no-link TFs:", ", ".join(PT[PT["no links"]].TF))
for lab in ("|z| >= 2", "top 10 % of |score| per TF"):
    for c in ("precision %", "recall %"): v = PT[f"{lab}: {c}"].dropna(); print(lab, c, "median", round(v.median(), 2), "IQR", round(v.quantile(.25), 2), "-", round(v.quantile(.75), 2), "min", round(v.min(), 2), "max", round(v.max(), 2))
print("done")
