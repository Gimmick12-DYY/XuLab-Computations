# m57 Amendment 1: q-value-filtered lab peak sets, observed recall and precision only (no shuffle).
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/work/users/d/y/dyy12/XuLab/regulatome"; s = open(f"{R}/m48_stage1_top10k/m48_tables.py").read(); exec(s[:s.index("VERS = {")])  # TFS, UG, OK, STRAT, BROAD, rule_abc, window, V, LISTS (top 10,000 check)
O = f"{R}/m57_stage1_q_cutoff/out"; os.makedirs(O, exist_ok=True)
F = np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)
SC = pd.DataFrame(F["score"], index=F["tfs"].astype(str), columns=F["genes"].astype(str)).reindex(index=TFS, columns=UG).values; assert (np.isfinite(SC) == OK).all()
NT = len(TFS); PA = np.full(SC.shape, np.nan); PS = np.full(SC.shape, np.nan)
for i in range(NT): ok = OK[i]; n = ok.sum(); PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / n; PS[i, ok] = 100 * (rankdata(SC[i, ok]) - 0.5) / n
QS = [("q <= 0.05", 0.05), ("q <= 0.01", 0.01), ("q <= 1e-3", 1e-3), ("q <= 1e-5", 1e-5), ("q <= 1e-10", 1e-10), ("q <= 1e-20", 1e-20)]; TOPS = [("top 10,000", 10000), ("top 5,000", 5000)]
PADS_ = {"10 kb": 10000, "0": 0, "50 kb": 50000}; L = {}; NPK = {}
for t in TFS:
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 8, 9], names=["c", "s", "sig", "q", "off"]); P["rk"] = P.sig.rank(ascending=False, method="first")
    for nm, x in QS + TOPS:
        sub = P[P.q >= -np.log10(x)] if nm.startswith("q") else P[P.rk <= x]; NPK[(nm, t)] = len(sub)
        if len(sub): c5, p5 = sub.c.values.astype(str), (sub.s + sub.off).values; A, _ = rule_abc(c5, p5)
        for pd_, w in PADS_.items():
            L[(nm, pd_, t)] = (A | window(c5, p5, w) if w else A) if len(sub) else np.zeros(len(UG), bool)
    print(t, len(P), flush=True)
for pd_ in PADS_: assert all((L[("top 10,000", pd_, t)] == LISTS[pd_][t]).all() for t in TFS), "top 10,000 lists differ from m48"
PSETS = [n for n, _ in QS + TOPS]
# ---- peak-set table (before any overlap number)
PK, PTL = [], []
for nm in PSETS:
    npk = np.array([NPK[(nm, t)] for t in TFS]); r = {"peak set": nm, "peaks per TF, median": float(np.median(npk)), "25th": np.percentile(npk, 25), "75th": np.percentile(npk, 75), "min": int(npk.min()), "max": int(npk.max()), "TFs with 0 peaks": int((npk == 0).sum()), "TFs with < 100 peaks": int((npk < 100).sum()), "TFs with < 500 peaks": int((npk < 500).sum())}
    for pd_ in ("10 kb", "0", "50 kb"):
        g = np.array([(L[(nm, pd_, t)] & OK[i]).sum() for i, t in enumerate(TFS)]); r.update({f"genes per TF, {pd_}: median": float(np.median(g)), f"{pd_}: 25th": np.percentile(g, 25), f"{pd_}: 75th": np.percentile(g, 75), f"{pd_}: min": int(g.min()), f"{pd_}: max": int(g.max()), f"total links, {pd_}": int(g.sum())})
    PK.append(r)
    for i, t in enumerate(TFS): PTL.append({"peak set": nm, "TF": t, "peaks": int(npk[i]), **{f"candidate genes, {pd_}": int((L[(nm, pd_, t)] & OK[i]).sum()) for pd_ in ("10 kb", "0", "50 kb")}})
PK, PTL = pd.DataFrame(PK), pd.DataFrame(PTL); pd.set_option("display.width", 340); pd.set_option("display.max_columns", 40); print(PK.round(1).to_string(index=False), flush=True)
VS = []
for x in (10, 5, 2, 1): VS += [("all", f"top {x}%", OK & (PA > 100 - x), x), ("up", f"top {x}% up", OK & (PS > 100 - x), x), ("down", f"bottom {x}% down", OK & (PS < x), x)]
three = np.array([t in ("NRF1", "E4F1", "DNMT1") for t in TFS]); br = np.array([t in BROAD for t in TFS])
SETS = {"all 63 TFs (main)": np.ones(NT, bool), "55 non-broad TFs": ~br, "without NRF1, E4F1, DNMT1": ~three}; ROWS, PER = [], []
for nm in PSETS:
    for pd_ in ("10 kb", "0", "50 kb"):
        B = np.vstack([L[(nm, pd_, t)] for t in TFS]); ours = (B & OK).sum(1)
        for v, c, M, x in VS:
            de = M.sum(1); ob = (B & M).sum(1)
            for sn, sm in SETS.items():
                sel = sm & (de > 0); o = ob[sel].sum(); nP = de[sel].sum(); nO = ours[sel].sum()
                ROWS.append({"peak set": nm, "padding": pd_, "direction": v, "rule": c, "TF set": sn, "TFs": int(sel.sum()), "TFs with 0 links": int((ours[sel] == 0).sum()), "candidate links": int(nO), "Perturb-seq targets": int(nP), "links in both": int(o), "recall (%)": 100 * o / nP, "precision (%)": 100 * o / nO if nO else np.nan, "precision over chance (precision / X%)": (100 * o / nO) / x if nO else np.nan})
            if pd_ == "10 kb" and c == "top 5%":
                for k in range(NT): PER.append({"peak set": nm, "TF": TFS[k], "peaks": NPK[(nm, TFS[k])], "candidate target genes": int(ours[k]), "Perturb-seq targets": int(de[k]), "genes in both": int(ob[k]), "recall (%)": 100 * ob[k] / de[k], "precision (%)": 100 * ob[k] / ours[k] if ours[k] else np.nan})
T = pd.DataFrame(ROWS); PE = pd.DataFrame(PER)
with pd.ExcelWriter(f"{O}/stage1_q_cutoff_recall_precision_raw.xlsx") as w:
    PK.to_excel(w, sheet_name="peak sets", index=False); T[(T["TF set"] == "all 63 TFs (main)") & (T.direction == "all")].to_excel(w, sheet_name="pooled", index=False)
    T[(T["TF set"] == "all 63 TFs (main)") & (T.direction != "all")].to_excel(w, sheet_name="up_down", index=False); T[T["TF set"] != "all 63 TFs (main)"].to_excel(w, sheet_name="TF sets", index=False)
    PE.to_excel(w, sheet_name="per TF (10 kb, top 5%)", index=False); PTL.to_excel(w, sheet_name="peaks per TF", index=False)
    pd.DataFrame({"note": ["Observed only, no shuffle. Peak sets: lab MACS2 narrowPeak, kept if column 9 (-log10 q) >= -log10(X); top 10,000 / 5,000 by signalValue. Links: summit in a promoter or canonical Hi-C ABC-linked element, or within the padding of the TSS. Targets: top X % of |score| within the TF (paper per-gene score). Recall = in both / targets; precision = in both / candidate links of the same TFs; precision over chance = precision / (X/100)."]}).to_excel(w, sheet_name="notes", index=False)
M_ = T[(T["TF set"] == "all 63 TFs (main)")]; k = ["peak set", "padding", "rule", "candidate links", "Perturb-seq targets", "links in both", "recall (%)", "precision (%)", "precision over chance (precision / X%)", "TFs with 0 links"]
print(M_[(M_.padding == "10 kb") & (M_.direction == "all")][k].round(2).to_string(index=False)); print(M_[(M_.padding != "10 kb") & (M_.rule == "top 5%")][k].round(2).to_string(index=False)); print(M_[(M_.padding == "10 kb") & M_.rule.isin(["top 5% up", "bottom 5% down"])][k].round(2).to_string(index=False))
print("done")
