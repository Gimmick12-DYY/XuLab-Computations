# m57 Amendment 3: log2 fold-change target rules (alone, with |z| >= 2, within-TF percentile), observed recall and precision only.
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read(); exec(SRC50[:SRC50.index("# ---- responder counts")])  # TFS, UG, OK, Z, AZ, L1, LISTS (top 10,000), rule_abc, window, V
O = "/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/out"; NT = len(TFS); GN = OK.sum(1)
# q <= 0.05 lists (as m57)
PADS_ = {"10 kb": 10000, "0": 0, "50 kb": 50000}; LQ = {}
for t in TFS:
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 8, 9], names=["c", "s", "q", "off"]); sub = P[P.q >= -np.log10(0.05)]
    if len(sub): c5, p5 = sub.c.values.astype(str), (sub.s + sub.off).values; A, _ = rule_abc(c5, p5)
    for pd_, w in PADS_.items(): LQ[(pd_, t)] = (A | window(c5, p5, w) if w else A) if len(sub) else np.zeros(len(UG), bool)
LIST = {("top 10,000", p): np.vstack([LISTS[p][t] for t in TFS]) for p in PADS_}; LIST.update({("q <= 0.05", p): np.vstack([LQ[(p, t)] for t in TFS]) for p in PADS_})
F = np.nan_to_num(L1); AF = np.abs(F); P1 = np.full(F.shape, np.nan); PS1 = np.full(F.shape, np.nan)
for i in range(NT): ok = OK[i]; n = ok.sum(); P1[i, ok] = 100 * (rankdata(AF[i, ok]) - 0.5) / n; PS1[i, ok] = 100 * (rankdata(F[i, ok]) - 0.5) / n
RULES = []
for f in (1.1, 1.2, 1.3, 1.5, 2.0):
    c = np.log2(f)
    for d, M in (("either", OK & (AF >= c)), ("up", OK & (F >= c)), ("down", OK & (F <= -c))): RULES.append(("A fold change alone", f"fold >= {f}", d, M)); RULES.append(("B fold change and |z| >= 2", f"fold >= {f}", d, M & (AZ >= 2)))
for x in (10, 5, 2, 1):
    for d, M in (("either", OK & (P1 > 100 - x)), ("up", OK & (PS1 > 100 - x)), ("down", OK & (PS1 < x))): RULES.append(("C top % of |log2FC| within TF", f"top {x}%", d, M))
for c in (0.25, 0.5): RULES.append(("check vs m50 V1", f"|log2FC| >= {c}", "either", OK & (AF >= c)))
rows, per = [], []
for (ps, pd_), B in LIST.items():
    links = (B & OK).sum(1)
    for fam, rule, d, M in RULES:
        tg = M.sum(1); ib = (B & M).sum(1); sel = tg > 0; chance = (links * tg / GN).sum() / links.sum()
        rows.append({"peak set": ps, "padding": pd_, "rule family": fam, "rule": rule, "direction": d, "TFs with >= 1 target": int(sel.sum()), "targets per TF, median": float(np.median(tg)), "25th": np.percentile(tg, 25), "75th": np.percentile(tg, 75),
                     "total targets": int(tg.sum()), "links": int(links.sum()), "in both": int(ib.sum()), "recall (%)": 100 * ib[sel].sum() / tg[sel].sum() if sel.any() else np.nan, "precision (%)": 100 * ib.sum() / links.sum(), "chance precision (%)": 100 * chance, "precision / chance": (ib.sum() / links.sum()) / chance if chance else np.nan})
        if ps == "top 10,000" and pd_ == "10 kb" and fam != "check vs m50 V1":
            r = 100 * ib[sel] / tg[sel]; pr = 100 * ib / np.maximum(links, 1)
            per.append({"rule family": fam, "rule": rule, "direction": d, "recall median (%)": np.median(r) if sel.any() else np.nan, "recall 25th": np.percentile(r, 25) if sel.any() else np.nan, "recall 75th": np.percentile(r, 75) if sel.any() else np.nan, "precision median (%)": np.median(pr), "precision 25th": np.percentile(pr, 25), "precision 75th": np.percentile(pr, 75)})
T = pd.DataFrame(rows); PT = pd.DataFrame(per)
ck = T[(T["rule family"] == "check vs m50 V1") & (T["peak set"] == "top 10,000") & (T.padding == "10 kb")]; print(ck[["rule", "total targets", "recall (%)"]].round(2).to_string(index=False)); assert ck["total targets"].tolist() == [184237, 28070], ck
B10 = LIST[("top 10,000", "10 kb")]; FOLD = 2 ** AF; EDG = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, np.inf]; BD = []
for d, sg in (("up", F > 0), ("down", F < 0)):
    for lo, hi in zip(EDG[:-1], EDG[1:]):
        M = OK & sg & (FOLD >= lo) & (FOLD < hi); BD.append({"direction": d, "fold band": f"{lo}-{hi}" if np.isfinite(hi) else f">= {lo}", "TF-gene pairs": int(M.sum()), "linked": int((M & B10).sum()), "share linked (%)": 100 * (M & B10).sum() / max(M.sum(), 1)})
BD = pd.DataFrame(BD); allshare = 100 * (OK & B10).sum() / OK.sum()
with pd.ExcelWriter(f"{O}/stage1_fold_change_recall_precision_raw.xlsx") as w:
    T[T["rule family"] != "check vs m50 V1"].to_excel(w, sheet_name="pooled", index=False); PT.to_excel(w, sheet_name="per-TF median, top10k 10kb", index=False); BD.to_excel(w, sheet_name="fold bands", index=False); T[T["rule family"] == "check vs m50 V1"].to_excel(w, sheet_name="check vs m50", index=False)
    pd.DataFrame({"note": ["Observed only, no shuffle. log2FC = log2((mean CP10k in the TF's top-25% sgRNA cells + 0.1) / (GEM-weighted NTC mean CP10k + 0.1)) (m50). A: |log2FC| >= log2(fold) (up: log2FC >= ; down: <= -). B: A and |z| >= 2 (m23 z = KD - NTC log1p difference / standard error). C: top X % of |log2FC| within the TF (up / down: top / bottom X % of signed log2FC).",
      "Links: top 10,000 lab peaks by signalValue, or peaks with narrowPeak column 9 (-log10 q) >= -log10(0.05); summit in a promoter or canonical Hi-C ABC-linked element, or within the padding of the TSS. Recall = in both / targets (TFs with >= 1 target); precision = in both / links (all 63 TFs); chance precision = sum over TFs of links_TF x targets_TF / genes_TF, divided by total links; 63 TFs, own gene removed.",
      f"Fold bands: share of TF-gene pairs in the band that are Regulome links (top 10,000, 10 kb), pooled over TFs; all pairs: {allshare:.2f} %."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 25); k = ["rule family", "rule", "direction", "targets per TF, median", "25th", "75th", "total targets", "in both", "recall (%)", "precision (%)", "chance precision (%)", "precision / chance"]
X = T[(T["peak set"] == "top 10,000") & (T.padding == "10 kb") & (T["rule family"] != "check vs m50 V1")]; print(X[k].round(2).to_string(index=False)); print(PT.round(2).to_string(index=False)); print(BD.round(2).to_string(index=False)); print("all pairs share linked:", round(allshare, 2))
Q = T[(T["peak set"] == "q <= 0.05") & (T.padding == "10 kb") & (T["rule family"] == "A fold change alone")]; print(Q[k].round(2).to_string(index=False))
W = T[(T["peak set"] == "top 10,000") & (T.padding != "10 kb") & (T.direction == "either") & T.rule.isin(["fold >= 1.1", "fold >= 1.2", "top 5%"])]; print(W[["padding"] + k].round(2).to_string(index=False)); print("done")
