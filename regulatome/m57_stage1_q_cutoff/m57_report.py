# m57 report: one plain recall / precision table assembled from existing outputs (m56, m57 raw, looser targets, fold change, link type and rank, ABC vs distance); no new overlap computation.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{R}/m57_stage1_q_cutoff/out"; X = lambda f, sh: pd.read_excel(f"{O}/{f}", sheet_name=sh)
F = np.load(f"{R}/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True); SC = F["score"]; OK = np.isfinite(SC); NPAIR = int(OK.sum()); assert NPAIR == 707239
def pct_counts(x):
    n = []
    for i in range(SC.shape[0]):
        ok = OK[i]; pa = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / ok.sum(); n.append(int((pa > 100 - x).sum()))
    return float(np.median(n)), int(np.sum(n))
PK = X("stage1_q_cutoff_recall_precision_raw.xlsx", "peak sets").set_index("peak set")
RAW = X("stage1_q_cutoff_recall_precision_raw.xlsx", "pooled"); LO = X("stage1_looser_targets_recall_precision_raw.xlsx", "pooled"); LOUD = X("stage1_looser_targets_recall_precision_raw.xlsx", "up_down")
FC = X("stage1_fold_change_recall_precision_raw.xlsx", "pooled"); LT = X("stage1_link_type_and_rank_raw.xlsx", "link type"); RK = X("stage1_link_type_and_rank_raw.xlsx", "strength rank"); AD = X("stage1_abc_vs_distance_raw.xlsx", "rules A-C")
for d in (RAW, LO, LOUD, FC): d["padding"] = d["padding"].astype(str)
rows = []
def add(block, setting, peaks, genes_med, links, tmed, ttot, both, rec, prec, zero):
    rows.append({"block": block, "setting": setting, "peaks per TF (median)": peaks, "our linked genes per TF (median)": genes_med, "our links (total)": int(links), "Perturb-seq targets per TF (median)": tmed, "targets (total)": int(ttot),
                 "targets as % of all genes": 100 * ttot / NPAIR, "in both": int(both), "recall %": rec, "precision %": prec, "TFs with no links": zero})
def lo(rule, ps="top 10,000", pad="10 kb", src=None):
    d = (src if src is not None else LO); r = d[(d["peak set"] == ps) & (d.padding == pad) & (d.rule == rule)]; assert len(r) == 1, (rule, ps, pad, len(r)); return r.iloc[0]
def gm(ps, pad): return float(PK.loc[ps, f"genes per TF, {pad}: median"])
T5 = pct_counts(5)
h = lo("top 5%"); add("headline", "HEADLINE: top 10,000 peaks, promoter + ABC + 10 kb, top 5 % of genes per TF, 63 TFs", 10000, gm("top 10,000", "10 kb"), h["candidate links"], T5[0], h["Perturb-seq targets"], h["links in both"], h["recall (%)"], h["precision (%)"], 0)
for x in (50, 20, 10, 2, 1):
    r = lo(f"top {x}%"); c = pct_counts(x); add("Perturb-seq rule, same links", f"top {x} % of genes per TF", 10000, gm("top 10,000", "10 kb"), r["candidate links"], c[0], r["Perturb-seq targets"], r["links in both"], r["recall (%)"], r["precision (%)"], 0)
for rule, lab in (("top 5% up", "up only, top 5 %"), ("bottom 5% down", "down only, bottom 5 %")):
    r = lo(rule, src=LOUD); add("Perturb-seq rule, same links", lab, 10000, gm("top 10,000", "10 kb"), r["candidate links"], T5[0], r["Perturb-seq targets"], r["links in both"], r["recall (%)"], r["precision (%)"], 0)
for fam, d, lab in (("A fold change alone", "either", "fold >= 1.1"), ("B fold change and |z| >= 2", "either", "fold >= 1.1 and |z| >= 2"), ("B fold change and |z| >= 2", "up", "fold >= 1.1 and |z| >= 2, up only")):
    r = FC[(FC["peak set"] == "top 10,000") & (FC.padding == "10 kb") & (FC["rule family"] == fam) & (FC.rule == "fold >= 1.1") & (FC.direction == d)].iloc[0]
    add("Perturb-seq rule, same links", lab, 10000, gm("top 10,000", "10 kb"), r["links"], r["targets per TF, median"], r["total targets"], r["in both"], r["recall (%)"], r["precision (%)"], 0)
for ps in ("top 5,000", "q <= 0.05", "q <= 0.01", "q <= 1e-3", "q <= 1e-5"):
    r = RAW[(RAW["peak set"] == ps) & (RAW.padding == "10 kb") & (RAW.rule == "top 5%") & (RAW["TF set"] == "all 63 TFs (main)")].iloc[0]
    add("Peak set, top 5 % targets", ps, float(PK.loc[ps, "peaks per TF, median"]), gm(ps, "10 kb"), r["candidate links"], T5[0], r["Perturb-seq targets"], r["links in both"], r["recall (%)"], r["precision (%)"], int(r["TFs with 0 links"]))
for pad, lab in (("0", "padding 0 kb"), ("50 kb", "padding 50 kb")):
    r = lo("top 5%", pad=pad); add("Padding, top 5 %", lab, 10000, gm("top 10,000", pad), r["candidate links"], T5[0], r["Perturb-seq targets"], r["links in both"], r["recall (%)"], r["precision (%)"], 0)
for rule, lab in (("promoter + all distal ABC (0 kb rule)", "ABC only (promoter + all ABC elements, no window)"), ("summit within +-10 kb of TSS", "distance only, within 10 kb"), ("summit within +-20 kb of TSS", "distance only, within 20 kb"),
                  ("summit within +-50 kb of TSS", "distance only, within 50 kb"), ("current without distal ABC > 50 kb", "current rule without distal ABC > 50 kb")):
    r = AD[AD.rule == rule].iloc[0]; add("Link rule, top 5 %", lab, 10000, r["median genes per TF"], r["links"], T5[0], 35343, r["top 5% either: in both"], r["top 5% either: recall (%)"], r["top 5% either: precision (%)"], int((r["links"] == 0)))
for n in (100, 250, 500, 1000, 2000):
    r = RK[(RK.ranking == "strongest linked peak signalValue") & (RK["top N genes per TF"].astype(str) == str(n)) & (RK.targets == "top 5%") & (RK.direction == "either")].iloc[0]
    add("Strongest linked genes per TF (current rule, by peak signalValue), top 5 %", f"top {n:,} genes per TF", 10000, float(min(n, 2892)) if n < 2000 else np.nan, r["links"], T5[0], 35343, r["in both"], r["recall (%)"], r["precision (%)"], 0)
for cls, lab in (("(1) promoter of the gene", "own promoter"), ("(2) distal ABC, nearest bound element 0-10 kb", "distal ABC < 10 kb"), ("(2) distal ABC, nearest bound element 10-50 kb", "distal ABC 10-50 kb"),
                 ("(2) distal ABC, nearest bound element 50-200 kb", "distal ABC 50-200 kb"), ("(2) distal ABC, nearest bound element > 200 kb", "distal ABC > 200 kb"), ("(3) 10 kb window only", "10 kb window only")):
    r = LT[(LT["class"] == cls) & (LT.targets == "top 5%") & (LT.direction == "either")].iloc[0]
    add("By link type (current rule, top 5 %)", lab, 10000, r["median genes per TF"], r["links"], T5[0], 35343, r["in both"], r["recall contribution (%)"], r["precision (%)"], np.nan)
REP = pd.DataFrame(rows)
P = pd.read_excel(f"{R}/m56_stage1_top5pct/out/stage1_percentile_precision_recall.xlsx", sheet_name="per_TF_top5"); P = P[P.padding.astype(str) == "10 kb"]
PT = pd.DataFrame({"TF": P.TF, "peaks": 10000, "our linked genes": P["candidate target genes"], "Perturb-seq targets": P["top 5%: Perturb-seq targets"], "in both": P["top 5%: genes in both"], "recall %": P["top 5%: recall (%)"], "precision %": P["top 5%: precision (%)"]}).sort_values("precision %", ascending=False)
assert int(PT["in both"].sum()) == 11536
NOTES = pd.DataFrame({"block": ["definitions", "headline", "Perturb-seq rule, same links", "Peak set", "Padding", "Link rule", "Strongest linked genes", "By link type", "per_TF"], "source": [
  "Recall = in both / Perturb-seq targets; precision = in both / our links. 63 TFs, 11,228-gene universe (707,239 TF-gene pairs), own gene removed. Targets: top X % of |score| within the TF (paper per-gene score, full precision) unless stated. Links: top 10,000 lab peaks by signalValue; summit in a promoter or canonical Hi-C ABC-linked element, or within 10 kb of the TSS, unless stated.",
  "m57 looser-targets run, job 10493172, commit 83371a0 (identical to m56 job 10492565, commit a05dc07).",
  "Top X %: job 10493172 (83371a0). Up / down: same job. Fold change (log2((KD CP10k + 0.1)/(NTC + 0.1))) and |z| (m23 z): job 10493662, commit 790c25d; recall over TFs with >= 1 target, precision over all links.",
  "m57 raw run, job 10492917, commit f02b3a7. q <= X = narrowPeak column 9 (-log10 q) >= -log10(X); top 5,000 by signalValue.",
  "Job 10493172 (83371a0).", "ABC vs distance run, job 10494793, commit bba7b58.", "Link type and rank run, job 10494520, commit b2646cb; genes ranked by the strongest linking peak's signalValue.",
  "Job 10494520 (b2646cb). Exclusive classes: own promoter element; else distal ABC element (split by the nearest bound distal element's distance to the TSS); else 10 kb window only. Recall column = this class's contribution to the headline recall.",
  "Headline setting per TF, from m56 job 10492565 (commit a05dc07)."]})
with pd.ExcelWriter(f"{O}/stage1_precision_recall_report.xlsx") as w:
    REP.to_excel(w, sheet_name="report", index=False); PT.to_excel(w, sheet_name="per_TF", index=False); NOTES.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 60)
print(REP.round(2).to_string(index=False))
for c in ("recall %", "precision %"): v = PT[c]; print(c, "median", round(v.median(), 2), "IQR", round(v.quantile(.25), 2), "-", round(v.quantile(.75), 2), "min", round(v.min(), 2), v.loc[v.idxmin()] if False else PT.loc[v.idxmin(), "TF"], "max", round(v.max(), 2), PT.loc[v.idxmax(), "TF"])
print("done")
