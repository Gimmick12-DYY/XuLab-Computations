# m56 (SPEC.md + Amendments): Stage 1, top 10 / 5 / 2 / 1 % of |score| per TF on all 63 TFs (main), recall and precision; E-distance-gated 57 TFs as a side check.
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata
SRC50 = open("/work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/m50_stage1.py").read(); exec(SRC50[:SRC50.index("# ---- responder counts")]); exec(SRC50[SRC50.index("# ---- Stage 1"):SRC50.index("ST, LOO = stage1(")])  # TFS, UG, OK, LISTS, STRAT, BROAD, stage1()
O = "/work/users/d/y/dyy12/XuLab/regulatome/m56_stage1_top5pct/out"; os.makedirs(O, exist_ok=True)
F = np.load("/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True)
SC = pd.DataFrame(F["score"], index=F["tfs"].astype(str), columns=F["genes"].astype(str)).reindex(index=TFS, columns=UG).values; assert (np.isfinite(SC) == OK).all()
PA = np.full(SC.shape, np.nan); PS = np.full(SC.shape, np.nan)
for i in range(len(TFS)): ok = OK[i]; n = ok.sum(); PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / n; PS[i, ok] = 100 * (rankdata(SC[i, ok]) - 0.5) / n
ED = pd.read_excel("/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out/stage1_edistance.xlsx", sheet_name="E-distance per TF").set_index("TF")
q = ED["q BH (11,228 genes)"].reindex(TFS).values; assert np.isfinite(q).all(); G = q < 0.05; print("E-distance q < 0.05:", int(G.sum()), "| not:", [t for t, g in zip(TFS, G) if not g], flush=True)
three = np.array([t in ("NRF1", "E4F1", "DNMT1") for t in TFS]); br = np.array([t in BROAD for t in TFS])
SETS = {"all 63 TFs (main)": np.ones(len(TFS), bool), "55 non-broad TFs": ~br, "without NRF1, E4F1, DNMT1": ~three, "side check: 57 TFs with E-distance q < 0.05 (not the main analysis)": G}
VS = []
for x in (10, 5, 2, 1): VS += [("all (top % of |score|)", f"top {x}%", OK & (PA > 100 - x), SC), ("up (top % of signed score)", f"top {x}% up", OK & (PS > 100 - x), SC), ("down (bottom % of signed score)", f"bottom {x}% down", OK & (PS < x), SC)]
ST, LOO = stage1(VS, 20261118, False); ST = ST[ST.direction == "all"].drop(columns="direction")
# leave-one-TF-out over the 63 TFs at 10 kb (same procedure, own run)
SETS_LOO = {f"63 without {t}": np.arange(len(TFS)) != i for i, t in enumerate(TFS)}
SETS_BAK = SETS; SETS = SETS_LOO
L = stage1([v for v in VS if v[0].startswith("all")], 20261118, False)[0]; SETS = SETS_BAK
L = L[(L.direction == "all") & (L.padding == "10 kb")]
LS = L.groupby("cutoff").agg(**{"recall min (%)": ("recall (%)", "min"), "recall max (%)": ("recall (%)", "max"), "gap min (points)": ("gap (points)", "min"), "gap max (points)": ("gap (points)", "max"), "precision min (%)": ("precision (%)", "min"), "precision max (%)": ("precision (%)", "max"), "p max": ("p (one-sided)", "max")}).reset_index()
LS["TF left out at min gap"] = [L[L.cutoff == c].sort_values("gap (points)").iloc[0]["TF set"].replace("63 without ", "") for c in LS.cutoff]
# per TF and padding, top 5 % (all / up / down)
rng = np.random.default_rng(20261118); NP = 1000; PT = []
for pad in ("10 kb", "0", "50 kb"):
    B = np.vstack([LISTS[pad][t] for t in TFS])
    for i, t in enumerate(TFS):
        ok = OK[i]; b = B[i][ok]; s_ = STRAT[ok]; ours = int(b.sum()); r = {"padding": pad, "TF": t, "E-distance q (side check)": q[i], "E-distance q >= 0.05 (flag)": bool(not G[i]), "broad TF": bool(br[i]), "candidate target genes": ours}
        for nm, M in (("top 5%", PA > 95), ("top 5% up", PS > 95), ("bottom 5% down", PS < 5)):
            p_ = M[i][ok]; k = int(p_.sum()); o = int((b & p_).sum()); dr = np.zeros(NP)
            for st in np.unique(s_[p_]):
                m = s_ == st; nn = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
                if nb: dr += rng.hypergeometric(nb, nn - nb, kk, size=NP)
            r.update({f"{nm}: Perturb-seq targets": k, f"{nm}: genes in both": o, f"{nm}: recall (%)": 100 * o / k, f"{nm}: precision (%)": 100 * o / ours, f"{nm}: shuffle mean genes in both": dr.mean(), f"{nm}: shuffle recall (%)": 100 * dr.mean() / k, f"{nm}: shuffle precision (%)": 100 * dr.mean() / ours, f"{nm}: observed / shuffle": o / dr.mean(), f"{nm}: p (one-sided)": (1 + (dr >= o).sum()) / (NP + 1)})
        PT.append(r)
PT = pd.DataFrame(PT); o10 = PT[PT.padding == "10 kb"].set_index("TF")["top 5%: observed / shuffle"]; PT["_k"] = PT.TF.map(o10); PT = PT.sort_values(["_k", "padding"], ascending=[False, True]).drop(columns="_k")
cols = ["padding", "version", "cutoff", "TF set", "TFs contributing", "Perturb-seq genes", "our links", "in both", "recall (%)", "shuffle mean (%)", "gap (points)", "precision (%)", "precision shuffle (%)", "observed / shuffle", "shuffle 2.5-97.5 (%)", "p (one-sided)"]
ST = ST[cols]; ST["precision gap (points)"] = ST["precision (%)"] - ST["precision shuffle (%)"]
note = pd.DataFrame({"note": ["Responders: per TF, genes in the top 5 % (main) / 2 % of |score| within the TF (paper per-gene score, full precision); up = top X % of the signed score, down = bottom X %. TFs: E-distance q < 0.05 (m51) = 57 main; all 63 for comparison.",
  "Regulome: top 10,000 lab peaks per TF; summit in a promoter or canonical Hi-C ABC-linked element, plus padding (10 kb main; 0, 50 kb).",
  "Recall = in both / Perturb-seq genes; precision = in both / our links over the same TFs. Matched gene shuffle within NTC expression x promoter ATAC deciles, exact hypergeometric per TF and stratum, 1,000 shuffles, seed 20261118 (one run for the main table; recall and precision share the shuffle and the one-sided p = (1 + #>= observed) / 1001)."]})
REN = {"Perturb-seq genes": "Perturb-seq targets", "our links": "candidate links", "in both": "links in both", "recall (%)": "recall observed (%)", "shuffle mean (%)": "recall shuffle (%)", "precision (%)": "precision observed (%)", "precision shuffle (%)": "precision shuffle (%)"}
POOL = ST[ST.version.str.startswith("all")].rename(columns=REN); UD = ST[~ST.version.str.startswith("all")].rename(columns=REN)
note = pd.DataFrame({"term": ["score", "percentile", "recall", "precision", "matched shuffle", "TF sets", "candidate links"], "definition": [
  "Paper per-gene score: per top-25% sgRNA cell, (log1p CP10k - NTC mean of its GEM) / NTC SD of its GEM, averaged over the TF's cells; own gene removed.",
  "Within-TF percentile of |score| (100 (average rank - 0.5) / n over the TF's ~11,227 genes); top X % = Perturb-seq targets of that TF (top 5 % = 561 genes). Up / down: top X % / bottom X % of the signed score.",
  "Recall = links in both / Perturb-seq targets, pooled over the TFs of the set.",
  "Precision = links in both / candidate links of the same TFs.",
  "Perturb-seq labels redrawn within NTC expression decile x promoter ATAC decile strata per TF (exact hypergeometric), 1,000 shuffles, seed 20261118, one run per table; recall and precision of a row share it; one-sided p = (1 + #>= observed) / 1001.",
  "Main analysis = all 63 TFs. 55 non-broad = m45 grouping; without NRF1, E4F1, DNMT1 = of the 63. Side check only (not the main analysis): the 57 TFs with E-distance q < 0.05 (m51; excludes E2F3, CHAMP1, ZNF644, HOXB9, AKAP8L, HOXB6).",
  "Top 10,000 lab HITAG peaks per TF; gene linked if a summit is in its promoter or a canonical Hi-C ABC-linked element, or within the padding (0 / 10 / 50 kb) of its TSS."]})
with pd.ExcelWriter(f"{O}/stage1_percentile_precision_recall.xlsx") as w:
    POOL.to_excel(w, sheet_name="pooled", index=False); UD.to_excel(w, sheet_name="up_down", index=False); PT.to_excel(w, sheet_name="per_TF_top5", index=False); LS.to_excel(w, sheet_name="leave_one_out", index=False); note.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 30)
print(ST[ST.cutoff.isin(["top 10%", "top 5%", "top 2%", "top 1%", "top 5% up", "bottom 5% down"])].drop(columns=["shuffle 2.5-97.5 (%)", "version"]).round(3).to_string(index=False)); print(LS.round(3).to_string(index=False))
g = PT[PT.padding == "10 kb"]
for c in ("top 5%: recall (%)", "top 5%: precision (%)", "top 5%: observed / shuffle"): v = g[c]; print(c, "median", round(v.median(), 3), "IQR", round(v.quantile(.25), 3), "-", round(v.quantile(.75), 3), "min", round(v.min(), 3), "max", round(v.max(), 3))
print("TFs obs/shuf > 1:", int((g["top 5%: observed / shuffle"] > 1).sum()), "| p < 0.05:", int((g["top 5%: p (one-sided)"] < 0.05).sum()), "| up p < 0.05:", int((g["top 5% up: p (one-sided)"] < 0.05).sum()), "| down p < 0.05:", int((g["bottom 5% down: p (one-sided)"] < 0.05).sum()))
k = ["TF", "E-distance q >= 0.05 (flag)", "candidate target genes", "top 5%: genes in both", "top 5%: recall (%)", "top 5%: precision (%)", "top 5%: shuffle mean genes in both", "top 5%: observed / shuffle", "top 5%: p (one-sided)"]
print(g[k].head(10).round(3).to_string(index=False)); print(g[k].tail(10).round(3).to_string(index=False))
print("done")
