# m53 (SPEC.md): within-TF percentile of the paper per-gene score; score spread vs cells, target histograms with the matched gene shuffle, Stage 1 with percentile responders.
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
S51 = open("/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/m51_paper_score.py").read(); exec(S51[:S51.index("A = np.abs(np.nan_to_num(SC))")])  # SC full precision, TFS, UG, OK, LISTS, STRAT, BROAD, NC, stage1()
O = "/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out"; PL = "/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/plots"; os.makedirs(O, exist_ok=True); os.makedirs(PL, exist_ok=True)
old = pd.read_csv("/work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out/paper_score_tf_by_gene.tsv.gz", sep="\t", index_col=0).reindex(index=TFS, columns=UG).values.astype(float)
mx = float(np.nanmax(np.abs(SC - old))); print("full-precision score vs saved 4-decimal file, max abs diff:", mx, flush=True); assert mx <= 5.1e-5 and (np.isfinite(old) == OK).all()
np.savez_compressed(f"{O}/paper_score_full_precision.npz", score=SC.astype(np.float64), tfs=np.array(TFS), genes=np.array(UG))
NT = len(TFS); PA = np.full(SC.shape, np.nan); PS = np.full(SC.shape, np.nan)
for i in range(NT):
    ok = OK[i]; n = ok.sum(); PA[i, ok] = 100 * (rankdata(np.abs(SC[i, ok])) - 0.5) / n; PS[i, ok] = 100 * (rankdata(SC[i, ok]) - 0.5) / n
NG_ok = OK.sum(1); print("genes per TF (OK):", int(NG_ok.min()), "-", int(NG_ok.max()), "| top 10/5/2/1 % per TF:", [int((PA[0] > 100 - x).sum()) for x in (10, 5, 2, 1)], flush=True)
br = np.array([t in BROAD for t in TFS]); three = np.array([t in ("NRF1", "E4F1", "DNMT1") for t in TFS]); NCc = np.array(NC)
# ---- Part 1
q = lambda p: np.array([np.percentile(np.abs(SC[i, OK[i]]), p) for i in range(NT)])
P1 = pd.DataFrame({"TF": TFS, "top-25% cells": NCc, "SD of score": [SC[i, OK[i]].std() for i in range(NT)], "90th pct |score|": q(90), "95th pct |score|": q(95), "99th pct |score|": q(99), "genes with |score| >= 0.3": [(np.abs(SC[i, OK[i]]) >= .3).sum() for i in range(NT)]})
P1C = pd.DataFrame([{"measure": c, "Spearman vs cells": spearmanr(P1[c], NCc).statistic, "p": spearmanr(P1[c], NCc).pvalue} for c in ("SD of score", "90th pct |score|", "95th pct |score|", "99th pct |score|", "genes with |score| >= 0.3")])
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); print(P1C.round(4).to_string(index=False), flush=True)
# ---- Part 2 helpers
NP = 1000
def binidx(P, nb): return np.minimum((P / (100 / nb)).astype(int), nb - 1)
def hist_draws(P, B, nb, rng):
    obs = np.zeros((NT, nb)); dr = np.zeros((NT, NP, nb)); tg = np.zeros(NT)
    for i in range(NT):
        ok = OK[i]; bi = binidx(P[i, ok], nb); b = B[i][ok]; s_ = STRAT[ok]; obs[i] = np.bincount(bi[b], minlength=nb); tg[i] = b.sum()
        for st in np.unique(s_[b]):
            m = s_ == st; col = np.bincount(bi[m], minlength=nb); k = int(b[m].sum()); dr[i] += rng.multivariate_hypergeometric(col, k, size=NP)
    return obs, dr, tg
def pooled(obs, dr, tg, sel):
    T = tg[sel].sum(); o = 100 * obs[sel].sum(0) / T; d = 100 * dr[sel].sum(0) / T
    return o, d.mean(0), np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0), [(1 + (d[:, k] >= o[k]).sum()) / (NP + 1) for k in range(len(o))], [(1 + (d[:, k] <= o[k]).sum()) / (NP + 1) for k in range(len(o))], int(T)
SETS2 = {"all 63 TFs": np.ones(NT, bool), "55 non-broad TFs": ~br, "without NRF1, E4F1, DNMT1": ~three}
rng = np.random.default_rng(20261118); H, HD = [], {}
for pad in ("10 kb", "0", "50 kb"):
    B = np.vstack([LISTS[pad][t] for t in TFS])
    for kind, P in (("|score| percentile", PA), ("signed percentile", PS)):
        obs, dr, tg = hist_draws(P, B, 20, rng)
        for sn, sel in SETS2.items():
            if pad != "10 kb" and sn != "all 63 TFs": continue
            o, m, lo, hi, pu, pd_, T = pooled(obs, dr, tg, sel); HD[(pad, kind, sn)] = (o, m, lo, hi)
            for k in range(20): H.append({"padding": pad, "percentile": kind, "TF set": sn, "targets pooled": T, "bin": f"{5 * k}-{5 * k + 5}", "share of targets (%)": o[k], "shuffle mean (%)": m[k], "shuffle 2.5 (%)": lo[k], "shuffle 97.5 (%)": hi[k], "observed / shuffle": o[k] / m[k], "p above (one-sided)": pu[k], "p below (one-sided)": pd_[k]})
    print("hist", pad, flush=True)
H = pd.DataFrame(H); print(H[(H.padding == "10 kb") & (H["TF set"] == "all 63 TFs")][["percentile", "bin", "share of targets (%)", "shuffle mean (%)", "shuffle 2.5 (%)", "shuffle 97.5 (%)", "p above (one-sided)", "p below (one-sided)"]].round(3).to_string(index=False), flush=True)
# per TF: 10 bins + top-set table
rng = np.random.default_rng(20261118); B10 = np.vstack([LISTS["10 kb"][t] for t in TFS]); obs10, dr10, tg10 = hist_draws(PA, B10, 10, rng)
TOPS = [("top 10% |score|", PA > 90), ("top 5% |score|", PA > 95), ("top 1% |score|", PA > 99), ("signed top 5% (up)", PS > 95), ("signed bottom 5% (down)", PS < 5)]
rng = np.random.default_rng(20261118); PT = []
for i in range(NT):
    ok = OK[i]; b = B10[i][ok]; s_ = STRAT[ok]; r = {"TF": TFS[i], "top-25% cells": int(NCc[i]), "broad TF": bool(br[i]), "target genes (10 kb)": int(b.sum())}
    for nm, M in TOPS:
        p_ = M[i][ok]; o = int((b & p_).sum()); dr = np.zeros(NP)
        for st in np.unique(s_[p_]):
            m = s_ == st; nn = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
            if nb: dr += rng.hypergeometric(nb, nn - nb, kk, size=NP)
        r.update({f"{nm}: targets in set": o, f"{nm}: share of targets (%)": 100 * o / b.sum(), f"{nm}: shuffle mean (%)": 100 * dr.mean() / b.sum(), f"{nm}: observed / shuffle": o / dr.mean() if dr.mean() else np.nan, f"{nm}: p (one-sided)": (1 + (dr >= o).sum()) / (NP + 1)})
    PT.append(r)
PT = pd.DataFrame(PT).sort_values("top-25% cells").reset_index(drop=True)
# ---- Part 3
SETS = {"all 63 TFs": np.ones(NT, bool), "55 non-broad TFs": ~br, "without NRF1, E4F1, DNMT1": ~three}
VS = []
for x in (10, 5, 2, 1): VS += [("|score| percentile", f"top {x}%", OK & (PA > 100 - x), SC), ("signed percentile, up", f"top {x}% up", OK & (PS > 100 - x), SC), ("signed percentile, down", f"bottom {x}% down", OK & (PS < x), SC)]
ST, LOO = stage1(VS, 20261118, True); ST = ST[ST.direction == "all"].drop(columns="direction"); LOO = LOO[LOO.direction == "all"].drop(columns="direction")
def ratio_tf(M):
    out = np.full(NT, np.nan)
    for i in range(NT):
        ok = OK[i]; b = B10[i][ok]; p_ = M[i][ok]; s_ = STRAT[ok]; ex_ = 0.
        for st in np.unique(s_[p_]): m = s_ == st; ex_ += b[m].sum() * p_[m].sum() / m.sum()
        if ex_ > 0: out[i] = (b & p_).sum() / ex_
    return out
R5 = ratio_tf(OK & (PA > 95)); R03 = ratio_tf(OK & (np.abs(np.nan_to_num(SC)) >= .3)); N03 = (OK & (np.abs(np.nan_to_num(SC)) >= .3)).sum(1)
RC = pd.DataFrame([{"responder rule": nm, "TFs": int(np.isfinite(v).sum()), "Spearman per-TF observed / expected (10 kb) vs cells": spearmanr(v[np.isfinite(v)], NCc[np.isfinite(v)]).statistic, "p": spearmanr(v[np.isfinite(v)], NCc[np.isfinite(v)]).pvalue,
                     "Spearman responders per TF vs cells": spearmanr(nr, NCc).statistic if np.ptp(nr) else np.nan} for nm, v, nr in (("top 5% of |score|", R5, (OK & (PA > 95)).sum(1)), ("|score| >= 0.3", R03, N03))])
PTR = pd.DataFrame({"TF": TFS, "top-25% cells": NCc, "observed / expected, top 5% |score|": R5, "observed / expected, |score| >= 0.3": R03, "responders at |score| >= 0.3": N03})
print(ST[(ST.padding == "10 kb")][["version", "cutoff", "TF set", "TFs contributing", "Perturb-seq genes", "recall (%)", "shuffle mean (%)", "gap (points)", "observed / shuffle", "p (one-sided)", "precision (%)", "precision shuffle (%)"]].round(3).to_string(index=False))
print(LOO[["version", "cutoff", "recall min (%)", "recall max (%)", "gap min (points)", "gap max (points)", "TF left out at min gap"]].round(3).to_string(index=False)); print(RC.round(4).to_string(index=False), flush=True)
note = pd.DataFrame({"note": ["Score: m51 paper per-gene score, per top-25% cell (log1p CP10k - own-GEM NTC mean) / own-GEM NTC SD, averaged over the TF's cells; recomputed at full precision (max difference to the saved 4-decimal file reported in the log). Own gene removed; 11,228-gene universe.",
  "Within-TF percentile: average ranks over the TF's genes; 100 (rank - 0.5) / n; |score| version (100 = largest |score|) and signed version (0 = most down, 100 = most up). Top X % = percentile > 100 - X; bottom X % = percentile < X.",
  "Targets: Regulome links, top 10,000 lab peaks per TF, summit in a promoter or canonical Hi-C ABC-linked element or within the padding of the TSS (10 kb main; 0 and 50 kb). Matched gene shuffle: targets redrawn within NTC expression decile x promoter ATAC decile per TF (multivariate hypergeometric over bins for the histograms, hypergeometric for sets), 1,000 shuffles, seed 20261118.",
  "Stage 1: recall pooled over TFs; precision = in both / our links of those TFs; one shuffle run for the Stage 1 table; one-sided p = (1 + # shuffles >= observed) / 1001. Per-TF observed / expected uses the exact hypergeometric mean."]})
with pd.ExcelWriter(f"{O}/stage1_percentile.xlsx") as w:
    P1.to_excel(w, sheet_name="1 score spread per TF", index=False); P1C.to_excel(w, sheet_name="1 spread vs cells", index=False)
    H.to_excel(w, sheet_name="2 pooled histograms", index=False); PT.to_excel(w, sheet_name="2 per TF target sets", index=False)
    pd.DataFrame([{"TF": TFS[i], "top-25% cells": int(NCc[i]), "targets": int(tg10[i]), **{f"bin {10 * k}-{10 * k + 10} (%)": 100 * obs10[i, k] / tg10[i] for k in range(10)}, **{f"shuffle mean {10 * k}-{10 * k + 10} (%)": 100 * dr10[i, :, k].mean() / tg10[i] for k in range(10)}} for i in range(NT)]).sort_values("top-25% cells").to_excel(w, sheet_name="2 per TF histograms", index=False)
    ST[ST.padding == "10 kb"].to_excel(w, sheet_name="3 Stage 1 10 kb", index=False); ST[ST.padding != "10 kb"].to_excel(w, sheet_name="3 Stage 1 0 and 50 kb", index=False); LOO.to_excel(w, sheet_name="3 leave one TF out 10 kb", index=False)
    RC.to_excel(w, sheet_name="3 per-TF ratio vs cells", index=False); PTR.to_excel(w, sheet_name="3 per-TF ratios", index=False); note.to_excel(w, sheet_name="notes", index=False)
np.savez_compressed(f"{O}/plot_data.npz", **{f"{a}|{b}|{c}": np.vstack(v) for (a, b, c), v in HD.items()}, obs10=obs10, dr10_mean=dr10.mean(1), dr10_lo=np.percentile(dr10, 2.5, 1), dr10_hi=np.percentile(dr10, 97.5, 1), tg10=tg10, tfs=np.array(TFS), cells=NCc, p95=P1["95th pct |score|"].values)
print("done")
