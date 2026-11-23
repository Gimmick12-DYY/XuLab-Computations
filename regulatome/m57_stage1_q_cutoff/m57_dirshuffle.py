# m57 Amendment 7: matched gene shuffle for distal link sets by direction (nearest-gene, ABC distal, loops) beside proximal references.
import numpy as np, pandas as pd
SRC = open("/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/m57_distal.py").read(); exec(SRC[:SRC.index("TG = {}")])  # masks PROM, DIS, WIN, PD, LOOP, LOOPONLY, NEAR, NEAR100, NONPW, CUR; ex, at, dec, STRAT
O = "/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/out"
SETS = [("1 nearest-gene distal rule", "all", NEAR), ("1 nearest-gene distal rule", "<= 100 kb", NEAR100), ("1 nearest-gene distal rule", "excluding promoter / 10 kb genes", NEAR & NONPW), ("1 nearest-gene distal rule", "<= 100 kb, excluding promoter / 10 kb genes", NEAR100 & NONPW),
        ("2 pure distal ABC", "pure distal ABC links", PD), ("3 loops", "loop-supported pure distal ABC", PD & LOOP), ("3 loops", "loop-only rule, all", LOOPONLY), ("3 loops", "loop-only rule, excluding proximal", LOOPONLY & NONPW),
        ("4 reference", "distance within 10 kb", WIN), ("4 reference", "current rule", PROM | DIS | WIN), ("4 reference", "10 kb window-only class", WIN & ~PROM & ~DIS)]
SETS = [(a, b, M & OK) for a, b, M in SETS]
TG = {}
for x in (5, 1): TG[(x, "either")] = OK & (PA > 100 - x); TG[(x, "up")] = OK & (PS > 100 - x); TG[(x, "down")] = OK & (PS < x)
NP = 1000
def run(strat, sets, seed, loo_key=None):
    rng = np.random.default_rng(seed); rows = []; loo = None
    for fam, nm, B in sets:
        links = B.sum(1)
        for (x, d), T in TG.items():
            ib = (B & T).sum(1); dr = np.zeros((NT, NP))
            for i in range(NT):
                if not links[i]: continue
                s_ = strat[OK[i]]; b = B[i][OK[i]]; p_ = T[i][OK[i]]
                for st in np.unique(s_[p_]):
                    m = s_ == st; nn = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
                    if nb: dr[i] += rng.hypergeometric(nb, nn - nb, kk, size=NP)
            o = ib.sum(); dd = dr.sum(0); L_ = links.sum(); pu = (1 + (dd >= o).sum()) / (NP + 1); pl = (1 + (dd <= o).sum()) / (NP + 1)
            rows.append({"set": fam, "rule": nm, "targets": f"top {x}%", "direction": d, "links": int(L_), "in both": int(o), "recall (%)": 100 * o / T.sum(), "precision (%)": 100 * o / L_, "shuffle mean precision (%)": 100 * dd.mean() / L_,
                         "shuffle precision 2.5-97.5 (%)": f"{100 * np.percentile(dd, 2.5) / L_:.2f}-{100 * np.percentile(dd, 97.5) / L_:.2f}", "observed / shuffle": o / dd.mean(), "p upper (one-sided)": pu, "p lower (one-sided)": pl, "p two-sided": min(1.0, 2 * min(pu, pl))})
            if loo_key == (nm, x, d):
                loo = pd.DataFrame([{"left out": TFS[k], "observed / shuffle": ib[np.arange(NT) != k].sum() / dr[np.arange(NT) != k].sum(0).mean()} for k in range(NT)])
        print(fam, nm, flush=True)
    return pd.DataFrame(rows), loo
T1, LOO = run(STRAT, SETS, 20261118, loo_key=("all", 5, "down"))
# variant: expression x ATAC x nearest-neighbour TSS distance quintiles
q5 = lambda v: pd.qcut(pd.Series(v).rank(method="first"), 5, labels=False).values
nn_d = np.full(NGn, np.nan)
for ch, (tp, ix) in TB.items():
    if len(tp) < 2: nn_d[ix] = 1e9; continue
    d = np.full(len(tp), np.inf); d[1:] = np.minimum(d[1:], np.diff(tp)); d[:-1] = np.minimum(d[:-1], np.diff(tp)); nn_d[ix] = d
nn_d = np.nan_to_num(nn_d, nan=1e9); STRAT3 = q5(ex) * 25 + q5(at) * 5 + q5(nn_d)
print("variant strata used:", len(np.unique(STRAT3)), "| smallest stratum (genes):", int(pd.Series(STRAT3).value_counts().min()), flush=True)
T2, _ = run(STRAT3, [s for s in SETS if s[0].startswith("1")], 20261118)
DEXg = dec(ex); DATg = dec(at); MED = []
for fam, nm, B in SETS:
    if fam[0] in "14":
        ii, gg = np.nonzero(B); MED.append({"set": fam, "rule": nm, "links": int(B.sum()), "median NTC expression decile (1-10)": float(np.median(DEXg[gg]) + 1), "median promoter ATAC decile (1-10)": float(np.median(DATg[gg]) + 1),
                    "median TSS nearest-neighbour distance (kb)": float(np.median(nn_d[gg]) / 1000)})
MED.append({"set": "all tested pairs", "rule": "universe", "links": int(OK.sum()), "median NTC expression decile (1-10)": float(np.median(DEXg[np.nonzero(OK)[1]]) + 1), "median promoter ATAC decile (1-10)": float(np.median(DATg[np.nonzero(OK)[1]]) + 1), "median TSS nearest-neighbour distance (kb)": float(np.median(nn_d[np.nonzero(OK)[1]]) / 1000)})
MED = pd.DataFrame(MED); LS = pd.DataFrame([{"rule": "nearest-gene distal rule, all; down only; top 5 %", "observed / shuffle min": LOO["observed / shuffle"].min(), "max": LOO["observed / shuffle"].max(), "TF left out at min": LOO.loc[LOO["observed / shuffle"].idxmin(), "left out"], "TF left out at max": LOO.loc[LOO["observed / shuffle"].idxmax(), "left out"]}])
with pd.ExcelWriter(f"{O}/stage1_distal_direction_shuffle.xlsx") as w:
    T1.to_excel(w, sheet_name="shuffle expr x ATAC", index=False); T2.to_excel(w, sheet_name="nearest-gene, +gene density", index=False); MED.to_excel(w, sheet_name="linked gene composition", index=False); LS.to_excel(w, sheet_name="leave one TF out", index=False); LOO.to_excel(w, sheet_name="LOO per TF", index=False)
    pd.DataFrame({"note": ["Matched gene shuffle: Perturb-seq targets redrawn per TF within NTC expression decile x promoter ATAC decile (100 strata), exact hypergeometric per stratum, 1,000 shuffles, one random stream (seed 20261118) for all rows of the first sheet; precision shuffle = shuffle in-both / links.",
      "Variant: strata = quintiles of NTC expression x promoter ATAC x distance from the gene's TSS to the nearest other universe TSS (125 strata), own stream with the same seed.", "Link sets as in stage1_distal_links_raw.xlsx (m57_distal.py). 63 TFs, top 10,000 peaks, targets top X % of |score| within the TF; up / down from the signed score."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 45)
k = ["rule", "targets", "direction", "links", "in both", "recall (%)", "precision (%)", "shuffle mean precision (%)", "shuffle precision 2.5-97.5 (%)", "observed / shuffle", "p upper (one-sided)", "p lower (one-sided)"]
print(T1[k].round(3).to_string(index=False)); print(T2[k].round(3).to_string(index=False)); print(MED.round(2).to_string(index=False)); print(LS.round(3).to_string(index=False)); print("done")
