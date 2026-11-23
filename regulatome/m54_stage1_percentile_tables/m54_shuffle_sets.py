# m54: generated from m52_shuffle_sets.py (m52) by gen54.py.
# m48: matched gene shuffle by TF set and leave-one-TF-out for top 10,000 (from m45_shuffle_sets.py; bulk ChIP part removed, precision added).
# m45 Amendment 1 (SPEC.md): matched gene shuffle by TF set, leave-one-TF-out, bulk ChIP vs ours at padding 10 kb.
import os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M = f"{R}/m54_stage1_percentile_tables"
s = open(f"{M}/m54_tables.py").read(); exec(s[:s.index("VERS = {")])  # LISTS (paddings), TFS, OK, AZ, STRAT, BROAD, window, rule_abc, X
O = f"{M}/out"; rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1)
ti = {t: i for i, t in enumerate(TFS)}; EX3 = {"NRF1", "E4F1", "DNMT1"}
def per_tf(blist, tfs, c):
    global OURS_
    obs, den, dr = np.zeros(len(tfs)), np.zeros(len(tfs)), np.zeros((len(tfs), NP)); OURS_ = np.array([int(blist[t][OK[ti[t]]].sum()) for t in tfs])
    for k, t in enumerate(tfs):
        i = ti[t]; s_ = STRAT[OK[i]]; b = blist[t][OK[i]]; p_ = (AZ[i] >= c)[OK[i]]; obs[k] = (b & p_).sum(); den[k] = p_.sum()
        for st in np.unique(s_):
            m = s_ == st; n = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
            if kk and nb: dr[k] += rng.hypergeometric(nb, n - nb, kk, size=NP)
    return obs, den, dr
def row(obs, den, dr, sel):
    o = 100 * obs[sel].sum() / den[sel].sum(); dd = 100 * dr[sel].sum(0) / den[sel].sum()
    return {"TFs": int(sel.sum()), "Perturb-seq genes": int(den[sel].sum()), "in both": int(obs[sel].sum()), "recall observed (%)": o, "shuffle mean (%)": dd.mean(), "observed minus shuffle mean (points)": o - dd.mean(),
            "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd), "our links": int(OURS_[sel].sum()), "precision observed (%)": 100 * obs[sel].sum() / OURS_[sel].sum(), "precision shuffle (%)": dd.mean() * den[sel].sum() / OURS_[sel].sum(), "observed / shuffle": o / dd.mean()}
SETS = {"all 63": set(), "without NRF1, E4F1, DNMT1": EX3, "without broad TFs": BROAD, "without both": EX3 | BROAD, "without broad TFs, top-10k grouping": BROAD10, "without both, top-10k grouping": EX3 | BROAD10, "broad TFs only (8)": set(TFS) - BROAD}
A1, LOO, LOOS, BK = [], [], [], []
for pad in PADS:
    for c in (2, 3, 4, 5):
        obs, den, dr = per_tf(LISTS[pad], TFS, c)
        for sn, ex_ in SETS.items():
            sel = np.array([t not in ex_ for t in TFS]) & (den > 0); A1.append({"padding": pad, "cutoff": f"top {CUTL[c]}% of genes", "set": sn, **row(obs, den, dr, sel)})
        if pad == "10 kb" and c in (3, 4):
            for k, t in enumerate(TFS):
                sel = (np.arange(NT) != k) & (den > 0); LOO.append({"padding": pad, "cutoff": f"top {CUTL[c]}% of genes", "left out": t, **row(obs, den, dr, sel)})
    print("padding", pad, flush=True)
L = pd.DataFrame(LOO)
for c, g in L.groupby("cutoff"):
    full = [r for r in A1 if r["padding"] == "10 kb" and r["cutoff"] == c and r["set"] == "all 63"][0]; w = g.loc[g["observed minus shuffle mean (points)"].idxmin()]
    LOOS.append({"padding": "10 kb", "cutoff": c, "all 63, gap (points)": full["observed minus shuffle mean (points)"], "all 63, p": full["p (one-sided)"], "leave-one-out gap min": g["observed minus shuffle mean (points)"].min(), "leave-one-out gap max": g["observed minus shuffle mean (points)"].max(),
                 "leave-one-out p min": g["p (one-sided)"].min(), "leave-one-out p max": g["p (one-sided)"].max(), "TF whose removal lowers the gap most": w["left out"], "gap without it": w["observed minus shuffle mean (points)"], "p without it": w["p (one-sided)"],
                 "next 3 TFs": ", ".join(g.sort_values("observed minus shuffle mean (points)")["left out"].iloc[1:4])})
TT = {"6_shuffle_by_tf_set": pd.DataFrame(A1), "7_leave_one_tf_out_10kb": L, "7_leave_one_tf_out_summary": pd.DataFrame(LOOS)}
for n, d in TT.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_top10k_shuffle_sets.xlsx") as w:
    for n, d in TT.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40)
for n in ("6_shuffle_by_tf_set", "7_leave_one_tf_out_summary"): print("==", n); print(TT[n].round(3).to_string(index=False))
print("done")
