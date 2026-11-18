# m45 Amendment 1 (SPEC.md): matched gene shuffle by TF set, leave-one-TF-out, bulk ChIP vs ours at padding 10 kb.
import os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M = f"{R}/m45_stage1_summary"
s = open(f"{M}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])  # LISTS (paddings), TFS, OK, AZ, STRAT, BROAD, window, rule_abc, X
s40 = open(f"{R}/m40_overlap_reference/m40_perm.py").read(); a = s40.index("D33 = "); exec(s40[a:s40.index("T25 = pd.read_csv", a)])  # CH, BULK (A, G = A + 10 kb)
O = f"{M}/out"; rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1)
ti = {t: i for i, t in enumerate(TFS)}; EX3 = {"NRF1", "E4F1", "DNMT1"}
def per_tf(blist, tfs, c):
    obs, den, dr = np.zeros(len(tfs)), np.zeros(len(tfs)), np.zeros((len(tfs), NP))
    for k, t in enumerate(tfs):
        i = ti[t]; s_ = STRAT[OK[i]]; b = blist[t][OK[i]]; p_ = (AZ[i] >= c)[OK[i]]; obs[k] = (b & p_).sum(); den[k] = p_.sum()
        for st in np.unique(s_):
            m = s_ == st; n = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
            if kk and nb: dr[k] += rng.hypergeometric(nb, n - nb, kk, size=NP)
    return obs, den, dr
def row(obs, den, dr, sel):
    o = 100 * obs[sel].sum() / den[sel].sum(); dd = 100 * dr[sel].sum(0) / den[sel].sum()
    return {"TFs": int(sel.sum()), "Perturb-seq genes": int(den[sel].sum()), "in both": int(obs[sel].sum()), "recall observed (%)": o, "shuffle mean (%)": dd.mean(), "observed minus shuffle mean (points)": o - dd.mean(),
            "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}", "p (one-sided)": pval(o, dd)}
SETS = {"all 63": set(), "without NRF1, E4F1, DNMT1": EX3, "without broad TFs": BROAD, "without both": EX3 | BROAD}
A1, LOO, LOOS, BK = [], [], [], []
for pad in PADS:
    for c in (2, 3, 4, 5):
        obs, den, dr = per_tf(LISTS[pad], TFS, c)
        for sn, ex_ in SETS.items():
            sel = np.array([t not in ex_ for t in TFS]) & (den > 0); A1.append({"padding": pad, "cutoff": f"|z| >= {c}", "set": sn, **row(obs, den, dr, sel)})
        if pad == "10 kb" and c in (3, 4):
            for k, t in enumerate(TFS):
                sel = (np.arange(NT) != k) & (den > 0); LOO.append({"padding": pad, "cutoff": f"|z| >= {c}", "left out": t, **row(obs, den, dr, sel)})
        if pad == "10 kb":
            bt = [t for t, _ in CH if t in ti]; bo, bd, bdr = per_tf(BULK["G"], bt, c); oo, od, odr = per_tf(LISTS["10 kb"], bt, c)
            for k, t in enumerate(bt):
                if bd[k]: BK.append({"padding": pad, "cutoff": f"|z| >= {c}", "list": t, "bulk ChIP": True, **row(bo, bd, bdr, np.arange(len(bt)) == k)}); BK.append({"padding": pad, "cutoff": f"|z| >= {c}", "list": t, "bulk ChIP": False, **row(oo, od, odr, np.arange(len(bt)) == k)})
            for nm, (x, y, z) in (("bulk ChIP, 10 TFs pooled", (bo, bd, bdr)), ("our HITAG, same 10 TFs pooled", (oo, od, odr))):
                BK.append({"padding": pad, "cutoff": f"|z| >= {c}", "list": nm, "bulk ChIP": nm.startswith("bulk"), **row(x, y, z, y > 0)})
    print("padding", pad, flush=True)
L = pd.DataFrame(LOO)
for c, g in L.groupby("cutoff"):
    full = [r for r in A1 if r["padding"] == "10 kb" and r["cutoff"] == c and r["set"] == "all 63"][0]; w = g.loc[g["observed minus shuffle mean (points)"].idxmin()]
    LOOS.append({"padding": "10 kb", "cutoff": c, "all 63, gap (points)": full["observed minus shuffle mean (points)"], "all 63, p": full["p (one-sided)"], "leave-one-out gap min": g["observed minus shuffle mean (points)"].min(), "leave-one-out gap max": g["observed minus shuffle mean (points)"].max(),
                 "leave-one-out p min": g["p (one-sided)"].min(), "leave-one-out p max": g["p (one-sided)"].max(), "TF whose removal lowers the gap most": w["left out"], "gap without it": w["observed minus shuffle mean (points)"], "p without it": w["p (one-sided)"],
                 "next 3 TFs": ", ".join(g.sort_values("observed minus shuffle mean (points)")["left out"].iloc[1:4])})
TT = {"6_shuffle_by_tf_set": pd.DataFrame(A1), "7_leave_one_tf_out_10kb": L, "7_leave_one_tf_out_summary": pd.DataFrame(LOOS), "8_bulk_chip_vs_ours_10kb": pd.DataFrame(BK)}
for n, d in TT.items(): d.to_csv(f"{O}/{n}.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_shuffle_sets.xlsx") as w:
    for n, d in TT.items(): d.to_excel(w, sheet_name=n[:31], index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40)
for n in ("6_shuffle_by_tf_set", "7_leave_one_tf_out_summary", "8_bulk_chip_vs_ours_10kb"): print("==", n); print(TT[n].round(3).to_string(index=False))
print("done")
