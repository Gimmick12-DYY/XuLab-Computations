# m57 Amendment 8: proximal link sets under the standard and the gene-density-matched shuffle.
import numpy as np, pandas as pd
SRC = open("/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/m57_abcdist.py").read(); exec(SRC[:SRC.index("M_ = lambda M: M & OK")])  # PROM, DIS, WIN{}, S_prom, S_dall, S_win, PA, PS, STRAT, ex, at, TB
O = "/work/users/d/y/dyy12/XuLab/regulatome/m57_stage1_q_cutoff/out"
CUR = (PROM | DIS | WIN[10]) & OK; assert int(CUR.sum()) == 212087
S_cur = np.maximum.reduce([S_prom, S_dall, S_win[10]])
def topn(n):
    K = np.zeros_like(CUR)
    for i in range(NT): g = np.flatnonzero(CUR[i]); K[i, g[np.argsort(-S_cur[i, g], kind="stable")][:n]] = True
    return K
SETS = [("current rule", CUR), ("distance within 10 kb", WIN[10] & OK), ("own-promoter class", PROM & OK), ("10 kb window-only class", WIN[10] & ~PROM & ~DIS & OK), ("top 100 strongest genes per TF (current rule)", topn(100)), ("top 500 strongest genes per TF (current rule)", topn(500)), ("ABC only (0 kb rule)", (PROM | DIS) & OK)]
print({n: int(M.sum()) for n, M in SETS}, flush=True)
TG = {}
for x in (5, 1): TG[(x, "either")] = OK & (PA > 100 - x); TG[(x, "up")] = OK & (PS > 100 - x); TG[(x, "down")] = OK & (PS < x)
q5 = lambda v: pd.qcut(pd.Series(v).rank(method="first"), 5, labels=False).values
nn_d = np.full(NGn, 1e9)
for ch, (tp, ix) in TB.items():
    if len(tp) < 2: continue
    d = np.full(len(tp), np.inf); d[1:] = np.minimum(d[1:], np.diff(tp)); d[:-1] = np.minimum(d[:-1], np.diff(tp)); nn_d[ix] = d
STRAT3 = q5(ex) * 25 + q5(at) * 5 + q5(nn_d); print("density strata", len(np.unique(STRAT3)), "smallest", int(pd.Series(STRAT3).value_counts().min()), flush=True)
NP = 1000
def run(strat, tag):
    rng = np.random.default_rng(20261118); out = {}; loo = []
    for nm, B in SETS:
        links = B.sum(1)
        for (x, d), T in TG.items():
            ib = (B & T).sum(1); dr = np.zeros((NT, NP))
            for i in range(NT):
                s_ = strat[OK[i]]; b = B[i][OK[i]]; p_ = T[i][OK[i]]
                for st in np.unique(s_[p_]):
                    m = s_ == st; nn = int(m.sum()); nb = int(b[m].sum()); kk = int(p_[m].sum())
                    if nb: dr[i] += rng.hypergeometric(nb, nn - nb, kk, size=NP)
            o = ib.sum(); dd = dr.sum(0); L_ = links.sum(); pu = (1 + (dd >= o).sum()) / (NP + 1); pl = (1 + (dd <= o).sum()) / (NP + 1)
            out[(nm, x, d)] = {"links": int(L_), "in both": int(o), "precision (%)": 100 * o / L_, f"{tag}: shuffle mean precision (%)": 100 * dd.mean() / L_, f"{tag}: 2.5-97.5 (%)": f"{100 * np.percentile(dd, 2.5) / L_:.2f}-{100 * np.percentile(dd, 97.5) / L_:.2f}",
                               f"{tag}: obs / shuffle": o / dd.mean(), f"{tag}: p upper": pu, f"{tag}: p lower": pl}
            if nm == "current rule" and x == 5 and d in ("either", "up"):
                for k in range(NT): sel = np.arange(NT) != k; loo.append({"direction": d, "left out": TFS[k], "obs / shuffle": ib[sel].sum() / dr[sel].sum(0).mean()})
        print(tag, nm, flush=True)
    return out, pd.DataFrame(loo)
A_, _ = run(STRAT, "standard (expr x ATAC deciles)"); B_, LOO = run(STRAT3, "density (expr x ATAC x TSS-density quintiles)")
rows = [{"link set": k[0], "targets": f"top {k[1]}%", "direction": k[2], **A_[k], **{c: v for c, v in B_[k].items() if c not in ("links", "in both", "precision (%)")}} for k in A_]
T = pd.DataFrame(rows); LS = LOO.groupby("direction").agg(**{"obs / shuffle min": ("obs / shuffle", "min"), "max": ("obs / shuffle", "max")}).reset_index()
LS["TF left out at min"] = [LOO[LOO.direction == d].sort_values("obs / shuffle").iloc[0]["left out"] for d in LS.direction]
with pd.ExcelWriter(f"{O}/stage1_proximal_density_shuffle.xlsx") as w:
    T.to_excel(w, sheet_name="two shuffles", index=False); LS.to_excel(w, sheet_name="leave one TF out (density)", index=False); LOO.to_excel(w, sheet_name="LOO per TF", index=False)
    pd.DataFrame({"note": ["Two matched gene shuffles in one job, each 1,000 shuffles, own random stream (seed 20261118): standard = NTC expression decile x promoter ATAC decile (100 strata); density = expression quintile x ATAC quintile x quintile of the distance from the gene's TSS to the nearest other universe TSS (125 strata). Targets redrawn per TF within strata (exact hypergeometric); precision shuffle = shuffle in-both / links.",
      "Link sets: top 10,000 lab peaks, 63 TFs; current rule = promoter + ABC + 10 kb; own-promoter class = summit in the gene's ABC self-promoter element; window-only = within 10 kb, no promoter / distal ABC link; top N = genes ranked by the strongest linking peak signalValue under the current rule; ABC only = promoter + all ABC elements, no window."]}).to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 340); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 45)
print(T.round(3).to_string(index=False)); print(LS.round(3).to_string(index=False)); print("done")
