# PRC2-adjacent pooling control (Gate 1 v3, binned units). For the 8-TF pool and the pool minus EZH2/MTF2 (6 TFs), against
# 20 random pools each (members swapped for non-member TFs of the same library group within +/-4 cell-count ranks; total cells
# within +/-20% of the template):
#   reproducibility: excess = obs - pooled-panel null mean; rank of the real pool among the 21; how many random pools pass
#     reproducibility (obs > null p90 and BH q <= 0.10 within this family);
#   standard specificity (as in the module summary): margin vs non-member TFs + PRC2 core + NuRD core;
#   depth-matched specificity: margin vs the 20 random pools of matched total cells (own split-half r minus the best
#     cross-pool r, Pearson on residuals of shared eligible units); the same computed for every random pool vs the others.
import glob, json, os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; V3 = f"{X}/regulatome/gate1/v3_binned"; EMIN = 5.0
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r
def resid(OE, k, h):
    o, e = (OE[k, 0], OE[k, 1]) if h == "A" else (OE[k, 2], OE[k, 3]); return np.log2((o + .5) / (e + .5)), e >= EMIN
def pear(x, y, m):
    if m.sum() < 50: return np.nan
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 0 else np.nan
def margin(P, u, comps):
    ms = []
    for k in range(P[u].shape[0]):
        rA, eA = resid(P[u], k, "A"); rB, eB = resid(P[u], k, "B"); base = eA & eB; best = None
        for v in comps:
            if v == u or k >= P[v].shape[0]: continue
            rv, ev = resid(P[v], k, "B"); g = base & ev
            o_, x_ = pear(rA, rB, g), pear(rA, rv, g)
            if np.isfinite(o_) and np.isfinite(x_): best = o_ - x_ if best is None else min(best, o_ - x_)
        if best is not None: ms.append(best)
    return float(np.mean(ms)) if ms else np.nan, int(np.sum(np.array(ms) > 0)), len(ms)
rows = []
for ut in ["bin10k", "bin50k", "reprpc", "genebody"]:
    for real, pref in (("prc2adj", "randadj8_"), ("prc2adj_noEZH2MTF2", "randadj6_")):
        units = [real] + [f"{pref}{k:02d}" for k in range(1, 21)]
        R = pd.concat([pd.read_csv(f"{V3}/results_modules/{ut}.{u}.tsv", sep="\t") for u in units if os.path.exists(f"{V3}/results_modules/{ut}.{u}.tsv")])
        R["q"] = bh(R.p.values); R["reproducible"] = (R.obs > R.null_p90) & (R.q <= 0.10); R["excess"] = R.obs - R.null_mean
        P = {u: np.load(f"{V3}/results_modules/{ut}.{u}.profiles.npz")["OE"] for u in R.unit}
        for f in glob.glob(f"{V3}/results/{ut}.*.profiles.npz"):
            P[f.split("/")[-1][len(ut) + 1:-len(".profiles.npz")]] = np.load(f)["OE"]
        for c in ("PRC2core", "NuRDcore"): P[c] = np.load(f"{V3}/results_modules/{ut}.{c}.profiles.npz")["OE"]
        for r in R.itertuples():
            mem = set(r.members.split(","))
            std = margin(P, r.unit, [t for t in tfs if t in P and t not in mem] + ["PRC2core", "NuRDcore"])
            dm = margin(P, r.unit, [u for u in R.unit if u != r.unit])
            rows.append(dict(unit_type=ut, family=real, unit=r.unit, n_cells=r.n_cells, obs=r.obs, null_p90=r.null_p90, excess=r.excess,
                             q=r.q, reproducible=r.reproducible, std_margin=std[0], std_pos=f"{std[1]}/{std[2]}",
                             std_specific=std[0] > 0 and std[1] >= np.ceil(0.9 * std[2]),
                             dm_margin=dm[0], dm_pos=f"{dm[1]}/{dm[2]}", dm_specific=dm[0] > 0 and dm[1] >= np.ceil(0.9 * dm[2])))
        print(ut, real, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{V3}/prc2adj_pooling_control.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
for (ut, fam), d in D.groupby(["unit_type", "family"], sort=False):
    real = d[d.unit == fam].iloc[0]; rnd = d[d.unit != fam]
    print(f"{ut:9s} {fam:20s} excess {real.excess:+.3f} rank {int((rnd.excess >= real.excess).sum()) + 1}/21 "
          f"(random median {rnd.excess.median():+.3f}, max {rnd.excess.max():+.3f}); random reproducible {int(rnd.reproducible.sum())}/20, "
          f"random pass std test {int((rnd.reproducible & rnd.std_specific).sum())}/20; real std margin {real.std_margin:+.3f} ({real.std_pos}); "
          f"depth-matched margin real {real.dm_margin:+.3f} ({real.dm_pos}), rank {int((rnd.dm_margin >= real.dm_margin).sum()) + 1}/21, "
          f"random dm-specific {int(rnd.dm_specific.sum())}/20")
