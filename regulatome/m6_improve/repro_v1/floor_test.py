# repro_v1 per-TF floor test (README Amendments 7 and 7b, declared before the read). Observed v8@1000 d vs each TF's own 100 perm and
# 100 count-matched noise-floor draws. Writes per_tf_floor_test.tsv and prints the report numbers. DESCRIPTIVE. usage: floor_test.py
import numpy as np, pandas as pd
from scipy import stats
O = "/work/users/d/y/dyy12/XuLab/regulatome/m6_improve/repro_v1"
R10 = {"RBBP4", "NFYA", "SOX4", "XPA", "ZNF507", "RBBP7", "NME2", "E2F3", "MBD3", "DDIT3"}
F = pd.read_csv(f"{O}/noisefloor_per_draw.tsv", sep="\t"); V = pd.read_csv(f"{O}/run2c_per_tf.tsv", sep="\t"); V = V[V.method == "v8@1000"].set_index("tf")
assert F.groupby(["kind", "tf"]).size().eq(100).all() and len(V) == 79
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out
rows = []
for t, r in V.iterrows():
    x = dict(tf=t, cells=int(r.cells), tertile=r.depth_tertile, Jw=r.J_within, Jb=r.J_between, d=r["diff"])
    for k, ab in (("perm", "perm"), ("countmatched", "cm")):
        v = F[(F.kind == k) & (F.tf == t)]["diff"].values; sd = v.std(ddof=1)
        x[f"{ab}_mean"] = v.mean(); x[f"{ab}_q95"] = np.quantile(v, .95); x[f"p_{ab}"] = (1 + (v >= x["d"]).sum()) / 101
        x[f"z_{ab}"] = (x["d"] - v.mean()) / sd if sd > 0 else np.nan; x[f"pz_{ab}"] = stats.norm.sf(x[f"z_{ab}"]) if sd > 0 else np.nan
        x[f"shapiro_{ab}"] = stats.shapiro(v).pvalue; x[f"sd_{ab}"] = sd
    rows.append(x)
T = pd.DataFrame(rows)
def finish(T):
    T = T.copy(); T["q_perm"] = bh(T.p_perm); T["q_cm"] = bh(T.p_cm); T["clears"] = (T.p_perm <= .05) & (T.p_cm <= .05)
    T["sd_flag"] = (T.sd_perm < 1e-4) | (T.sd_cm < 1e-4); T["qz_perm"] = bh(T.pz_perm.fillna(1)); T["qz_cm"] = bh(T.pz_cm.fillna(1))
    T["called_fdr05"] = (T.qz_perm <= .05) & (T.qz_cm <= .05) & ~T.sd_flag; T["in_reproducible10"] = T.tf.isin(R10); return T
T = finish(T)
cols = ["tf", "cells", "tertile", "Jw", "Jb", "d", "perm_mean", "perm_q95", "p_perm", "cm_mean", "cm_q95", "p_cm", "q_perm", "q_cm", "clears", "in_reproducible10",
        "z_perm", "pz_perm", "qz_perm", "z_cm", "pz_cm", "qz_cm", "shapiro_perm", "shapiro_cm", "sd_flag", "called_fdr05"]
T.sort_values("d", ascending=False)[cols].to_csv(f"{O}/per_tf_floor_test.tsv", sep="\t", index=False, float_format="%.5g")
def report(T, lab):
    print(f"== {lab}: n={len(T)}")
    for c in ("clears", "called_fdr05"):
        s = T[T[c]].sort_values("d", ascending=False); tab = pd.crosstab(T[c], T.in_reproducible10).reindex(index=[True, False], columns=[True, False], fill_value=0)
        print(f"{c}: {len(s)}; by tertile {s.tertile.value_counts().to_dict()}; Fisher p {stats.fisher_exact(tab.values)[1]:.3g}\n{tab.to_string()}")
        print("  ", ", ".join(f"{a}({b:.3f})" for a, b in zip(s.tf, s.d)))
    print("exact-BH both q<=.05:", int(((T.q_perm <= .05) & (T.q_cm <= .05)).sum()), "| min q_perm", T.q_perm.min().round(4), "min q_cm", T.q_cm.min().round(4))
    print("shapiro p<.01 share perm", round((T.shapiro_perm < .01).mean(), 3), "cm", round((T.shapiro_cm < .01).mean(), 3), "| sd_flag:", list(T[T.sd_flag].tf))
report(T, "all 79"); report(finish(T[T.tf != "NRF1"]), "without NRF1")
