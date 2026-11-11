#!/usr/bin/env python3
"""Fig 3 v7 accessibility control, step 3: per TF and variant (uniform per-edge shuffle from fig3_v6/checks, atac, atac_density):
observed % of module edges >25 kb on a 0.7 loop, null median and 95% interval, O/E = observed / null median, empirical p (+1),
BH q across the 40 TFs, median spacing error of the matched draws. Writes matched_null_summary.tsv and matched_null_summary.png."""
import glob
from pathlib import Path
import numpy as np, pandas as pd
def bh(p):   # Benjamini-Hochberg q values
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; V6 = Path("/work/users/d/y/dyy12/XuLab/figures/fig3_v6/checks")
TFS = [t.strip() for t in open(V6 / "tfs40.txt") if t.strip()]; rows = []
for var, pat in (("uniform", str(V6 / "null_per_edge/{t}_*.npz")), ("atac", str(HERE / "null/atac_{t}_*.npz")), ("atac_density", str(HERE / "null/atac_density_{t}_*.npz"))):
    for t in TFS:
        fs = sorted(glob.glob(pat.format(t=t)))
        if not fs: continue
        z = [np.load(f) for f in fs]; nu = np.concatenate([q["null"] for q in z]); o = float(z[0]["obs"])
        er = np.concatenate([q["spacing_err"] for q in z]) if "spacing_err" in z[0].files else np.array([0.0])
        rows.append(dict(variant=var, tf=t, n_perm=len(nu), obs=100 * o, null_med=100 * np.median(nu), null_lo=100 * np.percentile(nu, 2.5),
                         null_hi=100 * np.percentile(nu, 97.5), oe=o / np.median(nu), p=(1 + (nu >= o).sum()) / (1 + len(nu)), spacing_err=float(np.median(er))))
R = pd.DataFrame(rows); R["q"] = R.groupby("variant").p.transform(lambda p: bh(p.values))
R.round(5).to_csv(HERE / "matched_null_summary.tsv", sep="\t", index=False)
for v, g in R.groupby("variant", sort=False):
    print(f"{v:13s} TFs {len(g)}  n_perm min {g.n_perm.min()}  pass q<0.05: {(g.q < 0.05).sum()}  median obs {g.obs.median():.1f}%  median null {g.null_med.median():.1f}%  median O/E {g.oe.median():.2f}  spacing err {g.spacing_err.median():.3f}")
low = ["SMARCA1", "FLYWCH1", "HOXA10", "ZNF512", "DDIT3"]
print(R[R.tf.isin(low)].pivot(index="tf", columns="variant", values="q").round(4).to_string())
fails = {v: sorted(g[g.q >= 0.05].tf) for v, g in R.groupby("variant")}; print("n.s.:", fails)
P = R.pivot(index="tf", columns="variant", values="oe"); order = P["uniform"].sort_values().index
fig, ax = plt.subplots(figsize=(3.4, 5.2)); y = np.arange(len(order)); C = dict(uniform="#1A1A1A", atac="#7A9CCB", atac_density="#B83A2E")
for v, dy in (("uniform", -0.2), ("atac", 0), ("atac_density", 0.2)):
    if v in P: ax.scatter(P.loc[order, v], y + dy, s=6, color=C[v], label={"uniform": "Uniform shuffle", "atac": "ATAC + promoter matched", "atac_density": "+ peak density matched"}[v], lw=0)
ax.axvline(1, color="#9E9E9E", lw=0.6, ls="--"); ax.set_yticks(y); ax.set_yticklabels(order, fontsize=4.5); ax.set_xlabel("Observed / null median", fontsize=7)
ax.tick_params(axis="x", labelsize=6); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=5.5, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(HERE / "matched_null_summary.png", dpi=300)
