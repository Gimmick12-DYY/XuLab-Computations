#!/usr/bin/env python3
"""Fig 3 v7 accessibility control for panel b: is the promoter share of regions in modules of 1/2/3-4/5-9/>=10 TFs above what open,
multi-TF (HOT-like) regions give? Each module region (inputs/hub_regions.tsv: regions merged <=500 bp, n_tfs = TFs whose modules
contain it) is matched to universe regions (build_universe.py) with the same ATAC decile (0-10) and the same number of TFs with a
Cicero peak there (bins 1, 2, 3-4, 5-9, 10-19, >=20). Expected promoter share = matched draws (1,000 draws of one universe region per
module region, same stratum); promoter = RefSeq Select TSS within +-1 kb, the same rule for module and universe regions.
Writes panel_b_hot_check.tsv and panel_b_hot_check.png."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; I = HERE / "inputs"
U = pd.read_csv(I / "universe.tsv.gz", sep="\t"); H = pd.read_csv(I / "hub_regions.tsv", sep="\t", usecols=["hub_region", "n_tfs"])
H["c"] = H.hub_region.str.split(":").str[0]; se = H.hub_region.str.split(":").str[1].str.split("-"); H["s"] = se.str[0].astype(int); H["e"] = se.str[1].astype(int)
TSS = {}
for f in glob.glob(str(I / "rs/*.json")):
    v = json.load(open(f))["ncbiRefSeqSelect"]
    if v: TSS[Path(f).stem] = np.sort([(g["txStart"] if g["strand"] == "+" else g["txEnd"]) for g in v])
def prom(c, s, e):
    t = TSS.get(c); 
    if t is None: return 0
    j = np.searchsorted(t, s - 1000); return int(j < len(t) and t[j] <= e + 1000)
H["prom"] = [prom(c, s, e) for c, s, e in zip(H.c, H.s, H.e)]
# map each module region to the universe regions it overlaps: ATAC decile = max, TF-peak count = max
H["atac_dec"] = 0; H["n_tf"] = 0
for c, g in H.groupby("c"):
    u = U[U.c == c].sort_values("s"); us, ue = u.s.values, u.e.values
    for i, s, e in zip(g.index, g.s.values, g.e.values):
        lo = np.searchsorted(ue, s); hi = np.searchsorted(us, e, side="right")
        if hi > lo: H.loc[i, "atac_dec"] = u.atac_dec.values[lo:hi].max(); H.loc[i, "n_tf"] = u.n_tf.values[lo:hi].max()
cut = lambda n: pd.cut(n, [0, 1, 2, 4, 9, 19, 1000], labels=["1", "2", "3-4", "5-9", "10-19", ">=20"])
H["tfbin"] = cut(H.n_tf.clip(lower=1)); U["tfbin"] = cut(U.n_tf.clip(lower=1))
U["st"] = U.atac_dec.astype(str) + "_" + U.tfbin.astype(str); H["st"] = H.atac_dec.astype(str) + "_" + H.tfbin.astype(str)
pr = U.groupby("st").prom.agg(["mean", "size"]); H["exp"] = H.st.map(pr["mean"])
H["bin"] = pd.cut(H.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1", "2", "3-4", "5-9", ">=10"])
rng = np.random.default_rng(7); idx = {k: v.values for k, v in U.groupby("st").prom}
rows = []
for b, g in H.groupby("bin", observed=True):
    sims = np.array([[idx[st][rng.integers(0, len(idx[st]))] for st in g.st.values] for _ in range(1000)]).mean(1)
    rows.append(dict(bin=b, n=len(g), obs_prom_pct=100 * g.prom.mean(), exp_prom_pct=100 * g.exp.mean(), null_lo=100 * np.percentile(sims, 2.5),
                     null_hi=100 * np.percentile(sims, 97.5), ratio=g.prom.mean() / g.exp.mean(), p_emp=(1 + (sims >= g.prom.mean()).sum()) / 1001,
                     in_atac_pct=100 * (g.atac_dec > 0).mean(), median_tf_peaks=float(g.n_tf.median())))
R = pd.DataFrame(rows); R.round(3).to_csv(HERE / "panel_b_hot_check.tsv", sep="\t", index=False); print(R.round(2).to_string(index=False))
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures"); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(3.2, 2.2)); x = np.arange(len(R))
ax.bar(x - 0.18, R.obs_prom_pct, 0.36, color="#B03A2E", label="Observed"); ax.bar(x + 0.18, R.exp_prom_pct, 0.36, color="#BDBDBD", label="Matched open, same no. of TF peaks")
ax.errorbar(x + 0.18, R.exp_prom_pct, yerr=[R.exp_prom_pct - R.null_lo, R.null_hi - R.exp_prom_pct], fmt="none", ecolor="#555555", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(R.bin, fontsize=6); ax.set_xlabel("Regions in modules of N TFs", fontsize=7); ax.set_ylabel("Promoter (%)", fontsize=7)
ax.tick_params(labelsize=6); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=5.5, frameon=False); fig.tight_layout(); fig.savefig(HERE / "panel_b_hot_check.png", dpi=300)
