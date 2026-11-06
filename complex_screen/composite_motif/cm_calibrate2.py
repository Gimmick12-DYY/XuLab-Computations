# Calibration follow-up: stricter FIMO cutoffs, and the with/without motif comparison stratified on expected reads (20 mu bins), so elements are compared only with others
# of similar accessibility, width and GC. Stratified shift = with-motif obs/exp over without-motif obs/exp, bootstrap CI over elements.
import numpy as np, pandas as pd
from cm_common import *
E = load_elements(); Z = np.load(f"{O}/calibration/ctcf_residual.npz"); y, mu = Z["y"], Z["mu"]
F = pd.read_csv(f"{O}/calibration/fimo.tsv", sep="\t", comment="#"); F = F[F["p-value"].notna()]
bins = np.minimum((pd.Series(mu).rank(pct=True).values * 20).astype(int), 19); rows = []
def strat(h, yv, muv, b):
    o1 = e1 = o0 = e0 = 0.0
    for k in range(20):
        s = b == k; a = s & h; c = s & ~h
        if a.sum() < 5 or c.sum() < 5: continue
        w = min(a.sum(), c.sum()); o1 += yv[a].mean() * w; e1 += muv[a].mean() * w; o0 += yv[c].mean() * w; e0 += muv[c].mean() * w
    return (o1 / e1) / (o0 / e0)
r = np.random.default_rng(2)
for pth in (1e-4, 1e-5, 1e-6, 1e-7):
    for m in ["MA0139.2", "shuffle1", "shuffle2", "shuffle3"]:
        h = E.eid.isin(F.loc[(F.motif_id == m) & (F["p-value"] <= pth), "sequence_name"]).values
        if h.sum() < 50: rows.append(dict(p=pth, motif=m, elements_with=int(h.sum()))); continue
        s = strat(h, y, mu, bins); bs = []
        for _ in range(500):
            i = r.integers(0, len(y), len(y)); bs.append(strat(h[i], y[i], mu[i], bins[i]))
        rows.append(dict(p=pth, motif=m, elements_with=int(h.sum()), frac=round(h.mean(), 4), reads_with=int(y[h].sum()), raw_obs_exp_with=round(y[h].sum() / mu[h].sum(), 3), raw_obs_exp_without=round(y[~h].sum() / mu[~h].sum(), 3),
                         stratified_ratio=round(s, 3), lo=round(np.percentile(bs, 2.5), 3), hi=round(np.percentile(bs, 97.5), 3)))
T = pd.DataFrame(rows); T.to_csv(f"{O}/calibration/ctcf_calibration_strict.tsv", sep="\t", index=False); pd.set_option("display.width", 200); print(T.to_string(index=False))
