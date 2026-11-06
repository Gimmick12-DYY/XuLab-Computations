"""Hybrid step 1: v2 TF-bridging inputs (unchanged) + Peakachu WT 0.7 loops as stochastic soft restraints.

Loops: union of peakachu_5000 and peakachu_10000 .loops.0.7.bedpe with both anchors inside the window; anchor bead =
bead holding the anchor centre; duplicates (both anchor beads within +-5 beads = 10 kb of an already kept loop, kept in
order of Peakachu score) are dropped; loops with anchors < 3 beads apart are dropped.
Strength: ICE O/E of the loop's 10 kb pixel (observed / mean of its diagonal over valid pairs); p_on = 0.2 + 0.6 * rank
percentile of O/E among the kept loops (0.2 weakest -> 0.8 strongest).
Writes inputs/ (v2 beads.tsv, tf_species.tsv, hic_10kb.npz, pairs_*.tsv copied) + loop_restraints.tsv.
"""
import argparse, shutil
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--v2", required=True)
ap.add_argument("--loops", nargs="+", required=True)
ap.add_argument("--region", default="chr16:29730000-31550000")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for f in ["beads.tsv", "tf_species.tsv", "hic_10kb.npz", "pairs_cicero.tsv", "pairs_loops.tsv"]:
    shutil.copy(Path(a.v2) / "inputs" / f, out / f)
chrom, se = a.region.split(":"); L0, L1 = map(int, se.split("-")); nb = (L1 - L0) // 2000
H = np.load(out / "hic_10kb.npz"); ice, keep = H["ice"], H["keep"]; nbin = ice.shape[0]
exp = np.full(nbin, np.nan)
for d in range(nbin):
    i = np.arange(nbin - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(ice[i, j])
    if ok.any(): exp[d] = ice[i[ok], j[ok]].mean()
rows = []
for f in a.loops:
    for ln in open(f):
        x = ln.split()
        if x[0] != chrom or x[3] != chrom: continue
        s1, e1, s2, e2 = map(int, (x[1], x[2], x[4], x[5]))
        if min(s1, s2) < L0 or max(e1, e2) > L1: continue
        c1, c2 = sorted(((s1 + e1) // 2, (s2 + e2) // 2))
        rows.append(dict(src=Path(f).name, a1=f"{chrom}:{s1}-{e1}", a2=f"{chrom}:{s2}-{e2}", c1=c1, c2=c2,
                         b1=(c1 - L0) // 2000, b2=(c2 - L0) // 2000, score=float(x[6]) if len(x) > 6 else np.nan))
Lp = pd.DataFrame(rows).sort_values("score", ascending=False)
kept = []
for r in Lp.itertuples():
    if r.b2 - r.b1 < 3: continue
    if any(abs(r.b1 - k.b1) <= 5 and abs(r.b2 - k.b2) <= 5 for k in kept): continue
    kept.append(r)
K = pd.DataFrame(kept).drop(columns="Index")
i, j = (K.c1 - L0) // 10000, (K.c2 - L0) // 10000
K["oe"] = [ice[p, q] / exp[q - p] if keep[p] and keep[q] and np.isfinite(ice[p, q]) else np.nan for p, q in zip(i, j)]
K["oe"] = K.oe.fillna(K.oe.median())
K["p_on"] = 0.2 + 0.6 * K.oe.rank(pct=True, method="average").sub(1 / len(K)).div(1 - 1 / len(K))
K["bin1"], K["bin2"] = i.values, j.values
K.to_csv(out / "loop_restraints.tsv", sep="\t", index=False)
print(f"loops in window: {len(Lp)} calls (5 kb + 10 kb) -> {len(K)} restraints after de-duplication; "
      f"span {(K.b2 - K.b1).min() * 2}-{(K.b2 - K.b1).max() * 2} kb (median {(K.b2 - K.b1).median() * 2:.0f}); "
      f"O/E median {K.oe.median():.2f}; mean p_on {K.p_on.mean():.2f}")
