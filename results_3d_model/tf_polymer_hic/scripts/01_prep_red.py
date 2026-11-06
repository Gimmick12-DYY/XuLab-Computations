"""Hi-C-first arms: restraint sets for the chr16 window.
  loops_only : the 165 Peakachu 0.7 loop restraints of the hybrid (loop_restraints.tsv), unchanged
  red_only / red_tf : the 18 "red" hub edges of the browser figure (shared clique edges >25 kb on a 0.7 loop,
               hub_chr16_edges.tsv on_loop_0.7 == 1); anchor bead = bead of each element's midpoint; p_on from the ICE O/E
               of the edge's 10 kb pixel, as the percentile among the 165 loops' O/E mapped to 0.2-0.8 (same scale as loops)
Writes <out>/<arm>/inputs/ (v2 beads.tsv etc. + loop_restraints.tsv).
"""
import argparse, shutil
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--hyb", required=True, help="tf_polymer_hic folder (inputs/)")
ap.add_argument("--hub-edges", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
I = Path(a.hyb) / "inputs"; L0 = 29730000
LR = pd.read_csv(I / "loop_restraints.tsv", sep="\t")
H = np.load(I / "hic_10kb.npz"); ice, keep = H["ice"], H["keep"]; n = ice.shape[0]
exp = np.array([np.nanmean(ice[np.arange(n - d), np.arange(n - d) + d][keep[:n - d] & keep[d:]]) if (keep[:n - d] & keep[d:]).any() else np.nan for d in range(n)])
E = pd.read_csv(a.hub_edges, sep="\t"); E = E[E["on_loop_0.7"] == 1].reset_index(drop=True)
def mid(r): s, e = map(int, r.split(":")[1].split("-")); return (s + e) // 2
rows = []
for r in E.itertuples():
    c1, c2 = sorted((mid(r.region1), mid(r.region2))); i, j = (c1 - L0) // 10000, (c2 - L0) // 10000
    oe = ice[i, j] / exp[j - i] if keep[i] and keep[j] and np.isfinite(ice[i, j]) else np.nan
    rows.append(dict(a1=r.region1, a2=r.region2, c1=c1, c2=c2, b1=(c1 - L0) // 2000, b2=(c2 - L0) // 2000, bin1=i, bin2=j, oe=oe,
                     n_tfs=r.n_tfs, tfs=r.tfs))
R = pd.DataFrame(rows); R["oe"] = R.oe.fillna(LR.oe.median())
R["p_on"] = 0.2 + 0.6 * np.array([(LR.oe < v).mean() for v in R.oe])
for arm, tab in (("loops_only", LR), ("red_only", R), ("red_tf", R)):
    d = Path(a.out) / arm / "inputs"; d.mkdir(parents=True, exist_ok=True)
    for f in ["beads.tsv", "tf_species.tsv", "hic_10kb.npz", "pairs_cicero.tsv", "pairs_loops.tsv"]: shutil.copy(I / f, d / f)
    tab.to_csv(d / "loop_restraints.tsv", sep="\t", index=False)
print(f"red edges: {len(R)}; span {(R.b2 - R.b1).min() * 2}-{(R.b2 - R.b1).max() * 2} kb; O/E median {R.oe.median():.2f}; p_on {R.p_on.round(2).tolist()}")
