"""Hi-C-first model, step 1: restraint set from WT Hi-C for chr16:29.73-31.55 Mb (2 kb beads).

Hi-C-restrained polymer in the spirit of maximum-entropy / data-driven restraint models (Zhang & Wolynes 2015 PNAS;
Di Pierro et al. 2016 PNAS): enriched Hi-C contacts become soft attractive restraints whose ON probability scales with
the contact enrichment, so the ensemble mixes rather than closing every contact at once.
  pixel restraints : every 10 kb ICE pixel (both bins kept, 2-100 bins apart) with O/E >= --oe-min; 20% of them are
                     withheld at random (seed 7) for validation and NOT restrained; restrained bond between the central
                     beads of the two bins; p_on = 0.1 + 0.5 * O/E rank percentile; flat-bottom harmonic, r0 = 1.5 sigma, k = 3
  loop restraints  : the 165 Peakachu 0.7 loops of the hybrid run (p_on 0.2-0.8 by O/E rank), r0 = 1, k = 10
  red edges        : the 18 hub shared clique edges on a 0.7 loop (browser figure), ALWAYS on, r0 = 1, k = 10
Pixels that coincide with (+-1 bin) a loop or red edge are not duplicated as pixel restraints. Held-out pixels never
coincide with a loop/red edge (+-1 bin).
Writes inputs/ (v2 beads etc.), restraints.tsv (kind, b1, b2, bin1, bin2, oe, p_on, k, r0) and heldout_pixels.tsv.
"""
import argparse, shutil
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--hyb", required=True, help="tf_polymer_hic (inputs/loop_restraints.tsv, v2 inputs)")
ap.add_argument("--hub-edges", required=True)
ap.add_argument("--oe-min", type=float, default=2.0)
ap.add_argument("--max-sep", type=int, default=100)
ap.add_argument("--holdout", type=float, default=0.2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
I = Path(a.hyb) / "inputs"; O = Path(a.out); O.mkdir(parents=True, exist_ok=True); L0 = 29730000
for f in ["beads.tsv", "tf_species.tsv", "hic_10kb.npz", "pairs_cicero.tsv", "pairs_loops.tsv"]: shutil.copy(I / f, O / f)
H = np.load(I / "hic_10kb.npz"); ice, keep = H["ice"], H["keep"]; n = ice.shape[0]
exp = np.full(n, np.nan)
for d in range(n):
    i = np.arange(n - d); j = i + d; ok = keep[i] & keep[j] & np.isfinite(ice[i, j])
    if ok.any(): exp[d] = ice[i[ok], j[ok]].mean()
LR = pd.read_csv(I / "loop_restraints.tsv", sep="\t")
E = pd.read_csv(a.hub_edges, sep="\t"); E = E[E["on_loop_0.7"] == 1]
def mid(r): s, e = map(int, r.split(":")[1].split("-")); return (s + e) // 2
red = []
for r in E.itertuples():
    c1, c2 = sorted((mid(r.region1), mid(r.region2))); i, j = (c1 - L0) // 10000, (c2 - L0) // 10000
    red.append(dict(kind="red_edge", b1=(c1 - L0) // 2000, b2=(c2 - L0) // 2000, bin1=i, bin2=j,
                    oe=ice[i, j] / exp[j - i] if keep[i] and keep[j] else np.nan, p_on=1.0, k=10.0, r0=1.0))
loops = [dict(kind="loop", b1=r.b1, b2=r.b2, bin1=r.bin1, bin2=r.bin2, oe=r.oe, p_on=r.p_on, k=10.0, r0=1.0) for r in LR.itertuples()]
taken = np.zeros((n, n), bool)
for x in red + loops: taken[max(0, x["bin1"] - 1):x["bin1"] + 2, max(0, x["bin2"] - 1):x["bin2"] + 2] = True
pix = []
for d in range(2, min(a.max_sep, n - 1) + 1):
    for i in range(n - d):
        j = i + d
        if not (keep[i] and keep[j]) or not np.isfinite(ice[i, j]) or taken[i, j]: continue
        oe = ice[i, j] / exp[d]
        if oe >= a.oe_min: pix.append(dict(bin1=i, bin2=j, oe=oe))
P = pd.DataFrame(pix)
rng = np.random.default_rng(7); ho = rng.random(len(P)) < a.holdout
P[ho].to_csv(O / "heldout_pixels.tsv", sep="\t", index=False)
P = P[~ho].copy()
P["p_on"] = 0.1 + 0.5 * P.oe.rank(pct=True); P["kind"] = "pixel"; P["k"] = 3.0; P["r0"] = 1.5
P["b1"] = P.bin1 * 5 + 2; P["b2"] = P.bin2 * 5 + 2
R = pd.concat([pd.DataFrame(red), pd.DataFrame(loops), P[["kind", "b1", "b2", "bin1", "bin2", "oe", "p_on", "k", "r0"]]], ignore_index=True)
R.to_csv(O / "restraints.tsv", sep="\t", index=False)
print(f"pixels O/E>={a.oe_min} (2-{a.max_sep} bins, not on a loop/red edge): {len(P) + ho.sum()}; held out {ho.sum()}; restrained {len(P)} "
      f"(mean p_on {P.p_on.mean():.2f}); loops {len(loops)} (mean p_on {LR.p_on.mean():.2f}); red edges {len(red)} always on; "
      f"expected ON per trajectory ~{P.p_on.sum() + LR.p_on.sum() + len(red):.0f}")
