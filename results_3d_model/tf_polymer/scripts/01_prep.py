"""Step 1: bead table and validation pair sets for the TF-binding polymer of chr16:29,730,000-31,550,000 (hg38).

Binding beads come ONLY from MACS2 summits of the 79 TFs (no Hi-C, loops or Cicero edges enter the model).
Hi-C, Peakachu loops and Cicero edges are written here for the held-out validation only.

Outputs (inputs/):
  beads.tsv          910 x 2 kb beads: n_tfs with a summit in the bead, TF list, strength class (0 none, 1: 1 TF, 2: 2-4, 3: >=5),
                     hub-core label (the 9 highlighted v2 cores)
  hic_10kb.npz       ICE-balanced and raw 10 kb Hi-C of the locus (validation only), keep mask from the PASTIS prep
  pairs_cicero.tsv   hub-component edges (24), shared clique edges (>=2 TFs, >25 kb) and Cicero step-1 pairs, as bead pairs
  pairs_loops.tsv    Peakachu 0.7 loops as 10 kb bin pairs
"""
import argparse, glob, os, re
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, cooler

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="chr16:29730000-31550000")
ap.add_argument("--bead", type=int, default=2000)
ap.add_argument("--res", type=int, default=10000)
ap.add_argument("--peaks-dir", required=True)
ap.add_argument("--mcool", required=True)
ap.add_argument("--prev", required=True, help="results_3d_model (out/ and v2/ of the PASTIS model)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
P = Path(a.prev)
chrom, se = a.region.split(":"); L0, L1 = map(int, se.split("-"))
nb = (L1 - L0) // a.bead
tobead = lambda pos: int((pos - L0) // a.bead)
def mid(r):
    c, s = r.split(":"); s, e = map(int, s.split("-")); return (s + e) // 2

# summits per bead
tfs_in = defaultdict(set)
files = sorted(glob.glob(os.path.join(a.peaks_dir, "TF.*_peaks.q005.narrowPeak")))
for f in files:
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.rstrip("\n").split("\t")
        if x[0] != chrom: continue
        s = int(x[1]) + int(x[9])
        if L0 <= s < L1: tfs_in[tobead(s)].add(tf)
B = pd.DataFrame(dict(bead=np.arange(nb), start=L0 + a.bead * np.arange(nb)))
B["end"] = B.start + a.bead
B["n_tfs"] = [len(tfs_in[i]) for i in range(nb)]
B["tfs"] = [",".join(sorted(tfs_in[i])) for i in range(nb)]
B["cls"] = np.select([B.n_tfs >= 5, B.n_tfs >= 2, B.n_tfs == 1], [3, 2, 1], 0)
B["bin10"] = (B.start - L0) // a.res
print(f"{len(files)} TF summit files; beads {nb}; bound {int((B.cls > 0).sum())} "
      f"(class1 {int((B.cls == 1).sum())}, class2 {int((B.cls == 2).sum())}, class3 {int((B.cls == 3).sum())})")

# 9 highlighted hub cores (v2) -> bead of the core midpoint
bins = pd.read_csv(P / "out/bins.tsv", sep="\t")
hc = pd.read_csv(P / "v2/highlighted_cores.tsv", sep="\t")
B["hub_label"] = ""
for r in hc.itertuples():
    core = bins.loc[bins.bin == r.bin, "hub_core"].values[0]
    b = tobead(mid(core))
    B.at[b, "hub_label"] = r.label
    print(f"hub core {r.label}: {core} -> bead {b} (n_tfs at bead {B.n_tfs[b]}, class {B.cls[b]})")
B.to_csv(out / "beads.tsv", sep="\t", index=False)

# validation-only Hi-C
clr = cooler.Cooler(f"{a.mcool}::/resolutions/{a.res}")
ice = clr.matrix(balance=True).fetch(a.region)
raw = clr.matrix(balance=False).fetch(a.region)
np.savez_compressed(out / "hic_10kb.npz", ice=ice, raw=raw, keep=bins.keep.values.astype(bool))

# Cicero / clique pair sets as bead pairs
rows = []
hub = pd.read_csv(P / "v2/hub_edges_with_groups.tsv", sep="\t")
for r in hub.itertuples():
    b1, b2 = sorted((tobead(mid(r.region1)), tobead(mid(r.region2))))
    rows.append(dict(set="hub_edges", b1=b1, b2=b2, group=r.group, color=r.color, n_tfs_edge=r.n_tfs, label=f"{r.region1}|{r.region2}"))
sh = pd.read_csv(P / "out/shared_edges.tsv", sep="\t")
for r in sh.itertuples():
    b1, b2 = sorted((tobead(mid(r.region1)), tobead(mid(r.region2))))
    rows.append(dict(set="shared_edges", b1=b1, b2=b2, group=r.clique_group, color="", n_tfs_edge=r.n_tfs, label=f"{r.region1}|{r.region2}"))
ci = pd.read_csv(P / "out/cicero_pairs.tsv", sep="\t")
for r in ci.itertuples():
    b1, b2 = sorted((tobead(mid(r.peak1)), tobead(mid(r.peak2))))
    rows.append(dict(set="cicero_step1", b1=b1, b2=b2, group=-1, color="", n_tfs_edge=1, label=r.tf))
C = pd.DataFrame(rows)
C = C[C.b2 > C.b1].drop_duplicates(["set", "b1", "b2"])
C.to_csv(out / "pairs_cicero.tsv", sep="\t", index=False)
print("Cicero/clique bead pairs:", C.groupby("set").size().to_dict())
Lp = pd.read_csv(P / "out/loops.tsv", sep="\t")[["bin1", "bin2", "score", "res"]]
Lp.to_csv(out / "pairs_loops.tsv", sep="\t", index=False)
print("Peakachu 0.7 loops (10 kb bin pairs):", len(Lp))
