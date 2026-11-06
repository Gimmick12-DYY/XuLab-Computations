"""v2 step 1: sparse, species-specific binding map for the TF-switching polymer of chr16:29,730,000-31,550,000 (hg38).

Binding beads (sparse): beads bound by >=5 TFs, plus the top beads by summed MACS2 -log10 q of the summits in the bead,
up to --n-bind beads in total (>=5-TF beads, which include the 9 hub cores, are always kept).
TF species: the TFs with summits in the selected beads are grouped by k-means (--k species) on their L2-normalised
binary profiles over those beads. Bead b attracts species k with weight min(1, n_bk / --sat), where n_bk = number of TFs
of group k with a summit in b; default sat = 1, i.e. binary: any TF of group k (beads bridge only via shared species).
No Hi-C, loops or Cicero edges are used. Validation inputs (hic_10kb.npz, pairs_*.tsv) are copied from the v1 prep.

Outputs: beads.tsv (n_tfs, tfs, cls (v1 class), bind, w_0..w_{k-1}), tf_species.tsv
"""
import argparse, glob, os, re, shutil
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="chr16:29730000-31550000")
ap.add_argument("--bead", type=int, default=2000)
ap.add_argument("--peaks-dir", required=True)
ap.add_argument("--v1", required=True, help="tf_polymer (v1) folder: inputs/beads.tsv, hic_10kb.npz, pairs_*.tsv")
ap.add_argument("--n-bind", type=int, default=140)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--sat", type=float, default=1.0, help="weight = min(1, n_bk/sat); 1 = binary (any TF of group k)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
chrom, se = a.region.split(":"); L0, L1 = map(int, se.split("-")); nb = (L1 - L0) // a.bead

rows = []
for f in sorted(glob.glob(os.path.join(a.peaks_dir, "TF.*_peaks.q005.narrowPeak"))):
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.rstrip("\n").split("\t")
        if x[0] != chrom: continue
        s = int(x[1]) + int(x[9])
        if L0 <= s < L1: rows.append((tf, (s - L0) // a.bead, float(x[8])))
P = pd.DataFrame(rows, columns=["tf", "bead", "mlog10q"])
M = pd.crosstab(P.bead, P.tf).clip(upper=1)
n = M.sum(1); score = P.groupby("bead").mlog10q.sum()
B = pd.read_csv(Path(a.v1) / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""})
rank = score.rank(ascending=False, method="first")
# keep all >=5-TF beads, fill to n_bind by score
must = set(n.index[n >= 5]); rest = [b for b in rank.sort_values().index if b not in must]
sel = sorted(must | set(rest[:max(0, a.n_bind - len(must))]))
B["bind"] = B.bead.isin(sel)
assert B.loc[B.hub_label != "", "bind"].all()
Ms = M.loc[sel]; tfs = Ms.columns[Ms.sum(0) > 0]; Ms = Ms[tfs]
X = Ms.T.values.astype(float); X /= np.linalg.norm(X, axis=1, keepdims=True)
lab = KMeans(a.k, n_init=50, random_state=0).fit_predict(X)
pd.DataFrame(dict(tf=tfs, species=lab, n_bind_beads=Ms.sum(0).values)).sort_values(["species", "tf"]).to_csv(out / "tf_species.tsv", sep="\t", index=False)
nk = Ms.values @ np.eye(a.k)[lab]          # beads x species TF counts
W = np.zeros((len(B), a.k)); W[np.array(sel)] = np.minimum(1.0, nk / a.sat)
for k in range(a.k): B[f"w_{k}"] = W[:, k]
B["n_species"] = (W > 0).sum(1)
B["score_mlog10q"] = B.bead.map(score).fillna(0)
B.to_csv(out / "beads.tsv", sep="\t", index=False)
for f in ["hic_10kb.npz", "pairs_cicero.tsv", "pairs_loops.tsv"]: shutil.copy(Path(a.v1) / "inputs" / f, out / f)
print(f"binding beads {len(sel)}/{nb} ({100*len(sel)/nb:.1f}%); >=5-TF beads {len(must)}; TFs in species: {len(tfs)}; "
      f"TFs per species {np.bincount(lab).tolist()}; beads per species {(W > 0).sum(0).tolist()}; "
      f"mean weight per species {np.round(W[np.array(sel)].mean(0), 2).tolist()}; species per binding bead {np.bincount(B.n_species[B.bind]).tolist()}")
hb = B[B.hub_label != ""]
print(hb[["bead", "hub_label", "n_tfs"] + [f"w_{k}" for k in range(a.k)]].round(2).to_string(index=False))
