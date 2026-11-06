"""Step 1: extract the chr16p11.2 locus at 10 kb from WT HEK293T Hi-C and collect per-bin / per-pair annotations.

Outputs (out/):
  counts_raw.npz      raw contact counts (all locus bins, symmetric, diagonal kept)
  bins.tsv            bin table: start, end, ICE weight, raw coverage, keep flag, n_tfs (MACS2 q<0.05), TF list,
                      hub-core promoter label
  shared_edges.tsv    shared Cicero clique edges (>=2 TFs, >25 kb) inside the locus, with bin indices and clique group
  cicero_pairs.tsv    Cicero step-1 pairs (>25 kb) from the TFs that share an edge in the locus
  loops.tsv           Peakachu 0.7 loops (5 kb U 10 kb) with both anchors in the locus
"""
import argparse, glob, os, re
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, cooler

ap = argparse.ArgumentParser()
ap.add_argument("--mcool", required=True)
ap.add_argument("--region", default="chr16:29730000-31550000")
ap.add_argument("--res", type=int, default=10000)
ap.add_argument("--loops", nargs="+", required=True)
ap.add_argument("--shared-edges", required=True)
ap.add_argument("--hub-cores", required=True)
ap.add_argument("--peaks-dir", required=True)
ap.add_argument("--cicero-dir", required=True, help="cobinding/work/ourrun_k50 (per-TF <TF>/<TF>.peak_edges.tsv)")
ap.add_argument("--rbbp4-edges", required=True)
ap.add_argument("--min-span", type=int, default=25000)
ap.add_argument("--min-cov-frac", type=float, default=0.2, help="drop bins with raw cis coverage below this fraction of the locus median")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

chrom, se = a.region.split(":"); L0, L1 = map(int, se.split("-"))
clr = cooler.Cooler(f"{a.mcool}::/resolutions/{a.res}")
raw = clr.matrix(balance=False).fetch(a.region).astype(float)
bins = clr.bins().fetch(a.region).reset_index(drop=True)
n = len(bins)
cov = raw.sum(1) - np.diag(raw)
w = bins["weight"].values
med = np.median(cov[cov > 0])
keep = np.isfinite(w) & (cov >= a.min_cov_frac * med)
bins["raw_cov"] = cov.astype(int); bins["keep"] = keep
np.savez_compressed(out / "counts_raw.npz", counts=raw)
print(f"{n} bins; {keep.sum()} kept; dropped: " + ", ".join(f"{s/1e6:.2f}" for s in bins.start[~keep]))

def tobin(pos):
    i = (np.asarray(pos) - bins.start.iloc[0]) // a.res
    return i.astype(int)
def parse(r):
    c, s = r.split(":"); s, e = s.split("-"); return c, int(s), int(e)
def inloc(c, s, e):
    return c == chrom and s >= L0 and e <= L1

# per-bin TF count from MACS2 q<0.05 peaks
tf_in_bin = defaultdict(set)
files = sorted(glob.glob(os.path.join(a.peaks_dir, "TF.*_peaks.q005.narrowPeak")))
for f in files:
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.split("\t")
        if x[0] != chrom: continue
        s, e = int(x[1]), int(x[2])
        if e <= L0 or s >= L1: continue
        for b in range(max(0, (s - L0) // a.res), min(n - 1, (e - 1 - L0) // a.res) + 1):
            tf_in_bin[b].add(tf)
bins["n_tfs"] = [len(tf_in_bin[i]) for i in range(n)]
bins["tfs"] = [",".join(sorted(tf_in_bin[i])) for i in range(n)]
print(f"MACS2 files: {len(files)} TFs")

# hub cores (anchor-centred cores in >=5 TFs' cliques) with promoter labels
hc = pd.read_csv(a.hub_cores, sep="\t")
bins["hub_core"] = ""; bins["hub_core_ntfs"] = 0; bins["hub_label"] = ""
for r in hc.itertuples():
    c, s, e = parse(r.core)
    if not inloc(c, s, e): continue
    b = int(tobin((s + e) // 2))
    if r.n_tfs > bins.at[b, "hub_core_ntfs"]:
        bins.at[b, "hub_core"] = r.core; bins.at[b, "hub_core_ntfs"] = r.n_tfs
        bins.at[b, "hub_label"] = r.promoter_gene if isinstance(r.promoter_gene, str) else ""
bins.to_csv(out / "bins.tsv", sep="\t", index_label="bin")

# shared clique edges in locus; clique group = connected components of edges that share a TF clique id
se_ = pd.read_csv(a.shared_edges, sep="\t")
rows = []
for r in se_.itertuples():
    p1, p2 = parse(r.region1), parse(r.region2)
    if not (inloc(*p1) and inloc(*p2)) or r.distance_kb * 1000 <= a.min_span: continue
    b1, b2 = sorted((int(tobin((p1[1] + p1[2]) // 2)), int(tobin((p2[1] + p2[2]) // 2))))
    rows.append(dict(region1=r.region1, region2=r.region2, distance_kb=r.distance_kb, n_tfs=r.n_tfs, tfs=r.tfs,
                     on_loop_07=r._6, bin1=b1, bin2=b2, cliques=r.cliques))
E = pd.DataFrame(rows)
parent = list(range(len(E)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
owner = {}
for i, cl in enumerate(E.cliques):
    for c in cl.split(";"):
        if c in owner: parent[find(i)] = find(owner[c])
        else: owner[c] = i
roots = [find(i) for i in range(len(E))]
# number groups by size (largest first), then genomic position
sizes = pd.Series(roots).value_counts()
order = sorted(sizes.index, key=lambda r: (-sizes[r], E.bin1[r]))
gid = {r: k for k, r in enumerate(order)}
E["clique_group"] = [gid[r] for r in roots]
E.drop(columns="cliques").to_csv(out / "shared_edges.tsv", sep="\t", index=False)
print(f"shared edges >25 kb in locus: {len(E)}; clique groups: {E.clique_group.nunique()} sizes {sizes.values.tolist()}")
fig_tfs = sorted({t for s in E.tfs for t in s.split(",")})

# Cicero step-1 pairs (>25 kb) from those TFs
crow = []
for tf in fig_tfs:
    f = a.rbbp4_edges if tf == "RBBP4" else os.path.join(a.cicero_dir, tf, f"{tf}.peak_edges.tsv")
    if not os.path.exists(f):
        print("missing", f); continue
    d = pd.read_csv(f, sep="\t", usecols=["peak1", "peak2", "coaccess"])
    d = d[d.peak1.str.startswith(chrom + ":") & d.peak2.str.startswith(chrom + ":")]
    for r in d.itertuples():
        p1, p2 = parse(r.peak1), parse(r.peak2)
        if not (inloc(*p1) and inloc(*p2)): continue
        m1, m2 = (p1[1] + p1[2]) / 2, (p2[1] + p2[2]) / 2
        if abs(m2 - m1) <= a.min_span: continue
        b1, b2 = sorted((int(tobin(m1)), int(tobin(m2))))
        crow.append(dict(tf=tf, peak1=r.peak1, peak2=r.peak2, coaccess=r.coaccess, span_kb=abs(m2 - m1) / 1000, bin1=b1, bin2=b2))
C = pd.DataFrame(crow)
C.to_csv(out / "cicero_pairs.tsv", sep="\t", index=False)
print(f"Cicero step-1 pairs >25 kb in locus from {len(fig_tfs)} TFs: {len(C)} ({C[['bin1','bin2']].drop_duplicates().shape[0]} distinct bin pairs)")

# Peakachu 0.7 loops, 5 kb U 10 kb, deduplicated at 10 kb bin level
lrow = []
for f in a.loops:
    res = re.search(r"peakachu_(\d+)\.", f).group(1)
    for ln in open(f):
        x = ln.split()
        if x[0] != chrom or x[3] != chrom: continue
        s1, e1, s2, e2 = map(int, (x[1], x[2], x[4], x[5]))
        if not (inloc(chrom, s1, e1) and inloc(chrom, s2, e2)): continue
        b1, b2 = sorted((int(tobin((s1 + e1) // 2)), int(tobin((s2 + e2) // 2))))
        lrow.append(dict(res=res, a1=f"{chrom}:{s1}-{e1}", a2=f"{chrom}:{s2}-{e2}", score=float(x[6]) if len(x) > 6 else np.nan, bin1=b1, bin2=b2))
Lp = pd.DataFrame(lrow).sort_values("score", ascending=False).drop_duplicates(["bin1", "bin2"])
Lp = Lp[Lp.bin2 > Lp.bin1]
Lp.to_csv(out / "loops.tsv", sep="\t", index=False)
print(f"Peakachu 0.7 loops in locus: {len(lrow)} calls -> {len(Lp)} distinct 10 kb bin pairs")
