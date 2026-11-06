"""v2 step 1 for another hub locus: everything 01_prep_v2.py needs, built from scratch for any region (no PASTIS prep).

Model inputs (TF binding only): 2 kb beads, MACS2 summits of the 79 TFs, sparse binding beads (all beads with >=5 TFs
plus the top beads by summed -log10 q, --frac-bind of the beads in total), 6 TF species (k-means), binary weights.
Validation-only inputs: 10 kb ICE Hi-C + keep mask, Peakachu 0.7 loops, shared clique edges (>=2 TFs, >25 kb),
Cicero step-1 pairs (>25 kb) of the TFs in those edges, and the hub-component edges of the browser figure.
Hub cores = the elements of the browser figure's hub edges; label = gene with a TSS within 2 kb of the element (as in
the browser gene track), else "".
"""
import argparse, glob, os, re
from pathlib import Path
import numpy as np, pandas as pd, cooler
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--region", required=True)
ap.add_argument("--bead", type=int, default=2000)
ap.add_argument("--res", type=int, default=10000)
ap.add_argument("--peaks-dir", required=True)
ap.add_argument("--mcool", required=True)
ap.add_argument("--loops", nargs="+", required=True)
ap.add_argument("--hub-edges", required=True, help="hub_<locus>_edges.tsv of the browser figure")
ap.add_argument("--genes", required=True, help="genes_<locus>.tsv of the browser figure")
ap.add_argument("--shared-edges", required=True)
ap.add_argument("--cicero-dir", required=True)
ap.add_argument("--rbbp4-edges", required=True)
ap.add_argument("--frac-bind", type=float, default=0.154)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--min-span", type=int, default=25000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
chrom, se = a.region.split(":"); L0, L1 = map(int, se.replace(",", "").split("-")); nb = (L1 - L0) // a.bead
assert (L1 - L0) % a.res == 0
def parse(r):
    c, s = r.split(":"); s, e = s.split("-"); return c, int(s), int(e)
def mid(r): _, s, e = parse(r); return (s + e) // 2
def inloc(r): c, s, e = parse(r); return c == chrom and s >= L0 and e <= L1
tobead = lambda p: int((p - L0) // a.bead)

# summits
rows = []
for f in sorted(glob.glob(os.path.join(a.peaks_dir, "TF.*_peaks.q005.narrowPeak"))):
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.rstrip("\n").split("\t")
        if x[0] != chrom: continue
        s = int(x[1]) + int(x[9])
        if L0 <= s < L1: rows.append((tf, tobead(s), float(x[8])))
P = pd.DataFrame(rows, columns=["tf", "bead", "mlog10q"])
M = pd.crosstab(P.bead, P.tf).clip(upper=1); n = M.sum(1); score = P.groupby("bead").mlog10q.sum()
B = pd.DataFrame(dict(bead=np.arange(nb), start=L0 + a.bead * np.arange(nb))); B["end"] = B.start + a.bead
B["n_tfs"] = B.bead.map(n).fillna(0).astype(int)
B["tfs"] = [",".join(sorted(M.columns[M.loc[b] > 0])) if b in M.index else "" for b in B.bead]
B["cls"] = np.select([B.n_tfs >= 5, B.n_tfs >= 2, B.n_tfs == 1], [3, 2, 1], 0)

# hub cores = browser hub-edge elements, labelled by promoter gene
G = pd.read_csv(a.genes, sep="\t"); G["tss"] = np.where(G.strand == "+", G.txStart, G.txEnd)
HE = pd.read_csv(a.hub_edges, sep="\t")
elems = sorted({r for c in ("region1", "region2") for r in HE[c] if inloc(r)}, key=mid)
B["hub_label"] = ""; B["hub_element"] = ""
lab_of = {}
for r in elems:
    _, s, e = parse(r); d = np.minimum(np.abs(G.tss - s), np.abs(G.tss - e)); d[(G.tss >= s) & (G.tss <= e)] = 0
    g = G.loc[d <= 2000, "name"]; lab = "/".join(sorted(set(g)))[:24] if len(g) else ""
    b = tobead(mid(r)); lab_of[r] = lab
    B.at[b, "hub_element"] = r; B.at[b, "hub_label"] = lab if lab else f"{mid(r)/1e6:.3f} Mb"

# sparse binding beads: all >=5-TF beads (and every hub element) + top by score
n_bind = int(round(a.frac_bind * nb))
must = set(n.index[n >= 5]) | set(B.bead[B.hub_label != ""])
must = {b for b in must if b in M.index}
rest = [b for b in score.rank(ascending=False, method="first").sort_values().index if b not in must]
sel = sorted(must | set(rest[:max(0, n_bind - len(must))]))
B["bind"] = B.bead.isin(sel)
Ms = M.loc[sel]; tfs = Ms.columns[Ms.sum(0) > 0]; Ms = Ms[tfs]
X = Ms.T.values.astype(float); X /= np.linalg.norm(X, axis=1, keepdims=True)
lab = KMeans(a.k, n_init=50, random_state=0).fit_predict(X)
pd.DataFrame(dict(tf=tfs, species=lab)).sort_values(["species", "tf"]).to_csv(out / "tf_species.tsv", sep="\t", index=False)
W = np.zeros((nb, a.k)); W[np.array(sel)] = (Ms.values @ np.eye(a.k)[lab]) > 0
for k in range(a.k): B[f"w_{k}"] = W[:, k]
B["n_species"] = (W > 0).sum(1)
B.to_csv(out / "beads.tsv", sep="\t", index=False)
print(f"{a.region}: {nb} beads; binding {len(sel)} ({100*len(sel)/nb:.1f}%); >=5-TF beads {int((n >= 5).sum())}; "
      f"TFs per species {np.bincount(lab).tolist()}; beads per species {(W > 0).sum(0).astype(int).tolist()}")
print("hub elements:", [(r, lab_of[r]) for r in elems])

# validation-only: Hi-C
clr = cooler.Cooler(f"{a.mcool}::/resolutions/{a.res}")
reg = f"{chrom}:{L0}-{L1}"
ice = clr.matrix(balance=True).fetch(reg); raw = clr.matrix(balance=False).fetch(reg)
cov = raw.sum(1) - np.diag(raw); w = clr.bins().fetch(reg)["weight"].values
keep = np.isfinite(w) & (cov >= 0.2 * np.median(cov[cov > 0]))
np.savez_compressed(out / "hic_10kb.npz", ice=ice, raw=raw, keep=keep)
# pairs
rows = []
LOOPC, EDGE = "#C0392B", "#4A6FA5"
for _, r in HE.iterrows():
    if not (inloc(r.region1) and inloc(r.region2)): continue
    b1, b2 = sorted((tobead(mid(r.region1)), tobead(mid(r.region2))))
    rows.append(dict(set="hub_edges", b1=b1, b2=b2, group=int(r["on_loop_0.7"]), color=LOOPC if r["on_loop_0.7"] == 1 else EDGE,
                     n_tfs_edge=r.n_tfs, label=f"{r.region1}|{r.region2}"))
SE = pd.read_csv(a.shared_edges, sep="\t")
sh = []
for r in SE.itertuples():
    if not (inloc(r.region1) and inloc(r.region2)) or r.distance_kb * 1000 <= a.min_span: continue
    b1, b2 = sorted((tobead(mid(r.region1)), tobead(mid(r.region2))))
    rows.append(dict(set="shared_edges", b1=b1, b2=b2, group=-1, color="", n_tfs_edge=r.n_tfs, label=f"{r.region1}|{r.region2}"))
    sh.append(r.tfs)
fig_tfs = sorted({t for s in sh for t in s.split(",")} | {t for s in HE.tfs for t in s.split(",")})
for tf in fig_tfs:
    f = a.rbbp4_edges if tf == "RBBP4" else os.path.join(a.cicero_dir, tf, f"{tf}.peak_edges.tsv")
    if not os.path.exists(f): continue
    d = pd.read_csv(f, sep="\t", usecols=["peak1", "peak2"])
    d = d[d.peak1.str.startswith(chrom + ":") & d.peak2.str.startswith(chrom + ":")]
    for r in d.itertuples():
        if not (inloc(r.peak1) and inloc(r.peak2)) or abs(mid(r.peak2) - mid(r.peak1)) <= a.min_span: continue
        b1, b2 = sorted((tobead(mid(r.peak1)), tobead(mid(r.peak2))))
        rows.append(dict(set="cicero_step1", b1=b1, b2=b2, group=-1, color="", n_tfs_edge=1, label=tf))
C = pd.DataFrame(rows); C = C[C.b2 > C.b1].drop_duplicates(["set", "b1", "b2"])
C.to_csv(out / "pairs_cicero.tsv", sep="\t", index=False)
print("pairs:", C.groupby("set").size().to_dict())
lrow = []
for f in a.loops:
    for ln in open(f):
        x = ln.split()
        if x[0] != chrom or x[3] != chrom: continue
        s1, e1, s2, e2 = map(int, (x[1], x[2], x[4], x[5]))
        if s1 < L0 or e2 > L1 or s2 < L0 or e1 > L1: continue
        b1, b2 = sorted((((s1 + e1) // 2 - L0) // a.res, ((s2 + e2) // 2 - L0) // a.res))
        lrow.append(dict(bin1=b1, bin2=b2, score=float(x[6]) if len(x) > 6 else np.nan))
Lp = pd.DataFrame(lrow).sort_values("score", ascending=False).drop_duplicates(["bin1", "bin2"]); Lp = Lp[Lp.bin2 > Lp.bin1]
Lp.to_csv(out / "pairs_loops.tsv", sep="\t", index=False)
print("loops:", len(Lp))
