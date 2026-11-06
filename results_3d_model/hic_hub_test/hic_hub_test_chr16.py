"""Direct test of the chr16p11.2 promoter hub in WT HEK293T Hi-C (GSE216399; HEK293TWT-MboI-R1-filtered.mcool), no polymer model.

Per resolution (--res 10000 and 5000): ICE-balanced chr16 matrix; expected = mean balanced contact per diagonal over valid bin
pairs of chr16 (cooltools expected_cis without smoothing; cooltools is not installed in the env); O/E = observed / expected.
Pairs closer than 25 kb are excluded throughout.
Bin annotation (chr16): TF count = number of the 79 TFs with a MACS2 q<0.05 summit in the bin; class 0 / 1-4 / 5-9 / >=10;
active promoter = bin holding a GENCODE v46 basic protein-coding or lncRNA TSS and >= 1 TF summit.
Hub = 23 elements of the browser figure (hub_chr16_edges.tsv) -> bins; 24 hub edges (18 loop-edges on a 0.7 loop, 6 Cicero-only),
plus the 58 shared clique edges >25 kb in chr16:29.73-31.55 Mb.
1. Edge level (1,000 draws): median O/E of each set vs (a) random chr16 pairs at distance +-10%; (b) pairs of active promoters with
   the same TF-class pair at distance +-10%. ratio = observed median / mean null median; one-sided empirical p.
2. Hub level (1,000 draws): mean O/E over ALL pairs of hub bins (>25 kb) vs sets built in other chr16 windows of the same span
   (not overlapping the hub locus): each hub bin is replaced by a bin within +-30 kb of the same relative position that is
   (a) any valid bin (distance-matched) or (b) an active promoter of the same TF class (distance + TF matched; window
   widened to +-90 / +-180 kb when no such promoter lies within +-30 kb). Also after
   removing the 24 Cicero edge pairs (and the same index pairs in the nulls).
3. Transitivity: triplets with A-B and B-C loop-edges and A-C not a hub edge; median O/E of A-C vs distance-matched random pairs.
4. APA (10 kb only, +-5 bins): mean O/E pileup of each pair set and of 1,000 distance-matched random pairs;
   APA score = centre / mean of the lower-left 3x3 corner.
"""
import argparse, glob, gzip, json, os, re, itertools
from pathlib import Path
import numpy as np, pandas as pd, cooler

ap = argparse.ArgumentParser()
ap.add_argument("--mcool", required=True); ap.add_argument("--peaks-dir", required=True); ap.add_argument("--gtf", required=True)
ap.add_argument("--hub-edges", required=True); ap.add_argument("--shared-edges", required=True)
ap.add_argument("--res", type=int, default=10000); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out", required=True)
a = ap.parse_args()
O = Path(a.out); O.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(7); RES = a.res; CH = "chr16"
L0, L1 = 29730000, 31550000; MIN = int(np.ceil(25000 / RES)); DMAX = int(3_000_000 / RES)
clr = cooler.Cooler(f"{a.mcool}::/resolutions/{RES}")
M = clr.matrix(balance=True).fetch(CH).astype(np.float32); nb = M.shape[0]
valid = np.isfinite(clr.bins().fetch(CH)["weight"].values)
exp = np.full(DMAX + 1, np.nan)
for d in range(DMAX + 1):
    v = np.diagonal(M, d); ok = valid[:nb - d] & valid[d:] & np.isfinite(v)
    if ok.any(): exp[d] = v[ok].mean()
def oe(i, j):
    i, j = np.minimum(i, j), np.maximum(i, j); d = j - i
    with np.errstate(invalid="ignore"):
        r = M[i, j] / exp[np.minimum(d, DMAX)]
    return np.where(valid[i] & valid[j] & (d <= DMAX), r, np.nan)
# annotations
tf = np.zeros(nb, int)
for f in sorted(glob.glob(os.path.join(a.peaks_dir, "TF.*_peaks.q005.narrowPeak"))):
    s = set()
    for ln in open(f):
        x = ln.split("\t")
        if x[0] == CH: s.add((int(x[1]) + int(x[9])) // RES)
    for b in s:
        if b < nb: tf[b] += 1
cls = np.select([tf >= 10, tf >= 5, tf >= 1], [3, 2, 1], 0)
tss = np.zeros(nb, bool)
with gzip.open(a.gtf, "rt") as g:
    for ln in g:
        if not ln.startswith(CH + "\t"): continue
        x = ln.split("\t", 9)
        if x[2] != "gene" or not re.search(r'gene_type "(protein_coding|lncRNA)"', x[8]): continue
        p = int(x[3]) if x[6] == "+" else int(x[4]); b = p // RES
        if b < nb: tss[b] = True
active = tss & (tf >= 1) & valid
def mid(r): s, e = map(int, r.split(":")[1].split("-")); return (s + e) // 2
HE = pd.read_csv(a.hub_edges, sep="\t")
HE["i"] = [min(mid(x), mid(y)) // RES for x, y in zip(HE.region1, HE.region2)]; HE["j"] = [max(mid(x), mid(y)) // RES for x, y in zip(HE.region1, HE.region2)]
SE = pd.read_csv(a.shared_edges, sep="\t")
SE = SE[[r1.startswith(CH + ":") and r2.startswith(CH + ":") and L0 <= mid(r1) < L1 and L0 <= mid(r2) < L1 and d > 25 for r1, r2, d in zip(SE.region1, SE.region2, SE.distance_kb)]].copy()
SE["i"] = [min(mid(x), mid(y)) // RES for x, y in zip(SE.region1, SE.region2)]; SE["j"] = [max(mid(x), mid(y)) // RES for x, y in zip(SE.region1, SE.region2)]
hub = np.array(sorted({mid(r) // RES for r in pd.concat([HE.region1, HE.region2])}))
hub = hub[valid[hub]]
sets = {"loop_edges_18": HE[HE["on_loop_0.7"] == 1], "cicero_only_6": HE[HE["on_loop_0.7"] == 0], "hub_edges_24": HE, "shared_edges_locus": SE}
res = dict(res=RES, n_hub_bins=int(len(hub)), hub_tf_classes=np.bincount(cls[hub], minlength=4).tolist(),
           n_active_promoter_bins_chr16=int(active.sum()))
act_idx = np.where(active)[0]; val_idx = np.where(valid)[0]
def rand_pair_dist(d):
    lo, hi = max(MIN, int(d * 0.9)), max(MIN, int(np.ceil(d * 1.1)))
    for _ in range(100):
        dd = rng.integers(lo, hi + 1); i = rng.choice(val_idx[val_idx < nb - dd]); j = i + dd
        if valid[j]: return i, j
    return None
cache = {}
def rand_pair_cls(i0, j0):
    d = j0 - i0; key = (min(cls[i0], cls[j0]), max(cls[i0], cls[j0]), d)
    if key not in cache:
        lo, hi = max(MIN, int(d * 0.9)), int(np.ceil(d * 1.1)); A = act_idx
        D = A[None, :] - A[:, None]; ii, jj = np.where((D >= lo) & (D <= hi))
        ci, cj = cls[A[ii]], cls[A[jj]]; m = ((ci == cls[i0]) & (cj == cls[j0])) | ((ci == cls[j0]) & (cj == cls[i0]))
        cache[key] = np.c_[A[ii[m]], A[jj[m]]]
    c = cache[key]; return tuple(c[rng.integers(len(c))]) if len(c) else None
# 1. edge level
edge_rows = []
for nm, E in sets.items():
    pr = [(i, j) for i, j in zip(E.i, E.j) if j - i >= MIN and valid[i] and valid[j]]
    obs = np.array([float(oe(i, j)) for i, j in pr]); om = np.nanmedian(obs)
    nA, nB = [], []
    for _ in range(a.n):
        va = [rand_pair_dist(j - i) for i, j in pr]; vb = [rand_pair_cls(i, j) for i, j in pr]
        nA.append(np.nanmedian([float(oe(*x)) for x in va if x])); nB.append(np.nanmedian([float(oe(*x)) for x in vb if x]))
    nA, nB = np.array(nA), np.array(nB)
    res[f"edges_{nm}"] = dict(n=len(pr), median_oe=om, mean_oe=float(np.nanmean(obs)), null_dist=float(nA.mean()), ratio_dist=float(om / nA.mean()),
                              p_dist=float((1 + (nA >= om).sum()) / (a.n + 1)), null_tf=float(nB.mean()), ratio_tf=float(om / nB.mean()),
                              p_tf=float((1 + (nB >= om).sum()) / (a.n + 1)))
    for (i, j), v in zip(pr, obs): edge_rows.append(dict(set=nm, i=i, j=j, kb=(j - i) * RES / 1000, oe=v, tf_i=tf[i], tf_j=tf[j]))
pd.DataFrame(edge_rows).to_csv(O / f"edges_oe_{RES // 1000}kb.tsv", sep="\t", index=False)
# 2. hub level (TF-matched null: same-class active promoter within +-30 kb, widened to +-90 / +-180 kb when none)
edge_bins = {(min(i, j), max(i, j)) for i, j in zip(HE.i, HE.j)}
pairs = [(p, q) for p, q in itertools.combinations(range(len(hub)), 2) if hub[q] - hub[p] >= MIN]
nonedge = [(p, q) for p, q in pairs if (hub[p], hub[q]) not in edge_bins]
def setstat(b, P): return float(np.nanmean([float(oe(b[p], b[q])) for p, q in P]))
obs_all, obs_ne = setstat(hub, pairs), setstat(hub, nonedge)
off = hub - hub[0]; span = off[-1]; W = int(30000 / RES)
lo_loc, hi_loc = L0 // RES, L1 // RES
nulls = {k: [] for k in ("dist_all", "dist_nonedge", "tf_all", "tf_nonedge")}
for kind in ("dist", "tf"):
    got = 0; tries = 0
    while got < a.n and tries < a.n * 200:
        tries += 1; s0 = rng.integers(0, nb - span - W)
        if s0 + span + W >= lo_loc and s0 - W <= hi_loc: continue
        b = []
        for k_, o in enumerate(off):
            c0 = s0 + o; win = np.array([], int)
            for Wk in ((W,) if kind == "dist" else (W, 3 * W, 6 * W)):
                w_ = np.arange(max(0, c0 - Wk), min(nb, c0 + Wk + 1))
                w_ = w_[valid[w_]] if kind == "dist" else w_[active[w_] & (cls[w_] == cls[hub[k_]])]
                win = np.setdiff1d(w_, b)
                if len(win): break
            if not len(win): break
            b.append(rng.choice(win))
        if len(b) < len(hub): continue
        b = np.array(b); nulls[f"{kind}_all"].append(setstat(b, pairs)); nulls[f"{kind}_nonedge"].append(setstat(b, nonedge)); got += 1
for k in nulls: nulls[k] = np.array(nulls[k]); np.save(O / f"hub_null_{k}_{RES // 1000}kb.npy", nulls[k])
for k, ob in (("all", obs_all), ("nonedge", obs_ne)):
    for kind in ("dist", "tf"):
        nn = nulls[f"{kind}_{k}"]
        if not len(nn): res[f"hub_{k}_{kind}"] = dict(obs=ob, n_null=0); continue
        res[f"hub_{k}_{kind}"] = dict(obs=ob, n_pairs=len(pairs if k == "all" else nonedge), n_null=len(nn), null_mean=float(nn.mean()),
                                      ratio=float(ob / nn.mean()), p=float((1 + (nn >= ob).sum()) / (len(nn) + 1)), null_q95=float(np.quantile(nn, 0.95)))
np.save(O / f"hub_oe_submatrix_{RES // 1000}kb.npy", np.array([[float(oe(x, y)) if x != y else np.nan for y in hub] for x in hub]))
pd.DataFrame(dict(bin=hub, start=hub * RES, tf=tf[hub], cls=cls[hub])).to_csv(O / f"hub_bins_{RES // 1000}kb.tsv", sep="\t", index=False)
# 3. transitivity
LE = sets["loop_edges_18"]; adj = {}
for i, j in zip(LE.i, LE.j): adj.setdefault(i, set()).add(j); adj.setdefault(j, set()).add(i)
tri = set()
for bb, nbs in adj.items():
    for x, y in itertools.combinations(sorted(nbs), 2):
        p, q = min(x, y), max(x, y)
        if q - p >= MIN and (p, q) not in edge_bins: tri.add((p, q))
tri = sorted(tri)
if tri:
    ob = np.nanmedian([float(oe(p, q)) for p, q in tri]); nT = []
    for _ in range(a.n): nT.append(np.nanmedian([float(oe(*x)) for x in (rand_pair_dist(q - p) for p, q in tri) if x]))
    nT = np.array(nT); res["transitive_pairs"] = dict(n=len(tri), median_oe=ob, null_dist=float(nT.mean()), ratio=float(ob / nT.mean()), p=float((1 + (nT >= ob).sum()) / (a.n + 1)))
# 4. APA (10 kb)
if RES == 10000:
    K = 5
    def pile(P):
        acc = [M[i - K:i + K + 1, j - K:j + K + 1] / exp[np.clip((np.arange(j - K, j + K + 1)[None] - np.arange(i - K, i + K + 1)[:, None]), 0, DMAX)]
               for i, j in P if i - K >= 0 and j + K < nb and j - i >= 2 * K]
        return (np.nanmean(acc, 0) if acc else None), len(acc)
    def score(P_): c = P_[K, K]; ll = P_[K + 2:K + 5, K - 4:K - 1]; return float(c / np.nanmean(ll))
    apa = {}
    groups = {"loop_edges_18": list(zip(LE.i, LE.j)), "cicero_only_6": list(zip(sets["cicero_only_6"].i, sets["cicero_only_6"].j)),
              "hub_all_pairs": [(hub[p], hub[q]) for p, q in pairs], "hub_nonedge_pairs": [(hub[p], hub[q]) for p, q in nonedge]}
    for nm, P in groups.items():
        pm, k = pile(P)
        if pm is None: continue
        ctrl = [rand_pair_dist(j - i) for i, j in P for _ in range(max(1, a.n // max(1, len(P))))]
        cm, kc = pile([x for x in ctrl if x])
        np.save(O / f"apa_{nm}.npy", pm); np.save(O / f"apa_{nm}_control.npy", cm)
        apa[nm] = dict(n=k, apa=score(pm), centre=float(pm[K, K]), control_apa=score(cm), control_centre=float(cm[K, K]), n_control=kc)
    res["apa_10kb"] = apa
json.dump(res, open(O / f"results_{RES // 1000}kb.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))
