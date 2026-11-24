"""APA of RBBP4 Multi-region Coordinated Cluster (MCC) edges (2026-11-24). WT HEK293T Hi-C (GSE216399), balanced, 10 kb, +-100 kb (21 x 21).
Sets (all intra-chromosomal, element midpoints; pairs closer than 150 kb dropped so the window stays off the diagonal):
  MCC edges        : unique region pairs inside one RBBP4 MCC (RBBP4_sweep_k50_i10000_s555/cliques.tsv), > 25 kb.
  non-MCC pairs    : RBBP4 coordinated binding pairs (first-step Cicero p <= 0.05, > 25 kb) that are not MCC edges, distance-matched.
  random peak pairs: random pairs of RBBP4 peaks (Cicero input peaks) on the same chromosome, distance-matched.
Distance matching: 20 log-spaced distance bins; each control set drawn per bin in the MCC-edge proportions, 5x the MCC count (with
replacement only when a bin runs short). Per set: mean balanced-contact pile-up and mean O/E pile-up (O/E against the chromosome-wide
distance mean). APA scores: P2LL = centre pixel / mean of the 3 x 3 lower-left corner (closest to the diagonal), on the observed pile-up;
centre O/E = mean O/E of the centre pixel over pairs; also centre 3 x 3 O/E. Draws a 3-panel figure (mean O/E, shared colour scale).
  python rbbp4_mcc_apa.py <outdir>   (Longleaf, hic env)"""
import glob, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import h5py, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
X = Path("/work/users/d/y/dyy12/XuLab"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
for f in glob.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
MCOOL = X / "hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool"; RES, FL, MIN_D, MAX_D = 10000, 10, 150_000, 1_000_000
rng = np.random.default_rng(2026)
def iv(x): c, se = x.strip().split(":"); s, e = se.split("-"); return c, int(s), int(e)
mid = lambda p: (p[1] + p[2]) // 2

E = pd.read_csv(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv", sep="\t"); E = E[E.pval <= 0.05]
C = pd.read_csv(X / "cobinding/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv", sep="\t")
key = lambda a, b: tuple(sorted((a, b)))
mcc = {key(a, b) for rr in C.regions.astype(str) for a, b in combinations(rr.split(";"), 2)}
def tup(a, b):
    p, q = iv(a), iv(b)
    if p[0] != q[0]: return None
    m1, m2 = sorted((mid(p), mid(q))); return (p[0], m1, m2)
S = {"MCC edges": [], "non-MCC pairs": []}
for a, b in zip(E.peak1, E.peak2):
    t = tup(a, b)
    if t is None or t[2] - t[1] <= 25000: continue
    S["MCC edges" if key(a, b) in mcc else "non-MCC pairs"].append(t)
n_mcc_all = len(S["MCC edges"])
for k in S: S[k] = [t for t in S[k] if MIN_D <= t[2] - t[1] <= MAX_D]
peaks = pd.read_csv(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/cicero/cds/regions_used_canonical.tsv.gz", sep="\t", header=None)[0].astype(str)
PM = defaultdict(list)
for r in peaks:
    if r.startswith("chr"): PM[iv(r)[0]].append(mid(iv(r)))
PM = {c: np.sort(v) for c, v in PM.items() if len(v) > 10}
bins = np.logspace(np.log10(MIN_D), np.log10(MAX_D + 1), 21)
dist = lambda L: np.array([t[2] - t[1] for t in L])
target = np.histogram(dist(S["MCC edges"]), bins)[0] * 5
# non-MCC: draw per bin
nm = S["non-MCC pairs"]; dn = np.digitize(dist(nm), bins) - 1; pick = []
for b_, need in enumerate(target):
    pool = np.flatnonzero(dn == b_)
    if need and len(pool): pick += list(rng.choice(pool, need, replace=len(pool) < need))
S["non-MCC pairs"] = [nm[i] for i in pick]
# random peak pairs: chromosome weighted by MCC edges, distance drawn from the bin, nearest peak to the far end
chrs = [t[0] for t in S["MCC edges"]]; rp = []
for b_, need in enumerate(target):
    got = 0
    while got < need:
        c = chrs[rng.integers(len(chrs))]; P = PM.get(c)
        if P is None: continue
        a = P[rng.integers(len(P))]; d = rng.uniform(bins[b_], bins[b_ + 1]); j = np.searchsorted(P, a + d)
        if j >= len(P): continue
        m2 = P[j]
        if bins[b_] <= m2 - a < bins[b_ + 1]: rp.append((c, int(a), int(m2))); got += 1
S["random peak pairs"] = rp

f = h5py.File(MCOOL, "r"); g = f[f"resolutions/{RES}"]; names = [x.decode() for x in g["chroms/name"][:]]; co = g["indexes/chrom_offset"][:]
D = MAX_D // RES + 2 * FL + 4
def band(c):
    i = names.index(c); c0, c1 = int(co[i]), int(co[i + 1]); n = c1 - c0; w = g["bins/weight"][c0:c1]
    B = np.zeros((D + 1, n)); lo, hi = int(g["indexes/bin1_offset"][c0]), int(g["indexes/bin1_offset"][c1]); step = 20_000_000
    for a0 in range(lo, hi, step):
        a = g["pixels/bin1_id"][a0:min(hi, a0 + step)] - c0; b = g["pixels/bin2_id"][a0:min(hi, a0 + step)] - c0; v = g["pixels/count"][a0:min(hi, a0 + step)]
        d = b - a; m = (d <= D) & (b < n); a, d, b, v = a[m], d[m], b[m], v[m]; B[d, a] = v * w[a] * w[b]
    ok = np.isfinite(w)
    for d in range(D + 1):
        valid = np.zeros(n, bool); valid[: n - d] = ok[: n - d] & ok[d:]; B[d, ~valid] = np.nan
    exp = np.nanmean(B, axis=1); return B, exp
res = {}
for name, L in S.items():
    obs, oe, cen = [], [], []
    byc = defaultdict(list)
    for t in L: byc[t[0]].append(t)
    for c, ts in byc.items():
        if c not in names: continue
        B, exp = band(c); n = B.shape[1]
        for _, m1, m2 in ts:
            i, j = m1 // RES, m2 // RES
            if i - FL < 0 or j + FL >= n: continue
            O = np.full((2 * FL + 1, 2 * FL + 1), np.nan); Q = np.full_like(O, np.nan)
            for a in range(-FL, FL + 1):
                for b in range(-FL, FL + 1):
                    d = (j + b) - (i + a)
                    if 0 < d <= D: O[a + FL, b + FL] = B[d, i + a]; Q[a + FL, b + FL] = B[d, i + a] / exp[d]
            obs.append(O); oe.append(Q)
    obs, oe = np.array(obs), np.array(oe)
    Om, Qm = np.nanmean(obs, 0), np.nanmean(oe, 0)
    ll = Om[-3:, :3]   # rows = upstream anchor +7..+10 bins, cols = downstream anchor -10..-7: closest to the diagonal
    res[name] = dict(n=len(obs), P2LL=Om[FL, FL] / np.nanmean(ll), centre_OE=Qm[FL, FL], centre3x3_OE=np.nanmean(Qm[FL - 1:FL + 2, FL - 1:FL + 2]),
                     median_dist_kb=np.median(dist(L)) / 1e3, Qm=Qm, Om=Om)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items() if k not in ("Qm", "Om")}, flush=True)
T = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk not in ("Qm", "Om")} for k, v in res.items()}).T
T.loc["MCC edges", "n_before_distance_filter"] = n_mcc_all
T.to_csv(OUT / "rbbp4_mcc_apa.tsv", sep="\t")
np.savez_compressed(OUT / "rbbp4_mcc_apa_pileups.npz", **{f"{k.replace(' ', '_')}_{m}": v[m] for k, v in res.items() for m in ("Qm", "Om")})
vmax = max(np.nanmax(v["Qm"]) for v in res.values()); vmin = min(np.nanmin(v["Qm"]) for v in res.values())
fig, axs = plt.subplots(1, 3, figsize=(150 / 25.4, 55 / 25.4), gridspec_kw=dict(wspace=0.25))
for ax, (name, v) in zip(axs, res.items()):
    im = ax.imshow(v["Qm"], cmap="Reds", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks([0, FL, 2 * FL]); ax.set_xticklabels(["−100 kb", "B", "+100 kb"]); ax.set_yticks([0, FL, 2 * FL]); ax.set_yticklabels(["−100 kb", "A", "+100 kb"])
    ax.set_title(f"{'RBBP4 MCC edges' if name == 'MCC edges' else 'RBBP4 ' + name if name == 'non-MCC pairs' else 'Random RBBP4 peak pairs'}\n"
                 f"n = {v['n']:,}; P2LL {v['P2LL']:.2f}; centre O/E {v['centre_OE']:.2f}", fontsize=7)
cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02); cb.set_label("Mean observed / expected", fontsize=7)
fig.suptitle("APA, WT HEK293T Hi-C 10 kb; MCC, Multi-region Coordinated Cluster; controls distance-matched", fontsize=7, x=0.01, ha="left")
for ext in ("png", "svg"): fig.savefig(OUT / f"rbbp4_mcc_apa.{ext}", dpi=300, bbox_inches="tight")
print(T.round(3).to_string())
