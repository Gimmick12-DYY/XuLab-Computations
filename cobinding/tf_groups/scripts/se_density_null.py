"""Density-matched null for SE capture by cliques. Each SEdb SE (342) is matched to non-SE windows of the same size whose
total network-node count (sum over TFs of each TF's Cicero network nodes in the window) is within +-10% (min +-2) of the
SE's, same chromosome when possible (else genome-wide; tolerance widened if <50 candidates). 1,000 null sets, one window
per SE each. Statistics: fraction of SEs with clique regions of >=1/2/5/10 TFs and mean number of clique TFs per SE.
p = (#null >= observed + 1) / 1001 (one-sided). args: location_dir outdir"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/work/users/d/y/dyy12/XuLab"); LOC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); rng = np.random.default_rng(11); NS = 1000
SE = pd.read_csv(ROOT / "results_3d_model/se_sedb/HEK293T_WT_SEdb_SE_02_1951_hg38.bed", sep="\t", header=None).iloc[:, :4]; SE.columns = ["chrom", "start", "end", "se"]
SE = SE[SE.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])].reset_index(drop=True); print(len(SE), "SEs on chr1-22/X")
N = pd.read_csv(LOC / "nodes_annotated.tsv.gz", sep="\t", usecols=["tf", "chrom", "start", "end", "clique"]); N["mid"] = (N.start + N.end) // 2
tfs = sorted(N.tf.unique()); chroms = sorted(set(N.chrom) & set(SE.chrom))
allm = {c: np.sort(g.mid.values) for c, g in N.groupby("chrom")}
clq = {(t, c): np.sort(g.mid.values) for (t, c), g in N[N.clique == 1].groupby(["tf", "chrom"])}
sem = {c: g for c, g in SE.groupby("chrom")}
def count(c, s, e): m = allm.get(c, np.array([])); return np.searchsorted(m, e, side="right") - np.searchsorted(m, s)
def nclique(c, s, e):
    n = np.zeros(len(s), int)
    for t in tfs:
        m = clq.get((t, c))
        if m is not None: n += (np.searchsorted(m, e, side="right") - np.searchsorted(m, s)) > 0
    return n
def overlaps_se(c, s, e):
    if c not in sem: return np.zeros(len(s), bool)
    g = sem[c]; return np.array([((g.start.values < b) & (g.end.values > a)).any() for a, b in zip(s, e)])
def candidates(c, w, k=4000):
    m = allm[c]; mid = rng.choice(m, k) + rng.integers(-w // 2, w // 2 + 1, k); s = mid - w // 2; e = s + w
    keep = ~overlaps_se(c, s, e); s, e = s[keep], e[keep]; return c, s, e, count(c, s, e)
obs_n = np.array([nclique(c, np.array([s]), np.array([e]))[0] for c, s, e in zip(SE.chrom, SE.start, SE.end)])
obs_d = np.array([count(c, s, e) for c, s, e in zip(SE.chrom, SE.start, SE.end)])
pools, info = [], []
for i, (c, s, e) in enumerate(zip(SE.chrom, SE.start, SE.end)):
    w = e - s; d = obs_d[i]; tol = max(2, 0.1 * d); scope = "chrom"
    cc, cs, ce, cd = candidates(c, w)
    ok = np.abs(cd - d) <= tol
    if ok.sum() < 50:      # genome-wide
        scope = "genome"; parts = [candidates(x, w, 1500) for x in chroms]
        cc = np.concatenate([[p[0]] * len(p[1]) for p in parts]); cs = np.concatenate([p[1] for p in parts]); ce = np.concatenate([p[2] for p in parts]); cd = np.concatenate([p[3] for p in parts])
        ok = np.abs(cd - d) <= tol
        while ok.sum() < 50 and tol < 10 * max(d, 5): tol *= 1.5; ok = np.abs(cd - d) <= tol
    cc = np.asarray(cc) if np.ndim(cc) else np.array([c] * len(cs))
    sel = np.where(ok)[0]
    ncl = np.concatenate([nclique(x, cs[sel][cc[sel] == x], ce[sel][cc[sel] == x]) for x in np.unique(cc[sel])])
    order = np.concatenate([np.where(cc[sel] == x)[0] for x in np.unique(cc[sel])])
    pool_n = np.empty(len(sel), int); pool_n[order] = ncl
    pools.append((pool_n, cd[sel])); info.append({"se": SE.se[i], "density": d, "n_clique_TF": obs_n[i], "pool": len(sel), "scope": scope, "tol": tol,
                                                 "null_mean_clique_TF": pool_n.mean(), "null_mean_density": cd[sel].mean()})
I = pd.DataFrame(info); I.to_csv(OUT / "se_density_null_per_se.tsv", sep="\t", index=False)
null = np.array([[p[0][rng.integers(len(p[0]))] for p in pools] for _ in range(NS)])
stats = {}
for k in (1, 2, 5, 10):
    o = (obs_n >= k).mean(); nv = (null >= k).mean(1); stats[f"frac_SE_clique_ge{k}TF"] = (o, nv.mean(), np.quantile(nv, .025), np.quantile(nv, .975), (np.sum(nv >= o) + 1) / (NS + 1))
o = obs_n.mean(); nv = null.mean(1); stats["mean_clique_TFs_per_SE"] = (o, nv.mean(), np.quantile(nv, .025), np.quantile(nv, .975), (np.sum(nv >= o) + 1) / (NS + 1))
R = pd.DataFrame(stats, index=["observed", "null_mean", "null_lo95", "null_hi95", "p_one_sided"]).T; R["fold"] = R.observed / R.null_mean
R.to_csv(OUT / "se_density_matched_null.tsv", sep="\t"); print(R.round(4).to_string())
print("matching check: SE density mean", obs_d.mean().round(1), "null window density mean", I.null_mean_density.mean().round(1), "; scope", I.scope.value_counts().to_dict(), "; median pool", int(I.pool.median()))
np.save(OUT / "se_density_null_draws.npy", null)
