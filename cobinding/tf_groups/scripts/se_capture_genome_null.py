"""Capture claim: do our network peaks and cliques hit SEdb HEK293T SEs more than chance? SEs (chr1-22/X) re-placed
1,000 times as same-size windows on the same chromosome, uniformly over non-gap sequence (ChromHMM-covered bases) and not
overlapping an SE. Capture = fraction of SEs/windows holding (a) any TF's network peak, (b) a clique region of >=1/2/5/10
TFs. Also per-TF: fraction of SEs with that TF's network peak vs null. p = (#null >= obs + 1)/1001.
args: location_dir outdir"""
import sys, gzip, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/work/users/d/y/dyy12/XuLab"); LOC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); rng = np.random.default_rng(5); NS = 1000
KEEP = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
SE = pd.read_csv(ROOT / "results_3d_model/se_sedb/HEK293T_WT_SEdb_SE_02_1951_hg38.bed", sep="\t", header=None).iloc[:, :3]; SE.columns = ["chrom", "start", "end"]
SE = SE[SE.chrom.isin(KEEP)].reset_index(drop=True)
H = pd.read_csv(ROOT / "data/HEK293T_chromHMM18.bed.gz", sep="\t", header=None, usecols=[0, 1, 2]); H.columns = ["chrom", "s", "e"]
seg = {c: (g.s.values, g.e.values, np.cumsum(g.e.values - g.s.values)) for c, g in H.groupby("chrom")}
N = pd.read_csv(LOC / "nodes_annotated.tsv.gz", sep="\t", usecols=["tf", "chrom", "start", "end", "clique"]); N["mid"] = (N.start + N.end) // 2
tfs = sorted(N.tf.unique())
allm = {c: np.sort(g.mid.values) for c, g in N.groupby("chrom")}
tfm = {(t, c): np.sort(g.mid.values) for (t, c), g in N.groupby(["tf", "chrom"])}
clq = {(t, c): np.sort(g.mid.values) for (t, c), g in N[N.clique == 1].groupby(["tf", "chrom"])}
def hit(d, c, s, e):
    m = d.get(c); return np.zeros(len(s), bool) if m is None else (np.searchsorted(m, e, side="right") - np.searchsorted(m, s)) > 0
def stats(c, s, e):
    out = {"any_network_peak": hit(allm, c, s, e)}
    n = np.zeros(len(s), int)
    for t in tfs: n += hit(clq, (t, c), s, e) if False else (lambda m: np.zeros(len(s), bool) if m is None else (np.searchsorted(m, e, side="right") - np.searchsorted(m, s)) > 0)(clq.get((t, c)))
    for k in (1, 2, 5, 10): out[f"clique_ge{k}TF"] = n >= k
    pt = np.array([(lambda m: np.zeros(len(s), bool) if m is None else (np.searchsorted(m, e, side="right") - np.searchsorted(m, s)) > 0)(tfm.get((t, c))) for t in tfs])
    return out, pt
def run(W):
    acc, pts = {}, []
    for c, g in W.groupby("chrom", sort=False):
        o, pt = stats(c, g.start.values, g.end.values)
        for k, v in o.items(): acc.setdefault(k, []).append(v)
        pts.append(pt)
    return {k: np.concatenate(v).mean() for k, v in acc.items()}, np.concatenate(pts, axis=1).mean(1)
obs, obs_tf = run(SE)
sem = {c: g for c, g in SE.groupby("chrom")}
def place(c, w, k):
    S, E, C = seg[c]; out = []
    while len(out) < k:
        u = rng.integers(0, C[-1], 4 * k); i = np.searchsorted(C, u, side="right"); pos = E[i] - (C[i] - u)
        s = pos - w // 2; e = s + w; g = sem[c]
        ok = ~((g.start.values[None, :] < e[:, None]) & (g.end.values[None, :] > s[:, None])).any(1)
        out += list(zip(s[ok], e[ok]))
    return out[:k]
pre = [place(c, e - s, NS) for c, s, e in zip(SE.chrom, SE.start, SE.end)]
null, null_tf = [], []
for j in range(NS):
    W = pd.DataFrame([(c, p[j][0], p[j][1]) for c, p in zip(SE.chrom, pre)], columns=["chrom", "start", "end"])
    a, b = run(W); null.append(a); null_tf.append(b)
R = pd.DataFrame(null); res = pd.DataFrame({"observed": pd.Series(obs), "null_mean": R.mean(), "null_hi95": R.quantile(.975),
                                           "fold": pd.Series(obs) / R.mean(), "p": pd.Series({k: (np.sum(R[k] >= obs[k]) + 1) / (NS + 1) for k in obs})})
res.to_csv(OUT / "se_capture_vs_genome_null.tsv", sep="\t"); print(len(SE), "SEs"); print(res.round(4).to_string())
NT = np.array(null_tf); T = pd.DataFrame({"tf": tfs, "observed": obs_tf, "null_mean": NT.mean(0), "fold": obs_tf / np.maximum(NT.mean(0), 1e-9),
                                         "p": (np.sum(NT >= obs_tf, 0) + 1) / (NS + 1)}).sort_values("fold", ascending=False)
T.to_csv(OUT / "se_capture_per_tf_vs_genome.tsv", sep="\t", index=False)
print("TFs with p<0.05 (Bonferroni-free):", (T.p < 0.05).sum(), "of", len(T), "; median fold", T.fold.median().round(2)); print(T.round(3).head(8).to_string(index=False)); print(T.round(3).tail(5).to_string(index=False))
