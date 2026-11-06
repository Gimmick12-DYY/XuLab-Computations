"""Which Cicero pairs (>25 kb) are shared by more TFs than expected if TFs called pairs independently?
For each shared pair and TF t: tested_t = t has a Cicero input peak overlapping both ends (within the 1 Mb window);
chance rate p_t = t's called pairs / t's tested links (Cicero n_links). Expected TFs = sum_t tested_t * p_t.
P(observed or more) from the Poisson-binomial distribution; Bonferroni over all tested links (conservative)."""
import glob, gzip
from pathlib import Path
import numpy as np, pandas as pd
S = pd.read_csv("shared_pairs.tsv", sep="\t")
info = pd.read_csv("cicero_info_all.tsv", sep="\t", header=None, index_col=0, names=["tf", "n_links", "n_peaks"])
called = {}
for f in glob.glob("pairs/**/*.peak_edges.tsv", recursive=True):
    tf = Path(f).name.split(".")[0]
    if tf not in called:
        d = pd.read_csv(f, sep="\t", usecols=["pval"]); called[tf] = int((d.pval <= 0.05).sum())
def iv(r):
    r = r.strip()
    if ":" in r:
        c, se = r.split(":"); s, e = se.split("-")
    else:
        c, s, e = r.split("_")
    return c, int(s), int(e)
peaks = {}
for f in glob.glob("regs/*/cicero/mm/regions.tsv.gz"):
    tf = f.split("/")[1]; d = {}
    with gzip.open(f, "rt") as fh:
        for ln in fh:
            if ln.strip():
                c, s, e = iv(ln); d.setdefault(c, []).append((s, e))
    peaks[tf] = {c: (np.array(sorted(v))[:, 0], np.array(sorted(v))[:, 1]) for c, v in d.items()}
tfs = [t for t in peaks if t in info.index and t in called and info.loc[t, "n_links"] > 0]
p = {t: called[t] / info.loc[t, "n_links"] for t in tfs}
def has(t, c, s, e):
    if c not in peaks[t]: return False
    st, en = peaks[t][c]; i = np.searchsorted(st, e)
    return i > 0 and (en[max(0, i - 20):i] > s).any()
def pb_tail(ps, k):  # P(K >= k), Poisson-binomial by DP
    dist = np.zeros(len(ps) + 1); dist[0] = 1
    for q in ps:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q; dist[0] *= (1 - q)
    return dist[k:].sum()
M = sum(info.loc[t, "n_links"] for t in tfs)
rows = []
for r in S.itertuples():
    c1, s1, e1 = iv(r.region1); c2, s2, e2 = iv(r.region2)
    obs = [t for t in r.tfs.split(",") if t in p]
    tested = [t for t in tfs if has(t, c1, s1, e1) and has(t, c2, s2, e2)]
    ps = [p[t] for t in tested]
    k = len(obs)
    pv = pb_tail(np.array(ps), k) if k <= len(ps) else 0.0
    rows.append({"region1": r.region1, "region2": r.region2, "distance_kb": r.distance_kb, "n_tfs": r.n_tfs, "n_tfs_tested": len(tested),
                 "expected_tfs": round(sum(ps), 3), "fold": round(k / max(sum(ps), 1e-9), 1), "p": pv, "p_bonf": min(1.0, pv * M), "tfs": r.tfs})
R = pd.DataFrame(rows).sort_values(["p", "n_tfs"], ascending=[True, False])
R.to_csv("pair_concurrency.tsv", sep="\t", index=False)
print("TFs in model", len(tfs), "; tested links total", M)
print("pairs with Bonferroni p < 0.05:", (R.p_bonf < 0.05).sum(), "of", len(R))
print(R.head(20)[["region1", "region2", "distance_kb", "n_tfs", "n_tfs_tested", "expected_tfs", "fold", "p_bonf"]].to_string(index=False))
