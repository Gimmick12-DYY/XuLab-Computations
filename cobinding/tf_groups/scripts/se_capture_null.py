"""Null for SE capture: the 342 SEdb SEs re-placed (same size, same chromosome) centred on random active ChromHMM
segments (TssA, EnhA1/2, EnhG1/2), 200 times; capture = SEs/windows holding a clique region of >=1/2/5/10 TFs.
args: location_dir outdir"""
import sys, gzip, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/work/users/d/y/dyy12/XuLab"); LOC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); rng = np.random.default_rng(3)
SE = pd.read_csv(ROOT / "results_3d_model/se_sedb/HEK293T_WT_SEdb_SE_02_1951_hg38.bed", sep="\t", header=None).iloc[:, :3]; SE.columns = ["chrom", "start", "end"]
H = pd.read_csv(ROOT / "data/HEK293T_chromHMM18.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 3]); H.columns = ["chrom", "s", "e", "st"]
H = H[H.st.isin(["TssA", "EnhA1", "EnhA2", "EnhG1", "EnhG2"])]; Hm = {c: ((g.s + g.e) // 2).values for c, g in H.groupby("chrom")}
N = pd.read_csv(LOC / "nodes_annotated.tsv.gz", sep="\t", usecols=["tf", "chrom", "start", "end", "clique"]); N = N[N.clique == 1]
idx = {c: (np.sort(((g.start + g.end) // 2).values), g.assign(m=(g.start + g.end) // 2).sort_values("m").tf.values) for c, g in N.groupby("chrom")}
def capture(W):
    n = []
    for c, s, e in zip(W.chrom, W.start, W.end):
        if c not in idx: n.append(0); continue
        m, t = idx[c]; n.append(len(set(t[np.searchsorted(m, s):np.searchsorted(m, e, side="right")])))
    n = np.array(n); return {k: (n >= k).mean() for k in (1, 2, 5, 10)}
obs = capture(SE); null = []
for _ in range(200):
    rows = []
    for c, s, e in zip(SE.chrom, SE.start, SE.end):
        mid = rng.choice(Hm[c]) if c in Hm else (s + e) // 2; w = e - s; rows.append((c, mid - w // 2, mid - w // 2 + w))
    null.append(capture(pd.DataFrame(rows, columns=["chrom", "start", "end"])))
R = pd.DataFrame(null); res = pd.DataFrame({"observed": pd.Series(obs), "null_mean": R.mean(), "null_p95": R.quantile(.95), "emp_p": pd.Series({k: (R[k] >= obs[k]).mean() for k in obs})})
res.index = [f"clique_ge{k}TF" for k in obs]; res.to_csv(OUT / "se_capture_vs_active_null.tsv", sep="\t"); print(res.round(3).to_string())
