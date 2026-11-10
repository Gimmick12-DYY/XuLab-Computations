# Within-local-cluster percentile for the corrected tables: average-linkage tree on 1 - corrected Pearson r for each element set; the local cluster is the smallest subtree containing both TFs; the pair's corrected percentile is ranked among all TF pairs inside it. No free parameter.
import json
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
D_ = "../../repressive_chromhmm_corrected/data_min20"; SETS = {"enhancer": "enhancer_biv", "repressive": "repressive", "polycomb": "polycomb"}
def rd(p):
    m = pd.read_csv(p, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
OUT = {}
for a, b in (("EZH2", "MTF2"), ("MLX", "SREBF2")):
    OUT[f"{a}-{b}"] = {}
    for k, f in SETS.items():
        R = rd(f"{D_}/pearson_{f}.tsv"); P = rd(f"{D_}/pct_{f}.tsv").loc[R.index, R.index]; T = list(R.index); Dm = 1 - R.values; Dm = (Dm + Dm.T) / 2; np.fill_diagonal(Dm, 0); root = to_tree(linkage(squareform(Dm, checks=False), "average")); ia, ib = T.index(a), T.index(b); node = root
        while True:
            nx = next((c for c in (node.left, node.right) if c is not None and {ia, ib} <= set(c.pre_order())), None)
            if nx is None: break
            node = nx
        ids = node.pre_order(); sub = P.values[np.ix_(ids, ids)]; iu = np.triu_indices(len(ids), 1); v = sub[iu]; x = float(P.loc[a, b]); OUT[f"{a}-{b}"][k] = dict(pct=x, local=float((v <= x).mean()), n_tfs=len(ids), n_pairs=len(v), members=sorted(T[i] for i in ids), r=float(R.loc[a, b])); print(a, b, k, "pct", round(x, 3), "local", round(OUT[f"{a}-{b}"][k]["local"], 3), "TFs", len(ids), "pairs", len(v), sorted(T[i] for i in ids)[:60])
json.dump(OUT, open("fig/local_cluster.json", "w"), indent=1)
M = rd("../../scripts/complex_catalog/mats/pct_all.tsv"); print("indices 0-based in slide matrix:", {t: list(M.index).index(t) for t in ("EZH2", "MTF2", "MLX", "SREBF2")})
