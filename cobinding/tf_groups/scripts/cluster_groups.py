"""Cluster TFs on depth/popularity-corrected sharing (curveball z). Leiden (modularity, weights = z clipped at
ZMIN, only TFs with >= MIN items) + split-half (odd vs even chromosomes) reproducibility.
args: indir tag(e.g. pair_all or pair_all_strat) min_items [resolution]"""
import sys, numpy as np, pandas as pd, igraph as ig, leidenalg as la
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
IN = Path(sys.argv[1]); TAG = sys.argv[2]; MIN = int(sys.argv[3]); ZMIN = 3.0
LEVEL = TAG.split("_")[0]; SUFX = "_strat" if TAG.endswith("_strat") else ""
RES = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0; OTAG = TAG + (f"_r{RES:g}" if RES != 1 else "")
def load(tag, nm): return pd.read_csv(IN / f"null_{tag}_{nm}.tsv.gz", sep="\t", index_col=0)
dep = pd.read_csv(IN / f"null_{TAG}_depth.tsv", sep="\t", header=None, index_col=0)[1]
keep = dep.index[dep >= MIN]
def part(z, seeds=50):
    z = z.loc[keep, keep].values.copy(); np.fill_diagonal(z, 0); W = np.where(z > ZMIN, z, 0)
    g = ig.Graph.Weighted_Adjacency(W.tolist(), mode="undirected", attr="weight", loops=False)
    best = max((la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=RES, seed=s) for s in range(seeds)), key=lambda p: p.quality())
    co = np.zeros((len(keep), len(keep)))
    for s in range(seeds):
        m = np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=RES, seed=100 + s).membership); co += m[:, None] == m[None, :]
    return np.array(best.membership), co / seeds
z = load(TAG, "z"); m, co = part(z)
out = pd.DataFrame({"tf": keep, "group": m, "depth": dep[keep].values})
# rename groups by size, singletons -> -1
sz = out.group.value_counts(); order = {g: i + 1 for i, g in enumerate(sz[sz >= 2].index)}
out["group"] = out.group.map(lambda g: order.get(g, 0))
res = {}
if (IN / f"null_{LEVEL}_odd{SUFX}_z.tsv.gz").exists():
    zo, ze = load(f"{LEVEL}_odd{SUFX}", "z").loc[keep, keep], load(f"{LEVEL}_even{SUFX}", "z").loc[keep, keep]
    iu = np.triu_indices(len(keep), 1)
    res["split_half_pearson_z"] = np.corrcoef(zo.values[iu], ze.values[iu])[0, 1]
    mo, _ = part(zo); me, _ = part(ze)
    res["split_half_ARI"] = adjusted_rand_score(mo, me); res["ARI_odd_vs_full"] = adjusted_rand_score(mo, m); res["ARI_even_vs_full"] = adjusted_rand_score(me, m)
    out["odd_group"] = mo; out["even_group"] = me
# per-TF stability = mean co-assignment with own group mates across Leiden seeds
out["seed_stability"] = [co[i, (out.group.values == out.group.values[i]) & (np.arange(len(out)) != i)].mean() if out.group.values[i] else np.nan for i in range(len(out))]
out.sort_values(["group", "depth"], ascending=[True, False]).to_csv(IN / f"groups_{OTAG}.tsv", sep="\t", index=False)
pd.Series(res).to_csv(IN / f"groups_{OTAG}_reproducibility.tsv", sep="\t", header=False)
print(res); print(out.groupby("group").tf.apply(lambda s: ",".join(s)).to_string())
