"""SEdb HEK293T super-enhancers (SE_02_1951, hg38, 342) vs the co-binding network.
Capture: SEs touched (any overlap) by (a) any TF's network nodes, (b) clique regions, (c) clique regions of >=2 / >=5 TFs,
(d) shared Cicero pairs (>=2 TFs, >25 kb). Enrichment: per TF, clique vs non-clique nodes in SEs, Mantel-Haenszel over
TF x local-density decile strata (as in clique_location_density.py); plus per-SE number of TFs with a clique there vs
expected from stratum rates. args: location_dir pairs_results_dir outdir"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm, binomtest
from statsmodels.stats.contingency_tables import StratifiedTable
ROOT = Path("/work/users/d/y/dyy12/XuLab"); LOC, PR, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True)
SE = pd.read_csv(ROOT / "results_3d_model/se_sedb/HEK293T_WT_SEdb_SE_02_1951_hg38.bed", sep="\t", header=None).iloc[:, :4]
SE.columns = ["chrom", "start", "end", "se"]
def hits(df):   # index of SE overlapped by each row (-1 if none)
    out = np.full(len(df), -1)
    for c, g in SE.groupby("chrom"):
        s, e = g.start.values, g.end.values; m = (df.chrom == c).values; idx = np.where(m)[0]
        for i, a, b in zip(idx, df.start.values[m], df.end.values[m]):
            j = np.where((s < b) & (e > a))[0]
            if len(j): out[i] = g.index[j[0]]
    return out
N = pd.read_csv(LOC / "nodes_annotated.tsv.gz", sep="\t", keep_default_na=False); N["se"] = hits(N)
N["mid"] = (N.start + N.end) // 2; dens = np.zeros(len(N))
for (tf, c), idx in N.groupby(["tf", "chrom"]).groups.items():
    m = np.sort(N.mid.values[idx]); x = N.mid.values[idx]; dens[idx] = np.searchsorted(m, x + 5e5, side="right") - np.searchsorted(m, x - 5e5)
N["dens"] = dens
inSE = N[N.se >= 0]
cap = {"SEs": len(SE), "any_network_node": inSE.se.nunique(), "clique_region_any_TF": inSE[inSE.clique == 1].se.nunique()}
ntf = inSE[inSE.clique == 1].groupby("se").tf.nunique()
for k in (2, 5, 10): cap[f"clique_regions_ge{k}TF"] = int((ntf >= k).sum())
reg = pd.read_csv(PR / "regions.tsv", sep="\t"); import scipy.sparse as sp
A = sp.load_npz(PR / "inc_pair.npz").tocsc(); pairs = pd.read_csv(PR / "pairs.tsv", sep="\t"); cs = np.asarray(A.sum(0)).ravel()
rse = hits(reg); sh = cs >= 2
for lab, mask in [("cicero_pair_anchor_any_TF", cs >= 1), ("shared_pair_anchor_ge2TF", sh)]:
    a = np.concatenate([pairs.r1.values[mask], pairs.r2.values[mask]]); cap[lab] = len(set(rse[a][rse[a] >= 0]))
both = sh & (rse[pairs.r1.values] >= 0) & (rse[pairs.r2.values] >= 0) & (rse[pairs.r1.values] != rse[pairs.r2.values])
cap["shared_pairs_linking_two_SEs"] = int(both.sum())
C = pd.Series(cap); C.to_csv(OUT / "se_capture.tsv", sep="\t", header=False); print(C.to_string())
tfc = N.groupby("tf").clique.agg(["sum", "size"]); tfs = tfc.index[(tfc["sum"] >= 30) & (tfc["size"] - tfc["sum"] >= 30)]
M = N[N.tf.isin(tfs)].copy(); M["dec"] = M.groupby("tf").dens.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
M["stratum"] = M.tf + "_" + M.dec.astype(str); M["p_exp"] = M.groupby("stratum").clique.transform("mean")
def mh(df):
    tabs = []
    for _, g in df.groupby("stratum"):
        h = g.se.values >= 0; c = g.clique.values == 1
        t = np.array([[(h & c).sum(), (~h & c).sum()], [(h & ~c).sum(), (~h & ~c).sum()]])
        if t.sum(0).min() > 0 and t.sum(1).min() > 0: tabs.append(t)
    if not tabs: return np.nan, np.nan
    st = StratifiedTable(tabs); return np.log2(st.oddsratio_pooled), st.test_null_odds().pvalue
l2, p = mh(M); crude = [(M[(M.tf == t) & (M.clique == 1)].se >= 0).mean() / max((M[(M.tf == t) & (M.clique == 0)].se >= 0).mean(), 1e-9) for t in tfs]
per = pd.DataFrame([(t, *mh(M[M.tf == t]), (M[(M.tf == t)].se >= 0).sum(), (M[(M.tf == t) & (M.clique == 1)].se >= 0).sum()) for t in tfs],
                   columns=["tf", "MH_log2OR", "MH_p", "nodes_in_SE", "clique_nodes_in_SE"])
per["crude_ratio"] = crude; per.to_csv(OUT / "se_enrichment_per_tf.tsv", sep="\t", index=False)
up = (per.MH_log2OR > 0).sum(); n = per.MH_log2OR.notna().sum()
print(f"\nclique vs non-clique node in SE: crude median ratio {np.median(crude):.2f}; density-matched pooled MH log2OR {l2:.3f} (p={p:.2g}); TFs up {up}/{n} (sign p={binomtest(int(up), int(n)).pvalue:.2g})")
X = M[M.se >= 0].groupby(["se", "tf"]).agg(c=("clique", "max"), e=("p_exp", "max")).reset_index()
S = X.groupby("se").agg(n_tf_node=("tf", "size"), n_tf_clique=("c", "sum"), exp=("e", "sum"), var=("e", lambda q: (q * (1 - q)).sum()))
S["z"] = (S.n_tf_clique - S.exp) / np.sqrt(S["var"] + .25); S = S.join(SE.set_index(SE.index)).sort_values("z", ascending=False)
S.to_csv(OUT / "se_per_locus.tsv", sep="\t"); print(f"SEs with clique excess z>2: {(S.z > 2).sum()} / {len(S)}; total clique TFs obs {S.n_tf_clique.sum()} vs exp {S.exp.sum():.1f}")
print(S.head(15)[["se", "chrom", "start", "end", "n_tf_node", "n_tf_clique", "exp", "z"]].round(2).to_string())
