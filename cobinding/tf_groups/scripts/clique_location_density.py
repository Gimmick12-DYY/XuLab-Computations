"""Density-matched version of clique_location.py. A clique needs >=3 mutually linked nodes within 1 Mb, so nodes in
node-dense (gene-dense) neighbourhoods form cliques more easily. Each node gets its TF's local density (same-TF network
nodes within +-500 kb) and is stratified by TF x density decile; expected clique probability = stratum rate.
(1) chromatin class / A-B: Mantel-Haenszel OR over TF x decile strata (+ per-TF MH OR, sign test);
(2) 1 Mb hotspots and (3) promoter genes re-scored with the stratum expectations.
args: outdir (holding nodes_annotated.tsv.gz from clique_location.py)"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable
OUT = Path(sys.argv[1]); N = pd.read_csv(OUT / "nodes_annotated.tsv.gz", sep="\t", keep_default_na=False)
N["mid"] = (N.start + N.end) // 2; dens = np.zeros(len(N))
for (tf, c), idx in N.groupby(["tf", "chrom"]).groups.items():
    m = np.sort(N.mid.values[idx]); x = N.mid.values[idx]
    dens[idx] = np.searchsorted(m, x + 500_000, side="right") - np.searchsorted(m, x - 500_000)
N["dens"] = dens
tfc = N.groupby("tf").clique.agg(["sum", "size"]); tfs = tfc.index[(tfc["sum"] >= 30) & (tfc["size"] - tfc["sum"] >= 30)]
N = N[N.tf.isin(tfs)].copy()
N["dec"] = N.groupby("tf").dens.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
N["stratum"] = N.tf + "_" + N.dec.astype(str); N["p_exp"] = N.groupby("stratum").clique.transform("mean")
print("clique rate by density decile (median over TFs):", N.groupby(["tf", "dec"]).clique.mean().groupby("dec").median().round(3).tolist())
def mh(df, hit):
    tabs = []
    for _, g in df.groupby("stratum"):
        h = hit[g.index]; c = g.clique.values == 1
        t = np.array([[(h & c).sum(), (~h & c).sum()], [(h & ~c).sum(), (~h & ~c).sum()]])
        if t.sum(0).min() > 0 and t.sum(1).min() > 0: tabs.append(t)
    if not tabs: return np.nan, np.nan
    st = StratifiedTable(tabs); return np.log2(st.oddsratio_pooled), st.test_null_odds().pvalue
res, per = [], []
for cat, col in [(c, "cls") for c in sorted(N.cls.unique())] + [("A compartment", "AB")]:
    D = N[N.AB != "NA"] if col == "AB" else N
    hit = (D.AB == "A") if col == "AB" else (D.cls == cat)
    l2, p = mh(D, hit)
    pt = [mh(D[D.tf == t], hit[D.tf == t])[0] for t in tfs]; pt = np.array([v for v in pt if v == v])
    res.append({"category": cat, "MH_log2OR_pooled": l2, "MH_p": p, "median_tf_log2OR": np.median(pt), "n_tf_up": int((pt > 0).sum()), "n_tf": len(pt),
                "sign_p": binomtest(int((pt > 0).sum()), len(pt)).pvalue})
R = pd.DataFrame(res).sort_values("MH_log2OR_pooled", ascending=False); R.to_csv(OUT / "class_enrichment_density_matched.tsv", sep="\t", index=False)
print(R.round(4).to_string(index=False))
N["bin"] = N.chrom + ":" + (N.mid // 1_000_000).astype(str)
g = N.groupby(["bin", "tf"]).agg(obs=("clique", "sum"), exp=("p_exp", "sum"), var=("p_exp", lambda p: (p * (1 - p)).sum())).reset_index()
g["z"] = (g.obs - g.exp) / np.sqrt(g["var"] + 0.25); g["sig"] = norm.sf(g.z) < 0.01
H = g.groupby("bin").agg(n_tf_present=("tf", "size"), n_tf_sig=("sig", "sum"), obs=("obs", "sum"), exp=("exp", "sum"), mean_z=("z", "mean"))
H = H[H.n_tf_present >= 10]; H["p_bin"] = [binomtest(int(k), int(n), 0.01, alternative="greater").pvalue for k, n in zip(H.n_tf_sig, H.n_tf_present)]
H["q_bin"] = multipletests(H.p_bin, method="fdr_bh")[1]
genes = N[N.clique == 1].groupby("bin").gene.apply(lambda s: ",".join(sorted(set(x for v in s if v for x in v.split(",")))[:20]))
H["clique_promoter_genes"] = genes.reindex(H.index).fillna(""); H = H.sort_values(["q_bin", "mean_z"], ascending=[True, False])
H.to_csv(OUT / "hotspots_1Mb_density_matched.tsv", sep="\t")
print((H.q_bin < 0.05).sum(), "of", len(H), "bins q<0.05 after density matching"); print(H.head(20).drop(columns="p_bin").round(3).to_string())
P = N[N.gene != ""].assign(gene=lambda d: d.gene.str.split(",")).explode("gene").groupby(["gene", "tf"]).agg(c=("clique", "max"), e=("p_exp", "max")).reset_index()
GG = P.groupby("gene").agg(n_tf_node=("tf", "size"), n_tf_clique=("c", "sum"), exp=("e", "sum"), var=("e", lambda p: (p * (1 - p)).sum()))
GG["z"] = (GG.n_tf_clique - GG.exp) / np.sqrt(GG["var"] + 0.25); GG["p"] = norm.sf(GG.z); GG["q"] = multipletests(GG.p, method="fdr_bh")[1]
GG.sort_values("p").to_csv(OUT / "promoter_genes_clique_excess_density_matched.tsv", sep="\t")
(OUT / "go").mkdir(exist_ok=True); (OUT / "go/clique_excess_dm_q05.txt").write_text("\n".join(GG.index[GG.q < 0.05]) + "\n")
(OUT / "go/clique_excess_dm_top200.txt").write_text("\n".join(GG.sort_values("p").index[:200]) + "\n")
print((GG.q < 0.05).sum(), "genes q<0.05"); print(GG.sort_values("p").head(25).round(3).to_string())
