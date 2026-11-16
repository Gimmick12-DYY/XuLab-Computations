# m31 Amendment 1: test-split subset of the supported-edge tables. Descriptive only.
import numpy as np, pandas as pd
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"; O31 = f"{RG_}/m31_perturbseq_supported_edges/out"
GS = pd.read_csv(f"{RG_}/m7_step2/splits/gene_split_wide.tsv", sep="\t"); TEST = set(GS.gene[GS.set == "test"])
SP = pd.read_csv(f"{RG_}/m6_improve/split_v1.tsv", sep="\t"); HO = set(SP.tf[SP.set == "heldout"]); DEV = set(SP.tf[SP.set == "development"])
S = pd.read_csv(f"{O31}/supported_edges.tsv", sep="\t"); G = pd.read_csv(f"{O31}/per_gene_summary.tsv", sep="\t", usecols=["gene"]); UGt = set(G.gene) & TEST
import sys; sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])
TFL = list(TF); gi_ = {g: k for k, g in enumerate(UG)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
E = pd.read_csv(f"{RG_}/m22_edge_table/out/regulome_edges.tsv.gz", sep="\t", usecols=["tf", "gene"]); E = E[E.gene.isin(UGt)]
tmask = np.array([g in UGt for g in UG]); Nt = int(tmask.sum()); St = S[S.gene.isin(UGt)]; P = []
for i, t in enumerate(TFL):
    z = ZA[i].copy()
    for o in OWN.get(t, [t]):
        if o in gi_: z[gi_[o]] = np.nan
    resp = tmask & np.isfinite(z) & (np.abs(np.nan_to_num(z)) >= 2); b = E.gene[E.tf == t].nunique(); s = St[St.TF == t]
    P.append(dict(label="descriptive, test chromosomes", TF=t, tf_split="heldout" if t in HO else ("development" if t in DEV else "not in split"), test_genes=Nt, n_regulome_targets=b, n_responding_genes=int(resp.sum()), n_supported_edges=len(s),
                  n_down=int((s.knockdown_z < 0).sum()), n_up=int((s.knockdown_z > 0).sum()), n_reproducible=int((s.reproducible_both_halves == "yes").sum()), pct_of_all_genes_bound=100 * b / Nt, pct_of_responding_genes_bound=100 * len(s) / resp.sum() if resp.sum() else np.nan))
P = pd.DataFrame(P).sort_values("n_supported_edges", ascending=False); P.to_csv(f"{O31}/per_tf_summary_test_chrom.tsv", sep="\t", index=False, float_format="%.3g")
St[St.TF.isin(HO)].to_csv(f"{O31}/supported_edges_test.tsv", sep="\t", index=False, float_format="%.4g")
print("test-chromosome genes", Nt, "supported edges on test chrom", len(St), "of which held-out TFs", int(St.TF.isin(HO).sum()), "held-out TFs", len(HO))
h = P[P.tf_split == "heldout"]; print(h.drop(columns="label").round(1).to_string(index=False))
for nm, g in P.groupby("tf_split"): print(nm, "TFs", len(g), "median % responding bound", round(g.pct_of_responding_genes_bound.median(), 1), "median % all bound", round(g.pct_of_all_genes_bound.median(), 1))
