# usage: gene2tfs.py GENE  -> out/lookup_gene_<GENE>.tsv: every TF bound at GENE in the Regulome edge table (our data), ordered by predicted-impact share, with external knockdown columns beside it.
import sys, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m22_edge_table/out"; g = sys.argv[1]
E = pd.read_csv(f"{O}/regulome_edges.tsv.gz", sep="\t"); d = E[E.gene == g].sort_values("predicted_impact_share_at_gene", ascending=False)
d.to_csv(f"{O}/lookup_gene_{g}.tsv", sep="\t", index=False); print(len(d), "TFs bound at", g, "->", f"{O}/lookup_gene_{g}.tsv")
