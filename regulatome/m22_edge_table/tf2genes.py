# usage: tf2genes.py TF  -> out/lookup_tf_<TF>.tsv: every target gene of TF in the Regulome edge table (our data), with external knockdown columns beside it.
import sys, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m22_edge_table/out"; t = sys.argv[1]
E = pd.read_csv(f"{O}/regulome_edges.tsv.gz", sep="\t"); d = E[E.tf == t].sort_values("predicted_impact", ascending=False)
d.to_csv(f"{O}/lookup_tf_{t}.tsv", sep="\t", index=False); print(len(d), "target genes of", t, "->", f"{O}/lookup_tf_{t}.tsv")
