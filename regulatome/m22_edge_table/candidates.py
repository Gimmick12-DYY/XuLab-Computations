# m22 Amendment 4, step 5: candidate example genes (table only). Genes with >= 3 bound TFs (our data) where a bound TF has a reproducible external knockdown effect.
import pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m22_edge_table/out"; RP = "observed_kd_reproducible [external]"; ZC = "observed_kd_z [external: X-Atlas/Orion HEK293T]"
E = pd.read_csv(f"{O}/regulome_edges.tsv.gz", sep="\t", low_memory=False); E[RP] = E[RP].fillna(False).astype(bool)
nb = E.groupby("gene").tf.size(); g3 = set(nb[nb >= 3].index); hit = E[E.gene.isin(g3) & E[RP]]
cols = ["gene", "tf", "promoter_peak", "n_distal_elements_with_peak", "score_signal_x_abc", "within_tf_percentile", "predicted_impact_share_at_gene", "shared_by", ZC, "observed_kd_z_half1 [external]", "observed_kd_z_half2 [external]", "tier"]
C = hit[cols].copy(); C["n_tfs_bound_at_gene"] = C.gene.map(nb); C["n_reproducible_bound_tfs_at_gene"] = C.gene.map(hit.groupby("gene").size())
C = C.sort_values(["n_reproducible_bound_tfs_at_gene", "within_tf_percentile"], ascending=[False, False]); C.to_csv(f"{O}/candidate_examples.tsv", sep="\t", index=False, float_format="%.4g")
print("candidate rows", len(C), "genes", C.gene.nunique(), "TFs", C.tf.nunique()); print(C.tf.value_counts().head(10).to_string()); print(C.head(25)[["gene", "tf", "promoter_peak", "within_tf_percentile", "shared_by", ZC, "n_tfs_bound_at_gene", "n_reproducible_bound_tfs_at_gene"]].round(2).to_string(index=False))
