# Per-panel-TF guide pairs (library) and HEK293T cell counts (guide calls table)
import pandas as pd, sys
panel = pd.read_csv(sys.argv[1], sep="\t")
lib = pd.read_csv(sys.argv[2])
calls = pd.read_csv(sys.argv[3], usecols=["cell_barcode", "feature_call", "gene_target", "pass_guide_filter"])
calls = calls[calls.pass_guide_filter.astype(str) == "True"]
print("HEK293T cells passing guide filter:", len(calls), "targets:", calls.gene_target.nunique())
lib_p = lib.groupby("target_gene_name").agg(n_guide_pairs=("id_ab", "nunique"))
lib_p["n_guides_library"] = lib.groupby("target_gene_name").apply(lambda d: len(set(d.id_a) | set(d.id_b)))
c = calls.groupby("gene_target").agg(n_cells=("cell_barcode", "size"), n_guide_pairs_detected=("feature_call", "nunique"))
out = panel.merge(lib_p, left_on="gene_symbol", right_index=True, how="left").merge(c, left_on="gene_symbol", right_index=True, how="left")
out["n_cells"] = out.n_cells.fillna(0).astype(int)
out["present_hek293t"] = out.n_cells > 0
out.to_csv(sys.argv[4], sep="\t", index=False)
print(out.to_string())
print("NTC cells:", (calls.gene_target == "Non-Targeting").sum())
