# m36: simple-overlap tables (no expected values on the main sheets; reference columns moved to a separate sheet).
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{R}/m36_binary_overlap/out"; S = f"{R}/m35_threshold_sweep/out"
REF = ["expected_both", "expected_recovery_pct", "expected_precision_pct", "fold_both_over_expected", "expected_pct", "fold", "pooled_fold", "median_fold_tfs_ge10", "n_tfs_fold_gt_1_2", "tfs_fold_gt_1_2", "expected_recovery_pct"]
def split(path_tsv, out_xlsx, keys):
    d = pd.read_csv(path_tsv, sep="\t"); ref = [c for c in d.columns if c in REF]
    with pd.ExcelWriter(out_xlsx) as w:
        d.drop(columns=ref).to_excel(w, sheet_name="simple_overlap", index=False); d[keys + ref].to_excel(w, sheet_name="reference_not_used", index=False)
    print(out_xlsx, "moved", ref)
split(f"{O}/overlap_per_tf.tsv", f"{O}/overlap_per_tf.xlsx", ["TF", "cutoff"])
split(f"{O}/overlap_pooled.tsv", f"{O}/overlap_pooled.xlsx", ["cutoff"])
split(f"{S}/recovery_top25_per_tf.tsv", f"{S}/recovery_top25_per_tf.xlsx", ["TF", "source", "reference_cell_line", "threshold"])
split(f"{S}/recovery_top25_pooled.tsv", f"{S}/recovery_top25_pooled.xlsx", ["set", "threshold"])
T = pd.read_csv(f"{O}/overlap_per_tf.tsv", sep="\t")
with pd.ExcelWriter(f"{O}/simple_overlap_top25.xlsx") as w:
    for c, name in (("|z| >= 2", "z2"), ("|z| >= 3", "z3")):
        g = T[T.cutoff == c].rename(columns={"ours": "our_genes", "perturbseq": "perturbseq_genes", "recovery_pct": "recovery_pct (both / Perturb-seq genes)", "precision_pct": "precision_pct (both / our genes)"})
        g = g[["TF", "our_genes", "perturbseq_genes", "both", "recovery_pct (both / Perturb-seq genes)", "precision_pct (both / our genes)"]].sort_values("both", ascending=False)
        g.to_excel(w, sheet_name=name, index=False); g.to_csv(f"{O}/simple_overlap_top25_{name}.tsv", sep="\t", index=False, float_format="%.1f")
print(pd.read_csv(f"{O}/simple_overlap_top25_z2.tsv", sep="\t").to_string(index=False))
