# m36: plain-overlap tables for the user. Main sheets use plain labels only; reference columns (expected, fold) kept on a separate sheet.
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{R}/m36_binary_overlap/out"; S = f"{R}/m35_threshold_sweep/out"
REF = ["expected_both", "expected_recovery_pct", "expected_precision_pct", "fold_both_over_expected", "expected_pct", "fold", "pooled_fold", "median_fold_tfs_ge10", "n_tfs_fold_gt_1_2", "tfs_fold_gt_1_2"]
PLAIN = {"ours": "our genes", "perturbseq": "Perturb-seq genes", "both": "in both", "ours_only": "our genes only", "perturbseq_only": "Perturb-seq genes only",
         "recovery_pct": "share of Perturb-seq genes found in our Regulome (%)", "precision_pct": "share of our Regulome genes found in Perturb-seq (%)",
         "links": "Perturb-seq genes", "found": "in both", "pct_found": "share of Perturb-seq genes found in the binding list (%)", "n_tfs_ge10_links": "TFs with >= 10 Perturb-seq genes", "ge10_links": ">= 10 Perturb-seq genes",
         "ge10_responders": ">= 10 Perturb-seq genes"}
def split(path_tsv, out_xlsx, keys):
    d = pd.read_csv(path_tsv, sep="\t"); ref = [c for c in d.columns if c in REF]
    with pd.ExcelWriter(out_xlsx) as w:
        d.drop(columns=ref).rename(columns=PLAIN).to_excel(w, sheet_name="simple_overlap", index=False); d[keys + ref].to_excel(w, sheet_name="reference_not_used", index=False)
    print(out_xlsx, list(d.drop(columns=ref).rename(columns=PLAIN).columns)[:9])
split(f"{O}/overlap_per_tf.tsv", f"{O}/overlap_per_tf.xlsx", ["TF", "cutoff"])
split(f"{O}/overlap_pooled.tsv", f"{O}/overlap_pooled.xlsx", ["cutoff"])
split(f"{S}/recovery_top25_per_tf.tsv", f"{S}/recovery_top25_per_tf.xlsx", ["TF", "source", "reference_cell_line", "threshold"])
split(f"{S}/recovery_top25_pooled.tsv", f"{S}/recovery_top25_pooled.xlsx", ["set", "threshold"])
T = pd.read_csv(f"{O}/overlap_per_tf.tsv", sep="\t"); cols = ["TF", "our genes", "Perturb-seq genes", "in both", "share of Perturb-seq genes found in our Regulome (%)", "share of our Regulome genes found in Perturb-seq (%)"]
with pd.ExcelWriter(f"{O}/simple_overlap_top25.xlsx") as w:
    for c, name in (("|z| >= 2", "z2"), ("|z| >= 3", "z3")):
        g = T[T.cutoff == c].rename(columns=PLAIN)[cols].sort_values("in both", ascending=False); g.to_excel(w, sheet_name=name, index=False); g.to_csv(f"{O}/simple_overlap_top25_{name}.tsv", sep="\t", index=False, float_format="%.1f")
print("ok")
