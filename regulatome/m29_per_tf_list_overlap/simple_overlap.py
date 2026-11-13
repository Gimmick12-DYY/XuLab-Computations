# Plain per-TF overlap table from per_tf_overlap.tsv, cutoff C1 |z| >= 2. Diagnostic.
import pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m29_per_tf_list_overlap/out"; N = 11228
T = pd.read_csv(f"{O}/per_tf_overlap.tsv", sep="\t"); T = T[T.cutoff == "C1 |z|>=2"].copy()
S = pd.DataFrame(dict(TF=T.tf, n_regulome_targets=T.n_regulome, n_kd_responding_genes=T.n_perturb, n_shared=T.overlap,
    pct_of_responding_genes_that_are_bound=(100 * T.overlap / T.n_perturb).round(1), pct_of_bound_genes_that_respond=(100 * T.overlap / T.n_regulome).round(1),
    expected_shared=(T.n_perturb * T.n_regulome / N).round(1), fold=(T.overlap / (T.n_perturb * T.n_regulome / N)).round(2))).sort_values("n_shared", ascending=False)
f = f"{O}/simple_overlap_z2.tsv"; S.to_csv(f, sep="\t", index=False)
with open(f, "a") as h: h.write(f"# base rates over {N} genes: median % bound per TF = {100 * T.n_regulome.median() / N:.1f}; median % responding (|z| >= 2) per TF = {100 * T.n_perturb.median() / N:.1f}\n")
print(len(S))
