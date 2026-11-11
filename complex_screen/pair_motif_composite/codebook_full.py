# Match de novo families (per_tf_raw, per_tf_matched) against the full Codebook artifact list (37 motifs: 18 non-endogenous + NFI, SP/KLF,
# ETS, nuclear-receptor "endogenous HEK293 TF" categories). Output <dir>/denovo_families_codebook_full.tsv
import pandas as pd, subprocess, sys
O = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"; F0 = "/work/users/d/y/dyy12/XuLab/motif_analysis/cache/motifdb/codebook_artifacts.meme"
for d in ["per_tf_raw", "per_tf_matched"]:
    tt = subprocess.run(f"module load meme/5.5.7 >/dev/null 2>&1; tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 0.05 {O}/{d}/denovo_cmp/reps.meme {F0}", shell=True, capture_output=True, text=True, executable="/bin/bash").stdout
    open("/tmp/dyy_tt.tsv", "w").write(tt); t = pd.read_csv("/tmp/dyy_tt.tsv", sep="\t", comment="#").sort_values("q-value").drop_duplicates("Query_ID").set_index("Query_ID")
    F = pd.read_csv(f"{O}/{d}/denovo_families.tsv", sep="\t"); F["art"] = ("fam" + F.family.astype(str)).map(t["Target_ID"]); F["cat"] = F.art.str.replace(r"_motif-\d+", "", regex=True); F["generic"] = F.n_tf > 10
    endo = F.cat.str.contains("endogenous", na=False)
    print(f"== {d}: families {len(F)}; match any Codebook artifact {int(F.art.notna().sum())} (endogenous-TF categories {int(endo.sum())}, non-endogenous {int((F.art.notna() & ~endo).sum())})")
    print(f"   generic (>10 TFs) {int(F.generic.sum())}: matched {int((F.generic & F.art.notna()).sum())}; non-generic {int((~F.generic).sum())}: matched {int((~F.generic & F.art.notna()).sum())}")
    print(F[F.generic].sort_values("n_tf", ascending=False).head(15)[["n_tf", "cons", "novel", "guess", "cat"]].to_string(index=False))
    C = pd.read_csv(f"{O}/{d}/shared_motif_candidates.tsv", sep="\t"); s = C[C.emp_p <= .05]; cons = set(F.cons[F.art.notna()])
    print("   candidates emp<=0.05:", len(s), "with a shared word on the full list:", int(s.heldout_nongeneric.fillna("").apply(lambda x: any(c + "(" in x for c in cons)).sum()))
    F[["family", "n_tf", "cons", "novel", "guess", "art"]].to_csv(f"{O}/{d}/denovo_families_codebook_full.tsv", sep="\t", index=False)
