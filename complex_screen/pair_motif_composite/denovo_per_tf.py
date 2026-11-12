# Novel / TF-specific de novo motifs per TF from the ATAC-masked raw run (per_tf_raw; HOMER de novo len 8/10/12 on odd chromosomes, fold >= 1.5,
# p <= 1e-10, families by similarity >= 0.6). A family counts for TF t when: t has a member; held-out (even + X) FIMO enrichment in t's sites
# vs its matched background fold > 1 with p < 0.05; present in <= 3 TFs (TF-specific); no Tomtom match to the known DB (novel) or a match
# (known); not matching the full Codebook artifact list. Output motif_tables/denovo_per_tf.tsv, motif_tables/denovo_specific_families.tsv
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"; G = f"{O}/per_tf_raw"
F = pd.read_csv(f"{G}/denovo_families.tsv", sep="\t"); A = pd.read_csv(f"{G}/denovo_families_codebook_full.tsv", sep="\t").set_index("family"); V = pd.read_csv(f"{G}/family_val.tsv", sep="\t")
F["artifact"] = F.family.map(A.art).notna(); ok = {(t, k): (fo > 1 and p < .05, fo, p) for t, k, fo, p in zip(V.tf, V.family, V.fold, V.p)}
cells = pd.read_csv("/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
rows, fam = [], []
for t in cells.index:
    mine = F[F.tfs.str.split(",").apply(lambda L: t in L)]
    r = dict(tf=t, cells=int(cells[t]), n_families=len(mine))
    v = mine[np.array([ok.get((t, k), (False,))[0] for k in mine.family], dtype=bool)]; r["n_heldout"] = len(v)
    s = v[(v.n_tf <= 3) & ~v.artifact]; r["n_specific_heldout"] = len(s); r["n_specific_heldout_novel"] = int(s.novel.sum()); r["n_specific_heldout_known"] = int((~s.novel).sum())
    for x in s.itertuples(): fam.append(dict(tf=t, family=x.family, consensus=x.cons, n_tf=x.n_tf, tfs=x.tfs, novel=x.novel, db_match=x.guess, heldout_fold=round(ok[(t, x.family)][1], 2), heldout_p=float(ok[(t, x.family)][2])))
    rows.append(r)
R = pd.DataFrame(rows).sort_values("n_specific_heldout", ascending=False); R.to_csv(f"{O}/motif_tables/denovo_per_tf.tsv", sep="\t", index=False)
Fm = pd.DataFrame(fam).sort_values("heldout_p"); Fm.to_csv(f"{O}/motif_tables/denovo_specific_families.tsv", sep="\t", index=False)
print("families total", len(F), "| held-out pass any TF", len({k for (t, k), v in ok.items() if v[0]}))
print("TFs with >= 1 TF-specific (<=3 TFs) held-out de novo family:", int((R.n_specific_heldout > 0).sum()), "of 79; novel:", int((R.n_specific_heldout_novel > 0).sum()))
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60); print(R.head(25).to_string(index=False)); print(Fm.head(40).to_string(index=False))
# multiple-testing context: expected false passes of the held-out test at p<0.05 across TF x family pairs considered
n_tests = sum(1 for t in cells.index for k in F[F.tfs.str.split(",").apply(lambda L: t in L)].family); print("TF x own-family held-out tests:", n_tests, "expected chance passes at p<0.05 if null:", round(0.05 * n_tests, 1), "observed:", int(R.n_heldout.sum()))
