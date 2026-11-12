# Count frozen candidates (1,732) with >= 1 motif enriched in both TFs, at several per-TF enrichment levels, no pair-level test.
# Per-TF enrichment = motif_valid fisher z (FIMO presence, lab summits top 1,000 in open chromatin vs matched background), discovery (odd chr);
# "+held-out" also requires z > 0 in both on even + X. Generic = z >= 3 in > 10 of 79 TFs; Codebook artifact-like flagged separately.
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; G = f"{X}/complex_screen/pair_motif_composite/motif_valid"
Fd = pd.read_csv(f"{G}/scores_fisher_disc.tsv", sep="\t", index_col=0); Fv = pd.read_csv(f"{G}/scores_fisher_val.tsv", sep="\t", index_col=0); Pd = pd.read_csv(f"{G}/scores_panel_disc.tsv", sep="\t", index_col=0)
gen = (Fd >= 3).sum(1) > 10
FZ = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); FZ = FZ[FZ.candidate.astype(str) == "True"]
FZ = FZ[FZ.tf_a.isin(Fd.columns) & FZ.tf_b.isin(Fd.columns)]; rows = []
for thr in [2, 3, 4, 5]:
    for ng in [False, True]:
        for ho in [False, True]:
            n = []; 
            for a, b in zip(FZ.tf_a, FZ.tf_b):
                s = (Fd[a] >= thr) & (Fd[b] >= thr)
                if ng: s &= ~gen
                if ho: s &= (Fv[a] > 0) & (Fv[b] > 0)
                n.append(int(s.sum()))
            n = np.array(n); rows.append(dict(z_both=f">={thr}", generic_excluded=ng, heldout_same_sign=ho, pairs_with_shared=int((n > 0).sum()), pct=round((n > 0).mean() * 100, 1), median_n_shared_if_any=float(np.median(n[n > 0])) if (n > 0).any() else 0))
T = pd.DataFrame(rows); print(len(FZ), "candidates scored"); print(T.to_string(index=False)); T.to_csv(f"{G}/shared_counts_by_level.tsv", sep="\t", index=False)
print("generic motifs:", int(gen.sum()), "e.g.", list(Fd.index[gen][:12]))
