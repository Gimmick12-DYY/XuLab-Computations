# m46: Stage 1 with imputed binding (does imputed binding do better than lab peaks?)

User (2026-11-18): why does Stage 1 not use the imputed binding data; with padding, imputed data should do better. Test it in the Stage 1 framework (m45), nothing else changed.
Fixed as m45: HEK293T; top-25% sgRNA matrix (m23); 63 knockdown TFs; 11,228 genes; own gene removed; |z| >= 2, 3, 4, 5; link rule = bin centre in a promoter element or a canonical Hi-C ABC-linked element of the gene (m38 rule_abc), plus padding 0 / 10 kb / 50 kb around the TSS (m38 window).
Imputed input: unified imputer output unified/work/<tf>/impute/matrix_csr.npz (regions x cells, binarized, raw kept, imputed calls masked post hoc to the 104,738 HEK293T union-ATAC 1 kb bins). Per TF and bin: n positive cells; continuous score S = pseudobulk CPM of positive cells within the mask (as complex_screen/motif_shared_20261103/build_fg.py). Bin centre = bin start + 500. The number of the 63 knockdown TFs with an imputed matrix is reported.
Imputed Regulome lists, fixed here:
- I1a: all bins with >= 1% of the TF's cells positive (at least 3 cells); I1b: >= 5% of cells.
- I2: top bins by S, as many as needed for the same number of linked genes as the TF's lab-peak list at the same padding (equal list size; smallest N reaching the lab list size).
- I2loo: as I2 but ranked by the TF-specific score loo = log2((S_t + e) / (mean S of the other 78 TFs + e)) (recall/imputed_scores.npz); sensitivity.
- I3: top 5,000 bins by S.
Lab lists: m45 (top 5,000 lab MACS2 peaks), same TFs.
Per version x padding x cutoff, pooled over TF sets all 63 / without NRF1, E4F1, DNMT1 / without the broad TFs (m42/m45: padding-0 lab list > 4,000 genes): our genes, in both, recall, share of our genes found in Perturb-seq, matched gene shuffle (expression x promoter-ATAC strata, exact hypergeometric per TF and stratum, 1,000 shuffles, seed 20261118): mean, 2.5-97.5, one-sided p; and the lab lists the same way. Per TF: recall and shuffle p for lab and imputed lists. Jaccard between each TF's imputed and lab list (per padding, version). Own list vs the 5 other TFs nearest in list size (pool = that version's lists for all 79 TFs).
Verdict, stated before numbers: imputed beats lab peaks on recall at equal list size if I2 pooled recall > lab recall; against the shuffle if I2's observed-minus-shuffle-mean gap (points) is larger than lab's and p < 0.05. Context only: m9 (imputed promoter binding lost NRF1/E4F1 signal), m7 and Fig 2 audit (imputed not TF-specific).
