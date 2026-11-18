# m42_without_broad_tfs: overlap and TF-specificity tables without the broad TFs

Question: what happens to the overlap, own-list vs other-TF, matched-gene and dose-response tables (m36/m38, m40 refs 1-2) when TFs with broad gene lists are removed?

Broad TFs (defined before numbers):
- Main: TFs whose rule-A list (top 5,000 peaks, promoter/ABC, m38) has more than 4,000 genes. The list, with sizes and whether each has a usable top-25% knockdown (among the 63), is printed and saved as out/broad_tfs.tsv.
- Sensitivity: TFs with more than 40% of their top-5,000 peak summits in promoters (inside a GeneList.TSS1kb interval of the ABC run, i.e. +/-500 bp... as defined by that file).

Setup: top-25% sgRNA cells, 63 TFs, 11,228 genes, own genes removed; rules A and G; |z| >= 2, 3, 4, 5.
Versions: (i) all TFs (must reproduce m38/m40); (ii) broad TFs removed from the evaluated TFs AND from the pool of other TFs' lists; (iii) only the broad TFs (evaluated), pool = all 79. Versions (ii) and (iii) for both definitions.
Tables:
1. Pooled simple overlap: our genes, Perturb-seq genes, in both, share of Perturb-seq genes found in our Regulome, share of our Regulome genes found in Perturb-seq, TFs and genes left.
2. Own list vs the 5 nearest-size other TFs in the pool; median rank of the own list among the pool + own (number of lists stated); TFs with own list in the top 5 and in the top 10%.
3. Changed genes vs matched not-changed genes (|z| < 1, same NTC expression decile x promoter ATAC decile) vs all not-changed genes, share bound (as m40 ref 1).
4. Dose-response: share bound by |z| bin 0-1, 1-2, 2-3, 3-4, 4-5, >= 5.
Descriptive; no p-values.
