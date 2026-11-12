m19_binary_matrices (SPEC.md + Amendment 1, commits a7b56e7, 308ce70). Diagnostic, benchmark-developed. HEK293T.
Binary Regulome TF x gene matrix vs binary Perturb-seq responder matrix (76 TFs with an Arm 2 knockdown x 11,228 genes; own gene removed).
Versions looked at: 3 Regulome definitions (B1 any linked q005 peak; B2 promoter peak; B3 top-N by peak signal x ABC, N = 1,000, also 500 and 2,000)
x 4 responder definitions (P1 top 5% |z|; P2 |z| >= 3; P3 BH q < 0.05; P4 |z| >= 2 in both halves, same sign) x 3 directions (all, down, up) = 36 (+24 for B3 N = 500/2,000). All are reported; none selected.
responder_counts.tsv            responders per TF per P x direction (read first; < 20 = too few)
all_combinations_per_tf.tsv     every TF x version: n list, n responders, OR [95% multinomial-resample interval], Jaccard [interval], controls (swapped TF, pseudo-TF, ATAC x CpG matched genes)
summary_by_combination.tsv      per version: medians over TFs and controls
pooled_MH.tsv                   Mantel-Haenszel OR for the 10 strong-phenotype TFs and the other 66
grid_specificity.tsv            TF x TF grid: mean diagonal minus off-diagonal OR (and log OR), TF-label permutation p, own-best TFs
fig_B1xP4_strong10.png/.svg     B1 and P4 matrices for the 10 strong TFs on the union of called genes, and the OR grid
