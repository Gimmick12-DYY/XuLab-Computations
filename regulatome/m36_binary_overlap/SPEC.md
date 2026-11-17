# m36: binary Regulome vs binary Perturb-seq, 2x2 overlap per TF (descriptive, no significance tests)

User (2026-11-17): each TF maps to genes as 1/0 in the Regulome; binarize Perturb-seq the same way and check the overlap.
- Genes: the 11,228-gene universe; TFs: the 63 with a top-25% sgRNA knockdown (m23, all >= 25 cells); own gene excluded (HP1 -> CBX1/3/5, ZUFSP -> ZUP1) and genes without a z for that TF excluded per TF.
- regulome_binary: 1 if the TF has an edge to the gene in the frozen Regulome (m22/m34 regulome_v1: one of its top 5,000 HITAG peaks in the promoter element or a canonical Hi-C ABC-linked element, ABC >= 0.02), else 0.
- perturbseq_binary: 1 if |z| >= 2 in the top-25% matrix (default); second version |z| >= 3.
- Per TF and cutoff: both (1,1), ours only (1,0), Perturb-seq only (0,1), neither (0,0); recovery = both / Perturb-seq 1s; precision = both / our 1s; expected both = our 1s x Perturb-seq 1s / genes; Jaccard = both / (both + ours only + Perturb-seq only). Pooled = sums over the 63 TFs.
- Cutoff table: |z| >= 2, 2.5, 3, 4 and top 5% of genes per TF by |z|: median responders per TF, TFs with >= 10 responders, pooled recovery and precision vs expected. No cutoff is chosen by agreement.
- Panel j: per-TF stacked bars (both / ours only / Perturb-seq only) at |z| >= 2 for the TFs with the most "both" (all TFs in the table), recovery % at the bar end; pooled summary with the expected overlap marked. CTCF* = known exception (untested).
