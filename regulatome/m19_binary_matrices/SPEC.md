# m19_binary_matrices: binary ABC target matrix vs binary Perturb-seq responder matrix (spec, 2026-11-12)

Label: diagnostic, benchmark already read. HEK293T only (our HITAG binding, canonical HEK293T ABC, X-Atlas/Orion HEK293T Perturb-seq). Written before any number is computed.

## Matrices (TF x gene, the 76 TFs with a knockdown x the 11,228-gene universe of m9; each TF's own gene removed)
ABC (binding) matrix, 1 = gene is on the TF's ABC target list:
- B1 (primary): the TF has a lab q005 MACS2 peak (strongest summit) in any canonical ABC-linked element of the gene (ABC >= 0.02, self-promoter included) = m11 long_q0.05 any_peak.
- B2: promoter element only (promoter_peak). B3: distal elements only (n_linked_with_peak > 0 and no promoter peak).
Perturb-seq (response) matrix, 1 = gene responds to the TF's knockdown (Arm 2):
- P1 (primary): |z| >= 2 in half 1 and in half 2 with the same sign (m15 raw per-TF z halves; the confirmation does not use the binding).
- P2: |z| >= 3, all cells (raw per-TF z, m9 r1).
- P3: top 5% |z| per TF, all cells (as m16).
TFs with fewer than 20 responders under a cutoff are excluded for that cutoff and listed.

## Comparisons (each B x P combination; B1 x P1 is the primary reading)
1. Per TF: 2 x 2 table, Haldane OR with 95% CI, Fisher p, BH q over TFs, Jaccard, n bound, n responders, n both.
2. Pooled over TFs: Mantel-Haenszel OR stratified by TF (Robins-Breslow-Greenland 95% CI); and stratified by TF x gene stratum (promoter ATAC decile x CpG island) to remove the general tendency of accessible genes to respond to any knockdown.
3. TF specificity: the 76 x 76 matrix of log OR(ABC list of TF i, responders of TF j); diagonal mean minus off-diagonal mean, with a permutation p (10,000 shuffles of row labels); the rank of TF i's own knockdown among the 76 columns for TF i's list; count of TFs ranked first. Same restricted to the 10 strong-phenotype TFs (m16).
4. Direction among overlapping entries: fraction of bound responders that go down on knockdown, per TF and pooled.
Results reported for all TFs, the 10 strong-phenotype TFs and the other 66 separately. Numbers are reported with their comparison; untested causes are written as unexplained.

## Amendment 1 (2026-11-12): definitions replaced by the requested design, before any number is computed
Label for everything: diagnostic, benchmark-developed. The matrices and statistics below replace sections "Matrices" and "Comparisons" above.
1. Genes: the 11,228-gene universe (>= 10% NTC detection, in the ABC gene list); the knocked-down gene removed per TF. TFs: the 76 with an Arm 2 knockdown, reported as the 10 strong-phenotype TFs (m16) and the other 66.
2. Regulome binary matrix (reused builds, m18 pair_real features, which the m20 independent recount matched exactly): B1 = q005 summit in any canonical-ABC-linked element (self-promoter included); B2 = promoter element only; B3 = top N genes by summed peak signalValue x ABC, N = 500, 1,000, 2,000 (N = 1,000 is the B3 used in counts of versions; 500 and 2,000 are also tabulated).
3. Perturb-seq binary matrix (raw per-TF z): P1 = top 5% |z|; P2 = |z| >= 3; P3 = Benjamini-Hochberg q < 0.05 on two-sided normal p from z, within the TF; P4 = same sign and |z| >= 2 in both halves of cells (m15 halves). Each as all, down-only (z < 0) and up-only (z > 0). Responder counts per TF are reported first; fewer than 20 responders = "too few", no OR.
4. Statistics: per TF, OR (Haldane) and Jaccard with 95% intervals from 2,000 multinomial resamples of the 2 x 2 counts (equivalent to a gene bootstrap); Mantel-Haenszel OR (Robins-Breslow-Greenland interval) for the 10 strong TFs and for the other 66. Grid: OR(Regulome row of TF i, Perturb-seq row of TF j) over all TF pairs with enough responders; mean diagonal OR minus mean off-diagonal OR (also on log OR), TF-label permutation p (10,000 shuffles), number of TFs whose own knockdown is their best column. Per-TF controls on the same genes: swapped TF (m7 mis0, same depth tertile; m18 pair_swap), pseudo-TF (m18 pair_pseudo0), and ATAC-decile x CpG-matched gene lists of the same size (median OR of 200 draws).
5. Reporting: one table with every combination (3 B x 4 P x 3 directions = 36 versions, plus B3 at N = 500 and 2,000); no selection. Figure for the 10 strong TFs under B1 x P4 (all directions): the two binary matrices side by side on the union of called genes, and the 10 x 10 OR grid.
