# m43_specificity_matrix: TF x TF matrix of Perturb-seq genes found in each TF's Regulome list

Goal: show TF specificity as a matrix. A bright diagonal = a TF's own list finds its own Perturb-seq genes better than other TFs' lists do.

Inputs: top-25% sgRNA cells (m23), 63 TFs with a knockdown, 11,228 genes, own genes removed. Regulome lists: rule A (main) and rule G (second set), top 5,000 peaks, exactly as m38/m40.
Matrix: rows = knocked-down TF (its Perturb-seq genes at |z| >= c); columns = TF whose Regulome list is used. Square version: the 63 TFs in both (main figures). The 79-column version is saved as a table.
Rows kept: TFs with >= 10 Perturb-seq genes at that cutoff (n stated). Cutoffs: |z| >= 2 and 4 (main), 3 and 5 (extra).
Cell values:
- (a) raw: share of the row TF's Perturb-seq genes found in the column TF's list (%).
- (b) adjusted: (a) minus the share of the row TF's non-changing genes (|z| < 1) found in the same column's list, in percentage points (diverging, centred at 0).
- (c) extra: (a) standardised within each column (z across rows), table + figure.
- Spearman version (rule A, all 63 rows, no cutoff): Spearman correlation across genes (usable in the row TF's knockdown) between the column TF's binding score and the row TF's |z|. Binding score of a gene = max over its ABC-linked elements (ABC >= 0.02, self-promoter included) of (max signalValue of the TF's top-5,000 peak summits in that element) x ABC score; 0 if not linked.
Order: hierarchical clustering (average linkage) of the 63 TFs on Jaccard distance of their rule-A lists, computed once and reused everywhere. Broad TFs (rule-A list > 4,000 genes: RBBP4, XPA, RBBP7, NFYA, E2F3, ZNF507, DDIT3, SOX4, NME2, ZNF703) marked with a side colour bar; one version with broad TFs removed from rows and columns.
Summaries (adjusted version): median diagonal vs median off-diagonal value; per TF, rank of the diagonal within its row; TFs whose diagonal is in the top 5 of the row; with and without broad TFs.
Style: Arial, palette option C (figures_paletteC/regulome_palette.py), diagonal cells outlined thinly; paper version with TF names >= 5 pt; slide version 13.33 x 7.5 in. Descriptive, no p-values.

## Amendment 1: double standardisation (to make a diagonal easier to see)
(d) Starting from the raw share matrix (rows = knocked-down TFs kept at that cutoff, columns = the square TF set), remove the column effect (list size / breadth) and the row effect (how broad the knockdown is) by double centring: x - row mean - column mean + grand mean (iterated to convergence, median polish style with means), then divide each row by its standard deviation. A cell then reads "how much better than expected for this list and this knockdown, in row SDs". Computed for all TFs and with broad TFs removed (rows and columns), rules A and G, |z| >= 2-5. Summaries as before (median diagonal vs off-diagonal, rank of own list in its row). Figures with the same order and diagonal outline; a compact printed version for |z| >= 4 without broad TFs (square: TFs that are both rows and columns).
