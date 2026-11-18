# m41_partner_added_genes: do a TF's partners add better genes than non-partners, per added gene?

Why: the m40 partner comparison used total coverage. A real partner is picked by binding similarity, so it overlaps the TF's own list more and adds fewer new genes than a non-partner of the same list size; broad TFs give coverage to anyone; added genes can reflect open chromatin and depth. So compare per added gene.

Setup: top-25% sgRNA cells (m23), 63 TFs with a knockdown, 11,228 genes, own genes removed. Our lists: rules A and G, exactly as m38 (top 5,000 peaks). Perturb-seq genes of TF X: |z| >= 2 and >= 3 in X's knockdown. Partner sets: P1 Figure 4 trio, P2 AF3-supported (strong), P4 top 1 and top 3 by co-binding (as m39). Pairs: X in the 63 TFs, P any of the 79 with a list, P != X.
Non-partner pool for X: the 79 TFs minus X and minus all X's partners in that set. 100 draws per pair, seed 20261118. Draw schemes:
- (a) same list size (+/-20%, else the 5 nearest by size), as in m40;
- (b) matched on number of genes added to X (+/-20%, else the 5 nearest);
- (c) as (b), excluding the 9 broadest TFs (list > 4,000 genes under rule A) from the pool.
Real-partner pairs are also reported split by whether the partner is one of the 9 broad TFs.

1. Overlap: Jaccard of P and X lists, share of P's list already in X's list, number of genes added (in P, not in X, usable in X's knockdown). Real vs scheme (a).
2. Added genes: share that are X's Perturb-seq genes; real vs (a), (b), (c). Reference: share among X's own genes, and among genes bound by neither X nor P.
3. Same as 2 but against matched genes: for each added gene, the rate of X's Perturb-seq genes among genes bound by neither X nor P in the same stratum (NTC expression decile x promoter ATAC decile, as m40). Report observed, matched, and observed minus matched.
4. Shared targets: genes bound by X and P vs genes bound by X only: share of X's Perturb-seq genes. Non-partners matched on number of shared genes (+/-20%, else 5 nearest).
5. Co-regulation (P with a usable knockdown, non-partner pool also restricted to the 63 with a knockdown, scheme (b)): among added genes, share that change on P's knockdown, share that change on both, and sign agreement of the two knockdowns among genes changed in both.
Pooled = sum over pairs / sum over pairs; non-partners as mean of draws and 5th-95th range. Per-pair tables for P1 and P2. Plain wording; no p-values in the headline.

## Amendment 1 (co-regulation, part 5 only)
Concern: AF3 partners with a knockdown are dominated by RBBP7 and XPA, whose knockdowns change many genes, so "changed on partner's knockdown" can be higher for that reason alone.
Redo part 5 for P1 (trio) and P2 (AF3 pairs), |z| >= 2 and 3, rules A and G (script m41_coreg.py). Added genes = in P's list, not in X's list, usable in both knockdowns.
Non-partners (100 draws per pair, seed 20261118): TFs with a knockdown, not X, not X's partners, same broad / non-broad class as P (broad = rule A list > 4,000 genes), number of Perturb-seq genes within +/-20% of P's, and number of added genes within +/-20% of P's. If none: drop the added-genes condition; if still none: the 5 nearest in Perturb-seq gene count within the class.
Report pooled (sum over pairs): share of added genes changed on P's knockdown, share changed on both, same-direction share; non-partners as mean and 5th-95th of draws. Subsets: all pairs; pairs involving RBBP7 removed; pairs with a broad partner removed.
Per pair: observed "changed on both" count beside expected = added x (share changed on X's knockdown) x (share changed on P's knockdown) within the added genes, and observed / expected. Pooled observed / expected also given (sum / sum).
List the AF3 pairs with a partner knockdown and their partners.
