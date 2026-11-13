# m28: how similar are the ABC builds? (diagnostic, tables only)

Question (user, 2026-11-12): a simple correlation showing how similar the simple ABC model is to the others.

Builds (all ABC v1.1.2, threshold 0.02 where thresholded):
- S  = m2_abc Predictions_powerlaw (simple: ATAC only, power-law contact, tiled elements_v2)
- K  = m2b_abc_k27ac Predictions_powerlaw (S + H3K27ac in activity)
- CP = m2c_abc_canonical HEK293T_powerlaw (canonical Snakemake, MACS2 candidates, power-law)
- CH = m2c_abc_canonical HEK293T_hicfixed (canonical + Hi-C; used by the m22 Regulome)
Scores: ABC.Score from EnhancerPredictionsAllPutative.tsv.gz + EnhancerPredictionsAllPutativeNonExpressedGenes.tsv.gz (all genes), genes restricted to those present in all four.

Elements differ between builds (tiled vs MACS2), so element-gene pairs are matched through element overlap (bedtools intersect, >= 1 bp): a pair (element e of A, gene g) takes, in B, the max ABC over B elements overlapping e for gene g (0 if none).

For every pair of builds (6 pairs), report:
1. Link level, Spearman and Pearson of ABC (A's element-gene pairs, matched as above), (a) over all putative pairs, (b) over pairs with ABC >= 0.02 in A or B (the part that makes links; (a) is inflated by distance alone).
2. Thresholded links: fraction of A's links (ABC >= 0.02) that are also links in B, both directions, and Jaccard on A's element basis.
3. Gene level, Spearman across shared genes of (i) number of links >= 0.02 per gene, (ii) sum of ABC over links >= 0.02 per gene.
No null model: these are descriptive similarities. Results labelled diagnostic.
