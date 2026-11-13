# How GO comparisons across gene lists are usually shown (2026-11-13)

1. clusterProfiler compareCluster dot plot (Yu et al. 2012 OMICS; Wu et al. 2021 Innovation): one plot, gene lists as columns,
   GO terms as rows (one shared axis), dot size = gene ratio, colour = adjusted p; only enriched cells drawn; redundant terms
   removed with simplify(). -> A_comparecluster_dotplot.png (our data).
2. Metascape multi-list heatmap (Zhou et al. 2019 Nat Commun): terms x lists heat map of -log10 p, clustered; shows terms shared
   by all lists vs selective to one list. -> B_metascape_heatmap.png (our data).
3. EnrichmentMap networks (Reimand et al. 2019 Nat Protoc): terms as nodes, edges = shared genes, node slices coloured by list;
   used for many terms / many lists.
4. Redundancy reduction before plotting: REVIGO (Supek et al. 2011 PLoS ONE) or clusterProfiler::simplify.
5. Background matters: use the genes that could have been selected (here: genes with a promoter peak in any TF's network), not the
   whole genome; Timmons et al. 2015 Genome Biol show that a mismatched reference list creates spurious enrichment.
Script: scripts/j_layout_examples.py (reads j_gene_set_dotplot_by_tf_count.tsv).
