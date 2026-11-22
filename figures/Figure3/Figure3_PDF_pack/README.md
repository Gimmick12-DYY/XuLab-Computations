# Figure 3 – editable PDF pack (2026-11-18)

Final panels as single-page PDFs for Adobe Illustrator: text is live (embedded TrueType Arial, editable), shapes are vector.
Rebuilt 2026-11-21 directly by matplotlib from the panel scripts in Longleaf `XuLab/paper_figures/figure3/` (no SVG conversion). The unlettered copies (clique_pairs_by_genomic_class, go_terms, module_edges_hic_support, rbbp4_chromosomes, rbbp4_network, shared_legend_for_3D_structure, super_enhancer_*, top_shared_clique_edges_chord) were replaced with the same new panels. a is unchanged.

| File | Panel |
|---|---|
| a_coordinated_binding_schematic.pdf | a – what coordinated binding is (schematic) |
| b1_rbbp4_network.pdf | b (option 1) – RBBP4 network |
| b2_rbbp4_hic_circos.pdf | b (option 1) – RBBP4 Hi-C loop circos |
| b_rbbp4_characterization.pdf | b (option 2) – RBBP4 pairs/module sizes with Hi-C support + pair classes |
| b_rbbp4_pair_classes.pdf | b – RBBP4 pair classes (P–P, P–E, E–E, P–R, R–R, other): 100% stacked bars for RBBP4 pairs, RBBP4 module edges, all TFs (68 TFs with pairs, pooled) and Hi-C loops |
| c_rbbp4_chromosomes.pdf | c – RBBP4 pairs and module edges along all chromosomes, supporting loops |
| d_module_edges_hic_support.pdf | d – module edges in 1 Mb bins + modules with loop edges + shuffle null |
| e_module_genomic_class.pdf | e – genomic class of module regions |
| f_top_shared_edges_chord.pdf | f – most widely shared module edges (chord) |
| g_se_browser.pdf | g – super-enhancer hub browser view |
| h_se_3d_structure.pdf | h – SE 3D structure |
| i_se_flower.pdf | i – SE network (flower) |
| ghi_shared_legend.pdf | legend shared by g, h, i |
| hic_ets2_locus.pdf | Hi-C locus: WT HEK293T Hi-C 10 kb triangle with A–B / A–C / B–C contacts, coordinated binding edges of an ETS2 3-region module (ACBD6 promoter – distal – LINC01732 promoter), TF counts, genes |
| hic_rbbp4_locus.pdf | Hi-C locus, RBBP4 version: 3-region RBBP4 module C00057 on chr1 (distal – THBS3 promoter – FDPS promoter), all edges on Hi-C loops |
| j_go_specific_vs_conserved_summary.pdf | j – GO terms of TF-specific vs conserved module elements |
| supplementary/go_per_tf_great_terms.pdf | GO term x TF matrix (all TFs with terms) |
| supplementary/j_gene_set_dotplot.pdf | earlier pooled promoter GO comparison |

Panel b = b_rbbp4_pair_classes (chosen 2026-11-22). b1, b2 and b_rbbp4_characterization are kept for now as alternatives.
Raster parts (embedded images, everything else vector): d – heatmap cells; g – Hi-C contact triangle; h – 3D structure render; hic_ets2_locus and hic_rbbp4_locus – Hi-C contact triangle (600 dpi).
Place g, h, i at 100% (drawn at print size); b1, b2, c and f are large originals and can be scaled down.
Figure legend notes: `Figure3/figure_legend_notes.md`.
