# Figure 3 – editable PDF pack (2026-11-18)

Final panels as single-page PDFs for Adobe Illustrator: text is live (embedded TrueType Arial, editable), shapes are vector.
Made from the approved SVGs by `Figure3/scripts/pdf_pack.py` (headless Chrome print, exact SVG page size); check table: `pack_check.tsv`.

| File | Panel |
|---|---|
| a_coordinated_binding_schematic.pdf | a – what coordinated binding is (schematic) |
| b1_rbbp4_network.pdf | b (option 1) – RBBP4 network |
| b2_rbbp4_hic_circos.pdf | b (option 1) – RBBP4 Hi-C loop circos |
| b_rbbp4_characterization.pdf | b (option 2) – RBBP4 pairs/module sizes with Hi-C support + pair classes |
| c_rbbp4_chromosomes.pdf | c – RBBP4 pairs and module edges along all chromosomes, supporting loops |
| d_module_edges_hic_support.pdf | d – module edges in 1 Mb bins + modules with loop edges + shuffle null |
| e_module_genomic_class.pdf | e – genomic class of module regions |
| f_top_shared_edges_chord.pdf | f – most widely shared module edges (chord) |
| g_se_browser.pdf | g – super-enhancer hub browser view |
| h_se_3d_structure.pdf | h – SE 3D structure |
| i_se_flower.pdf | i – SE network (flower) |
| ghi_shared_legend.pdf | legend shared by g, h, i |
| j_go_specific_vs_conserved_summary.pdf | j – GO terms of TF-specific vs conserved module elements |
| supplementary/go_per_tf_great_terms.pdf | GO term x TF matrix (all TFs with terms) |
| supplementary/j_gene_set_dotplot.pdf | earlier pooled promoter GO comparison |

Panel b has two options (b1 + b2, or b_rbbp4_characterization); choose one.
Raster parts (embedded images, everything else vector): d – heatmap cells; g – Hi-C contact triangle; h – 3D structure render.
Place g, h, i at 100% (drawn at print size); b1, b2, c and f are large originals and can be scaled down.
Figure legend notes: `Figure3/figure_legend_notes.md`.
