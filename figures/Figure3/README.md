# Figure 3: panel collection (2026-11-11)

One folder for Fig 3, collected for the new plan (a-j). No layout: arrange the panels yourself. Each panel has `.svg` (live Arial text,
Illustrator-safe font attributes, cropped to content), `.pdf` and a 300 dpi `.png`. Figure text says "higher-order coordinated binding module"
/ "module edge", never "clique". Rebuild everything with `collect_figure3.py` on Longleaf (`XuLab/figures/Figure3/`).

| Panel | Plan | File(s) | Source (Longleaf XuLab unless noted) | Script / commit |
|---|---|---|---|---|
| a | What coordinated binding is (schematic, a possible rosette) | `a_coordinated_binding_rosette` (120 x 52 mm) | illustration, not data | `a_rosette_schematic.py` |
| b | RBBP4 example: network | `b1_rbbp4_network` | `hic/work_wt293t/loops/union_5k_10k_0.7/RBBP4_sweep_k50_i10000_s555_pad30kb/publication_plots_v2/tf_peak_network_cliques_loops.svg` (deck slide 169) | `hic/scripts/plot_clique_network_loops_v2.py`; text fix here |
| b | RBBP4 example: Hi-C loop plot | `b2_rbbp4_hic_circos` | same folder, `RBBP4_clique_loop_edges_circos_tf_top.svg` (slide 170) | `hic/scripts/plot_clique_circos_tf_top.py` |
| c | RBBP4 chromosome plot (module edges above, loops below) | `c_rbbp4_chromosomes_plain`, `c_rbbp4_chromosomes_highlight` | same folder, `RBBP4_clique_edges_above_loops_below_chromosomes.{plain,highlight}.svg` (slide 172) | `hic/scripts/plot_pairs_vs_loops_chromosomes_v2.py` |
| d | All-TF module loop-edge support + per-edge shuffle null | `d_module_edges_hic_support` | Fig 3 v7 panel a | `figures/fig3_v7/a_module_edges_v7.py` (9766509) |
| e | All-TF module distribution (genomic class) | `e_module_genomic_class` | Fig 3 v7 panel b | `figures/fig3_v7/b_class_bars_v7.py` (9766509) |
| f | Chord of the most widely shared module edges | `f_top_shared_edges_chord` | Mac `hub_figures/basic/clique_edge_overlap/fig_chord_tf_top_shared_edges.svg` (slide 72, md5 7902fef4 = deck) | title/subtitle dropped here |
| g | Super-enhancer hub example (browser) | `g_se_browser` (print width 181 mm) | Fig 3 v7 panel c | `figures/fig3_v7/c_se_browser_v7.py` (1a3c955) |
| h | SE 3D structure | `h_se_3d_structure` (91 mm) | Fig 3 v7 panel d (approved rosette, slide 22) | `figures/fig3_v7/prep_panels_v7.py` |
| i | SE network illustration (flower) | `i_se_flower` (84 mm) | Fig 3 v7 panel e | `figures/fig3_v7/flower_v7.py` |
| g-i | Shared legend for g, h, i | `ghi_shared_legend` | Fig 3 v7 legend strip, letters updated | `figures/fig3_v7/cde_legend_v7.py` |
| j | Module genes / function | `j_module_genes_go` | Fig 3 v7 extra panel | `figures/fig3_v7/fig3_geneset_dotplot.py` (14d6893) |

g, h and i are drawn on larger canvases with fonts preset for the print sizes given; place them at 100% of the SVG/PDF size.
b, c and f are slide-sized originals (b1 444 mm, b2 311 mm, c 419 mm, f 281 mm wide): scale them down when placing.

**Missing:** none (a is a first draft of the schematic: linear module -> rosette in 3D).

**Not certain:** "RBBP4 chromosome combined plot" (c) is taken as the chromosome-track plot (slide 172, plain and highlight versions);
the circos (slide 170) is filed as b's Hi-C plot. Swap if you meant the other way round.

`supplementary/`: pairs with Hi-C loop support + per-pair null (approved D1), whole-module block-shuffle null (stricter; 35/40),
RBBP4 pair/edge permutation (S1D, slide 16), SE edge-count table (slide 21).
`checks/`: ATAC + promoter (and peak-density) matched null for d (27/40 and 26/40 TFs, median O/E 1.19), HOT-region check for e (promoter
rise = matched expectation), per-edge vs block null per TF, SVG-vs-PDF render check, input dependency audit (panel letters there are v7's).
`figure_legend_notes.md`: draft legend text with the new letters.
