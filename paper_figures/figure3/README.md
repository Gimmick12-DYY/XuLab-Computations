# Figure 3: code and source data

Every Figure 3 panel is drawn by one script in `panels/` from the tables in `source_data/`. Text stays live, editable text in all
vector files: PDFs embed TrueType Arial (`pdf.fonttype = 42`), SVGs keep `<text>` elements with explicit Arial font attributes
(`svg.fonttype = "none"`). Panels are drawn at their print size unless noted (sizes below are the PDF page sizes), so font sizes in Illustrator are the
printed sizes. Large originals (b1, b2, f, Hi-C loci) keep their slide canvas and must be scaled down when placed.

## Rebuild

```
cd paper_figures/figure3
python prepare_source_data.py     # once, after the analysis pipelines: copies/derives source_data/ + MANIFEST.tsv (hic env: needs h5py)
python make_figure3.py            # all panels -> output/ (cobinding env); or: python make_figure3.py g h i
python check_outputs.py           # every PDF: embedded TrueType Arial + extractable text; every SVG: <text>, no glyph outlines
```

Environments (Longleaf): panels `conda/envs/cobinding` (Python 3.10, matplotlib 3.10, numpy 2.2, pandas 2.3, scipy 1.15,
networkx, python-igraph 1.0); `prepare_source_data.py` `conda/envs/hic` (adds h5py). Arial TrueType files are read from
`$FIG_FONT_DIR` (default `/work/users/d/y/dyy12/fonts/arial`). Shared style and palette C: `../common/style.py`.
A full build takes about 70 s on one core.

## Panels

| Output (`output/`) | Script (`panels/`) | Source data (`source_data/`) | Print size |
|---|---|---|---|
| a_coordinated_binding_schematic | none (see below) | | 183 mm |
| rbbp4_network | rbbp4_network.py (original colour code, no loop overlay) | rbbp4_peak_edges, rbbp4_modules | 24 in canvas, scale down |
| rbbp4_hic_circos | rbbp4_hic_circos.py | rbbp4_modules, rbbp4_module_edges_on_loops | 17 in canvas, scale down |
| rbbp4_characterization | rbbp4_characterization.py (+ _rbbp4.py, _genomics.py) | rbbp4_*, peakachu loops, refseq_select_tss, hek293t_chromhmm18, rbbp4_shuffle_null | 120 x 70 mm |
| rbbp4_square (slot option) | rbbp4_square.py | as rbbp4_characterization | 60 x 60 mm |
| rbbp4_chromosomes | rbbp4_chromosomes.py (+ _genomics.py) | rbbp4_peak_edges, rbbp4_modules, peakachu loops | 180 x 150 mm |
| module_edges_hic_support | module_edges_hic_support.py | d_module_edge_bins_1000kb, d_module_loop_coverage, d_module_edge_null | 183 x 95 mm |
| module_genomic_class | module_genomic_class.py | class_annotation_counts_chromhmm, class_module_regions_chromhmm | 70 x 64 mm |
| top_shared_edges_chord | top_shared_edges_chord.py | f_top_shared_module_edges, f_gene_symbols | 16 x 13 in canvas, scale down |
| se_browser | se_browser.py | se_hic_10kb, peakachu loops, se_hub_edges, se_partner_bands, se_tf_coverage_1kb, se_chromhmm_window, se_refseq_select | 155 x 85 mm (chr19:0.43-1.95 Mb) |
| se_3d_structure | se_3d_structure.py | se_rosette_render (PyMOL), se_rosette_nodes | 91 x 51 mm |
| se_flower | se_flower.py | se_hub_edges, se_partners, se_enhancer_names | 86 x 58 mm |
| se_shared_legend | se_shared_legend.py | none | 177 x 13 mm |
| go_specific_vs_conserved_summary | go_specific_vs_conserved_summary.py | go_surviving_per_tf, go_per_tf_controls | 200 x 82 mm |
| hic_ets2_locus, hic_rbbp4_locus | hic_loci.py | hic_loci.json, hic_locus_<name>_{10kb.npz,peaks,genes} | 8.3 x 7.5 in, scale down |
| supplementary/S_pairs_loop_support, S_module_edge_block_null | S_pairs_and_block_null.py | d_per_tf_null_summary, s_shuffle_null_percentiles, d_module_loop_coverage | 146 x 123 mm; 112 x 90 mm |
| supplementary/S_rbbp4_permutation | S_rbbp4_permutation.py | rbbp4_shuffle_null | 107 x 59 mm |
| supplementary/S_se_edge_table | S_se_edge_table.py | se_hub_edges, se_hic_test_10kb | 79 x 146 mm |
| supplementary/S_go_per_tf_great_terms | S_go_per_tf_great_terms.py | go_*, tf_panel | 351 x 253 mm |
| supplementary/S_j_gene_set_dotplot | S_j_gene_set_dotplot.py | go_pooled_*, go_summary_*, go_top_term_jaccard | 183 x 66 mm |

Raster parts (everything else is vector): d heatmap cells; g and the Hi-C locus panels, Hi-C contact pixels (600 dpi);
h, the PyMOL render of the 3D structure.

Panel a (`a_coordinated_binding_schematic`, the coordinated binding pair / hub illustration) was drawn outside this code and is
not rebuilt here.

## Source data

`source_data/MANIFEST.tsv` lists every input with its origin (pipeline output path in this repo, or the workstation path for
tables made there), the script that made it, and its sha256. `prepare_source_data.py` copies pipeline outputs unchanged and
derives a few small tables (per-edge shuffle nulls stacked per TF, null percentiles, Hi-C locus windows, GO term themes, gene
symbols). `source_scripts/` holds the analysis scripts that make the Figure 3-specific tables (module-edge bins, GO tests, the
RBBP4 locus screen, GO robustness checks); the per-edge module null comes from
`figures/fig3_v6/checks/module_edge_null_check.py` and the ATAC-matched control from `figures/fig3_v7/checks/atac_matched_null/`;
the other pipelines live in `cobinding/`, `hic/` and `results_3d_model/`.

Shared definitions (`panels/_genomics.py`): a pair is on a Hi-C loop when each end, extended by +-30 kb, overlaps a different
anchor of one Peakachu score >= 0.7 loop (5 kb loops first, then 10 kb; wild-type HEK293T, GSE216399); pairs and module edges
count when their region midpoints are > 25 kb apart.

## Reproducibility notes

- b1 layout: force-directed per component with a fixed seed and nodes in sorted order. The earlier slide version used an
  unseeded layout, so node positions differ from it; nodes, edges, colours and labels are the same.
- Ties are broken explicitly: d rows with equal module counts alphabetically; j terms with equal q by higher fold enrichment;
  the GO supplementary matrix keeps table order for equal q. Earlier versions left ties to the platform's sort routine.
- d heatmap columns are clustered (Jaccard, average linkage) with many tied distances; their order can differ from the earlier
  version, the cells are the same.

Legend text: `figure_legend_notes.md`.
