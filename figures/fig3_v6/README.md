# Fig 3 v6: built from the approved deck figures

Rebuilt 2026-11-09 after v5 was rejected. Every panel is the figure already approved in the local deck
(`~/Desktop/HEK293T-ATGTAG/293T Yuyang local.pptx`, read only from a temp copy), traced to its source file by
image hash (md5 of the slide picture = md5 of the PNG on disk). No TF clustering matrix. v3 to v5 untouched.

Page: 183 x 247 mm, 8 pt bold Arial letters, text live (Arial), each panel one named group (`panel_<stem>`).
Outputs: `Fig3_v6.svg`, `Fig3_v6.pdf`, `Fig3_v6_preview.png` (300 dpi), `panels/` (per-panel SVGs as placed).

## Slide -> panel -> source

| Panel | Deck slide (picture md5) | Source file (original) | v6 file |
|---|---|---|---|
| a network | 169 "Coordinated binding", left picture (ad3eaef6, exact match) | Longleaf `hic/work_wt293t/loops/union_5k_10k_0.7/RBBP4_sweep_k50_i10000_s555_pad30kb/publication_plots_v2/tf_peak_network_cliques_loops.svg` (from `hic/scripts/plot_clique_network_loops_v2.py`, `cobinding/slurm/run_rbbp4_loop_plots_wt293t_v2.sbatch`) | `src/approved_tf_peak_network_cliques_loops.svg` -> `panels/a_network.svg` |
| b genomic class bars | 11 "Fig 3c" (bars of 02069735, figC_genomic_annotation_chromhmm_paletteC); same data as S1C slide 60 (e6dc7798 = `hub_figures/basic/descriptive/figC1_class_bars.png`) | `figures_paletteC/figC1_class_bars_paletteC.svg` (from `hub_figures/basic/descriptive/plot_figC_split_paletteC.py`) | `panels/b_class_bars.svg` |
| c module Hi-C support | 15, right (515f24d6) + new null panel | `figures_paletteC/figD2_cliques_loop_coverage_paletteC.svg` (`plot_s1d_split_paletteC.py`); right: module-edge observed vs own 10,000-shuffle null drawn with the D1 null code | `panels/c_module_loop_support.svg` |
| supplementary (out of Fig 3) | 15, left (7d7a2e64) | `figures_paletteC/figD1_pairs_loop_support_paletteC.svg` | `supplementary/S_pairs_loop_support.svg/.pdf/.png` |
| d SE browser | 21 "4E-H11", left (11c4ef61) | `results_3d_model/se_sedb/figure_gold/browser_final.svg` (`se_sedb/scripts/28_browser_gold.py`) | `panels/d1_se_browser.svg` |
| d SE edge table | 21, right (87822a32) | `results_3d_model/se_sedb/figure_gold/browser_side_final.svg` (same script) | `panels/d2_se_edge_table.svg` |
| e SE 3D rosette | 22 "3D Structure of the super-enhancer hub" (0c7c77be) | `results_3d_model/se_sedb/figure_rosette_own_beads/se_rosette.svg` (`se_sedb/scripts/47_rosette_own_beads.py`) | `panels/e1_se_rosette.svg` |
| e SE flower | 23 "All Cicero edges in the super-enhancer hub" (86bd1010) | `results_3d_model/se_sedb/figure_chord_tf/flower_final.svg` (`se_sedb/scripts/42_chord_tf.py`) | `panels/e2_se_flower.svg` |
| (not used) circos | 170 "Coordinated binding" (92e4febf) | `.../publication_plots_v2/RBBP4_clique_loop_edges_circos_tf_top.svg` | separate slide, not placed |
| (not used) chr16 hub | 13 "Fig 3e" (ad389e09) | `figures_paletteC/hub_chr16_browser_paletteC.svg` | no room (see below) |

Slide 169's right picture is a zoom of the same network; slide 60's bars are the same table as the palette C bars on slide 11.

## What was changed (nothing else)

- Wording, every "clique" in figure text: network legend ("higher-order co-binding module", "loop-supported module edge",
  "module with every edge on a loop (15)", "module peak on that loop"); D2 title, x label and legend title ("Modules by Hi-C
  loop-edge coverage (% = modules with >=1 loop edge)", "higher-order co-binding modules (with >=1 edge >25 kb)", "share of module
  edges on a 0.7 loop"); class bars ("all module regions", "in modules of N TF(s)", title "Genomic class of regions in
  higher-order co-binding modules").
- Slide-style titles removed: network title/subtitle, SE browser title, rosette title ("Most-closed structure ..."). Axis titles kept.
- SE flower: partner-partner edges now bulge OUTWARD from the ring as teardrop petals (were inner chords); TF pie charts and their
  legend removed (user, 2026-11-09): nodes are small solid dots in the type colour (blue promoter, vermillion enhancer, grey other);
  original ring radius, SE hexagon, spokes, colours and solid/dashed rules; labels pushed slightly outward past the petals.
- Network: the approved SVG has text as glyph outlines, so its text was converted to live Arial text in place (same anchors and
  sizes); its legend was redrawn beside the plot at 5 pt with the same handles. Re-running the script was tried
  (`network_v6.py`, `run_network_v6.sbatch`) but gives a different component arrangement (unseeded set order), so the approved
  SVG is used.
- D1, D2 and the class bars were re-run from their original scripts at their original sizes with only the wording changes
  (`b_hic_support_v6.py`, `e_class_bars_v6.py`); the class-bar percentages and n's match the slides.

## Panel c update (2026-11-09, user)

The pairs chart (D1) left Fig 3 for the supplement. Panel c = D2 (module loop-edge coverage) plus, at its right, observed vs own
null for module edges >25 kb (null column 1 of the S1D 10,000-shuffle npz; TFs with >=100 module edges; BH over those 40 TFs):
35 of 40 TFs q < 0.05, median observed 21.7% (same as v4's module null). Filled dot = q < 0.05, open = n.s., n/a = <100 edges.
Table: `panels_src/b2_module_edge_null.tsv`.

## Scales and legibility

The approved figures were drawn for slides, so they are scaled down to share one page: a 0.12, b 0.83, c 0.75, d 0.35,
e 0.28 (rows share one scale). Smallest printed text: c about 3.7 pt (bar % labels; TF names 4.1 pt), d 4.2 pt (12 pt labels; table footnote ~3.5 pt),
e about 3 pt (flower labels, legends), network coordinate labels about 1 pt (vector, zoomable). The chr16p11.2 hub (f) does not
fit at a readable size and is left out.

## Scripts (Longleaf `XuLab/figures/fig3_v6/`, git)

- `flower_v6.py` (Mac paths): flower from `42_chord_tf.py` with outward petals; needs `model/sim/eps0/traj182.npz` (conformation 9102).
- `b_hic_support_v6.py`, `e_class_bars_v6.py` (Mac paths): wording-only re-runs of the palette C scripts.
- `prep_panels_v6.py`, `svgtext.py` (Mac paths): text edits on the approved SVGs -> `panels/`.
- `build_fig3_v6.py` (Longleaf, hic env, rsvg-convert + Arial in `/work/users/d/y/dyy12/fonts/arial`): page assembly.
- `network_v6.py`, `run_network_v6.sbatch`: re-run attempt of the network (not used, see above).
