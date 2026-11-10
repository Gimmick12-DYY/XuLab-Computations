# Fig 3 v7 (2026-11-10)

Re-layout requested by the user: top = module edge percentages (a, left) and module distribution (b, right); middle = the
super-enhancer example (c, full width); bottom = 3D structure (d, left) then the flower illustration (e, right). Paper-quality legends.
v6 (`../fig3_v6/`) is untouched. Content = the approved deck figures (see `../fig3_v6/README.md` for slide -> source mapping).

Page 183 x 245.6 mm; `Fig3_v7.svg` (groups `panel_a` ... `panel_e`, `panel_cde_legend`), `Fig3_v7.pdf` (Arial TrueType embedded),
`Fig3_v7_preview.png` (300 dpi). Smallest printed text 5.0 pt (TF names and bar values in a, row labels in b); axis labels 7 pt,
ticks/legends 6 pt, letters 8 pt bold.

## Panels and what changed
- a `a_module_edges_v7.py`: approved D2 bars + observed-vs-own-null at its right, now the PER-EDGE shuffle (user decision
  2026-11-09: same procedure as the approved pairs null). 36 of 40 TFs BH q < 0.05. Drawn at final size; no titles/footnote.
- b `b_class_bars_v7.py`: approved class bars at final size; no title; 2-column legend; values only on segments >=10%.
- c `c_se_browser_v7.py`: approved SE browser (28_browser_gold.py) re-run on its own canvas with fonts preset for the 0.739x placement;
  vertically compressed (Hi-C shown to 150 kb, TF tracks 0.145 in, edge arcs 0.52 in); slide title and edge-count table removed
  (table -> supplementary, numbers in figure_legend_notes.md).
- legend c-e `cde_legend_v7.py`: one shared legend (edge colours/styles, element classes, 3D symbols, TF peak colours) replacing the
  separate browser table, rosette and flower legends.
- d `prep_panels_v7.py`: approved rosette SVG with title, legend column and footnote removed; labels enlarged to 6 pt at placement.
- e `flower_v7.py`: v6 flower (9ee9850) without its legend, labels 6 pt at the 89 mm placement.
- `build_fig3_v7.py`: assembly on Longleaf (rsvg-convert), reports scales and the smallest font per panel.

## Supplementary (`supplementary/`)
S_pairs_loop_support (approved D1), S_module_edge_block_null (stricter whole-module block shuffle; 35 of 40 TFs), S_rbbp4_network,
S_se_edge_table (approved browser side table).

## Conventions adopted from figures4papers (github.com/ChenLiu-1996/figures4papers)
Arial; top/right spines off; frameless legends; svg.fonttype none, pdf.fonttype 42; shared font sizes within a row; compact legends in
free plot space or a dedicated legend strip (c-e); thin dark edges (0.3 pt) on stacked bars; minimal grid (null panel x grid only);
vector SVG/PDF with the 3D render raster kept at its 600 dpi source; 300 dpi preview. Not adopted: its 16 pt fonts, 2.5 pt axes and its
palette (Palette C kept; axes 0.6 pt). The approved box/whisker null design was kept rather than shaded bands.
