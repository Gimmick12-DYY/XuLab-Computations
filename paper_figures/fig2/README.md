# Figure 2 (QC: single-cell vs bulk ChIP)

Rebuild (Longleaf):

    python downstream/fig2_qc/r21_fig2_source.py      # source_data/ (corr matrix table, heatmap arrays)
    python paper_figures/fig2/make_fig2.py             # output/

| Panel | Script | Source data | Origin |
|---|---|---|---|
| corr_matrix_6x6 | panels/a_corr_matrix.py | source_data/corr_matrix_6x6.tsv | round 3 summit-group Pearson (downstream/fig2_qc/r2_05_pearson.py), ZNF777 depth, 3 seeds |
| peak heatmaps V3 (main), V2, V1 | panels/b_heatmaps.py | source_data/heatmaps_<TF>.npz | downstream/fig2_qc/r17_kayaokur_heatmaps.py caches |

Style: common/style.py (Arial 7 pt, editable PDF/SVG text, 89/183 mm). Matrix colours diverging (centred at 0,
clipped at ±0.2); heatmaps turbo with one scale per panel (Kaya-Okur 2019 convention), bodies rasterised at 600 dpi.
Style choices and citations: figures_fig2_qc/paper_style_brief.md on the workstation.
