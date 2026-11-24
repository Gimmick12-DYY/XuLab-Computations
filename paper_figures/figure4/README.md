# Figure 4: code and source data

Panels are drawn by scripts in `panels/` from the tables in `source_data/`, with the shared style in `../common/style.py`
(Arial with live text: PDF TrueType `pdf.fonttype = 42`, SVG `<text>` with explicit font attributes). Panels are drawn at print size.
So far only panel c is built here; panels a, b and d are still the versions in the Mac `Figure4/` folder.

## Rebuild

```
cd paper_figures/figure4
python prepare_source_data.py     # Longleaf: needs bedtools and MEME (FIMO) modules; writes source_data/ + MANIFEST.tsv
python make_figure4.py            # all panels -> output/ ; or: python make_figure4.py c
python check_outputs.py           # PDFs: embedded TrueType Arial + extractable text; SVGs: <text>, no glyph outlines
```

## Panels

| Panel | Script | Source data | Upstream analysis |
|---|---|---|---|
| c, shared enriched motifs | `panels/c_shared_motifs.py` | `c_shared_motifs.tsv`, `c_logo_counts.tsv` | `complex_screen/pair_motif_composite` (pair table `motif_tables/pair_shared_motifs_p005.tsv`, HOMER known on `recall/homer/raw_lab`, FIMO enrichment in `motif_valid/`) |

Panel c: rows RBBP4-ZNF644, ADNP-MTF2, DR1-NFYA; three shared motifs per pair (p <= 0.05 in both TFs on discovery and held-out
chromosomes; generic and Tn5-background motifs excluded, dagger = generic filler when a pair has fewer than three). Each motif shows two logos in parallel, one from each TF's own
FIMO instances in its binding sites, labelled by the consensus of both; the line under each logo is the largest of the four p-values.
Logo letters are drawn as shapes (they are data marks); all other text is live text.
