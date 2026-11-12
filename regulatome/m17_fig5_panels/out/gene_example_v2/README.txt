Fig 5d gene example, version B: locus view + knockdown effect bars (m17_fig5_panels SPEC Amendment 3, commit 2ffd71a). Illustration on an already-read benchmark. HEK293T.
Script: make_gene_tf_locus.py. Genes: PGAM1, HSPB1. Files: fig5d_<gene>_locus.svg (editable Arial text) / .png; coverage_<gene>.tsv (the plotted tracks).
Tracks, top to bottom:
- ABC links (canonical, ABC >= 0.02) to the TSS; arc shade by ABC score.
- ATAC (OmniATAC merged tagAlign) and H3K27ac (SRX18002015/16 tagAlign): raw counts of read 5-prime ends per 50-bp bin.
- One HITAG row per TF with a confirmed (K = 10,000) effect on the gene and a linked q005 peak, plus confirmed TFs without a linked peak (HSPB1: DNMT1).
  Read 5-prime ends from the raw per-TF fragment files (/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment), 50-bp bins, CPM by the TF total reads, 3-bin running mean, same y scale for all TF rows.
  Black ticks = q005 peak summits; gold ticks = own-motif sites (corrected m10 FIMO scans, p < 1e-4; only TFs with a motif).
- Pooled CPM of all other panel TFs; GENCODE v46 basic transcript with the ABC TSS. Blue band = promoter element, orange bands = linked enhancers.
Right: the gene z under that TF knockdown (m15 double-centred, Arm 2, all cells) as a bar; white dots = half 1 and half 2.
Bottom strip: every TF with a linked q005 peak and knockdown data, ordered by z; dot area = peak-score binding percentile; colour = z; outlined = confirmed in both halves.
Weak points: E4F1 at PGAM1 and NME2 at HSPB1 have almost no reads in the window although their knockdown effects are confirmed; ETS2 and DNMT1 at HSPB1 are sparse.
