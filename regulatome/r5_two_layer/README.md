# R5 two-layer Regulome (prep only)
Status 2026-11-09: prep for the R5 build. A simplified spec is coming from the design thread (core score = raw binding CPM x ABC over all m2_abc links, promoter share and loop support as annotations). Nothing here is a frozen output; no manifest v7 yet. No Perturb-seq data is read.

- r5_prep.py: loop anchor -> promoter tables and the replicated H3K27ac flag. TSS table = ABC TargetGeneTSS per gene (the R2 gene universe).
- edges/loop_anchor_promoter.tsv.gz (Peakachu 0.7, 5 kb; anchor overlaps TSS +/- 2 kb) and edges/loop_anchor_promoter_10kb.tsv.gz (10 kb, sensitivity). Kept: feeds the clique table.
- edges/h3k27ac_flag_elements.tsv.gz and edges/h3k27ac_flag_atac.tsv.gz (GSE246314 peaks in >= 2 of 3 replicates). Written before the 2026-11-09 19:03 change of direction ("keep ABC simple"); the H3K27ac flag is NOT used by the core score and is kept only as a record.
