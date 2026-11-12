# m20_raw_binding_audit (spec, 2026-11-12)

Label: diagnostic. HEK293T HITAG binding; bulk ChIP references are labelled with their own cell line (HEK293T, HEK293 or K562). Written before any number is computed.
Audit TFs: CTCF, NRF1, NFYA, MAZ, ZNF143, E4F1.

Part 1, pipeline integrity
a. For each audit TF, list the file each analysis read (m11 peaks, m16 lists, m18 blocks B/BC/D and the raw element counts) and check the file -> TF label mapping, including the 79-column order of block D and of raw_counts.npy against m1_binding/v2/tfs.txt.
b. Genome build and coordinates: chromosome naming, coordinate ranges against hg38 chromosome sizes, 0/1-based starts, summit = start + offset; confirm that no hg19 input (TF1000cells pmat) is used.
c. Independent recomputation with bedtools in a new shell script (no m10/m11/m18 code): promoter reads (read 5' ends in TSS +/- 500 bp, CPM by total reads) and peak-in-linked-element calls (q005 summit inside a canonical ABC-linked element, ABC >= 0.02, self-promoter included) per gene; compared with the m18 feature table entry by entry (Pearson, fraction of exact matches; for calls, agreement and the 2 x 2 table).
d. Label check from the m18 feature columns only: for audit TF X, the own-motif promoter indicator of X against promoter reads of each of the 79 D columns (top 1,000 genes vs the rest, OR); the rank of column X among the 79. Same with the per-TF B_prom_peak rows.

Part 2, the earlier bulk overlap reproduced with controls
- Bulk ChIP sets on Longleaf (data/), cell line stated per set. The earlier figure: downstream/eval/summaries/tf_eval_summary.tsv (raw pipeline AUROC on bulk-bound cCRE bins vs random bins; downstream/run_tf_benchmark_bins.sh, compare_pos_neg.py), quoted as found.
- For each ChIP set: fraction of our q005 summits inside ChIP peaks and fraction of ChIP peaks containing one of our summits; AUROC of our pseudo-bulk read count per union-ATAC region (data/HEK293T_ATAC_union3.bed.gz) against ChIP overlap.
- Controls: (i) the same numbers for every other TF's HITAG data against this ChIP set, with the rank of the own TF among 79; (ii) ATAC alone (OmniATAC reads per region) as the predictor; (iii) promoter (within 1 kb of a GeneList TSS) vs distal regions.

Part 3, where the signal is lost (per audit TF with a ChIP set and a knockdown)
Rows: peaks vs ChIP -> elements with an ABC link -> genes (our gene list vs ChIP-derived gene list: ChIP peak in a linked element; Jaccard and OR) -> gene list vs knockdown responders (top 5% |z| and the reproducible set, |z| >= 2 in both halves with the same sign). Every row also for the swapped TF (m7 mis0, same depth tertile) and an ATAC-only list of the same size.
Within shared genes (bound by ours and by ChIP): Spearman of our peak score vs ChIP signalValue, and of ChIP signal vs |z|.
Reasons not tested are written as unexplained.
