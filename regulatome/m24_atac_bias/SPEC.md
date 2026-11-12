# m24_atac_bias: open-chromatin bias of HITAG peaks and post-peak ranking (spec, 2026-11-12)

Label: diagnostic. HEK293T. No Perturb-seq. Committed before any number is computed.

Part A, per TF (79), for the lab q005 peaks and, for comparison, the as-called p < 0.01 peaks:
- cells, fragments (read 5' ends); number of peaks; FRiP = fraction of the TF's read 5' ends inside its own q005 peaks; fraction of read 5' ends inside union ATAC peaks (data/HEK293T_ATAC_union3.bed.gz); fraction of peaks whose summit lies in a union ATAC peak; fraction of summits within 1 kb of a GeneList TSS.
- Height vs ATAC: over q005 peaks with the summit inside union ATAC, Spearman between peak height (signalValue; and the TF's read CPM in summit +/- 100 bp) and ATAC signal (OmniATAC tagAlign reads in the same window).
- Reference for a pure open-chromatin track: H3K27ac (SRX18002015/16 tagAlign) in the same windows of union ATAC peaks: fraction of H3K27ac reads in union ATAC peaks, and Spearman of H3K27ac vs ATAC over union ATAC peaks.
- One table and one figure (79 TFs ordered by fragments: FRiP, ATAC-overlap fraction with a 70-80% red-flag band, height-ATAC Spearman).

Part B, does ratio or residual ranking prioritise true sites?
- Per TF, q005 peaks with the summit inside union ATAC, ranked four ways: (1) signalValue; (2) ratio = (TF CPM + 0.5) / (ATAC CPM + 0.5) in summit +/- 100 bp; (3) residual of log(TF CPM + 0.5) on log(ATAC CPM + 0.5), linear fit per TF; (4) ATAC CPM alone (control, no TF information).
- For the top 500, 1,000 and 2,000 peaks of each ranking: fraction with the TF's own motif within +/- 50 bp of the summit (HOCOMOCO v11 motif as used in m10, FIMO p < 1e-4; sequence names without ":"; FIMO exit codes checked with pipefail and every chromosome present in the scanned set), against a background of the same number of union ATAC peak centres (+/- 50 bp) matched on GC decile, for the 31 TFs with a motif; and, for TFs with bulk ChIP, the fraction of summits inside ChIP peaks (CTCF HEK293T and HEK293, NFYA K562, MAZ/SETDB1/ZBTB7A/ZIC2/ZNF282/ZNF777 HEK293).
- Negative control: the same with a swapped motif (the next TF in the sorted list of motif TFs, cyclic).
- Verdict per TF and pooled, at top 1,000 (500 and 2,000 reported): paired difference ranking 2 minus ranking 1 and ranking 3 minus ranking 1 in own-motif fraction (and ChIP fraction), TF-bootstrap 95% interval (2,000 draws). "Ratio/residual ranking improves own-motif and ChIP agreement" only if the interval is above 0 AND the gain exceeds the same gain for the swapped motif (paired interval of the difference above 0).
- Positive controls that must behave: CTCF, NFYA, NRF1 own motif enriched over the GC-matched background under ranking 1.
- The report puts the new numbers beside the earlier attempts: m9 gene-level ATAC residual against knockdown, and the m11 control re-calls (pooled-other-TF and OmniATAC control).
