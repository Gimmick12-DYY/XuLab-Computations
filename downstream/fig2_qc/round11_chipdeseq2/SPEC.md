# Round 11: ChIP-only DESeq2 differential peaks, then the summit-group 6x6 and a heatmap at those peaks
Declared 2026-11-23, before any download or result; recipe fixed. User (21:21Z): "We work on the ChIP-seq data only
first to find the differential peaks". Outputs are alternatives (fig2_corr_matrix_6x6_chipdeseq2, _colz, imputed
version, fig2_heatmap_chipdeseq2) until the user picks.

## 1. ChIP replicates
The local ChIP BAMs are replicate 1 only. Replicate 2 is downloaded to /work/users/d/y/dyy12/XuLab/data/chip_reps/
(nothing under /vast):
| ChIP | experiment | rep 1 (local) | rep 2 | cell line |
|---|---|---|---|---|
| CTCF | ENCSR000DTW | ENCFF139DCW | ENCFF474ZKB | HEK293 |
| MAZ | ENCSR290SSQ | MAZ_HEK293.bam (expected ENCFF085SQS) | ENCFF298ZTU | HEK293 |
| ZBTB7A | ENCSR773REP | ZBTB7A_HEK293.bam (expected ENCFF559CGO) | ENCFF421CTJ | HEK293 |
| ZIC2 | ENCSR728MWW | ZIC2_HEK293.bam (expected ENCFF156XLC) | ENCFF042EMQ | HEK293 |
| ZNF777 | ENCSR295BIP | ZNF777_HEK293.bam (expected ENCFF436UNS) | ENCFF859NFR | HEK293 |
| ZNF282 | GSE78099 | GSM2466515 / SRR5197098 (local alignment) | GSM2466516 / SRR5197099 | HEK293T |
ENCODE rep 2 = GRCh38 filtered alignments from the same (2020) pipeline run as rep 1; md5 checked against ENCODE. The
local rep-1 files are checked against ENCODE's md5 for the expected accession (mismatch reported, not fatal).
ZNF282 rep 2 reads (ENA fastq of SRR5197099) are aligned exactly as rep 1 (bowtie2 2.5.4 --very-sensitive-local,
/proj/seq/data/hg38_UCSC Bowtie2Index, samtools sort).

## 2. Read filtering (same for all 12 BAMs)
MAPQ >= 30; unmapped, secondary, supplementary, QC-fail and duplicate-flagged reads excluded; one read per fragment
(read 1 for paired-end, every read for single-end), counted at its 5' end. ENCODE filtered BAMs already have duplicates
removed; the ZNF282 BAMs (both reps) are duplicate-marked with samtools markdup first, so all 12 are deduplicated.
Read type per BAM (SE/PE, read length) is reported; it differs between experiments (CTCF and ZNF282 single-end) but
not between the two replicates of one ChIP.

## 3. Regions
Union of the 6 ChIP peak sets used in Fig 2 (CTCF ENCFF314ZAL; MAZ, ZBTB7A, ZIC2, ZNF777 ENCODE IDR; ZNF282
GSM2466515 re-called), summit +-250 bp, merged (bedtools merge), chr1-22 + X, hg38 blacklist v2 removed. Each merged
region keeps, for each ChIP, the summit and signalValue of that ChIP's strongest peak inside it (if any).

## 4. QC before DESeq2 (reported)
Per-ChIP Pearson r of log2(CPM + 1) between rep 1 and rep 2 across all regions, and the full 12 x 12 table (do the
replicates of a ChIP correlate better with each other than with other ChIPs?).

## 5. DESeq2
DESeq2 (R 4.5), 12 samples, design ~ ChIP (6 levels). Standard median-of-ratios size factors; if every region has a
zero in some sample (size factors undefined), estimateSizeFactors(type = "poscounts") instead (reported). For each
ChIP a numeric contrast "this ChIP minus the mean of the other 5" (+1, -1/5 each), results(), lfcShrink(type = "ashr").
Differential for a ChIP = padj < 0.05 and shrunken LFC > 1. Reported: dispersion median, n differential per ChIP, and
the pairwise overlap of the 6 differential sets.

## 6. Summit-group 6x6 (raw and imputed)
The round 3 summit-group recipe with only the region selection changed (was: preferred summits, own bulk log2CPM minus
the mean of the other 5 > 1, top 5,000 by signalValue):
- For each ChIP: its differential regions, top 5,000 by shrunken LFC (all if fewer); each region is represented by that
  ChIP's summit in it; sorted by that peak's signalValue and split into 30 groups (180 groups in all).
- Bulk value per group: log2 CPM (+1) of ChIP reads in the regions of the group, from the DESeq2 count matrix (rep 1 +
  rep 2 summed; the old recipe used rep 1 within summit +-250 bp).
- Raw single-cell value: log2((C + .5)/(F + .5)) - the same for the 79-TF pool minus this TF; C = 5' ends within +-100 bp
  of the summits, F = 1-2 kb flanks; our TF thinned to ZNF777 depth (420,339 reads; seeds 11, 22, 33; r averaged).
- Imputed: r25 recipe at the same summits (Mdrop5 mask, raw-positive cells + model calls; 1 kb summit bin vs flank bins
  2-5 kb away, minus the pool).
- Pearson across the 180 groups; own-best rows/columns, diagonal minus off-diagonal, exact 720-permutation p, without
  CTCF (120), and odd vs even + X chromosome halves.

## 7. Heatmap
Rows = the same differential regions, one block per ChIP (top 5,000 by LFC; sorted by that ChIP's signal; rows
averaged in consecutive groups for display). Panels: ChIP (that block's ChIP, +-1 kb from the round 6 bigWigs) | our raw
for each of the 6 TFs (+-1 kb, 50 bp bins, 5' end CPM at full depth, divided by the pool profile's mean so each TF is on
an enrichment scale) | imputed for each of the 6 TFs (Mdrop5, 1 kb bins, +-5 kb, fraction of cells / pool). Question
answered: does each ChIP's block light up the matching TF more than the other TFs?
Scripts: downstream/fig2_qc/r11_*. Panel scripts: paper_figures/fig2/panels/a_corr_matrix.py (6x6),
g_heatmap_chipdeseq2.py. No changes after results.
