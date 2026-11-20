# m47: Stage 1 with different numbers of lab peaks per TF

User (2026-11-20): we use the top 5,000 peaks per TF; would the full peak set help?
Everything as m45 Stage 1 (code reused unchanged): lab MACS2 peaks on raw HITAG reads (bigwig/macs2/TF.<TF>_peaks.narrowPeak), ranked by signalValue (col 7), summit = start + col 10; link = summit in a promoter element or a canonical Hi-C ABC-linked element of the gene (m38 rule_abc), plus padding 0 / 10 kb / 50 kb around the TSS (m38 window); 11,228-gene universe; top-25% sgRNA matrix (m23); 63 knockdown TFs; own gene removed; |z| >= 2, 3, 4, 5.
Only the peak set per TF varies: top 2,000, 5,000, 10,000, 20,000, 50,000 by signalValue; all peaks with q < 0.05 (col 9 >= -log10 0.05); the full peak file. A TF with fewer peaks than N keeps all it has (number of such TFs reported).
Top 5,000 must reproduce m45 (recall 17.73 / 19.20 / 22.12 / 24.37 at padding 0; 22.74 / 26.47 / 31.63 / 36.56 at 10 kb).
Per peak set x padding: peaks per TF (median, range), linked genes per TF (median), total links, TFs with > 4,000 linked genes; per cutoff: pooled recall, matched gene shuffle (NTC expression decile x promoter ATAC decile, exact hypergeometric per TF and stratum, 1,000 shuffles, seed 20261118) mean, 2.5-97.5, gap (points), one-sided p = (1 + #shuffles >= observed) / 1001, and the number of single TFs above the shuffle (p < 0.05, >= 10 Perturb-seq genes).
TF sets: all 63; the 55 non-broad TFs, broad TFs fixed as in m45 (padding-0 top-5,000 list > 4,000 genes), the same grouping for every peak set.
Descriptive; no expected/fold wording.
