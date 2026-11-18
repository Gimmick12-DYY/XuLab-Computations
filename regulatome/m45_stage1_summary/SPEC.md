# m45: Regulome Stage 1 summary (tables, panels and slides)

User (2026-11-18): document the Stage 1 results in the local deck. Stage 1 = simple ABC linking with binary results, judged on recall against background. Coordinated-binding links and partners are Stage 2 and are left out.

Setup (fixed, as m36/m38/m40/m42): HEK293T; top-25% sgRNA matrix (m23); 63 TFs with a usable knockdown; 11,228-gene universe; own gene removed; genes without a z excluded per TF.
Regulome link (binary TF x gene): one of the TF's top 5,000 lab MACS2 peaks (signalValue rank, summit = start + col 10) with summit in a promoter element or a canonical Hi-C ABC-linked element of the gene (m38 rule A), optionally plus a summit within +/- padding of the gene's TSS (ABC GeneList tss).
Two dials only: |z| cutoff 2, 3, 4, 5; padding 0 (rule A), 10 kb (m38 rule B), 50 kb (m38 rule C).
Broad TFs (fixed, m42): the 10 TFs whose padding-0 list has > 4,000 genes (RBBP4, XPA, RBBP7, NFYA, E2F3, ZNF507, DDIT3, SOX4, NME2, ZNF703); 8 have a knockdown. Versions: all 63 TFs; without broad (55; broad lists also removed from the pool of other lists); broad only (8).

Tables per padding x cutoff x version, pooled (sum over TFs) and per TF:
1. Our genes, Perturb-seq genes, in both; recall = in both / Perturb-seq genes (share of Perturb-seq genes found in our Regulome); share of our genes found in Perturb-seq = in both / our genes. TFs left (>= 1 Perturb-seq gene). Median genes per TF.
2. Recall against background (m40 reference 1): share bound among changed genes vs matched background (the same TF's genes with |z| < 1 in the same NTC-expression decile x promoter-ATAC decile, averaged over changed genes) vs all non-changing genes (|z| < 1).
3. Dose response: share bound by |z| bin 0-1, 1-2, 2-3, 3-4, 4-5, >= 5.
4. Own list vs the 5 other TFs nearest in list size (pool = all 79 lists, or without broad lists in the without-broad version); median rank; TFs with own list in top 5.
5. Gene shuffle (m40 Amendment 1, Perm 1), all three paddings: per TF, which genes are its Perturb-seq genes is shuffled within the expression x ATAC strata (exact hypergeometric draw per stratum), 1,000 shuffles, seed 20261118; unstratified shuffle as reference. Pooled observed recall, shuffle mean, 2.5-97.5 range, one-sided p = (1 + #shuffles >= observed) / 1001; all, without broad, broad only.

Checks: padding 0 must reproduce recall 17.7 / 19.2 / 22.1 / 24.4 % and matched 18.2 / 19.4 / 20.5 / 21.4 (m42); padding 10 kb recall 22.7 / 26.5 / 31.6 / 36.6 (m38 B); median genes per TF 1,429 / 1,909 / 3,511; Perturb-seq genes 63,954 / 11,237 / 2,188 / 558 and TFs 63 / 63 / 57 / 33.
Descriptive; no expected/fold/precision wording.

## Amendment 1 (2026-11-18, written before any number): matched gene shuffle by TF set
Same matched gene shuffle as table 5 (expression x promoter-ATAC strata, exact hypergeometric draw per TF and stratum, 1,000 shuffles, seed 20261118). Per padding and cutoff the per-TF observed counts and shuffle draws are made once; every TF set is the sum over its TFs.
Sets: all 63; without NRF1, E4F1, DNMT1 (60); without the broad TFs (55); without both (52). Paddings 0, 10 kb, 50 kb; |z| >= 2, 3, 4, 5.
Leave-one-TF-out (padding 10 kb, |z| >= 3 and >= 4): for each of the 63 TFs, the all-63 set without that TF; report the range of p and of observed minus shuffle mean (points) and the TF whose removal lowers the gap most.
Bulk ChIP side by side (padding 10 kb, |z| >= 2-5): the 10 bulk ChIP lists of m40 (m35 references, top 5,000 peaks, same link path, + 10 kb) vs our HITAG lists for the same 10 TFs; per list and pooled.
Report observed recall, shuffle mean, 2.5-97.5 range, one-sided p = (1 + #shuffles >= observed) / 1001.
