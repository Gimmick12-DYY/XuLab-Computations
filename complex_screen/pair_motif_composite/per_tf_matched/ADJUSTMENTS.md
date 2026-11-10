# Track B adjustments log (per-TF matched-background motif tables, novel-pair sharing)

Each step lists what changed, the sanity check (own motif recovered in the per-TF table) and, once pairs are scored, the random-pair pass rate.

## Step 0: imputed top-10k bins, matched background (per_tf_matched/, arrays 8752479 + 8752631), 2026-11-10
- Foreground: imputed CPM minus the 79-TF mean, top 10,000 eligible union-ATAC bins, centre +/- 100 bp; odd autosomes = discovery.
- Background: non-top eligible bins, equal counts per GC 0.025 x CpG o/e quintile x promoter stratum. GC gap <= 0.001 for every TF.
- Sanity: own HOMER motif in the top 10 with p <= 1e-5 in 0 of 21 TFs that have one. NFYA is the best (rank 17/460, p 1e-11); CTCF ranks 443. Tables are topped by Sp/KLF/E2F/NRF promoter words.
- Verdict: fails. Pair sharing on these tables would reflect generic promoter grammar, so it was not scored as evidence.

## Step 1 (fallback ii): raw MACS2 summits (per_tf_raw/, array 8754925)
- Foreground: lab MACS2 summits at p <= 1e-5, top 5,000, +/- 100 bp, blacklist / >50% exonic / N removed.
- Background: union-ATAC bins not overlapping the TF's peaks, matched the same way.
- Sanity: own motif in the top 10 with p <= 1e-5 in 2 of 21 TFs. NFYA (NFY rank 1, p 1e-306) and NRF1 (rank 1, p 1e-15) pass. MAZ is rank 10 (p 3e-4), CTCF rank 32 (p 2e-4, fold 1.43), DDIT3/Chop rank 65. Sp1/KLF words still top most tables.
- Pair scoring (novel_pairs.py, TFBASE=per_tf_raw): there are 617 / 508 / 738 novel pairs in the pct >= 0.90 / 0.80-0.90 / 0.60-0.80 tiers.
  - Empirical p <= 0.05 vs 200 depth- and pct-matched random pairs: strict 14 / 14 / 30, loose 24 / 22 / 21. That is 2-4% per tier, at or below the 5% expected by chance, so there is no excess of sharing among novel pairs.
  - AF3 strong/weak with empirical p <= 0.05: DDIT3-RBBP7, DDIT3-RBBP4 (CRE/ATF bZIP family shared and validated on even chromosomes; 12 of 79 TFs pass CRE), XPA-ZIC2, RBBP4-ZIC2 (KLF/NRF promoter words), RBPJ-ZNF507.
  - Controls: CTCF-STAG2 shares only Sp1; MAZ-ZNF143 and NFYA-ZNF644 share nothing at strict; RBBP7-SRCAP (negative) shares NRF1. None has empirical p < 0.37.
- Verdict: own-motif recovery improves for deep promoter TFs, but pair-level sharing is at the null rate. DDIT3-RBBP4/7 + CRE is the only biologically coherent call (DDIT3 is bZIP), and it does not survive the across-tier null.

## Step 2 (fallback iii): distal-only raw summits (per_tf_raw_distal/, array 8756641)
- The promoter stratum is dropped from the raw pools (TSS within 1 kb), so matching is preserved. 76 TFs keep >= 300 discovery windows; NFYA (272) and SOX4 (230) drop out.
- Sanity: own motif in the top 10 with p <= 1e-5 in 0 of 19 TFs. It is worse than with promoters (CTCF rank 46, MAZ 28, NRF1 307).
- Pair scoring: there are 572 / 464 / 684 novel pairs per tier. Empirical p <= 0.05: strict 16 / 14 / 12, loose 18 / 15 / 19 (2-3%, at or below chance). Controls share nothing strict.
- AF3 + empirical p <= 0.05: DDIT3-RBBP7 (emp 0.005; shares the CTCF/BORIS motif distally, a motif only 5 of 76 TFs pass, validated on even chromosomes in both), FOXJ3-FOXK1 (Usf2, 3 TFs), E2F3-MBD3 (KLF/Sp).
- Verdict: distal restriction does not recover own motifs; pair-level rate stays at the null. Imputed and distal de novo tasks that had not started were cancelled to give the raw de novo runs priority.

## Step 3 (fallback iv): de novo-only sharing on raw summits
- Each TF's de novo motifs (HOMER len 8/10/12, fold >= 1.5, p <= 1e-10) are compared between the two TFs (similarity >= 0.6), with the random-pair rate from the same matched null.
- Result (all 79 TFs): 4,844 de novo motifs form 393 families (median 6 TFs per family); 280 are non-generic (<= 10 TFs) and 364 have no Tomtom match (q < 0.05).
- Share of pairs with any shared non-generic family / with empirical p <= 0.05:
  - novel pct >= 0.90: 54% / 2.8%
  - 0.80-0.90: 58% / 3.0%
  - 0.60-0.80: 56% / 2.7%
  - all pairs: 56% / 2.7%
  This is below the 5% chance rate.
- Controls: CTCF-STAG2 shares 1 family (emp 0.51), MAZ-ZNF143 2 (0.26), NFYA-ZNF644 1 (0.46), EZH2-MTF2 0, the RBBP7-SRCAP negative 1 (0.54).
- The shared words are 8-mers that recur in 6-10 TFs (e.g. ATGGCGTG, TAGATCGA, AKTACGTT). They look like sequence-composition words of the raw single-cell peaks, not pair-specific grammar.
- Verdict: at the null rate.

## Step 4 (added, not in the pre-approved list): reversed design (reversed/, FIMO array 8759406)
- Sequence defines the sets. FIMO p <= 1e-4 of 2,528 known motifs (JASPAR + HOCOMOCO v12 + Codebook + 2 CAP-SELEX) over 95,044 eligible bin centres.
- Per TF: imputed signal (minus 79-TF mean, rank-normal) is residualized on the GC x CpG x promoter strata. z = mean residual at hit bins x sqrt(n) / sd, computed separately on discovery (odd) and validation (even + X) chromosomes.
- Pass: strict z_disc >= 4 and z_val >= 2; loose 3 / 1.5. Motifs passing in > 20 TFs are dropped as generic.
- Sanity: NFYA own motif z 7.8 / 6.5 (rank 20 of 2,427), NRF1 3.9 / 2.6 (rank 16), HES1 5.2 / 0.7, CTCF 2.1 / 2.2 (rank 256), MAZ 1.2, ZNF143 0.2. Tops are KLF/SP and KZFP words.
- Pair rates (any shared / empirical p <= 0.05 vs depth- and pct-matched random pairs):
  - novel pct >= 0.90: strict 34% / 5.2%, loose 65% / 5.0%
  - novel 0.80-0.90: 34% / 5.7%
  - novel 0.60-0.80: 30% / 5.0%
  - novel < 0.60: 29% / 5.8%
  - all pairs: 31% / 5.4%
  So the rate is the same across correlation tiers and equals the 5% expected under the null.
- Controls: CTCF-STAG2, MAZ-ZNF143, NFYA-ZNF644 and EZH2-MTF2 share nothing strict; the RBBP7-SRCAP negative shares 3 loose.
- AF3-strong pairs at empirical p <= 0.05: HES1-NR1D2, NR1D2-TBX3, NR1D2-ZNF703, RBBP7-SAFB. NR1D2 passes many KZFP/repeat motifs, so these reflect a TF-wide sequence preference rather than shared grammar.
- Verdict: at the null rate.
- Lead from step 4: the NFY motif (NFYA.H12CORE, the motif with the cleanest own-TF signal) is enriched in the signal of 11 TFs, validated on even chromosomes: SOX4 z 9.4 / 8.6, ZNF507 7.8 / 5.5, NFYA 7.8 / 6.5, E2F3 7.5 / 4.9, GLYR1, FLYWCH1, HOXA10, NME2, MLX, RBBP4, HOXB9.
  - Novel correlation pairs among them: NFYA-SOX4 (pct 0.96, shares 101 strict motifs, emp p 0.005), NFYA-ZNF507 (pct 1.00, emp 0.04), NFYA-RBBP4 (0.99), MLX-NFYA (0.85). AF3 is none for all of them.
  - This fits the user's logic steps 1 and 2 (correlation, then a motif enriched in both TFs) without structural support. It could also be shared CCAAT-box promoter occupancy rather than a complex.

## Track A guard on DDIT3-NFYA (nfya_null/, jobs 8764920, 8764923, 8764924)
- DDIT3-NFYA: NFY vs DDIT3-only 2.35x, p 5e-13, held-out 1.74x (p 3e-5). NFY-only permutation p = 0.005 over 200 permutations, which is the floor. NFY vs NFYA-only is only 1.23x (held-out 1.07x).
- NFYA-specific null (NFYA with 20 random partners matched on top-set overlap):
  - NFY vs partner-only reaches fold >= 1.3 and p <= 1e-3 in 15/20, and fold >= 1.5 with p <= 1e-5 in 10/20; 13/20 also hold on held-out chromosomes.
  - DDIT3-NFYA ranks 3 of 21 by fold. NFYA-ZNF512B (2.99x), NFYA-ZBTB7A (2.58x) and NFYA-FOXJ3 (2.18x) are as high or higher.
- DDIT3 null (DDIT3 with 20 random partners): NFY 0/20 and Chop 0/20 at any level. DDIT3 never shows its own motif.
- CentriMo (bin centre +/- 100 bp): no central enrichment of NFY or Ddit3::Cebpa in the co-bound, DDIT3-only or NFYA-only pool (all E >= 0.19). Bin centres are not summits, so this check has little power.
- Verdict: the DDIT3-NFYA shared-motif pass is NFYA's own motif carried into any NFYA co-bound pool. It is not specific to DDIT3 and is downgraded to "not specific". The shared-level random-pair null (0/20) was too easy because none of those pairs contained NFYA.

## Consolidated table (novel_pairs_ranked.tsv, controls.tsv; novel_final.py)
- Novel pairs with pct >= 0.60: 1,863. Each motif layer has its own empirical p vs 200 random pairs matched on depth and pct bin.
- Share of pairs with empirical p <= 0.05, by tier (pct >= 0.90 / 0.80-0.90 / 0.60-0.80):
  - raw-summit known: 5.0 / 5.9 / 5.8%
  - reversed design: 6.3 / 6.9 / 6.6%
  - de novo families: 2.8 / 3.0 / 2.7%
  - >= 2 layers: 1.1 / 0.8 / 0.7%, about what independent layers give by chance (~0.9%)
- The 16 pairs with >= 2 layers are mostly the NFY / promoter cluster (NFYA-RBBP4, NFYA-SOX4, RBBP4-SOX4, NFYA-XPA, HES1-XPA, NME2-RBBP4, MBD3-SOX4, ...). None is AF3 strong or weak.
- AF3-strong pairs reach at most 1 layer: HES1-NR1D2, NR1D2-TBX3, RBBP4-ZIC2, DDIT3-TBX3, DDIT3-RBBP7, NR1D2-ZNF703, RBBP7-SAFB, DDIT3-RBBP4, XPA-ZIC2.
- Controls (CTCF-STAG2, EZH2-MTF2, MAZ-ZNF143, NFYA-ZNF644, RBBP7-SRCAP): 0 layers, best empirical p 0.26-0.51.

## Overall verdict (2026-11-10 morning)
- Under every foreground tried (imputed bins, raw summits, distal-only, sequence-defined reversed design, de novo families) and at loosened thresholds, the share of novel high-correlation pairs that share motifs beyond depth- and pct-matched random pairs stays at the null rate.
- The only reproducible own-motif signal is NFY (plus NRF1, partly HES1). The one co-bound pass (DDIT3-NFYA) is NFYA's own motif and is not partner-specific.
- At this depth, motif enrichment cannot serve as the second evidence layer for novel pairs. The candidate ranking rests on correlation + AF3.

## Step 5 (2026-11-10): NFY depth-matched check (reversed/nfy_depth_matched.tsv; rev_depth_counts.py, rev_depth_z.py; jobs 8767671, 8767725)
- Why: 9 of the 10 TFs with validated NFY enrichment in the reversed design are among the 20 deepest of 79, so the signal may be depth convergence at promoters.
- (a) The reversed-design z was recomputed on raw reads (read starts in the 1 kb union-ATAC bins). Cells were subsampled per TF to 5,903 (the 20th-deepest TF, HOXB9) and to 3,000, with 3 seeds each and z averaged. TFs with fewer cells keep all of them. Steps as before: CPM minus the 79-TF mean, rank-normal, residual within GC x CpG x promoter strata, odd/even chromosomes.
- (b) Each TF's NFY z is compared with the 10 TFs nearest in cell count: depth-matched z = (z - mean of the 10) / sd.
- (c) Survive = raw strict (z_disc >= 4, z_val >= 2) at both depths AND imputed depth-matched z >= 2 on disc and val.
- Depth dependence of NFY z (Spearman vs cell count): imputed 0.60 (p 4e-9); raw at 5,903 cells 0.43; raw at 3,000 cells 0.37.
- Result: 0 of the 10 hits survive both.
  - Raw strict at both depths: none, NFYA included. At the loose level (3 / 1.5) at both depths, only NFYA and ZNF507 pass.
  - Imputed depth-matched z >= 2: none of the ten. SOX4 reaches 1.38 / 2.06 and FLYWCH1 2.06 / 1.46.
  - ZNF507 is the only hit with a raw signal: z 5.54 / 3.38 at 5,903 cells and 3.20 / 2.93 at 3,000. NFY is its top raw motif (rank 1 of 2,427), and it passes the raw depth-matched null (3.71 / 3.16). Its imputed depth-matched z is 1.29 / 1.45, because its deep neighbours all carry imputed NFY.
  - NFYA: NFY is rank 2 (5,903 cells) and rank 1 (3,000 cells) of 2,427 on raw reads, so the own-motif signal is real. Its imputed depth-matched z is only 0.95 / 1.20, because its deep neighbours (SOX4, XPA, RBBP7, ZNF507, E2F3, ...) all show NFY.
- References: NRF1 raw z_disc 5.0 at both depths (rank 1), but its validation z is only 0.4-0.5. CTCF raw 1.2 / 1.0 (rank 242).
- Verdict: the imputed NFY convergence across the ten TFs is mostly depth / promoter convergence. Only NFYA (own motif) and ZNF507 keep an NFY signal on depth-equalized raw reads. The NFYA-SOX4 / NFYA-RBBP4 / MLX-NFYA motif leads are withdrawn. NFYA-ZNF507 stays as a weak lead (raw loose at both depths, AF3 none).

## Step 6 (2026-11-10): external bulk route abandoned at the user's request
- The bulk_anchor/ route (ENCODE / ChIP-Atlas HEK293 peaks as anchors) was started: ENCODE search and one peak file per TF downloaded, no tests run.
- User 17:45: drop it, because the study is HEK293T, not HEK293. The folder was deleted, and nothing of it was committed.
- Coverage before the stop, for the record: HEK293 bulk exists for 6 ENCODE TFs (CTCF, MAZ, ZBTB7A, ZIC2, ZNF768, ZNF777) plus about 13 ChIP-Atlas GEO sets. 58 of 117 AF3/triple pairs had one partner covered.
- Replaced by the threshold-free motif profile (profile/).
