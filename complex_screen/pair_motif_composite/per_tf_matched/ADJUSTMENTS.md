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
- Pending: raw de novo tasks (array 8754925).

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
