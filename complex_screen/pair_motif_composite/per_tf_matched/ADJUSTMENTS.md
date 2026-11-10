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
