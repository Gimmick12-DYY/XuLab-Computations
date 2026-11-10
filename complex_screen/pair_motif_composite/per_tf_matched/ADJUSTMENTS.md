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
- Sanity: pending.
