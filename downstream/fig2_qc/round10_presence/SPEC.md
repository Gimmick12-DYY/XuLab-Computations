# Round 10: peak-presence heatmap, 4 bulk ChIPs vs our data (declared 2026-11-23, before any result)

User request 2026-11-23: combine 4 bulk ChIPs (drop MAZ and one ZNF), rows = genome-wide peak regions, mark presence,
remove regions shared by all 4, and show our data side by side to see whether the pattern matches.
- **ChIP sets.** Main: CTCF (ENCODE HEK293 ENCFF314ZAL), ZBTB7A, ZIC2 (ENCODE HEK293), ZNF282 (GSM2466515 HEK293T).
  ZNF777 is dropped from the main set because its ChIP is the weakest (FRiP 0.004, RSC < 0.8). Variant: ZNF777 in place
  of ZNF282.
- **Rows.** Union of the 4 ChIPs' summit +-250 bp intervals, merged (bedtools merge), chr1-22 + X, hg38 blacklist v2
  removed. Presence per ChIP = the row overlaps that ChIP's summit +-250 bp. Rows present in all 4 are removed. Rows are
  grouped by presence pattern: the 4 single-ChIP blocks first (in panel order), then the 6 pairs, then the 4 triples.
  Within a block, rows are ordered by the summed bulk ChIP CPM of the 4 ChIPs over the row (bamCoverage bigWigs, mean
  CPM x row length), highest first. n per pattern is reported.
- **Our data, same rows, same 4 TFs.**
  (A) presence: a lab MACS2 peak (q < 0.05, /vast/.../bigwig/macs2/TF.<TF>) overlaps the row.
  (B) signal: raw read 5' ends in the row; log2(((x + 0.5) / T_tf) / ((p + 0.5) / T_pool)), x = the TF's reads,
      p = reads of the other 78 TFs (leave-one-out pool), T = total reads; this removes the component shared by all TFs.
  (C) imputed presence: the Mdrop5-masked imputed fraction of cells in the 1 kb bin holding the row midpoint is at or
      above that TF's 95th percentile over the Mdrop5 bins.
- **Display.** Left: 4 ChIP presence columns; right: (A), (B), (C), 4 columns each. Rows averaged in consecutive groups
  within each block (group size = ceil(total rows / 600), at least one drawn row per block), so presence columns show
  the fraction present (white 0 to black 1); (B) on a diverging map centred at 0, capped at the 95th percentile of
  |value|. Block labels with n on the left.
- **Table per block.** Mean (B) and fraction present (A, C) for each of our 4 TFs. Expected if our data match: in each
  single-ChIP block, the matching TF has the highest value.
Scripts: downstream/fig2_qc/r10_presence.py; paper_figures/fig2/panels/f_presence.py.
