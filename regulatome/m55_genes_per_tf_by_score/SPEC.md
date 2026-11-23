# m55: genes left per TF at each per-gene score level (spec, 2026-11-23, written before any number)
User: "I want to know on average how many genes will be left for each TF at each gene score level."
Input: m53 full-precision paper per-gene score, 63 TFs x 11,226-11,227 genes (own gene removed). For |score| cutoffs 0.05 to 1.00 in steps of 0.05: per TF the number of genes with |score| >= cutoff (and split up / down by sign); across the 63 TFs mean, median, 25th / 75th percentile, min, max, TFs with 0 and with < 10 genes; mean per cell tertile (25-42 / 43-73 / 74-216 top-25% cells). One figure: x = |score| cutoff, y = genes per TF (log scale), median with IQR band and min-max range, mean as a second line. Per-TF table (63 x cutoffs). Descriptive; no Stage 1 overlap.

## Amendment 1 (2026-11-23, layout requested in the thread; written before the new figure)
Cutoffs 0.05 to 0.80 in steps of 0.05. Summary: mean, median, min, 25th, 75th, max of genes left per TF, and the median within the cell groups 25-42 / 43-73 / 74-216 top-25% cells. Per-TF sheet (TF, cells, genes left at every cutoff). Second sheet: the |score| at the top 10 / 5 / 2 / 1 % boundary of each TF (mean, median, min, max over TFs). Figure "genes_left_per_TF_by_score_cutoff": x = |score| cutoff, y = genes left per TF (log), one thin grey line per TF, one bold line for the mean of the 63 TFs; axis labels and a two-entry legend only.

## Amendment 2 (2026-11-23, user: y = median genes, x = gene score)
The figure shows only the median number of genes per TF (over the 63 TFs) at each |score| cutoff 0.05-0.80 (log y; cutoffs where the median is 0 are not drawn). Tables unchanged.

## Amendment 3 (2026-11-23, replaces the plan in Amendment 2, which was not applied: m55.py is unchanged)
Separate figure "median_genes_per_TF_by_score" from the existing m55 table (no new analysis): bar chart, x = |score| cutoff 0.05-0.80 (16 bars), y = median genes per TF over the 63 TFs, linear from 0, value above each bar; the genes_left_per_TF_by_score_cutoff line figure is kept.

## Amendment 4 (2026-11-23, user: "How does gene score relate with fold change?"; written before any number)
Over the 63 TFs x genes (Stage 1 pairs): Spearman of score with log2FC (pseudocount 0.1, m50) and of |score| with |log2FC|; per |score| band of 0.05 (0-0.05 ... 0.75-0.80, >= 0.80) and per cutoff |score| >= 0.05 ... 0.80: pairs, median and IQR of |log2FC| and as % change; the same median split by NTC expression tercile (from the matched-shuffle expression deciles: 1-3 / 4-7 / 8-10). Table only (m55_score_vs_fc.py).
