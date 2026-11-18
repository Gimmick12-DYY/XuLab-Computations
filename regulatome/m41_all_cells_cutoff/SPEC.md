# m41: cutoff table with all cells instead of top-25% sgRNA cells

Question: with the current link rule (A, top 5,000 peaks, promoter/ABC), how does the |z| cutoff table change if the Perturb-seq z-scores come from all cells (m15 "all" matrix, ZA) instead of the top-25% sgRNA cells (m23)?

- Our lists: rule A exactly as in m38 (also rule G for reference).
- Perturb-seq genes: |ZA| >= 2, 3, 4, 5; own genes removed as in m15.
- Report per cutoff: Perturb-seq genes left, TFs with any gene left, in both, share of Perturb-seq genes found, share of our genes found.
- Two TF sets: all TFs in the all-cell matrix, and the same 63 TFs as the top-25% table.
- Top-25% numbers printed beside for comparison (recomputed, must match m38: 17.7 / 19.2 / 22.1 / 24.4% for rule A).
