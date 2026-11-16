# m31: Perturb-seq-supported Regulome edges (descriptive tables only, no significance tests)

User direction (2026-11-16): for Figure 5, Perturb-seq is taken as the gold standard and no significance tests are run; describe which Regulome edges it supports and list other existing evidence for our data.

Definitions:
- Regulome edge: a (TF, gene) row of the primary edge table m22/out/regulome_edges.tsv.gz (commit 9b8d8c1: top 5,000 peaks per TF, summit in the promoter element or a canonical Hi-C ABC-linked element, ABC >= 0.02).
- Knockdown z: Arm 2 raw per-TF z, all cells (m15 "raw" ZA, as m29), halves Z1/Z2; TF's own gene removed (HP1 -> CBX1/3/5, ZUFSP -> ZUP1). Genes: the 11,228-gene universe. TFs: the 75 with both a knockdown and edges.
- Supported edge: a Regulome edge whose gene has |z| >= 2 for that TF's knockdown.
- Direction: z < 0 = gene goes down on knockdown (TF activates it); z > 0 = up (TF represses it).
- Reproducible: |z| >= 2 in both cell halves with the same sign.
- Peak location: promoter (promoter peak only), distal (linked distal element only), both.
- Score percentile within TF: the edge table's within_tf_percentile (percentile of signalValue x ABC among the TF's bound genes); promoter_signalValue also given.
- ABC: best_abc (ABC of the strongest linked element; self-promoter included).
- n other panel TFs bound: shared_by - 1.
Outputs: supported_edges.tsv; per_tf_summary.tsv (n Regulome targets, n responding genes, n supported, n down, n up, n reproducible, % of all genes bound, % of responding genes bound); per_gene_summary.tsv (n TFs bound, n TFs changing it, n both, names with z), top 30 by n both reported.
Evidence inventory (evidence_inventory.tsv), numbers copied from existing outputs only, nothing rerun: m20 part2_bulk_overlap (fraction of bulk ChIP peaks with an own summit, rank among 79); m25 choice_chip and choice_motif for the top-5,000 definition (own vs other-TF median; own motif fraction vs background); m12 per-cell on-target summaries (CTCF, NFYA, NRF1, ZNF143); Fig 2 QC sc-vs-bulk own-reference Pearson r from RESEARCH_JOURNAL.md (2026-11-10). One row per TF and evidence line, with the source folder.
No p values. Results labelled descriptive.
