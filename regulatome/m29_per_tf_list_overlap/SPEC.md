# m29: per-TF list overlap, Regulome gene list vs Perturb-seq responder list (diagnostic, tables only)

User request (2026-11-13): binarise Perturb-seq into a list of genes regulated by each TF using a cutoff, and compute the simple overlap per TF with the Regulome (ABC) gene list.

Lists (fixed before any overlap is computed):
- Regulome list per TF: genes with an edge in the primary edge table (m22 regulome_edges.tsv.gz, commit 9b8d8c1: top 5,000 peaks per TF, summit in the promoter element or a canonical Hi-C ABC-linked element, ABC >= 0.02). Plain gene list, no score.
- Perturb-seq list per TF: Arm 2 raw per-TF z (all cells; m15 "raw" matrices: ZA all cells, Z1/Z2 the two cell halves), the TF's own gene removed (HP1 -> CBX1/3/5, ZUFSP -> ZUP1).
  C1 |z| >= 2; C1_down z <= -2; C1_up z >= 2; C2 |z| >= 3; C3 top 200 genes by |z|; C4 reproducible responders (|z| >= 2 in both halves, same sign).
- Universe: genes in the m9 Regulome universe (11,228) that are expressed in HEK293T, defined as detected in >= 10% of NTC cells (the m15 rule); N reported.
- Testable: TF in both tables with both lists >= 5 genes in the universe.

Per TF and cutoff: list sizes, overlap, expected overlap (a*b/N), fold over expected, Jaccard, one-sided hypergeometric p (enrichment), BH q across the testable TFs of that cutoff.
Nulls per TF:
(a) swapped TF: Regulome lists of the other TFs in the same depth tertile (m6 per_tf_floor_test tertile) against this TF's Perturb-seq list; report their median fold, the own TF's rank, and empirical p = (1 + #others with fold >= own) / (1 + #others).
(b) matched genes: 1,000 random gene sets of the Perturb-seq list's size, matched on NTC expression decile x promoter ATAC decile x CpG island (as m22 Amendment 2), overlapped with the own Regulome list; mean overlap and empirical p = (1 + #draws >= observed) / 1,001.
Summary per cutoff: TFs testable; TFs with q < 0.05; of those, TFs that also have swapped-TF p <= 0.05 and matched-gene p <= 0.05 ("pass"); median fold own vs median of swapped; list of passing TFs. Per-TF table flags the 17 TFs with reproducible knockdowns (m27 F3).
Note: hypergeometric p alone is expected to look good for many TFs, because both lists favour expressed, open-promoter genes; the swapped-TF and matched-gene nulls are what show TF-specific overlap. Results labelled diagnostic.
