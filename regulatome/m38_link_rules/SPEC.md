# m38: other ways of linking our peaks to genes, and the plain overlap with Perturb-seq (descriptive, tables only)

Question (user, 2026-11-18): with the current rule 17.7% of Perturb-seq genes are found in our Regulome, and most top-5,000 peaks of a typical TF are unused (m37). How does the plain overlap change under other linking rules, all from our own HITAG data?
Fixed as m36: 11,228-gene universe; m23 top-25% sgRNA matrix; the 63 TFs; own gene removed; genes without a z excluded per TF.
Peaks: lab MACS2 narrowPeak per TF, ranked by signalValue (col 7), summit = start + col 10. Elements and links: canonical Hi-C ABC (m2c HEK293T_hicfixed, ABC >= 0.02, self-promoter included). TSS: ABC GeneList "tss" per universe gene.
Rules (binary TF x gene):
A current: top-5,000 summit in a promoter element or an ABC-linked element of the gene (must reproduce m36: 117,845 links, 11,337 in both at |z| >= 2).
B A + any top-5,000 summit within +/-10 kb of the gene's TSS. C A + within +/-50 kb.
D A + every top-5,000 peak not already in a promoter or ABC-linked element assigned to its nearest TSS: D100 (only if <= 100 kb), Dall (no cap).
E A + Fig 3 coordinated-binding links: the TF's frozen Cicero edges (cobinding/work/ourrun_k50/<TF>/<TF>.peak_edges.tsv, as used for Fig 3), edges with peak distance > 25 kb; if one end lies within +/-1 kb of a universe TSS, the TF is linked to that gene (via its distal partner peak). Links added by E alone are reported.
F the A rule with top 10,000 peaks (F10k), top 20,000 (F20k; TFs with fewer peaks keep all) and all q < 0.05 peaks (Fq).
G B + E.
Per rule: at |z| >= 2 and >= 3, per TF and pooled: our genes, Perturb-seq genes, in both, share of Perturb-seq genes found in our Regulome (in both / Perturb-seq genes), share of our Regulome genes found in Perturb-seq (in both / our genes); per-TF medians. Pooled also at |z| >= 4 and >= 5 (with n TFs that have >= 1 Perturb-seq gene), and for down-regulated (z <= -2) and up-regulated (z >= 2) genes separately. No expected, fold, random-gene or swapped-TF columns.
