m22_edge_table: the HEK293T Regulome edge table (SPEC.md, commit b3bd84d). Scripts: m22_build.py, m22_analyse.py, tf2genes.py, gene2tfs.py, make_gene_view.py, make_tf_view.py.
DEFINITION. The Regulome is defined from our HITAG data only (lab q005 MACS2 peaks of 79 TFs), with HEK293T ATAC, H3K27ac and Hi-C as auxiliary inputs for element-gene links (canonical ABC, ABC >= 0.02, self-promoter included).
An edge = a (TF, gene) pair with one of the TF's top 5,000 lab peaks (by signalValue, as called) with its summit in the gene's promoter element or an ABC-linked element (Amendment 4, the user's choice). The q005 version is kept as regulome_edges_q005.tsv.gz. One definition, no classes; shared targets are kept.
Perturb-seq (X-Atlas/Orion HEK293T) never defines an edge. It appears only in columns marked [external], as a check.
regulome_edges.tsv.gz columns: binding (promoter peak and signalValue, distal elements with a peak, best element, ABC and distance, score = sum signalValue x ABC, percentile within TF, promoter reads CPM);
combined-panel results (shared_by, co_bound_tfs, panel_relative_ratio - descriptive only, never a filter; frozen partner TFs at the same element; Figure 3 coordinated binding module of the best element);
own_motif_promoter / own_motif_best_element (m10 corrected scans; empty = TF has no motif);
enrichment_over_background and background_p_min (reads vs an ATAC-based expectation fitted per TF; NOT VALIDATED, see check_background_verdict.tsv; predicted_impact uses plain signalValue x ABC);
predicted_impact (ours) and predicted_impact_share_at_gene (share of all TFs' scores at that gene); tier (m21); [external] knockdown z all cells, halves, reproducible flag, reliability, strong phenotype.
IgG: /vast/som/xujie_lab/CUT&Tag/IgG (N2.IgG, hg38) is NOT used; per the user (2026-11-12) it likely belongs to another project. Background = ATAC only.
Checks (diagnostic): check_background_column.tsv / check_background_verdict.tsv; agreement_check.tsv (predicted impact vs external knockdown, with swapped-TF control); tf_x_tf_jaccard.tsv (shared regulation; part of it reflects open-chromatin background, m20/m24).
Summaries: per_tf_summary.tsv, per_gene_summary.tsv. Lookups: lookup_tf_<TF>.tsv, lookup_gene_<GENE>.tsv. Views: views/gene_<GENE>.png/.svg, views/tf_<TF>.png/.svg.
