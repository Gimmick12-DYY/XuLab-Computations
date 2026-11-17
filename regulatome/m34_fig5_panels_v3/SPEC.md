# m34: Figure 5 (Regulome) tables, panels and slides (descriptive; Perturb-seq as reference, no significance tests)

User approved the proposal (2026-11-17): Regulome = binding layer (ours) + per-TF evidence + Perturb-seq layer (external) + combination layer.
Defaults (user approved as written): tiers at |z| >= 2, Tier 1 = reproducible in both cell halves; panel c includes K562/HepG2 references labelled by cell line;
examples NRF1 (TF view) and HSPB1 (gene view), ATP5MC2 and MDK as extra gene tables; ML model in the supplement only.

Frozen Regulome table (regulome_v1): every edge of m22 regulome_edges.tsv.gz (commit 9b8d8c1: top 5,000 HITAG peaks per TF, summit in the promoter element or a canonical Hi-C ABC-linked element, ABC >= 0.02) plus:
- knockdown_z (Arm 2 raw per-TF z, all cells, own gene removed; m15 "raw"), z_half1, z_half2, knockdown_available;
- tier: 1 = |z| >= 2 and |z| >= 2 in both halves with the same sign; 2 = |z| >= 2, not reproducible; 3 = bound only (|z| < 2, or no knockdown / gene not measured);
- sign: activates (z <= -2: gene goes down on knockdown), represses (z >= 2), none;
- n_other_tfs_bound = shared_by - 1;
- per-TF evidence: own-motif excess in top-5,000 edge elements (m25), bulk ChIP reference and recall/precision rank among 79 (m20), our vs bulk overlap fold through the identical path (m33), fragments on own motif vs swapped TF (m12), Fig 2 QC own-reference r; tf_evidence_lines = number of lines that are positive
  (motif excess >= 5 points; bulk ChIP recall or precision rank <= 10 of 79; own-motif fragment enrichment exceeds the swapped TF by >= 0.1; Fig 2 r >= 0.3); tf_evidence = "multiple lines" (>= 2), "one line", "binding only".
Panel tables: b (targets per TF with promoter/distal split; TFs per gene distribution), c (m33 Part B), d (NRF1 funnel and supported edges), e (HSPB1, ATP5MC2, MDK: all bound TFs with z, tier, sign), f (activator/repressor share per TF among supported edges, Tier 1 counts), g (Figure 4 pairs MLX-SREBF2, ETS2-RBBP7, DR1-NFYA and all AF3 5-seed "strong" pairs from complex_screen/candidates_frozen_v1: genes bound by both members, and each member's knockdown z at those genes; SREBF2's knockdown has 18 cells and is marked too few).
Panels drawn twice: paper versions (Arial 7 pt, palette option C, figures_paletteC/regulome_palette.py) and slide versions at the deck's picture frame size (12.9 x 5.8 in) with text >= 12 pt.

## Amendment 1 (2026-11-17, user request): panels h and i from the m35 threshold sweep (descriptive)
h: share of Perturb-seq links (TF knockdown changes the gene) found in the Regulome (that TF bound at the gene), pooled over TFs, at |z| >= 2, 3, 4, 5, with the edge-count reference (pooled share of genes bound, weighted by responders); annotated with n TFs with >= 10 responders and n links. Facet: the 10 TFs with bulk ChIP, ours vs bulk (one reference per TF as m35), each with its own reference.
i: per TF, % of Tier-1-level links found (both halves |z| >= 2, same sign) vs the TF's reference, ordered by the difference, TFs with < 10 such links greyed; second row the same at |z| >= 4.
high_confidence_links.tsv: every link at the Tier-1 level or at |z| >= 4, with found in Regulome (yes/no), peak location, ABC, z. Source: m35 per_tf_threshold_sweep.tsv and m34 regulome_v1.
