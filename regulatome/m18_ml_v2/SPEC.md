# m18_ml_v2: Regulome ML model version 2 with richer inputs (spec, 2026-11-11)

Status: HEK293T X-Atlas/Orion knockdown data already read; m7 test chromosomes chr3/11/17/20 are spent. Everything here is "benchmark-developed, cross-validated", never "independent validation".

TFs: primary = the 10 strong-phenotype TFs of m16 (CTCF, DNMT1, E4F1, MEF2A, NRF1, PA2G4, RBBP7, RBPJ, SAFB, SETDB1); secondary = all TFs with a knockdown, reported separately.

Target: raw per-TF z (Arm 2), signed, genes detected >= 10% in NTC, knocked-down gene removed; centred per TF on training folds only.

Binding: raw HITAG fragments and lab q005 peaks (primary); imputed as a labelled side arm. Links: canonical ABC (Hi-C), ABC >= 0.02.

Feature blocks:
A. Gene baseline: promoter ATAC, promoter H3K27ac, CpG island, number of linked elements, summed ABC score, ChromHMM state at promoter, Hi-C loop-anchor flag, H3K27ac summed over linked distal elements.
R. RNA-seq: gene expression and detection in HEK293T from the pseudo-bulk of the 218,838 NTC cells (mean log1p CP10k); the TF's own expression; expression of all 79 panel TFs (constant across genes). If a bulk HEK293T RNA-seq is on Longleaf, report its path; do not download one.
H. Histone modifications: inventory first what exists on Longleaf for HEK293T or HEK293 beyond H3K27ac (H3K4me3, H3K4me1, H3K27me3, H3K36me3, H3K9me3; path, source, cell line labelled exactly as HEK293T or HEK293); promoter and linked-element signal for each available mark. Missing marks are listed as missing, not substituted from another cell type. Report the inventory before downloading anything.
C. Motif (no HITAG): own-motif best score and count at the promoter and over linked distal elements (HOCOMOCO v11, the corrected complete scans; missing motif = indicator).
B. This TF at this gene (HITAG): promoter reads; promoter peak signalValue; ABC-weighted distal sum; strongest distal element; for the top 3 linked elements by reads x ABC: reads, ABC score, distance to TSS; distance from TSS to nearest q005 summit; the gene's percentile within the TF for promoter reads and peak score.
BC. Motif x binding: reads within 100 bp of an own-motif site at promoter and linked elements; motif under a q005 peak yes/no.
D. Context: for all 79 TFs, promoter reads and ABC-weighted distal sum at the gene (158 features); number of TFs with a peak linked to the gene.
F. Coordinated binding, reused as built from the Figure 3 analysis (nothing recomputed; locate the module / higher-order coordinated binding tables, Cicero edges, hub and super-enhancer annotations and report their paths): per linked element: module membership, number of panel TFs bound, hub membership, Cicero edge to the promoter and weight, super-enhancer overlap; summarised per gene; per (TF, gene): whether this TF is a module member at the gene's elements and how many other members are bound with it. Never write "clique"; say "higher-order coordinated binding".
G. Complex pairs (provisional, list not final): for each (TF, gene) and every candidate partner of the TF in the current frozen candidate list (complex_screen; record path and sha256): partner binding at the same element (min of the two) summed over the gene's elements with ABC weights; number of partners co-bound; flags AF3-strong and literature-known.
E. TF descriptors: cells, depth, on-target KD %.

Models: LightGBM main; elastic net as linear reference. No neural net unless LightGBM shows a gain.

Evaluation: grouped 5-fold cross-validation over whole chromosomes (fold file fixed with seed 20261112, balanced on gene count, sha256 committed before fitting); tuning inside training folds only; out-of-fold predictions for every gene. Statistic: per-TF Spearman between out-of-fold prediction and observed z, also as a fraction of that TF's reliability ceiling; mean over TFs with paired TF-bootstrap (2,000 draws).

Ladder (same folds): A; A+R; A+R+H; A+R+H+C (no HITAG); then +B+BC; +D; +F; +G (full). Declared readings: "the richer model adds" = full minus A, interval above 0. "HITAG contributes" = full minus (A+R+H+C), interval above 0, AND larger than the same with all HITAG-derived blocks (B, BC, D, F membership, G) replaced by a swapped TF's binding (same depth tertile) and by pseudo-TF binding. Each added rung also reported as its own paired difference ("coordinated binding adds" = +F; "complex pairs add" = +G). Unfitted references on the same genes: peak x ABC score, promoter reads, motif score. If "HITAG contributes" fails while the full model beats A, the result is written as "a model of chromatin, RNA and motif predicts knockdown response; our binding does not add", and nothing further is fitted to rescue it.

Outputs: folds file, per_tf_oof.tsv, ladder.tsv, per-TF SHAP summaries by block, out-of-fold predictions per (TF, gene) with each TF's contribution to each gene (for the gene-example panel), README; copies to /Users/dyy/Desktop/HEK293T-ATGTAG/perturbseq_matrix/ml_v2/.

Order: (1) commit SPEC; (2) inventories for R, H, F, G: report paths and what is missing, since data may need to be supplied; (3) build features, fit, ladder.
