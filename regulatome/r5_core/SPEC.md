# R5 Regulome core (manifest v7)
Spec from the design thread (source: regulome_validation/regulome_core_spec.md; supersedes the two-layer spec), key points as relayed 2026-11-09:

Score: S(t,g) = sum over m2_abc links (Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv, 104,071 links, promoters included) of CPM_te x ABC(e,g). Raw reads per TF at the m2_abc elements, counted as in m1_binding/v2, converted to CPM per TF. No gate, residual, EB, weights, propagation or imputation. All 79 TFs treated identically.
Companion columns: S_atac (union ATAC CPM x ABC); S_pool (leave-TF-out pooled panel CPM x ABC).
Per pair: N_e (elements with >= 1 read); promoter_share (class == promoter contribution / S) and distal_share; n_loop_elements (distal elements inside a Peakachu 5 kb 0.7 anchor whose partner overlaps the gene's TSS +/- 2 kb); max_ABC; distance of the top-contributing element.
Lists: lists/<TF>.tsv.gz = all genes with S > 0 ranked by S (ties by N_e); top_1000 flag requiring N_e >= 2, with the number of genes affected.
tf_summary.tsv per TF: genes with S > 0; top-1000 median promoter_share; fraction of the top 1000 with >= 1 loop-supported distal element; Spearman(S, S_atac) and Spearman(S, S_pool); split-half top-1000 Jaccard (v2.1 first split, seed 20261029 + TF index); overlap of the top 1000 with the frozen v3 direct list, the B3/B4 promoter-only list and the S_atac top 1000.
Checks (no Perturb-seq): 7 ChIP TFs, fraction of top-1000 genes with a bulk ChIP peak in any contributing element, split promoter vs distal, vs the same for the S_atac top 1000; HPA Spearman of S vs nTPM and of S_atac vs nTPM; promoter-only vs distal-only top-1000 Jaccard with each other and with the full list.
Edges: edges/loop_anchor_promoter.tsv.gz.
Run rules: manifest_v7.sha256 written last, run refuses if it exists; scripts under regulatome/r5_core/; 1 CPU, 8-16 GB, < 20 min; new folder only; no Perturb-seq.

## Implementation notes
- "Distal" = ABC class genic or intergenic (not promoter). Contributing element = linked element with >= 1 read (for S_atac: ATAC count > 0).
- n_loop_elements counts distal linked elements with >= 1 read in a promoter-paired 5 kb anchor of that gene.
- top_1000: the 1,000 highest S among genes with N_e >= 2 (stable sort); top1000_excluded_by_Ne = genes in the unrestricted top 1,000 that the N_e rule removes.
- Split-half lists use the same N_e >= 2 rule within each half. Promoter-only / distal-only lists rank the promoter and distal parts of S, with N_e >= 2 on the full data.
- B3/B4 promoter-only list = top 1,000 of m4_validation/arms/promoter_only/<TF>.tsv by S; v3 direct list = regulatome/r2_targets/top1000/<TF>.tsv.
