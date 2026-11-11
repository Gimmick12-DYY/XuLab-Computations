# ABC audit: canonical recipe vs m2_abc vs m2b_abc_k27ac (2026-11-10)

"Canonical" = EngreitzLab/broadinstitute ABC-Enhancer-Gene-Prediction run through its own Snakemake workflow with the repo's default
config (config/config.yaml), i.e. the published recipe (Fulco 2019; Nasser 2021; Gschwind 2023 for v1.x). Repo defaults are quoted from v1.1.2.

| Step | Canonical (repo default; m2c plan) | m2_abc (2026-10-28) | m2b_abc_k27ac (2026-11-10) |
|---|---|---|---|
| Code | v1.1.2 (tag, commit e2a88d2, 2024-03-22) = latest tagged release; `main` is at 92ac503 with later untagged commits. m2c: v1.1.2 via `snakemake` with the repo Snakefile and rules | v1.1.2 scripts called one by one (run.neighborhoods.py, predict.py, filter_predictions.py), not the Snakefile | same as m2_abc |
| Peak calling | MACS2 on accessibility: `-p 0.1 --shift -75 --extsize 150 --nomodel --keep-dup all --call-summits -g hs` | none (no MACS2 step) | none |
| Candidates | top 150,000 peaks by read count, summit +/-250 bp, merged, `GRCh38_unified_blacklist` removed, TSS +/-250 (TSS500bp) always included (makeCandidateRegions.py) | elements_v2 classes promoter + distal_active + bivalent (ATAC union3 + ChromHMM active / bivalent, >2 kb tiled to ~1 kb): 159,965 elements; no blocklist step, no 500 bp re-sizing, no TSS include-list | same 159,965 elements as m2_abc |
| Activity | sqrt(qnorm ATAC x qnorm H3K27ac) when both given (geometric mean); ATAC alone if no H3K27ac | qnorm ATAC only | sqrt(qnorm ATAC x qnorm H3K27ac) |
| Activity inputs | user data | omniatac_merged.tn5.tagAlign.gz (4 OmniATAC reps) | same ATAC + H3K27ac ChIP-Atlas SRX18002015/16 merged tagAlign (74.9M reads) |
| Normalisation | `use_qnorm: True`, K562 EnhancersQNormRef, promoters and non-promoters quantile-normalised separately | same | same |
| Contact | user Hi-C if given (`--scale_hic_using_powerlaw`, gamma 1.024, scale 5.96, pseudocount at 5 kb; power-law fills missing contacts); power-law only if no Hi-C | power-law only (Predictions_powerlaw, 104,071 links = the set used downstream); WT Hi-C 5 kb run kept as a side arm (Predictions_hic5kb, 100,875 links) | power-law only (Hi-C variant only as a follow-up) |
| Hi-C file | user .hic or juicebox dir | GSE216399 WT MboI R1 mcool 5 kb, `cooler --balanced` = ICE weights, <= 5 Mb, written in juicebox layout as `chrN.KRobserved.gz` (the name says KR; the values are ICE) | n/a (follow-up used the same files) |
| ABC score | A x C / sum(A x C) over all candidates within 5 Mb of the TSS (`window 5000000`) | same | same |
| Threshold | config `threshold: null` = automatic from reference/abc_thresholds.tsv: ATAC + H3K27ac + powerlaw = 0.016, ATAC + H3K27ac + intact_hic = 0.025; a juicebox Hi-C type has no row, so it falls back to DEFAULT_THRESHOLD 0.02. m2c uses 0.02 (spec) | 0.02 | 0.02 |
| Promoters | `include_self_promoter: True`; promoter elements = candidates overlapping TSS500bp; `only_expressed_genes: False` | include_self_promoter True, only_expressed_genes False; promoter status from ABC's own TSS overlap on elements_v2 | same |
| Expressed-gene rule | no RNA: gene "expressed" if promoter activity quantile >= 0.3 or ubiquitously expressed; with `only_expressed_genes: False` all genes are kept (the flag only splits output files) | same | same (PromoterActivityQuantile also uses H3K27ac at TSS) |
| Genes / TSS | repo CollapsedGeneBounds.hg38 (all gene types) + TSS500bp; m2c overrides with `alt_genes` / `alt_TSS` = GENCODE v46 protein-coding (genes_pc_v46.bed, 20023 genes) and its TSS +/-250 (spec: same TSS file as m2_abc) | genes_pc_v46.bed | genes_pc_v46.bed |
| Chrom sizes | repo GRCh38_EBV.no_alt; m2c uses m2_abc chrom_sizes.tsv (chr1-22, X, Y) because the ATAC / H3K27ac tagAligns are sorted in that order | m2_abc chrom_sizes.tsv | same |
| TF-target calling | not part of ABC. Literature practice: TF ChIP peaks overlapping ABC enhancers -> linked genes. m2c: lab MACS2 peaks q <= 0.05, summit in a candidate -> all genes with ABC >= 0.02 (binary target set, no ranking) | none in m2_abc; downstream R5 core = sc raw CPM per element x ABC summed per gene (top 1,000); B6 list e = raw CPM summed over linked elements (top 1,000) | benchmark only (same continuous scores, top 1,000) |
| Perturb-seq read-out | m2c: Fisher OR of targets vs top-5% abs(z) responders, nulls (matched elements, ATAC top-N), plus B6 effect / Wspec | B6 / m6: own-KD z vs 1,000 expression-matched random sets, Wspec, top-5% logOR | same as m6 |

Departures from the canonical recipe in m2_abc / m2b: (1) no MACS2 candidate step (elements_v2 instead, larger tiled elements, no blocklist,
no TSS include-list); (2) the downstream link set is power-law only, not Hi-C; (3) m2_abc has no H3K27ac; (4) TF targets were continuous
sc-binding scores, not peak-based. Shared by all three, not canonical defaults but declared: protein-coding GENCODE v46 genes, m2_abc
chrom sizes, threshold 0.02, ICE-balanced (not KR) Hi-C.

## Correction, 2026-11-11: Hi-C files were empty
`m2_abc/dump_hic.sh` wrote awk `$9`, but the balanced value is `$8`, so `m2_abc/inputs/hic5kb/*/*.KRobserved.gz` has no values on any chromosome. ABC ran without error, with contacts missing and replaced by the power-law curve
(m2_abc Predictions_hic5kb: 70% of non-self links equal to power-law, 30% contact 0; first m2c Hi-C arm: 100% equal). The Hi-C row above describes what the file was meant to hold. Real Hi-C entered only with
`inputs/hic5kb_fixed` (dump_hic_fixed.sbatch; ICE-balanced values, mean row sum about 0.82 from the 5 Mb cut, chrN.KRnorm.gz = vectors of 1.0 used only for matrix size). Power-law runs (m2_abc 104,071 links, m2b, B6 / m6 / manifest v8) do not use these files and are unaffected.
