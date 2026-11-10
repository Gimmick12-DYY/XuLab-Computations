# m2b_abc_k27ac: Regulome ABC with H3K27ac in the activity term

Same ABC v1.1.2 run as `../m2_abc` (power-law contact, the 104,071-link set used by B6 / m6_improve / manifest v8), with one change:
activity = sqrt(quantile-normalised ATAC x quantile-normalised H3K27ac) (Fulco 2019 geometric mean) instead of ATAC alone.
Everything else is unchanged: candidates `m2_abc/inputs/candidates_active.bed` (promoter + distal_active + bivalent from elements_v2),
genes `genes_pc_v46.bed`, ATAC `omniatac_merged.tn5.tagAlign.gz`, K562 `EnhancersQNormRef` (it has an H3K27ac.RPM column), power-law
gamma 1.024 / scale 5.96, 5 Mb window, ABC >= 0.02, self-promoters included, all genes (only_expressed_genes False).
Note: with H3K27ac supplied, ABC's PromoterActivityQuantile (used only for the expressed/non-expressed split) also uses H3K27ac at the TSS.

## H3K27ac source
ChIP-Atlas HEK293T WT H3K27ac, GEO GSM6672008 / GSM6672009 = SRX18002015 (SRR22020179, 27.1M pairs) and SRX18002016 (SRR22020178, 25.1M pairs),
paired-end, fastq from ENA. Chosen because we have no HEK293T histone data of our own and ChIP-Atlas does not include GSE246314.
Processing: bowtie2 2.x `-X 2000 --no-mixed --no-discordant` to hg38 (`atac_validation/ref/hg38`, built from `downstream/cache/hg38.fa`),
proper pairs, MAPQ >= 30, `samtools markdup -r`; both reps merged into one tagAlign (one line per mate, like the ATAC input) sorted in the
m2_abc chrom order. No input/IgG is used (ABC counts reads in candidate elements only).
Possible later upgrade: GSE246314 untreated reps 1-3 (GSM7866612-14; same cells as the SE side check) realigned from SRA the same way.

## Scripts (run in this order; chained with Slurm afterok)
| step | script | output |
|---|---|---|
| fetch | `k27ac_fetch.sbatch` (array 0-1) | `fastq/` |
| align | `k27ac_align.sbatch` (array 0-1) | `bam/*.q30.dedup.bam`, bowtie2 / markdup / flagstat logs |
| merge | `k27ac_merge.sbatch` | `inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz` (+ .tbi) |
| ABC | `abc_k27ac.sbatch` | `results/Neighborhoods`, `results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv` |
| sanity | `k27ac_sanity.sbatch` -> `k27ac_sanity.py` | `results/k27ac_sanity.tsv` (vs ChIP-Atlas bigWigs, FRiP in GSE246314 peaks, TSS share) |
| compare + benchmark | `post_k27ac.sbatch` -> `compare_abc.py`, `bench_k27ac.py`, `bench_summary.py` | `results/compare/`, `bench/` |

## Benchmark (descriptive, development TFs only)
`bench_k27ac.py` = m6_improve grid6 scoring unchanged (kd_response/out_v2, 1,000 expression-matched random sets per list, seed 20261109 + arm,
own z = B6 effect, specificity = own minus mean off-KD z over development KDs, own-KD rank, top-5% logOR), on the 52 development TFs of
`m6_improve/split_v1.tsv`. The 24 held-out TFs are spent (heldout_v1) and are not scored here. Gene scores, top 1,000 per TF:
(a) `oldX_raw` binding CPM x ABC over m2_abc links; (b) `newX_raw` binding x ABC over m2b links; (c) `oldS_raw` binding-only over m2_abc
links (B6 list e analogue, grid6 `abc_raw`); (d) `newS_raw` binding-only over m2b links. Controls on the same link set and weighting:
`CTRL_n_elem`, `CTRL_pool` (mean other-TF CPM), `CTRL_atac`. `bench_summary.py` = spec_summary rules (b3b4 ceiling weights, bootstrap 10,000,
seed 20261116), with and without NRF1, plus new - old contrasts.

## Results (2026-11-10; dev-set tuning, descriptive, not confirmed)
Source QC: SRR22020179 / SRR22020178 = 27.1M / 25.1M pairs, 94.1% / 94.0% aligned; after proper-pair MAPQ>=30 and markdup -r 38.9M / 36.1M
reads kept (duplicate pairs 13.6% / 12.9%); merged tagAlign 74.9M reads. Sanity (`results/k27ac_sanity.tsv`): Spearman of our counts per candidate
element vs the ChIP-Atlas bigWig 0.86 / 0.86 (ATAC vs the same bigWig 0.65); our H3K27ac vs ATAC 0.68; FRiP in GSE246314 union peaks 0.29
(54 Mb of peaks); 11% of reads at protein-coding TSS +-1 kb; promoter-element share of element reads 0.44 (ATAC 0.44). Signal is usable.

Links (`results/compare/`): old 104,071, new 98,544, shared 83,736, lost 20,335, gained 14,808 (Jaccard 0.70). Element activity Spearman old vs
new 0.82 (promoters 0.92, non-promoter 0.79). ABC on shared links Spearman 0.92 / Pearson 0.998 (non-self 0.77 / 0.88). By class: promoter
elements 27,486 -> 32,184 (lost 3,659, gained 8,357); distal_active 41,417 -> 31,451 (lost 14,746, gained 4,780); bivalent 10,386 -> 10,127;
self-promoter links unchanged (24,782). H3K27ac shifts links from distal to promoter elements. Per gene: median |delta rank| on shared links 0.5;
top non-self element changes for 27% of genes; 513 genes lose all non-self links, 285 gain some from none.

Benchmark (`bench/summary.tsv`, `bench/contrasts.tsv`), ceiling-weighted specificity (Wspec), Arm 2, 52 dev TFs (NRF1 in / out):
(a) binding x ABC_old 0.32 [-0.53, 1.47] / -0.22; (b) binding x ABC_k27ac 0.24 [-0.64, 1.46] / -0.33; (c) list e old links 0.66 [-0.01, 1.28] /
0.39; (d) binding-only new links 0.69 [-0.18, 1.59] / 0.26. Paired new - old: x ABC -0.08 [-0.21, 0.06] (NRF1 out -0.11); binding-only
+0.02 [-0.34, 0.39] (NRF1 out -0.13). Arm 1 (20 TFs): same picture (x ABC -0.05, binding-only +0.15, CIs span 0). Within-link-set controls:
element count, pooled other-TF and ATAC score as high as or higher than binding in most rows; ATAC x ABC is below 0. Top-1,000 lists overlap old vs
new 97% (x ABC) and 72% (binding-only). Under the margin rule (Monte Carlo noise about +-0.2 Wspec) H3K27ac gives no gain; the standing default
(binding-only over old links, list e) stays. Follow-ups in `followups/` (Hi-C contact, H3K27ac-only activity, ABC thresholds 0.01 / 0.05).
