# TF groups by shared co-binding (cobinding/tf_groups)

Question (2026-11-06): do groups of TFs share Cicero co-binding pairs/cliques beyond depth, and do the groups or the
clique locations carry biological meaning?

## Pipeline (`bash scripts/run_all.sh`, then the per-step commands below)
- `build_incidence.py OUT [cliques]` — TF x pair and TF x anchor-region incidence. 1st-step Our Run k50 Cicero (p<=0.05,
  span >25 kb, blacklist and chrY/M dropped, regions merged across TFs). RBBP4 from `work/RBBP4_sweep_k50_i10000_s555`.
  The `cliques` option keeps only pairs inside one of that TF's cliques (FDR<=0.05 edges, size>=3).
  All pairs: 68 TFs, 375,053 pairs, 6,473 shared; clique edges: 172,311 pairs, 626 shared; 4,292 clique regions in >=2 TFs.
- `null_z.py IN pair|region 1000 all|odd|even|top:1000 SEED [library_strata.tsv]` — curveball null that keeps each TF's
  item count and each item's TF count (and with strata, per library-quality stratum: low <0.105, mid, high >=0.15 fraction of reads in
  elements). Output: z = (obs - null mean)/null sd of shared items per TF pair.
- `cluster_groups.py IN TAG MIN [res]` — Leiden on z>3, plus split-half (odd vs even chromosomes) reproducibility.
- `annotate_groups.py IN TAG` — per group: anchor class (protein-coding TSS +-1 kb, else ChromHMM18), STRING, CAP-SELEX,
  vs 1,000 depth- and library-matched random groups; writes gene lists for `go_enrich.py`.
- `string_vs_z.py` — pair-level STRING support vs z (TF-label permutation).
- `clique_location.py` / `clique_location_density.py OUT` — clique vs non-clique network nodes of the same TF;
  the density version stratifies by TF x local node density decile (+-500 kb).
- `go_enrich.py` — hypergeometric GO BP (GO Consortium files in data/go, downloaded 2026-11-06).
- WT Hi-C compartments: hic/work_wt293t/compartments/compartments_100000.E1.bedGraph (04_compartments.sbatch, 100 kb).

## Results (2026-11-06)
1. Without the library-stratified null, TFs split into high- vs low-library blocks. The stratified null still leaves a
   core (SOX4, NFYA, E4F1, ZNF507, MLX, GLYR1, MBD3, NME2, XPA, E2F3, DDIT3, RBBP7). The z matrix reproduces across
   chromosome halves (r = 0.79 pairs, 0.98 regions), but the Leiden partitions do not (ARI 0.12-0.24).
2. Depth-matched (each TF's top 1,000 pairs), the core disappears (median z -0.34; r with full-depth z = 0.21). The core
   is the deepest TFs. What is left is the small low-library block CHAMP1/ZUFSP/THAP6/HOXC9/POGK/EZH2, at 3-7 pairs per TF pair.
3. Known interactions: STRING (>=0.4, >=0.7, experimental) pairs do not have higher z (perm p 0.11-0.79, all levels).
   No panel pair is a CAP-SELEX interaction. Groups are not enriched for STRING edges vs matched random groups.
4. Promoter/enhancer: groups match depth-matched random groups (pctile 0.2-0.8), except the low-library group, which has
   fewer promoters (pctile 0.003). GO for the deep-TF group: generic housekeeping terms (mRNA processing, splicing, cell cycle;
   1.3-1.6x). Clique group DR1/ZNF644/ZNF143/FOXJ3: protein folding / response to heat (22 genes; DNAJB1/HSPA6/BAG3-type
   loci, see the exon caution in the hub notes).
5. Clique location (52 TFs): crude clique-vs-non-clique enrichment for A compartment, active enhancer, TSS flank,
   transcribed, promoter; depletion for quiescent. After matching local node density, most of it is gone (pooled MH log2OR:
   promoter +0.14, active enh +0.10, A +0.05, quiescent -0.13; per-TF sign tests mostly n.s.). Cliques form where nodes are
   dense. Density-matched hotspots: 278/2,586 1 Mb bins (chr16p11.2, chr19p13.3, chr10q26 BAG3, chr13 HSPH1,
   pericentromeric chr16:34/46 Mb and chr1:143 Mb, a mappability caveat). Promoter genes with excess: TECR, BAG3, HSPA6 only. No GO term.
