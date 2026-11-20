# candidates_frozen_v2: the frozen candidate list (cutoff .75)

From 2026-11-20 this is the single source of candidates for all layers (correlation → motif → AF3). The cutoff is Yuyang's decision ("Use 0.75 for the correlation matrix").
- It supersedes candidates_frozen_v1 (cutoff .60).
- v1 is kept because the AF3 runs and earlier tables reference it.

## Rule
build_candidates.py is candidates_frozen_v1/build_candidates.py with the cutoff as a parameter (default 0.75).

- **best_pct:** for every panel pair, the highest percentile of the TF–TF Pearson r over the five ChromHMM element-set matrices.
  - The five matrices: all elements, no quiescent, TSS, enhancer (incl. bivalent), repressive.
  - The panel is 79 TFs, with HP1 = CBX1/3/5 merged and ZUFSP = ZUP1.
- **known:** any of the following (same as v1):
  - a curated CUR pair not labelled Novel (CUR_known, CUR_capselex)
  - STRING physical >= .7 (string_physical in r1_pairs_frozen)
  - CAP-SELEX called (Xie et al. 2025)
  - a STAG2-class pair
  - any pair with NEUROD4
- **candidate:** best_pct >= .75 and not known.
- **AF3 and Boltz columns:** annotation only; they play no part in selection.
  - af3_1seed_tier, af3_1seed_source, af3_5seed_tier and boltz_tier are as in v1.
  - New column af3_5seed_final: the final 5-seed call from af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv (strong / weak / none / not_run).
  - AF3 was only run on pairs with best_pct >= .60.
- **Check against v1:** the script asserts that pair names, best_pct and known_reason are identical to v1 for all 3,081 pairs. Only the cutoff differs.

## Counts
3,081 panel pairs -> 1,236 with best_pct >= .75 -> 1,198 candidates.

The 38 known pairs removed:

| Reason | Pairs |
|---|---|
| NEUROD4 | 14 |
| CUR_known + STRING | 9 |
| STRING | 8 |
| STAG2 | 3 |
| CUR_capselex + CAP-SELEX called | 2 |
| CUR_known | 2 |

- **Relation to v1:** all 1,198 were v1 candidates; 534 v1 candidates drop out.
- **AF3 5-seed among the 1,198:** 19 strong, 45 weak, 1 none, 1,133 not run.
- **AF3 5-seed strong pairs lost relative to v1 (best_pct < .75):** EPAS1-HES1, ETS2-RBBP7, HOXB9-RBBP7, MEF2A-RBBP7, RBPJ-ZUFSP.

## Files
- **build_candidates.py:** the build (sbatch job 9984663, `python build_candidates.py 0.75`).
- **candidates_all3081.tsv:** all panel pairs with percentiles, known flags, candidate (at .75) and the AF3/Boltz annotation.
- **candidates_frozen_v2.tsv:** the 1,236 pairs with best_pct >= .75. Filter candidate == True for the 1,198.

## Checksums
Paths are relative to complex_screen/ unless they start with regulatome/.

| md5 | sha256 | file |
|---|---|---|
| 77a304f15172b946ebed205a6ab4f3be | b854a4099cb18526a449a769a2ed866e74fab497c90d26470e77b94560c40989 | af3_corr060_20261104/align_corr_20261109/inputs/pct_all.tsv |
| d6609746456ef605796114d0bb95a2d3 | 9e3d7d7c55c3d1582a71f2f53463bbfd68e5d9edeb0df9047f64a3b6ea581532 | af3_corr060_20261104/align_corr_20261109/inputs/pct_noquies.tsv |
| fff868bd33c1036512162fe8edd0790c | 6121a6935d9bfb81f0d83fe7a9bbe47df58d9e00097ec273a7202a181c4e893d | af3_corr060_20261104/align_corr_20261109/inputs/pct_tss.tsv |
| d007e8af99a343291c31cdecc14b5f67 | 9c984430deeb976d0b7054d8f2020f6844fffeb49ef6d12fc4aab3b3681154e6 | af3_corr060_20261104/align_corr_20261109/inputs/pct_enhancer_biv.tsv |
| 2251adbc58637a2c1b84d3c769f80d22 | 396f73f2b174003fa31abc1020657d50a3a56029cce5f3274d02f3fe075f2144 | af3_corr060_20261104/align_corr_20261109/inputs/pct_repressive.tsv |
| 11b75704ae63991bcd9d8d3290a0efb5 | 0bb5c91ca1b96ffc827d94ff44cd645218fc204b530c0cb97523a9f4f35e22e2 | regulatome/r1_tftf/v2_final/r1_pairs_frozen.tsv |
| f780d17a27f5384175824d98e09c44fa | 6eb077f4aa3626216b9f57a2e38b0d2e1bf7f39f54d0dcc8c4581c20566b4223 | jie_figures/summary_table_data.py (CUR) |
| 73ef93ff5bde98cb2a211661a2085a74 | b5c4c23b5a68012dffe1fe3ea946b656f3e333cbde10fb593bee113f47444de1 | af3_corr060_20261104/rescore_5seed/tiers_5seed.tsv |
| e31d1fdf9bb253153431693e52ddf778 | 2e4911709697dc845b0c3f8a0cd8b36caeb82c6eb84058210422657826f7a49f | af3_corr060_20261104/seeds5/all.txt |
| 9fefbea894f2822252b46b8123e25efd | 0c7cba4173d0c21ff39cfb8c2a518abedf162dd98efb8e15da1be6d2caab8053 | af3_local/af3_scores.tsv |
| ecaaaea391c9dfb8693a93cbd8733150 | bb39b483581ace5783a960e15883c72acbe929256deccbc7af9a657f724d31cc | af3_novel_20261103/af3_scores.tsv |
| b99f9a82c5768e7f36824ee4521f375c | 9bebaef303d2a577b663e3503a1880bfb1985d1429bf01656f3aff2290a8693c | af3_novel_20261103/af3_scores_weak.tsv |
| 17514fa877893877145ea7dd91e30d03 | cf6eee64732b2a9cb14a2fc20e3594c1fca4e72e53c515d2041ae2d4c7da337b | af3_novel_20261103/af3_scores_srcap.tsv |
| 34e5101866edce28efa353d3f92ec4f0 | ea6935a59018faa98ce68b0041fbdc74ab79647c0d0813ae1327a240024d98a1 | screen/scores.tsv (Boltz) |
| cf86482d2e0e2201463e94c0a58f85eb (cat of 1,629 files) | | af3_corr060_20261104/scores_s1/*.tsv |
| 4b911d402bd2349d4bc2d70c7d928bdb (cat of 97 files) | | af3_corr060_20261104/scores_s5/*.tsv |
| e9cb2f711591b921c2fa719dfc816fb6 | 0d004032ddfb9542bfaf1fa31e620095e5c67cd75bc1124979a3aa8a61ba62f5 | candidates_frozen_v2/build_candidates.py |
| 869a8efe7b984115e72af706dec9a7f6 | f33f4e457cdb06fe4857c5060798dc7f6b9280a04497f26a821b4c85e7f9d6ba | candidates_frozen_v2/candidates_all3081.tsv |
| 7991f6e1be3df44c717276e4e4b370c9 | 8b1297077765f9ac675258c31f5a3250af8eceec8873fe70a10973b7575f5076 | candidates_frozen_v2/candidates_frozen_v2.tsv |

## Downstream (2026-11-20)
- **Motif table:** pair_motif_composite/pair_motifs_v2.py (job 9984664) -> motif_tables/pair_shared_motifs_v2_{all3081,candidates1198,AF3strong,summary}.tsv.
- **Funnel:** pair_funnel_v2.py (job 9984665) -> funnel_v2_20261120/pair_funnel_v2_20261120{,_fisher,_triple}.tsv.
- **The sweep that led to the choice:** cutoff_sweep.py -> cutoff_sweep_20261120/.
