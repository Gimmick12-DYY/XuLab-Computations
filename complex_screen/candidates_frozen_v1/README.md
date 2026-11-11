# candidates_frozen_v1: the frozen candidate list

Single source of candidates for all layers (correlation → motif → AF3), per Yuyang 2026-11-11.

## Rule (build_candidates.py; the selection logic of af3_corr060_20261104/build_list.py)
- For every panel pair (79 TFs, HP1 = CBX1/3/5 merged, ZUFSP = ZUP1), best_pct = the highest co-binding percentile over the five ChromHMM element-set matrices: all elements, no quiescent, TSS, enhancer (incl. bivalent), repressive.
- known = a curated CUR pair not labelled Novel (CUR_known, CUR_capselex), STRING physical >= .7 (string_physical in r1_pairs_frozen), CAP-SELEX called (Xie et al. 2025), a STAG2-class pair, or any pair with NEUROD4.
- candidate = best_pct >= .60 and not known.
- Unlike build_list.py, the 103 pairs already run in earlier AF3 rounds are NOT removed (that was scheduling, not a candidate rule).
- AF3 and Boltz columns are annotation only and play no part in selection. af3_1seed_tier comes from af3_corr060_20261104/scores_s1, else the best tier in af3_local / af3_novel_20261103 (those earlier rounds were partly 5-seed). af3_5seed_tier comes from scores_s5 (strong = >= 3/5 seeds), and is pending if the pair is in seeds5/all.txt but not yet scored, else not_run. The 5-seed column is a snapshot at build time.

## Counts
3,081 panel pairs -> 1,784 with best_pct >= .60 -> 1,732 candidates (52 known: NEUROD4 22, STRING 12, CUR_known + STRING 10, STAG2 4, CUR_known 2, CUR_capselex + CAP-SELEX called 2).

## Files
| file | sha256 |
|---|---|
| candidates_frozen_v1.tsv | 08fe206490f12ba533451a6f058fae9be265dd7073f62c60fb84dd8fd61c348a |
| candidates_all3081.tsv | ab58745101864dc8f53f6408f294c520f2e68321e6eb37655b1fd3dea31d6ca4 |

## Inputs (identical md5 to the files build_list.py read on the Mac)
| input | sha256 |
|---|---|
| complex_screen/af3_corr060_20261104/align_corr_20261109/inputs/pct_all.tsv | b854a4099cb18526a449a769a2ed866e74fab497c90d26470e77b94560c40989 |
| complex_screen/af3_corr060_20261104/align_corr_20261109/inputs/pct_noquies.tsv | 9e3d7d7c55c3d1582a71f2f53463bbfd68e5d9edeb0df9047f64a3b6ea581532 |
| complex_screen/af3_corr060_20261104/align_corr_20261109/inputs/pct_tss.tsv | 6121a6935d9bfb81f0d83fe7a9bbe47df58d9e00097ec273a7202a181c4e893d |
| complex_screen/af3_corr060_20261104/align_corr_20261109/inputs/pct_enhancer_biv.tsv | 9c984430deeb976d0b7054d8f2020f6844fffeb49ef6d12fc4aab3b3681154e6 |
| complex_screen/af3_corr060_20261104/align_corr_20261109/inputs/pct_repressive.tsv | 396f73f2b174003fa31abc1020657d50a3a56029cce5f3274d02f3fe075f2144 |
| regulatome/r1_tftf/v2_final/r1_pairs_frozen.tsv | 0bb5c91ca1b96ffc827d94ff44cd645218fc204b530c0cb97523a9f4f35e22e2 |
| complex_screen/jie_figures/summary_table_data.py | 6eb077f4aa3626216b9f57a2e38b0d2e1bf7f39f54d0dcc8c4581c20566b4223 |
| complex_screen/screen/scores.tsv | ea6935a59018faa98ce68b0041fbdc74ab79647c0d0813ae1327a240024d98a1 |
