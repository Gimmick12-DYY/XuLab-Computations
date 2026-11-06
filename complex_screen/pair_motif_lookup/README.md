# Pair shared-motif lookup (2026-11-06)

Built only from finished motif results; no new enrichment was run. The Longleaf inputs were read, not modified, and copied to `inputs/`.

Build on Longleaf, which reads the result trees in place (read only):
`python3 build_pair_motif_lookup.py <out_dir>`

Build from the Mac mirror:
`python3 -I scripts/build_pair_motif_lookup.py outputs inputs`

Either way it takes about 45 s and needs no Slurm. `tf_cells.tsv` holds the per-TF cell counts from TF1000cells_motifs.xlsx.

The Longleaf copy is in `complex_screen/pair_motif_lookup/`. The 250 MB full lookup is not committed there; rebuild it with the script.

## Result sets
| set | source | TFs |
|---|---|---|
| A_raw_homer | `/vast/som/xujie_lab/TAGATG-293T/bigwig/motif/homer/TF.<TF>_motif_NewBg/knownResults.txt` (raw MACS2 q0.05 peaks, -useNewBg) | 76 |
| A_raw_ame | same tree, `ame/<TF>/ame.tsv` (HOCOMOCO v11 core, shuffled control); 10 files are empty and count as missing | 66 |
| A_raw_denovo | same tree, `homerResults/motifN.motif` | 68 |
| B_imp_peaks_homer, _homer_ocr, _ame, _ame_ocr | `motif_analysis/motif/<tf>/...` (imputed MACS3 peaks) | 78 / 34 / 63 / 40 |
| C_imp_bins_homer, _ame | `motif_analysis/motif_bins/<tf>/...` (imputed top-20k 1 kb bins) | 78 / 78 |
| E_imp_denovo_unknown | `motif_analysis/motif_imputed/<tf>/homer/homerResults` (45 TFs with no known motif) | 45 |
| E_imp_denovo_peaks | `motif_analysis/motif/<tf>/homer/homerResults` | 78 |

`motif_pmat` is skipped because those are hg19 peaks run against hg38.

## Rules
- A known motif passes if HOMER p <= 1e-5 (natural-log Log P-value <= ln 1e-5) or AME adj p <= 0.05.
- A de novo motif passes if HOMER p <= 1e-5. Two de novo motifs match if their similarity is >= 0.6. The similarity follows HOMER: Pearson r of the two frequency matrices on a common span, padding with 0.25, maximised over offset and strand, with at least 6 bp of overlap. HOMER is not installed on the Mac, so this is a Python version of that score, not `compareMotifs.pl` itself.
- `n_panel_tfs_passing` is the number of panel TFs in which the motif passes in that set. For a de novo match it is the larger of the two motifs' counts, where each count is the number of TFs with a passing motif at similarity >= 0.6.
- A motif is `pair_specific` if `n_panel_tfs_passing` <= 10.
- `pct_cobinding` is the % of the smaller raw peak set that overlaps the other TF's raw peaks.
- HP1 is the same TF as CBX1, CBX3 and CBX5.

## Outputs
- `motif_panel_frequency.tsv`: every motif with its pass count per set.
- `pair_motif_lookup.tsv`: one row per (pair, set, motif) where the motif passes in both TFs. It is 250 MB. Two subsets: `_candidates.tsv` (the 115 AF3-run and gate pairs) and `_pair_specific.tsv`.
- `pair_verdict.tsv`: per pair and set, the verdict, n_shared, n_specific and top specific motifs. It also carries the AF3 tier, the R1 class, STRING, CAP-SELEX and the Pearson percentile columns.
