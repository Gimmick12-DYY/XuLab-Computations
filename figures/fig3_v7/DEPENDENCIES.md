# Fig 3 v7: input dependencies on the lab MACS2 per-TF peaks (audit 2026-11-11, read-only)

MACS2 route: `cobinding/slurm/run_ourrun_tf.sbatch` takes `bigwig/macs2/TF.<TF>/TF.<TF>_peaks.narrowPeak` (called at p = 0.01, kept at
q <= 0.05) -> `cobinding/scripts/build_tf_peak_mm_from_macs2.py` (cell x peak matrix from the TF's fragment BED) -> Cicero (k = 50, 1 Mb)
-> `cobinding/work/ourrun_k50/<TF>/<TF>.peak_edges.tsv`, `peak_annot.tsv` and `cobinding/results/ourrun_k50/<TF>/cliques.tsv` (modules).
RBBP4 is the exception: its modules (`results/RBBP4_sweep_k50_i10000_s555`) come from the lab-provided peak matrix (`work/RBBP4_labpmat`),
not from build_tf_peak_mm_from_macs2.py.

| Panel | Input | Depends on MACS2 peaks | Via |
|---|---|---|---|
| a bars | `hub_figures/basic/descriptive/clique_loop_coverage.tsv` (modules per TF, loop-edge share) | yes | module = Cicero clique on the MACS2 q<=0.05 peaks (ourrun_k50); RBBP4: lab peak matrix |
| a null | `fig3_v6/checks/null_per_edge/*.npz`, `s1d_per_tf_null.tsv` (observed rate) | yes | module edges >25 kb from ourrun_k50 cliques; shuffled positions are not peaks |
| a, c, rows | Peakachu loops `hic/work_wt293t/loops/peakachu_{5000,10000}.loops.0.7.bedpe` | no | wild-type HEK293T Hi-C (GSE216399) |
| b | `hub_regions.tsv` (module regions, n TFs) | yes | union of ourrun_k50 module regions across TFs |
| b | `annotation_counts_chromhmm.tsv` ("all Cicero peaks", "all module regions" bars) | yes | per-TF Cicero peaks = MACS2 q<=0.05 peaks |
| b | RefSeq Select TSS (`rs/*.json`), HEK293T ChromHMM 18-state | no | annotation only |
| c Hi-C triangle | `se_sedb/model/inputs/hic_10kb.npz` | no | ICE-balanced wild-type Hi-C |
| c TF tracks | `se_hub/figure_v5_tftracks/tf_cov_2kb.tsv` | no | raw single-cell fragment coverage (CPM, 2 kb) |
| c Cicero edges | `se_sedb/hub_edges_all.tsv` | yes | Cicero edges among ourrun_k50 peaks touching the SE |
| c partner bands / classes | `se_sedb/figure/edges_in_structure.tsv` (from `edges_exact_se.tsv`, 02_edges_exact.py) | yes | partners = Cicero partners of the SE (MACS2 peaks); class by RefSeq/ChromHMM |
| c SE band | `se_sedb/HEK293T_WT_SEdb_SE_02_1951_hg38.bed` | no | SEdb (H3K27ac ROSE) |
| c ChromHMM / genes | `se_hub/figure_v7_refseq/chromhmm_window.tsv`, `refseq_select_chr19_0.43-2.25Mb.json` | no | annotation |
| d 3D chain | Hi-C-first polymer model (`se_sedb/model/sim/eps0`, restraints from Hi-C pixels + Peakachu loops) | no (structure) | no TF term in the model |
| d which elements / labels | `rosette_nodes.tsv`, partner list | yes | elements = Cicero partners of the SE |
| d small spheres | `se_sedb/model/inputs/beads.tsv` (`tfs` column, 03_prep_model_se.py `--peaks-dir ... TF.*_peaks.q005.narrowPeak`) | yes, directly | MACS2 q<=0.05 peak overlap per 2 kb bead |
| e flower | `se_sedb/hub_edges_all.tsv`, `edges_exact_se.tsv` | yes | all 135 Cicero edges of the hub; node class by RefSeq/ChromHMM |
| legend c-e | TF colour list | yes (labels only) | TFs with MACS2 peaks at the elements |
| supp. pairs loop support (D1) | ourrun_k50 `peak_edges.tsv`, per-pair null | yes | Cicero pairs on MACS2 peaks |
| supp. block null | same as a, whole-module shuffle | yes | as a |
| supp. RBBP4 network | `RBBP4_sweep_k50_i10000_s555` cliques | no MACS2 route; lab peak matrix | RBBP4_labpmat |
| supp. SE edge table | `hub_edges_all.tsv`, `hic_test/results_10kb.json` | yes | Cicero edges; Hi-C contact test itself uses Hi-C only |
| check: ATAC-matched null | `HEK293T_ATAC_union3.bed.gz`, 3 bulk ATAC narrowPeaks, ourrun_k50 `peak_annot.tsv` | partly | ATAC independent; universe and density use the MACS2-based Cicero peaks |
| check: panel b HOT | universe above, `hub_regions.tsv` | yes | as b |

Independent of MACS2: Hi-C (matrix, loops, polymer structure), raw fragment coverage tracks, ATAC, ChromHMM, RefSeq, SEdb.
Everything that defines a co-binding module, an edge, a partner or a TF sphere goes back to the MACS2 q<=0.05 peaks (RBBP4: lab peak matrix).
