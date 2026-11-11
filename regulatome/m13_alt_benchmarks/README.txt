Coverage of alternative benchmark data for the 79 HITAG panel TFs (HEK293T). Coverage only: nothing here compares these data with our binding.
Made by regulatome/m13_alt_benchmarks/m13.py on Longleaf (SPEC.md there). Sources, download dates and sha256: sources.tsv. Public files are in XuLab/data/external/ on Longleaf.

coverage.tsv: one row per TF (HP1 = CBX1/CBX3/CBX5, ZUFSP = ZUP1). Columns:
- cells, depth_tertile, claim_tier_d_ge_0.04, kd_ceiling, benchmark_testable_m10: carried from our own analyses.
- encode_chip_<HEK293T|HEK293|other>_n: released ENCODE TF ChIP-seq experiments; accession(tag, GRCh38 peak file yes/no) for HEK293T/HEK293.
- chipatlas_<HEK293T|HEK293|other>_n: ChIP-Atlas hg38 experiments. ChIP-Atlas files both lines under cell type "293"; HEK293T is assigned only where the sample metadata says 293T, everything else under "293" is counted as HEK293.
- GSE78099_HEK293T_n: Imbeault/Trono KZFP ChIP-exo samples (HEK293T) matching the TF, with supplementary file counts.
- longleaf_bulk_chip, longleaf_chromnitron: data already on Longleaf.
- hocomoco_v11_motif, hocomoco_v12_core_n, jaspar2024_n, workbook_*: motif availability; the TF1000cells_motifs.xlsx workbook stores p-values per method, not ranks.
- collectri_*, dorothea_ABC_*, trrust_*: curated targets, in our 11,228-gene universe, signed.
- replogle_<K562_gwps|K562_essential|RPE1>: TF perturbed in Replogle 2022 (from the perturbation list in each pseudobulk file's metadata; no expression values read).
- knocktf_datasets: KnockTF 2.0 human datasets (Sample_ID) for the TF; cell type is not in the downloaded table.
- encode_perturb_rnaseq: ENCODE shRNA / CRISPRi / CRISPR RNA-seq experiments (assay:cell line:accession).
- ntc_detect, ntc_mean_log1pCP10k: the TF gene's detection and mean expression in our non-targeting Perturb-seq cells.
coverage_summary.tsv: per option, TFs covered of 79 and of the 20 claim-tier TFs, with names.
