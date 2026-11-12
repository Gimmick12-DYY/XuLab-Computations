#!/bin/bash
#SBATCH -J m23kd
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m23_sgrna_top25/logs/kd_%j.out
A=/work/users/d/y/dyy12/XuLab/data/x_atlas_orion; P=/work/users/d/y/dyy12/XuLab/regulatome/phase1; cd /work/users/d/y/dyy12/XuLab/regulatome/m23_sgrna_top25
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I kd_response_m23.py $A/panel_ntc_cells $A/hf_sample/gene_metadata.parquet $A/HEK293T_filtered_guide_calls_per_cell.csv.gz $P/guide_stats.tsv $P/panel_tf_xatlas_status.tsv /work/users/d/y/dyy12/XuLab/regulatome/m23_sgrna_top25/out/matrices
