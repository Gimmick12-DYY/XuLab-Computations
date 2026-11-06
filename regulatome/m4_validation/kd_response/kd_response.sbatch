#!/bin/bash
#SBATCH -J xatlas_b2
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/kd_response/logs/b2_%j.out
A=/work/users/d/y/dyy12/XuLab/data/x_atlas_orion; P=/work/users/d/y/dyy12/XuLab/regulatome/phase1; cd /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/kd_response
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I kd_response.py $A/panel_ntc_cells $A/hf_sample/gene_metadata.parquet $A/HEK293T_filtered_guide_calls_per_cell.csv.gz $P/guide_stats.tsv $P/panel_tf_xatlas_status.tsv /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/kd_response/out
