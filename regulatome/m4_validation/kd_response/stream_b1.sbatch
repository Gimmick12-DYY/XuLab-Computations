#!/bin/bash
#SBATCH -J xatlas_b1
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH --array=0-11
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/kd_response/logs/b1_%A_%a.out
A=/work/users/d/y/dyy12/XuLab/data/x_atlas_orion; P=/work/users/d/y/dyy12/XuLab/regulatome/phase1; cd /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/kd_response
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I stream_b1.py $P/panel_tf_genes.tsv $P/hek_parquet_files.tsv $SLURM_ARRAY_TASK_ID $A/stream_tmp_b1/task$SLURM_ARRAY_TASK_ID $A/panel_ntc_cells
