#!/bin/bash
#SBATCH -J xatlas_kd
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:59:00
#SBATCH --array=0-11
#SBATCH -o logs/kd_%A_%a.out
cd /work/users/d/y/dyy12/XuLab/regulatome/phase1
D=/work/users/d/y/dyy12/XuLab/data/x_atlas_orion
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I stream_kd.py panel_tf_genes.tsv $D/hf_sample/gene_metadata.parquet hek_parquet_files.tsv $SLURM_ARRAY_TASK_ID $D/stream_tmp/task$SLURM_ARRAY_TASK_ID kd_stream
