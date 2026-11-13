#!/bin/bash
#SBATCH -J m26stream
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 04:00:00
#SBATCH --array=0-11
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m26_perturbseq_known_pairs/logs/stream_%A_%a.out
A=/work/users/d/y/dyy12/XuLab/data/x_atlas_orion; P=/work/users/d/y/dyy12/XuLab/regulatome/phase1; cd /work/users/d/y/dyy12/XuLab/regulatome/m26_perturbseq_known_pairs
mkdir -p $A/cohesin_cells
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I stream_cohesin.py cohesin_genes.tsv $P/hek_parquet_files.tsv $SLURM_ARRAY_TASK_ID $A/stream_tmp_m26/task$SLURM_ARRAY_TASK_ID $A/cohesin_cells
