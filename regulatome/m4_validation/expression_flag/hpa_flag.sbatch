#!/bin/bash
#SBATCH -J hpa_flag
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/expression_flag/hpa_%j.out
cd /work/users/d/y/dyy12/XuLab/data/hpa; [ -s rna_celline.tsv.zip ] || curl -sSLf --retry 5 -o rna_celline.tsv.zip https://www.proteinatlas.org/download/tsv/rna_celline.tsv.zip
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/expression_flag/hpa_flag.py
