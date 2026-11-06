#!/bin/bash
#SBATCH -J motsum2
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103/logs/summ_%j.out
O=/work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103; PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python; module load meme/5.5.7; cd $O
$PY -I collect_denovo.py || exit 1
tomtom -dist pearson -min-overlap 5 -thresh 0.05 -no-ssc -oc tomtom_denovo2 db/denovo_top5.meme db/denovo_top5.meme; tomtom -dist pearson -min-overlap 5 -thresh 0.05 -no-ssc -oc tomtom_denovo2_db db/denovo_top5.meme db/db.meme
$PY -I summarize.py; echo rc $?
