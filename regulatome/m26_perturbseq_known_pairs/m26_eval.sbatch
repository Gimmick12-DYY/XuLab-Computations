#!/bin/bash
#SBATCH -J m26eval
#SBATCH -p general
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m26_perturbseq_known_pairs/out/m26_eval_%j.log
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m26_perturbseq_known_pairs/m26_eval.py
