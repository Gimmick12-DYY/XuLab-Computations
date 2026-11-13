#!/bin/bash
#SBATCH -J m29
#SBATCH -p general
#SBATCH -t 03:00:00
#SBATCH --mem=48G
#SBATCH -c 2
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m29_per_tf_list_overlap/out/m29_%j.log
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m29_per_tf_list_overlap/m29.py
