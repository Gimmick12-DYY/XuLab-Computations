#!/bin/bash
#SBATCH -J m22a5
#SBATCH -p general
#SBATCH -t 04:00:00
#SBATCH --mem=48G
#SBATCH -c 2
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m22_edge_table/out/simple_abc_correlation_%j.log
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m22_edge_table/simple_abc_correlation.py
