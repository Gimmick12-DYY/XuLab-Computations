#!/bin/bash
#SBATCH -J m33
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m33_abc_pipeline_audit/out/m33_%j.log
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m33_abc_pipeline_audit/m33.py
