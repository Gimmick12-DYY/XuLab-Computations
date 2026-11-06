#!/bin/bash
#SBATCH -J hox_analyze
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/hox_check/logs/analyze_%j.out
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/hox_check/hox_check.py analyze 2>&1 | grep -v findfont
