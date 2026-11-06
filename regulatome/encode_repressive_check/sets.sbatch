#!/bin/bash
#SBATCH -J enc_sets
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/encode_repressive_check/logs/sets_%j.out
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/encode_repressive_check/encode_repressive.py sets
