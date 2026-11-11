#!/bin/bash
#SBATCH -J af3dnascore
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_dna_included_20261111/logs/%x_%j.out
cd /work/users/d/y/dyy12/XuLab/complex_screen/af3_dna_included_20261111 && /work/users/d/y/dyy12/conda/envs/snapatac2/bin/python score_dna.py
