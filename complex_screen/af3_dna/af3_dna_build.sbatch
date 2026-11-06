#!/bin/bash
#SBATCH -J af3dna_build
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_dna/logs/build_%j.out
python3 /work/users/d/y/dyy12/XuLab/complex_screen/af3_dna/af3_dna_make.py runs
