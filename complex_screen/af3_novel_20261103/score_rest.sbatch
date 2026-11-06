#!/bin/bash
#SBATCH -J af3nscr2
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o logs/score_rest_%j.log
source /work/users/d/y/dyy12/conda/etc/profile.d/conda.sh
conda activate snapatac2
python af3_score_srcap.py
python af3_score_weak.py
