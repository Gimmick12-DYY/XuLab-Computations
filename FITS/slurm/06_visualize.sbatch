#!/bin/bash
#SBATCH --job-name=fits_06_visualize
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab/FITS
source ~/.bashrc
conda activate cistopic
python scripts/06_visualize_fits.py --config /work/users/d/y/dyy12/XuLab/FITS/configs/default.yaml
