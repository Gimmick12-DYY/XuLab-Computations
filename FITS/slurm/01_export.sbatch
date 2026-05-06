#!/bin/bash
#SBATCH --job-name=fits_01_export
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab/FITS
CFG="/work/users/d/y/dyy12/XuLab/FITS/configs/default.yaml"
source "/nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh"
conda activate cistopic
python scripts/export_from_cfg.py --config ""
