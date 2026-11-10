#!/bin/bash
#SBATCH -J k27_sanity
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/logs/sanity_%j.out
set -euo pipefail
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/d/y/dyy12/envs/abc
mkdir -p $B/chipatlas_bw
for s in SRX18002015 SRX18002016; do
  [ -s $B/chipatlas_bw/$s.bw.ok ] || { curl -fsS --retry 3 -o $B/chipatlas_bw/$s.bw https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/$s.bw && touch $B/chipatlas_bw/$s.bw.ok; }
done
python $B/k27ac_sanity.py $B $B/chipatlas_bw
