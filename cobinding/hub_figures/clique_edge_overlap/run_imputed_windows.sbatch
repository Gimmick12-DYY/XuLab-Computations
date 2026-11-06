#!/bin/bash
#SBATCH -J imp_windows
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser/%x_%j.log
set -euo pipefail
OUT=/work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/users/d/y/dyy12/conda/envs/hic
TFS=$(python -c "
import pandas as pd
t=set()
for f in ['hub_chr16_edges.tsv','hub_chr17_edges.tsv']:
    t|={x for l in pd.read_csv('${OUT}/'+f,sep='\t').tfs.str.split(',') for x in l}
print(','.join(sorted(t)))")
python ${OUT}/imputed_windows.py ${OUT}/imputed_windows.tsv "$TFS" chr16:29700000-31600000 chr17:17100000-18950000
