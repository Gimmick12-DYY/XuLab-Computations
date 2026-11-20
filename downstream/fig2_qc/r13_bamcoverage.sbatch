#!/bin/bash
# Round 6 heatmaps: bulk ChIP CPM bigWigs (deepTools bamCoverage 3.5.6, duplicates ignored, reads extended to 200 bp,
# 50 bp bins). Task k = line k of the list below. Output results_293T_analysis/fig2_qc/round6_diffpeaks/heatmaps/bw/<name>.bw
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%A_%a.log
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab
OUT=$X/results_293T_analysis/fig2_qc/round6_diffpeaks/heatmaps/bw
mkdir -p $OUT
LIST="CTCF:$X/data/ENCFF139DCW.bam MAZ:$X/data/MAZ_HEK293.bam ZBTB7A:$X/data/ZBTB7A_HEK293.bam ZIC2:$X/data/ZIC2_HEK293.bam ZNF282:$X/data/ZNF282_HEK293.bam ZNF777:$X/data/ZNF777_HEK293.bam CTCF_nexus:$X/chip_gse103651/work/bam/gse103651.pooled.bam"
E=$(echo $LIST | tr " " "\n" | sed -n "${SLURM_ARRAY_TASK_ID}p")
N=${E%%:*}; B=${E#*:}
$X/.venv_deeptools/bin/bamCoverage -b $B -o $OUT/$N.bw --normalizeUsing CPM --ignoreDuplicates --extendReads 200 --binSize 50 -p 4
echo done $N
