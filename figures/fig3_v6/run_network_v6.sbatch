#!/bin/bash
#SBATCH --job-name=fig3v6_net
#SBATCH --output=/work/users/d/y/dyy12/XuLab/figures/fig3_v6/logs/net-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
# Fig 3 v6 panel a: re-run the approved RBBP4 network (same inputs as cobinding/slurm/run_rbbp4_loop_plots_wt293t_v2.sbatch, cut 0.7) with v6 wording.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; COB=${X}/cobinding; F=${X}/figures/fig3_v6
OUT=${X}/hic/work_wt293t/loops/union_5k_10k_0.7/RBBP4_sweep_k50_i10000_s555_pad30kb
RES=${COB}/results/RBBP4_sweep_k50_i10000_s555
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /work/users/d/y/dyy12/conda/envs/cobinding
cd ${X}/hic/scripts   # cluster_peaks import path is absolute; run where the original ran
LEGEND_FS=${LEGEND_FS:-36} python ${F}/network_v6.py --clique-loops-only --edges ${COB}/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv \
  --cliques ${RES}/cliques.tsv --nodes ${RES}/nodes.tsv --loop-pairs ${OUT}/RBBP4_pairs_as_loops.0.7.tsv --out-dir ${F}/panels_src
