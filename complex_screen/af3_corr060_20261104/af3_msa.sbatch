#!/bin/bash
#SBATCH -J af3c060m
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104/logs/msa_%A_%a.out
source /work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104/af3_common.sh
g=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $O/proteins.txt); ENVS=(); echo $g $(hostname)
run_af3 --json_path=/root/af/msa_in/$g.json --output_dir=/root/af/msa_out --norun_inference --jackhmmer_n_cpu=2 --nhmmer_n_cpu=2
echo rc $?; ls -la $O/msa_out/${g,,}/
