#!/bin/bash
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J boltz_dnacal
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/dna_calib/logs/dnacal_%j.out
# Boltz-2 protein-DNA calibration (8BYX, 1NKP): 1 seed, 5 samples, same low-memory settings as the screen
C=/work/users/d/y/dyy12/XuLab/complex_screen
D=$C/dna_calib
export HF_HOME=$C/cache/hf BOLTZ_CACHE=$C/cache/boltz XDG_CACHE_HOME=$C/cache/xdg TORCH_HOME=$C/cache/torch TRITON_CACHE_DIR=$C/cache/triton CUEQ_CACHE_DIR=$C/cache/cueq
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /work/users/d/y/dyy12/envs/boltz/bin/activate
boltz predict $D/inputs --out_dir $D/out --cache $C/cache/boltz --diffusion_samples 5 --seed 1 --output_format mmcif \
  --num_workers 2 --max_parallel_samples 1 > $D/logs/boltz.log 2>&1
echo rc $?
