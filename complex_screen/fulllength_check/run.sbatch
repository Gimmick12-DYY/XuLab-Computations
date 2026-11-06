#!/bin/bash
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J boltz_fullchk
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/fulllength_check/logs/run_%j.out
# full-length MTF2-RBBP4 and RBBP4-RBBP7, seeds 1-3 x 5 samples, low-memory settings
C=/work/users/d/y/dyy12/XuLab/complex_screen; O=$C/fulllength_check
export HF_HOME=$C/cache/hf BOLTZ_CACHE=$C/cache/boltz XDG_CACHE_HOME=$C/cache/xdg TORCH_HOME=$C/cache/torch TRITON_CACHE_DIR=$C/cache/triton CUEQ_CACHE_DIR=$C/cache/cueq
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /work/users/d/y/dyy12/envs/boltz/bin/activate
for s in 1 2 3; do
  boltz predict $O/inputs --out_dir $O/out_seed$s --cache $C/cache/boltz --diffusion_samples 5 --seed $s --output_format mmcif \
    --num_workers 2 --max_parallel_samples 1 > $O/logs/predict_seed$s.log 2>&1
  echo seed $s rc $?
done
