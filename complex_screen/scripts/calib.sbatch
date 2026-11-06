#!/bin/bash
#SBATCH -p l40-gpu,a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-7%4
#SBATCH -J boltz_calib
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/logs/calib_%A_%a.out
# Boltz-2 calibration on the 31 pilot pairs: 1 seed, 5 diffusion samples, precomputed unpaired MSAs
C=/work/users/d/y/dyy12/XuLab/complex_screen
export HF_HOME=$C/cache/hf BOLTZ_CACHE=$C/cache/boltz XDG_CACHE_HOME=$C/cache/xdg TORCH_HOME=$C/cache/torch TRITON_CACHE_DIR=$C/cache/triton CUEQ_CACHE_DIR=$C/cache/cueq
source /work/users/d/y/dyy12/envs/boltz/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
curl -s -m 20 -o /dev/null -w "colabfold http %{http_code}\n" https://api.colabfold.com/queue || echo "colabfold unreachable"
mapfile -t F < <(ls $C/calib/inputs/*.yaml | sort)
for ((i=SLURM_ARRAY_TASK_ID; i<${#F[@]}; i+=8)); do
  y=${F[$i]}; id=$(basename $y .yaml); t0=$(date +%s)
  boltz predict $y --out_dir $C/calib/out --cache $C/cache/boltz --diffusion_samples 5 --seed 1 \
    --output_format mmcif --num_workers 2 --override > $C/logs/calib_$id.log 2>&1
  echo -e "$id\t$?\t$(( $(date +%s) - t0 ))\t$(hostname)\t$(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $C/calib/timing.tsv
done
