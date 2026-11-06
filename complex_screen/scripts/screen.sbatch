#!/bin/bash
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:59:00
#SBATCH -J boltz_screen
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/logs/screen_%A_%a.out
# Boltz-2 pair screen: one batch (screen/batches/bNNN.txt) per array task, 1 seed, 5 diffusion samples
C=/work/users/d/y/dyy12/XuLab/complex_screen
export HF_HOME=$C/cache/hf BOLTZ_CACHE=$C/cache/boltz XDG_CACHE_HOME=$C/cache/xdg TORCH_HOME=$C/cache/torch TRITON_CACHE_DIR=$C/cache/triton CUEQ_CACHE_DIR=$C/cache/cueq
source /work/users/d/y/dyy12/envs/boltz/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=$(printf "b%03d" $SLURM_ARRAY_TASK_ID)
[ -f $C/screen/batch_scores/$B.tsv ] && { echo "$B already scored"; exit 0; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 10 > $C/logs/gpumem_$B.txt &
MON=$!
t0=$(date +%s)
boltz predict $C/screen/inputs/$B --out_dir $C/screen/out/$B --cache $C/cache/boltz --diffusion_samples 5 --seed 1 \
  --output_format mmcif --num_workers 2 --max_parallel_samples 1 > $C/logs/screen_$B.log 2>&1
rc=$?; kill $MON
echo -e "$B\t$rc\t$(( $(date +%s) - t0 ))\t$(sort -n $C/logs/gpumem_$B.txt | tail -1)\t$(hostname)" >> $C/screen/batch_timing.tsv
python -I $C/scripts/score_batch.py $C $B
rm -rf $C/screen/out/$B/boltz_results_$B/processed
python -I $C/scripts/merge_scores.py $C
grep -q missing $C/screen/batch_scores/$B.tsv && exit 1 || exit 0
