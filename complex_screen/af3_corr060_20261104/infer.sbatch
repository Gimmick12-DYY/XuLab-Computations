#!/bin/bash
#SBATCH -J af3c060
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda-570.86.15
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104/logs/%x_%A_%a.out
# usage: sbatch --array=... [--mem=128G] infer.sbatch pairs_file seeds(comma) tag
source /work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104/af3_common.sh
PF=$1; SD=$2; TAG=$3; n=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $O/$PF); l=${n,,}; echo $n $(hostname) seeds $SD; nvidia-smi -L
mkdir -p $O/out_$TAG $O/scores_$TAG; [ -s $O/scores_$TAG/$n.tsv ] && { echo already scored; exit 0; }
python3 $O/make_one.py $n $SD $O/pair_in/${n}_$TAG.json || exit 1
ENVS=(--env XLA_PYTHON_CLIENT_PREALLOCATE=false --env TF_FORCE_UNIFIED_MEMORY=true --env XLA_CLIENT_MEM_FRACTION=3.2)
run_af3 --json_path=/root/af/pair_in/${n}_$TAG.json --output_dir=/root/af/out_$TAG --norun_data_pipeline --jax_compilation_cache_dir=/root/af/jaxcache; rc=$?
echo rc $rc; rm -f $O/pair_in/${n}_$TAG.json
[ $rc -eq 0 ] || exit $rc
python3 $O/score_one.py $n $O/out_$TAG/$l $O/scores_$TAG && rm -f $O/out_$TAG/$l/${l}_data.json
