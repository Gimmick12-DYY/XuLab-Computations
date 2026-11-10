#!/bin/bash
#SBATCH -J iph
#SBATCH -p l40-gpu,a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH --array=0-11%12
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m6_improve/repro_v1/imputed_perhalf/logs/iph_%A_%a.out
# imputed_perhalf (README.md): one TF x half per task; everything refit inside the half. usage: sbatch --nice=500 perhalf.sbatch
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; U=$X/unified; I=$X/regulatome/m6_improve/repro_v1/imputed_perhalf
TFS=(ZNF79 PCGF6 ZNF143 E4F1 SOX4 XPA); T=${TFS[$((SLURM_ARRAY_TASK_ID / 2))]}; K=$((SLURM_ARRAY_TASK_ID % 2 + 1)); t=$(echo $T | tr A-Z a-z)
WD=$I/work/${t}_h$K; CFG=$U/configs/$t.yaml; [ -f $CFG ] || CFG=$U/configs/default.yaml
[ -s $WD/impute/matrix_csr.npz.done ] && { echo done; exit 0; }
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $I/prep_half.py $T $K $WD
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /work/users/d/y/dyy12/conda/envs/unified
module load cuda 2>/dev/null || true; module load cudnn 2>/dev/null || true
cd $U
python scripts/00_ingest.py --config $CFG --work-dir $WD
python scripts/01_train.py --config $CFG --work-dir $WD
python scripts/02_impute.py --config $CFG --work-dir $WD
python scripts/02b_apply_open_chromatin_mask.py --config $CFG --work-dir $WD --open-chromatin-bed $X/data/HEK293T_ATAC_union3.bed.gz
touch $WD/impute/matrix_csr.npz.done
