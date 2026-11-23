#!/bin/bash
#SBATCH -J drop5_all
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=24g
#SBATCH -t 01:00:00
#SBATCH -a 0-77%20
#SBATCH -o /work/users/d/y/dyy12/XuLab/downstream/share_293T_analysis/logs/drop5_%A_%a.log
# Accessibility-mask (ATAC CPM > 0.0781542, "Mdrop5") imputed matrices for every TF with a pre-mask matrix.
# Writes unified/work/<tf>/impute_atac_cpm_drop5/ ; does not touch impute/.
X=/work/users/d/y/dyy12/XuLab
mapfile -t TFS < <(ls -d $X/unified/work/*/impute/matrix_csr.npz.pre_openmask | awk -F/ "{print \$(NF-2)}")
tf=${TFS[$SLURM_ARRAY_TASK_ID]}
W=$X/unified/work/$tf
echo "task $SLURM_ARRAY_TASK_ID tf=$tf"
if [ "$tf" = ctcf ]; then echo "ctcf already built (2026-09-18), skip"; exit 0; fi
/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/python $X/downstream/apply_atac_cpm_drop5_mask.py \
  --pre-openmask $W/impute/matrix_csr.npz.pre_openmask --mm $W/mm --impute-dir $W/impute \
  --out-dir $W/impute_atac_cpm_drop5 --cpm-cache $X/unified/work/ctcf/impute_atac_cpm_drop5/bin_atac_mean_cpm.npy
