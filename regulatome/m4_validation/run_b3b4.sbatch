#!/bin/bash
#SBATCH -J b3b4
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/logs_b3b4/b3b4_%j.out
B=/work/users/d/y/dyy12/XuLab/regulatome/m4_validation; R=$B/../r1_tftf/v2_final_v3
sha256sum -c --quiet $R/frozen_manifest_v3.sha256 && echo manifest_v3_ok_before; (cd $B && sha256sum -c --quiet manifest_ablation.sha256 && echo ablation_ok_before)
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I $B/b3b4.py; echo rc $?
sha256sum -c --quiet $R/frozen_manifest_v3.sha256 && echo manifest_v3_ok_after; (cd $B && sha256sum -c --quiet manifest_ablation.sha256 && echo ablation_ok_after)
