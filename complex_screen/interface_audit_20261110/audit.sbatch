#!/bin/bash
#SBATCH -J if_audit
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/interface_audit_20261110/audit_%j.log
cd /work/users/d/y/dyy12/XuLab/complex_screen/interface_audit_20261110 && /work/users/d/y/dyy12/conda/envs/snapatac2/bin/python audit.py 8
