#!/bin/bash
#SBATCH -J motdiag2
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103/logs/diag2_%A_%a.out
O=/work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103; cd $O; module load meme/5.5.7; d=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" diag/todo.txt); rm -rf $d/sea; sea --p $d/fg.fa --n $d/bg.fa --m db/db.meme --thresh 1000 --noseqs --oc $d/sea >/dev/null 2>&1; echo $d rc $?
