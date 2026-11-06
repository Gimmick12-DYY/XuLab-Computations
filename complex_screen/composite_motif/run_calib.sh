O=/work/users/d/y/dyy12/XuLab/complex_screen/composite_motif; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python; mkdir -p $O/logs $O/union/counts
cd $O
P=$(sbatch --parsable -J cmprep -p general -c 1 --mem=32G -t 02:00:00 -o $O/logs/prep_%j.out --wrap "cd $O; module load bedtools; $PY cm_prep.py")
A=$(sbatch --parsable --dependency=afterok:$P -J cmcnt -p general -c 1 --mem=24G -t 01:00:00 --array=1-79%40 -o $O/logs/count_%A_%a.out --wrap "cd $O; $PY cm_count.py \$(sed -n \${SLURM_ARRAY_TASK_ID}p /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt)")
C=$(sbatch --parsable --dependency=afterok:$A -J cmcal -p general -c 1 --mem=8G -t 00:30:00 -o $O/logs/calib_%j.out --wrap "cd $O; $PY cm_calibrate.py")
echo prep $P count $A calib $C
