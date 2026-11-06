O=/work/users/d/y/dyy12/XuLab/regulatome/r2_targets; PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python
P=$(sbatch --parsable -J r2prep -p general -c 1 --mem=16G -t 01:00:00 -o $O/logs/prep_%j.out --wrap "$PY -I $O/r2_scores.py prep")
A=$(sbatch --parsable --dependency=afterok:$P --array=1-79%40 $O/r2.sbatch)
S=$(sbatch --parsable --dependency=afterok:$A -J r2sum -p general -c 1 --mem=4G -t 00:20:00 -o $O/logs/summary_%j.out --wrap "$PY -I $O/r2_summary.py")
echo prep $P array $A summary $S
